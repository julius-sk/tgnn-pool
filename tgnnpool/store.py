"""A versioned object store with the publish protocol the pool runtime uses.

The runtime publishes an object in four ordered passes, each followed by a
write-back and a fence:

    1. state  -> WRITING
    2. payload
    3. version and checksum
    4. state  -> READY

Only after pass 4 is the object observable as complete.  Two backends implement
this interface:

  MockStore  in-process, no hardware, with explicit interleaving points so a
             concurrent reader can be made to observe every intermediate state.
             Everything in B1 and the default mode of X1 runs on it.
  PoolStore  the real pool, via cxl_shm.  Same interface, same pass order.

Readers come in three strengths, and the difference between them is the whole
point of B1 and X1:

  NAIVE      reads whatever bytes are there.  Can observe a torn payload.
  COMMITTED  reads only objects whose state is READY.  Cannot tear, but can
             observe an object published after the reader's logical time -- in
             a temporal graph that is future information, which is a
             correctness bug, not a performance one.
  SNAPSHOT   reads only objects that are READY and carry version <= t.
"""
from __future__ import annotations

import threading
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterator


class State(IntEnum):
    EMPTY = 0
    WRITING = 1
    READY = 2
    RETIRED = 3


class Reader(IntEnum):
    NAIVE = 0
    COMMITTED = 1
    SNAPSHOT = 2


class TornRead(Exception):
    """The payload did not match its checksum."""


class UseAfterRetire(Exception):
    """A reader followed a reference to an object that was reclaimed.

    This is the third precondition of B1 and it is NOT currently enforced by
    the runtime: the API offers retire, and nothing stops a reader that is
    holding an offset from dereferencing it afterwards.
    """


@dataclass
class Obj:
    name: str
    state: State = State.EMPTY
    version: int = 0
    checksum: int = 0
    payload: bytes = b""


class MockStore:
    """In-process store that reproduces the publish protocol's pass order.

    `hook` is called between passes with the pass number just completed.  A
    test uses it to run the reader at a chosen interleaving instead of relying
    on thread scheduling, which makes the result reproducible.
    """

    def __init__(self, hook: Callable[[int, str], None] | None = None) -> None:
        self._objs: dict[str, Obj] = {}
        self._lock = threading.Lock()   # protects the dict, not the objects
        self.hook = hook or (lambda pass_no, name: None)
        self.torn_reads = 0
        self.future_reads = 0

    # ------------------------------------------------------------- writer

    def publish(self, name: str, payload: bytes, version: int) -> None:
        with self._lock:
            obj = self._objs.setdefault(name, Obj(name=name))

        obj.state = State.WRITING                      # pass 1
        self.hook(1, name)

        obj.payload = payload                          # pass 2
        self.hook(2, name)

        obj.version = version                          # pass 3
        obj.checksum = zlib.crc32(payload)
        self.hook(3, name)

        obj.state = State.READY                        # pass 4
        self.hook(4, name)

    def publish_steps(self, name: str, payload: bytes, version: int):
        """Same protocol as publish, but as a generator that yields after each
        pass.  A caller that advances this generator by hand can interleave the
        writer with a reader at any point; B1's checker enumerates every such
        interleaving.  publish() and publish_steps() must stay in lockstep --
        there is exactly one definition of the pass order and this is it."""
        with self._lock:
            obj = self._objs.setdefault(name, Obj(name=name))

        obj.state = State.WRITING
        yield 1
        obj.payload = payload
        yield 2
        obj.version = version
        obj.checksum = zlib.crc32(payload)
        yield 3
        obj.state = State.READY
        yield 4

    def revise(self, name: str, payload: bytes) -> None:
        """Rewrite an object that is already READY, in place.

        Nothing in the snapshot workload does this -- published snapshots are
        append-only.  B1's third case turns it on to show that the isolation
        argument depends on that and collapses without it (late-arriving edges
        for a past snapshot, or time-varying features, would both do this)."""
        with self._lock:
            obj = self._objs[name]
        obj.payload = payload
        obj.checksum = zlib.crc32(payload)

    def publish_torn(self, name: str, payload: bytes, version: int,
                     stop_after_pass: int) -> None:
        """Publish and stop, modelling a writer that died mid-publish."""
        with self._lock:
            obj = self._objs.setdefault(name, Obj(name=name))
        obj.state = State.WRITING
        if stop_after_pass < 2:
            return
        obj.payload = payload
        if stop_after_pass < 3:
            return
        obj.version = version
        obj.checksum = zlib.crc32(payload)

    def retire(self, name: str) -> None:
        with self._lock:
            obj = self._objs.get(name)
        if obj is not None:
            obj.state = State.RETIRED
            obj.payload = b"\xde\xad\xbe\xef" * (max(1, len(obj.payload) // 4))

    # ------------------------------------------------------------- reader

    def read(self, name: str, mode: Reader, at_version: int | None = None) -> bytes | None:
        with self._lock:
            obj = self._objs.get(name)
        if obj is None:
            return None

        if obj.state is State.RETIRED:
            raise UseAfterRetire(
                f"{name} was retired while a reader held a reference to it"
            )

        if mode is Reader.NAIVE:
            payload = obj.payload
            if obj.state is not State.READY or zlib.crc32(payload) != obj.checksum:
                self.torn_reads += 1
                raise TornRead(f"{name}: payload does not match its checksum")
            return payload

        if obj.state is not State.READY:
            return None                     # not yet publishable, skip it

        if mode is Reader.COMMITTED:
            if at_version is not None and obj.version > at_version:
                self.future_reads += 1      # observed, and deliberately allowed
            return obj.payload

        if at_version is not None and obj.version > at_version:
            return None                     # SNAPSHOT: invisible, as intended
        return obj.payload

    def names(self) -> Iterator[str]:
        with self._lock:
            return iter(sorted(self._objs))


class PoolStore:
    """The same interface over cxl_shm.  Imported lazily; cluster only.

    The pass order below must stay identical to MockStore.publish.  If the two
    drift, B1's argument is being checked against something the hardware does
    not do.
    """

    def __init__(self, pool_name_prefix: str = "tgnnpool") -> None:
        try:
            import cxl_shm  # noqa: F401
        except ImportError as exc:        # pragma: no cover - cluster only
            raise RuntimeError(
                "PoolStore needs cxl_shm.  On a laptop use MockStore; the "
                "derivations and X1's default mode never touch the pool."
            ) from exc
        self.prefix = pool_name_prefix
        raise NotImplementedError(
            "PoolStore is a declared interface, not a finished backend.  "
            "Filling it in is step 1 of X1 on the cluster; see docs/X1-leakage.md "
            "for the four calls it has to make and the two it must never make "
            "(cxl_shm_init and cxl_shm_finalize both wipe the pool)."
        )
