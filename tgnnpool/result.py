"""One record format for every experiment and derivation.

Each run writes results/<id>-<timestamp>.json.  The `evidence` field is not
decoration: a record is either `measured` (produced by running on hardware),
`derived` (computed from measured inputs by a stated formula) or `inferred`
(an argument, with no number behind it).  A plot or a table that mixes the
three without saying so is the failure this field exists to prevent.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

EVIDENCE = ("measured", "derived", "inferred")


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class Record:
    experiment: str                  # "X1", "B2", ...
    evidence: str                    # one of EVIDENCE
    summary: str                     # one sentence a reader can check
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    host: str = field(default_factory=socket.gethostname)
    python: str = field(default_factory=platform.python_version)
    git: str = field(default_factory=_git_revision)

    def __post_init__(self) -> None:
        if self.evidence not in EVIDENCE:
            raise ValueError(f"evidence must be one of {EVIDENCE}, got {self.evidence!r}")

    def write(self, directory: Path | None = None) -> Path:
        directory = directory or RESULTS
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started_at))
        path = directory / f"{self.experiment}-{stamp}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=False, default=str)
        return path


def banner(experiment: str, title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{experiment}  {title}\n{line}")
