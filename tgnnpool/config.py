"""Load the machine and workload descriptions.

Both files are plain data.  Code reads them; code never hardcodes a capacity, a
bandwidth or a dataset size, so re-running a derivation against a different
pool size or a different cluster is a config edit, not a code edit.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


class Missing(Exception):
    """A value the configuration deliberately leaves unmeasured was requested.

    The configs carry `null` for quantities nobody has measured yet.  Reading
    one raises instead of returning None, so a derivation cannot quietly
    substitute a guess and present the result as a number.
    """


class Config(dict):
    def require(self, *path: str) -> Any:
        node: Any = self
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"no such config key: {'.'.join(path)}")
            node = node[key]
        if node is None:
            raise Missing(
                f"{'.'.join(path)} is not measured yet. "
                f"The experiment that fills it in is named in {CONFIG_DIR}."
            )
        return node

    def get_or(self, default: Any, *path: str) -> Any:
        try:
            return self.require(*path)
        except (KeyError, Missing):
            return default


def load(name: str, override: str | None = None) -> Config:
    """Load configs/<name>.yaml, or the file named by `override`."""
    path = Path(override) if override else CONFIG_DIR / f"{name}.yaml"
    env = os.environ.get(f"TGNNPOOL_{name.upper()}_CONFIG")
    if override is None and env:
        path = Path(env)
    with open(path, "r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle))


def machine(override: str | None = None) -> Config:
    return load("machine", override)


def workload(override: str | None = None) -> Config:
    return load("workload", override)


# ---------------------------------------------------------------- formatting

def gb(num_bytes: float) -> str:
    """Decimal GB, which is how the capacity numbers in this project are quoted."""
    return f"{num_bytes / 1e9:,.1f} GB"


def gib(num_bytes: float) -> str:
    return f"{num_bytes / (1 << 30):,.1f} GiB"
