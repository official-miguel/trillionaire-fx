"""Minimal ``.env`` loader (no extra dependency).

Reads ``KEY=VALUE`` lines from the project's ``.env`` into ``os.environ`` without
overriding variables that are already set. Values are never logged or printed.
"""

from __future__ import annotations

import os
from pathlib import Path

from tfx.config import PROJECT_ROOT


def load_env(path: Path | None = None) -> list[str]:
    """Load ``path`` (default: ``<project>/.env``). Returns the names (not values) set."""
    p = path or PROJECT_ROOT / ".env"
    if not p.exists():
        return []
    loaded: list[str] = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
