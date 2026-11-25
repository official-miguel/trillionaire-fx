"""SQLite engine factory shared by every local database."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool


def sqlite_engine(url: str) -> Engine:
    """Thread-safe SQLite engine.

    In-memory databases use ONE shared connection (otherwise every thread of the web
    server would see its own empty database); file databases use normal pooling.
    """
    args = {"check_same_thread": False}
    if url in ("sqlite://", "sqlite:///:memory:") or url.endswith(":memory:"):
        return create_engine(url, future=True, connect_args=args, poolclass=StaticPool)
    return create_engine(url, future=True, connect_args=args)
