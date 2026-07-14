"""Canonical DuckDB connection setup for all GridMind storage modules."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb


@contextmanager
def connect_duckdb(
    path: Path | str, *, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection whose timestamp display and casts are always UTC."""
    connection = duckdb.connect(str(path), read_only=read_only)
    try:
        connection.execute("SET TimeZone='UTC'")
        yield connection
    finally:
        connection.close()
