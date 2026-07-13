"""Shared offline test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture()
def eia_payload() -> dict[str, Any]:
    """Load the checked-in EIA-shaped response fixture."""
    path = Path(__file__).parent / "fixtures" / "eia_region_response.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def hourly_frame() -> pd.DataFrame:
    """Create deterministic hourly test data long enough for weekly baselines."""
    timestamps = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "region": pd.Series(["PJM"] * len(timestamps), dtype="string"),
            "demand_mw": pd.Series([1000.0 + (index % 24) * 10 for index in range(240)]),
            "forecast_demand_mw": pd.Series([float("nan")] * 240, dtype="float64"),
            "net_generation_mw": pd.Series([float("nan")] * 240, dtype="float64"),
            "total_interchange_mw": pd.Series([float("nan")] * 240, dtype="float64"),
            "ingestion_timestamp_utc": pd.Timestamp("2024-02-01", tz="UTC"),
        }
    )


@pytest.fixture()
def ml_hourly_frame() -> pd.DataFrame:
    """Create deterministic two-region history long enough for default ML features."""
    timestamps = pd.date_range("2024-01-01", periods=420, freq="h", tz="UTC")
    frames = []
    for region, offset in (("PJM", 0.0), ("MISO", 300.0)):
        demand = [
            1000.0 + offset + (index % 24) * 8.0 + (index % 168) * 0.25
            for index in range(len(timestamps))
        ]
        frames.append(
            pd.DataFrame(
                {
                    "timestamp_utc": timestamps,
                    "region": pd.Series([region] * len(timestamps), dtype="string"),
                    "demand_mw": demand,
                    "forecast_demand_mw": float("nan"),
                    "net_generation_mw": float("nan"),
                    "total_interchange_mw": float("nan"),
                    "ingestion_timestamp_utc": pd.Timestamp("2024-02-01", tz="UTC"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["region", "timestamp_utc"], ignore_index=True
    )
