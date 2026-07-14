"""UTC DuckDB sessions and target-prediction timezone regressions."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from gridmind.config import Settings
from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.data.storage import DuckDBStorage
from gridmind.data.target_storage import TargetForecastStorage
from gridmind.exceptions import InsufficientHistoryError
from gridmind.models.serialization import load_model_bundle
from gridmind.models.target_factory import TargetForecaster, create_target_model
from gridmind.pipelines.predict_target import run_target_prediction
from gridmind.weather.schemas import REGION_WEATHER_COLUMNS
from gridmind.weather.storage import WeatherStorage

ORIGIN = pd.Timestamp("2026-07-14T05:00:00Z")


def _grid_history(*, regions: tuple[str, ...] = ("PJM",)) -> pd.DataFrame:
    timestamps = pd.date_range(end=ORIGIN, periods=200, freq="h", tz="UTC")
    frames: list[pd.DataFrame] = []
    for region_index, region in enumerate(regions):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp_utc": timestamps,
                    "region": region,
                    "demand_mw": [
                        1000.0 + region_index * 100 + index % 24 for index in range(len(timestamps))
                    ],
                    "forecast_demand_mw": float("nan"),
                    "net_generation_mw": float("nan"),
                    "total_interchange_mw": float("nan"),
                    "ingestion_timestamp_utc": pd.Timestamp("2026-07-14T05:05:00Z"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _forecast_weather(
    *,
    regions: tuple[str, ...] = ("PJM",),
    missing: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    # Use the provider's Asia/Kolkata representation deliberately. The first
    # required instant displays as 11:30+05:30 but is canonically 06:00Z.
    timestamps = pd.date_range("2026-07-14T00:00:00Z", periods=48, freq="h").tz_convert(
        "Asia/Kolkata"
    )
    rows: list[dict[str, object]] = []
    for region in regions:
        for timestamp in timestamps:
            if missing and region in missing and timestamp == missing[region]:
                continue
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "region": region,
                    "weather_data_type": "forecast",
                    "temperature_c": 25.0,
                    "apparent_temperature_c": 26.0,
                    "relative_humidity_pct": 55.0,
                    "precipitation_mm": 0.0,
                    "cloud_cover_pct": 20.0,
                    "wind_speed_10m_kph": 10.0,
                    "wind_direction_10m_deg": 90.0,
                    "wind_direction_sin": 1.0,
                    "wind_direction_cos": 0.0,
                    "shortwave_radiation_wm2": 100.0,
                    "direct_radiation_wm2": 60.0,
                    "diffuse_radiation_wm2": 40.0,
                    "temperature_min_c": 24.0,
                    "temperature_max_c": 26.0,
                    "temperature_spread_c": 2.0,
                    "wind_speed_spread_kph": 1.0,
                    "ingestion_timestamp_utc": pd.Timestamp("2026-07-14T05:05:00Z"),
                    "data_source": "fixture",
                }
            )
    return pd.DataFrame(rows, columns=REGION_WEATHER_COLUMNS)


def _fitted_model(history: pd.DataFrame) -> TargetForecaster:
    training = history[["region", "timestamp_utc", "demand_mw"]].copy()
    training["temperature_c"] = 20.0
    return create_target_model(
        "lightgbm",
        "demand_mw",
        weather_features=("temperature_c",),
        lags=(1, 24),
        rolling_windows=(3, 24),
        n_jobs=1,
        params={"n_estimators": 5},
    ).fit(training)


@pytest.mark.parametrize("session_timezone", ["Asia/Kolkata", "UTC"])
def test_connection_helper_forces_utc_and_preserves_instant(
    tmp_path: Path, session_timezone: str
) -> None:
    database = tmp_path / "timezone.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(f"SET TimeZone='{session_timezone}'")
        connection.execute("CREATE TABLE instants (timestamp_utc TIMESTAMPTZ)")
        connection.execute("INSERT INTO instants VALUES (TIMESTAMPTZ '2026-07-14 11:30:00+05:30')")
        displayed = connection.execute("SELECT timestamp_utc FROM instants").fetchone()
        assert displayed is not None
        expected_hour = 11 if session_timezone == "Asia/Kolkata" else 6
        assert displayed[0].hour == expected_hour

    with connect_duckdb(database, read_only=True) as connection:
        assert connection.execute("SELECT current_setting('TimeZone')").fetchone() == ("UTC",)
        value = connection.execute("SELECT timestamp_utc FROM instants").fetchone()
        assert value is not None
        assert pd.to_datetime(value[0], utc=True) == pd.Timestamp("2026-07-14T06:00:00Z")


def test_realistic_prediction_matches_kolkata_weather_and_persists_idempotently(
    tmp_path: Path,
) -> None:
    database = tmp_path / "gridmind.duckdb"
    history = _grid_history()
    DuckDBStorage(database).upsert(history)
    WeatherStorage(database).upsert_regions(_forecast_weather())
    with duckdb.connect(str(database), read_only=True) as connection:
        connection.execute("SET TimeZone='Asia/Kolkata'")
        displayed = connection.execute(
            "SELECT timestamp_utc FROM hourly_region_weather "
            "WHERE region='PJM' AND timestamp_utc=TIMESTAMPTZ '2026-07-14 06:00:00+00:00'"
        ).fetchone()
        assert displayed is not None
        assert displayed[0].isoformat() == "2026-07-14T11:30:00+05:30"

    model = _fitted_model(history)
    bundle_path = model.save(tmp_path / "bundle.joblib", metadata={"regions": ["PJM"]})
    settings = Settings(
        DUCKDB_PATH=database,
        WEATHER_LAGS="1,24",
        WEATHER_ROLLING_WINDOWS="3,24",
        _env_file=None,
    )
    first = run_target_prediction(
        settings,
        target="demand_mw",
        region="PJM",
        horizon=24,
        bundle=load_model_bundle(bundle_path),
        output_dir=tmp_path / "predictions",
    )
    second = run_target_prediction(
        settings,
        target="demand_mw",
        region="PJM",
        horizon=24,
        bundle=load_model_bundle(bundle_path),
        output_dir=tmp_path / "predictions",
    )

    assert first.forecasts["forecast_origin"].iloc[0] == ORIGIN
    assert first.forecasts["timestamp_utc"].iloc[0] == pd.Timestamp("2026-07-14T06:00:00Z")
    assert str(first.forecasts["timestamp_utc"].dt.tz) == "UTC"
    assert str(first.forecasts["created_at_utc"].dt.tz) == "UTC"
    assert first.duckdb_rows == second.duckdb_rows == 24
    stored = TargetForecastStorage(database).read(target="demand_mw")
    assert len(stored) == 24
    assert str(stored["forecast_origin"].dt.tz) == "UTC"


def test_genuine_missing_weather_lists_exact_utc_hour_and_isolates_region() -> None:
    histories = _grid_history(regions=("PJM", "MISO"))
    future = _forecast_weather(
        regions=("PJM", "MISO"),
        missing={"MISO": pd.Timestamp("2026-07-14T07:00:00Z").tz_convert("Asia/Kolkata")},
    )
    source = histories[["region", "timestamp_utc", "demand_mw"]].merge(
        future[["region", "timestamp_utc", "temperature_c"]],
        on=["region", "timestamp_utc"],
        how="outer",
    )
    model = create_target_model(
        "lightgbm",
        "demand_mw",
        weather_features=("temperature_c",),
        lags=(1, 24),
        rolling_windows=(3, 24),
        n_jobs=1,
        params={"n_estimators": 5},
    )

    with pytest.raises(InsufficientHistoryError) as captured:
        model.predict(source, horizon=24)
    message = str(captured.value)
    assert "for MISO" in message
    assert "forecast origin=2026-07-14T05:00:00Z" in message
    assert "required start=2026-07-14T06:00:00Z" in message
    assert "required end=2026-07-15T05:00:00Z" in message
    assert "required rows=24, matched rows=23" in message
    assert "missing UTC timestamps=[2026-07-14T07:00:00Z]" in message
    assert "for PJM" not in message
