"""Offline Milestone 3 ingestion, training, registry, and CLI pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
from mlflow import MlflowClient
from typer.testing import CliRunner

import gridmind.cli as cli_module
from gridmind.config import Settings
from gridmind.data.eia_client import EIAClient
from gridmind.data.storage import DuckDBStorage
from gridmind.features.weather import build_weather_features
from gridmind.models.serialization import load_model_bundle
from gridmind.models.target_factory import create_target_model
from gridmind.pipelines.predict_target import run_target_prediction
from gridmind.pipelines.renewable_ingest import run_renewable_ingestion
from gridmind.pipelines.train_target import run_target_training
from gridmind.pipelines.weather_ingest import run_weather_ingestion
from gridmind.weather.client import OPEN_METEO_FIELDS, WeatherClient
from gridmind.weather.storage import WeatherStorage

runner = CliRunner()


def _mapping(path: Path) -> None:
    path.write_text(
        """version: fixture-v1
source: offline fixture
regions:
  PJM:
    aggregation: weighted_mean
    rationale: test
    locations:
      - {name: A, latitude: 40, longitude: -75, weight: 1}
""",
        encoding="utf-8",
    )


def _weather_payload(periods: int = 48) -> dict[str, Any]:
    times = (
        pd.date_range("2024-01-01", periods=periods, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    )
    hourly: dict[str, Any] = {"time": times}
    for field in OPEN_METEO_FIELDS:
        if field == "wind_direction_10m":
            hourly[field] = [90.0] * periods
        elif field in {"precipitation"}:
            hourly[field] = [0.0] * periods
        else:
            hourly[field] = [10.0 + index / 100 for index in range(periods)]
    return {"latitude": 40.0, "longitude": -75.0, "hourly": hourly}


def _regional_weather(periods: int = 260) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps,
            "weather_data_type": "forecast",
            "temperature_c": 10.0,
            "apparent_temperature_c": 9.0,
            "relative_humidity_pct": 50.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 20.0,
            "wind_speed_10m_kph": 15.0,
            "wind_direction_10m_deg": 90.0,
            "wind_direction_sin": 1.0,
            "wind_direction_cos": 0.0,
            "shortwave_radiation_wm2": [
                100.0 if 6 <= item.hour <= 18 else 0.0 for item in timestamps
            ],
            "direct_radiation_wm2": 60.0,
            "diffuse_radiation_wm2": 40.0,
            "temperature_min_c": 9.0,
            "temperature_max_c": 11.0,
            "temperature_spread_c": 2.0,
            "wind_speed_spread_kph": 1.0,
            "ingestion_timestamp_utc": pd.Timestamp("2024-02-01", tz="UTC"),
            "data_source": "fixture",
        }
    )


def test_weather_ingestion_pipeline_is_cached_and_idempotent(tmp_path: Path) -> None:
    mapping = tmp_path / "locations.yaml"
    _mapping(mapping)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_weather_payload(), request=request)

    settings = Settings(
        GRID_LOCATION_CONFIG=mapping,
        WEATHER_CACHE_DIR=tmp_path / "weather",
        DATA_QUALITY_DIR=tmp_path / "quality",
        DUCKDB_PATH=tmp_path / "grid.duckdb",
        _env_file=None,
    )
    grid_timestamps = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    DuckDBStorage(settings.duckdb_path).upsert(
        pd.DataFrame(
            {
                "timestamp_utc": grid_timestamps,
                "region": "PJM",
                "demand_mw": 100.0,
                "forecast_demand_mw": 101.0,
                "net_generation_mw": 99.0,
                "total_interchange_mw": 1.0,
                "ingestion_timestamp_utc": pd.Timestamp("2024-02-01", tz="UTC"),
            }
        )
    )
    client = WeatherClient(
        historical_url="https://weather.test",
        forecast_url="https://weather.test",
        cache_dir=settings.weather_cache_dir,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    first = run_weather_ingestion(
        settings,
        region="PJM",
        start_date="2024-01-01",
        end_date="2024-01-02",
        client=client,
    )
    second = run_weather_ingestion(
        settings,
        region="PJM",
        start_date="2024-01-01",
        end_date="2024-01-02",
        client=client,
    )
    assert first.regional_rows == second.duckdb_rows == 48
    assert second.cache_hits == 1
    assert calls == 1
    assert first.report_path.parent != first.processed_path.parent
    weather_report = json.loads(first.report_path.read_text())
    assert weather_report["grid_weather_overlap"]["grid_rows"] == 48
    assert weather_report["grid_weather_overlap"]["regional_weather_rows"] == 48
    assert weather_report["grid_weather_overlap"]["overlap_rows"] == 48


def test_eia_renewable_request_and_ingestion_pipeline(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        payload = {
            "response": {
                "total": 2,
                "data": [
                    {
                        "period": "2024-01-01T00",
                        "respondent": "PJM",
                        "fueltype": "SUN",
                        "type-name": "Solar",
                        "value": 10,
                    },
                    {
                        "period": "2024-01-01T00",
                        "respondent": "PJM",
                        "fueltype": "WND",
                        "type-name": "Wind",
                        "value": 20,
                    },
                ],
            }
        }
        return httpx.Response(200, json=payload, request=request)

    client = EIAClient(
        "secret",
        "https://eia.test/v2",
        page_size=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    fetched = client.fetch_renewable_data("PJM", "2024-01-01", "2024-01-01")
    assert len(fetched.records) == 2
    assert "/fuel-type-data/" in str(captured[0].url)
    assert set(captured[0].url.params.get_list("facets[fueltype][]")) == {"SUN", "WND"}
    assert captured[0].url.params["sort[1][column]"] == "fueltype"

    settings = Settings(
        DATA_DIR=tmp_path / "data",
        DATA_QUALITY_DIR=tmp_path / "quality",
        DUCKDB_PATH=tmp_path / "grid.duckdb",
        _env_file=None,
    )
    result = run_renewable_ingestion(
        settings,
        region="PJM",
        start_date="2024-01-01",
        end_date="2024-01-01",
        client=client,
    )
    assert result.rows == result.duckdb_rows == 1
    assert result.quarantined_rows == 0
    assert result.processed_path.exists()
    report = json.loads(result.report_path.read_text())
    assert report["retained_rows"] == 1
    assert report["renewable_target_gap_count"] == 0
    assert report["demand_renewable_overlap"]["net_load_available_rows"] == 0
    assert result.report_path.parent == settings.data_quality_dir


def test_weather_aware_target_training_and_independent_registry(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=260, freq="h", tz="UTC")
    target = pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps,
            "demand_mw": [1000.0 + index % 24 for index in range(260)],
        }
    )
    settings = Settings(
        DATA_DIR=tmp_path / "data",
        MLFLOW_TRACKING_URI=f"sqlite:///{tmp_path / 'mlflow.db'}",
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        DEMAND_WEATHER_MODEL_NAME="weather-demand-test",
        MODEL_N_JOBS=1,
        SHAP_SAMPLE_SIZE=5,
        WEATHER_LAGS="1",
        WEATHER_ROLLING_WINDOWS="3",
        _env_file=None,
    )
    result = run_target_training(
        settings,
        target="demand_mw",
        region="PJM",
        weather_mode="realistic_forecast",
        model_names=["lightgbm"],
        horizon=2,
        validation_windows=1,
        step_size=2,
        mlflow_enabled=True,
        register_model=True,
        frame=target,
        weather=_regional_weather(),
        output_dir=tmp_path / "training",
    )
    assert result.selected_model == "lightgbm_demand_mw"
    assert result.model_version == "1"
    assert result.candidate_assigned
    assert (result.artifact_dir / "leaderboard.csv").exists()
    assert (result.artifact_dir / "window_selection.json").exists()
    evaluation = result.artifact_dir / "evaluation" / "lightgbm_demand_mw"
    assert (evaluation / "validation_predictions.parquet").exists()
    assert (evaluation / "window_metrics.csv").exists()
    assert (evaluation / "horizon_metrics.csv").exists()
    assert (evaluation / "region_metrics.csv").exists()
    assert (evaluation / "target_metrics.json").exists()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    assert str(client.get_model_version_by_alias("weather-demand-test", "candidate").version) == "1"
    try:
        client.get_registered_model(settings.mlflow_model_name)
    except Exception:
        pass
    else:
        raise AssertionError("Milestone 2 demand registry name must not be created or overwritten")


def test_milestone3_cli_commands_have_help_and_delegate(tmp_path: Path, monkeypatch: Any) -> None:
    for command in (
        "weather-ingest",
        "renewables-ingest",
        "train-target",
        "predict-target",
        "target-leaderboard",
    ):
        assert runner.invoke(cli_module.app, [command, "--help"]).exit_code == 0

    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli_module,
        "run_weather_ingestion",
        lambda *_args, **_kwargs: SimpleNamespace(
            location_rows=2, regional_rows=1, cache_hits=1, report_path=tmp_path / "weather.json"
        ),
    )
    weather = runner.invoke(
        cli_module.app,
        [
            "weather-ingest",
            "--region",
            "PJM",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
        ],
    )
    assert weather.exit_code == 0
    assert "cache hits: 1" in weather.output

    monkeypatch.setattr(
        cli_module,
        "run_renewable_ingestion",
        lambda *_args, **_kwargs: SimpleNamespace(
            rows=1, quarantined_rows=0, duckdb_rows=1, report_path=tmp_path / "renewable.json"
        ),
    )
    renewable = runner.invoke(
        cli_module.app,
        [
            "renewables-ingest",
            "--region",
            "PJM",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-01",
        ],
    )
    assert renewable.exit_code == 0
    assert "Renewable rows: 1" in renewable.output


def test_target_prediction_pipeline_persists_shared_contract(
    tmp_path: Path, hourly_frame: pd.DataFrame
) -> None:
    database = tmp_path / "grid.duckdb"
    history = hourly_frame.iloc[:80].copy()
    DuckDBStorage(database).upsert(history)
    weather = _regional_weather(82)
    WeatherStorage(database).upsert_regions(weather)
    settings = Settings(
        DUCKDB_PATH=database,
        WEATHER_LAGS="1",
        WEATHER_ROLLING_WINDOWS="3",
        _env_file=None,
    )
    built = build_weather_features(
        weather,
        lags=settings.weather_lags,
        rolling_windows=settings.weather_rolling_windows,
    )
    training = history.merge(
        built.frame[["region", "timestamp_utc", *built.feature_names]],
        on=["region", "timestamp_utc"],
    )
    model = create_target_model(
        "lightgbm",
        "demand_mw",
        weather_features=built.feature_names,
        lags=(1, 24),
        rolling_windows=(3, 24),
        n_jobs=1,
        params={"n_estimators": 5},
    ).fit(training)
    bundle_path = model.save(tmp_path / "bundle.joblib", metadata={"regions": ["PJM"]})
    result = run_target_prediction(
        settings,
        target="demand_mw",
        region="PJM",
        horizon=2,
        bundle=load_model_bundle(bundle_path),
        output_dir=tmp_path / "predictions",
    )
    assert result.duckdb_rows == 2
    assert result.parquet_path.exists()
    assert set(result.forecasts["target"]) == {"demand_mw"}
    repeated = run_target_prediction(
        settings,
        target="demand_mw",
        region="PJM",
        horizon=2,
        bundle=load_model_bundle(bundle_path),
        output_dir=tmp_path / "predictions",
    )
    assert repeated.duckdb_rows == 2


def test_direct_and_component_net_load_share_one_leaderboard(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=260, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "region": "PJM",
            "timestamp_utc": timestamps,
            "demand_mw": [1000.0 + index % 24 for index in range(260)],
            "solar_generation_mw": [
                max(0.0, 50.0 - abs(12 - item.hour) * 5) for item in timestamps
            ],
            "wind_generation_mw": [20.0 + index % 5 for index in range(260)],
        }
    )
    frame["total_renewable_generation_mw"] = (
        frame["solar_generation_mw"] + frame["wind_generation_mw"]
    )
    frame["net_load_mw"] = frame["demand_mw"] - frame["total_renewable_generation_mw"]
    settings = Settings(
        MODEL_N_JOBS=1,
        SHAP_SAMPLE_SIZE=5,
        WEATHER_LAGS="1",
        WEATHER_ROLLING_WINDOWS="3",
        _env_file=None,
    )
    result = run_target_training(
        settings,
        target="net_load_mw",
        region="PJM",
        model_names=["lightgbm"],
        horizon=2,
        validation_windows=1,
        step_size=2,
        mlflow_enabled=False,
        register_model=False,
        frame=frame,
        weather=_regional_weather(),
        output_dir=tmp_path / "net-load",
    )
    assert {"lightgbm_net_load_mw", "component_demand_solar_wind"}.issubset(
        set(result.leaderboard["model_name"])
    )
