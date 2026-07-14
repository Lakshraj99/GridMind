"""Offline anomaly pipeline, MLflow, and CLI integration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from mlflow import MlflowClient
from typer.testing import CliRunner

import gridmind.cli as cli_module
from gridmind.alerts.storage import AlertStorage
from gridmind.anomalies.contracts import make_anomaly
from gridmind.anomalies.storage import AnomalyStorage
from gridmind.config import Settings
from gridmind.data.storage import DuckDBStorage
from gridmind.pipelines.backtest_anomalies import run_anomaly_backtest
from gridmind.pipelines.detect_anomalies import run_anomaly_detection

runner = CliRunner()


def _store_history(path: Path, periods: int = 180) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    demand = [1000.0 + (index % 24) * 3 + (index % 7) for index in range(periods)]
    demand[145] = 2000.0
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "region": "PJM",
            "demand_mw": demand,
            "forecast_demand_mw": float("nan"),
            "net_generation_mw": float("nan"),
            "total_interchange_mw": float("nan"),
            "ingestion_timestamp_utc": pd.Timestamp("2025-02-01", tz="UTC"),
        }
    )
    DuckDBStorage(path).upsert(frame)
    return frame


def test_detection_pipeline_persists_events_alerts_detector_bundle_and_artifacts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "grid.duckdb"
    frame = _store_history(database)
    settings = Settings(
        DUCKDB_PATH=database,
        ANOMALY_LOOKBACK_HOURS=100,
        ANOMALY_MIN_TRAINING_ROWS=48,
        ANOMALY_CONTAMINATION=0.05,
        MLFLOW_ENABLED=False,
        _env_file=None,
    )
    result = run_anomaly_detection(
        settings,
        region="PJM",
        targets=("demand_mw",),
        start_date=frame["timestamp_utc"].iloc[120].isoformat(),
        end_date=frame["timestamp_utc"].iloc[160].isoformat(),
        detectors=("rules", "residual", "isolation_forest"),
        mlflow_enabled=False,
        artifact_root=tmp_path / "anomalies",
    )
    assert result.rows_evaluated == 41
    assert not result.anomalies.empty
    assert result.mlflow_run_id is None
    assert result.detector_report["demand_mw"]["residual_anomalies"] == 0
    assert result.detector_report["demand_mw"]["isolation_training_rows"] >= 48
    assert (result.artifact_dir / "demand_mw_detector.joblib").exists()
    for filename in (
        "anomaly_events.parquet",
        "anomaly_summary.json",
        "detector_metrics.json",
        "feature_schema.json",
        "threshold_report.json",
        "alert_summary.json",
        "anomaly_rate_report.csv",
        "lifecycle_summary.json",
    ):
        assert (result.artifact_dir / filename).exists()
    assert AnomalyStorage(database).count() == result.anomaly_rows
    assert AlertStorage(database).count() >= 1
    event_count = AnomalyStorage(database).count()
    alert_count = AlertStorage(database).count()
    history_count = len(AlertStorage(database).read_history())
    rerun = run_anomaly_detection(
        settings,
        region="PJM",
        targets=("demand_mw",),
        start_date=frame["timestamp_utc"].iloc[120].isoformat(),
        end_date=frame["timestamp_utc"].iloc[160].isoformat(),
        detectors=("rules", "residual", "isolation_forest"),
        mlflow_enabled=False,
        artifact_root=tmp_path / "anomalies",
    )
    assert AnomalyStorage(database).count() == event_count
    assert AlertStorage(database).count() == alert_count
    assert len(AlertStorage(database).read_history()) == history_count
    assert rerun.alerts_opened == 0
    assert rerun.alerts_updated == 0
    assert rerun.alerts_unchanged > 0
    assert set(rerun.anomaly_rate_report).issuperset(
        {
            "target",
            "detector_name",
            "anomaly_type",
            "severity",
            "day_utc",
            "evaluated_rows",
            "event_count",
            "anomaly_rate",
            "alerts_opened",
            "alerts_updated",
            "alerts_unchanged",
            "effective_isolation_rate",
        }
    )


def test_offline_backtest_writes_metrics_without_mutating_production_data(tmp_path: Path) -> None:
    database = tmp_path / "grid.duckdb"
    original = _store_history(database)
    settings = Settings(DUCKDB_PATH=database, MLFLOW_ENABLED=False, _env_file=None)
    result = run_anomaly_backtest(
        settings,
        region="PJM",
        target="demand_mw",
        start_date="2025-01-01",
        end_date="2025-01-07",
        seed=42,
        mlflow_enabled=False,
        artifact_root=tmp_path / "backtests",
    )
    assert result.injected_count == 6
    assert 0 <= result.metrics["precision"] <= 1
    assert 0 <= result.metrics["recall"] <= 1
    assert 0 <= result.metrics["f1"] <= 1
    assert (result.artifact_dir / "injected_anomalies.parquet").exists()
    assert (result.artifact_dir / "per_type_metrics.csv").exists()
    persisted = DuckDBStorage(database).read_data(
        regions=["PJM"],
        start_date="2025-01-01",
        end_date=original["timestamp_utc"].iloc[-1].isoformat(),
    )
    assert len(persisted) == len(original)
    assert persisted["demand_mw"].max() == original["demand_mw"].max()


def test_anomaly_mlflow_run_logs_offline_artifacts(tmp_path: Path) -> None:
    database = tmp_path / "grid.duckdb"
    frame = _store_history(database)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(
        DUCKDB_PATH=database,
        MLFLOW_TRACKING_URI=tracking_uri,
        MLFLOW_ARTIFACT_ROOT=tmp_path / "mlartifacts",
        MLFLOW_ENABLED=True,
        _env_file=None,
    )
    result = run_anomaly_detection(
        settings,
        region="PJM",
        targets=("demand_mw",),
        start_date=frame["timestamp_utc"].iloc[140].isoformat(),
        end_date=frame["timestamp_utc"].iloc[150].isoformat(),
        detectors=("rules",),
        mlflow_enabled=True,
        artifact_root=tmp_path / "anomalies",
    )
    assert result.mlflow_run_id
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(str(result.mlflow_run_id))
    assert run.data.params["region"] == "PJM"
    assert run.data.metrics["rows_evaluated"] == 11
    artifacts = client.list_artifacts(str(result.mlflow_run_id), "anomaly_detection")
    assert {item.path.rsplit("/", 1)[-1] for item in artifacts}.issuperset(
        {"anomaly_events.parquet", "anomaly_summary.json"}
    )


def test_anomaly_cli_help_success_invalid_and_failure_paths(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = Settings(DUCKDB_PATH=tmp_path / "grid.duckdb", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    for command in ("detect-anomalies", "anomaly-backtest", "anomalies", "alerts", "alert-update"):
        result = runner.invoke(cli_module.app, [command, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output
    fake = SimpleNamespace(
        rows_evaluated=10,
        anomalies=pd.DataFrame({"severity": ["warning"], "detector_name": ["rules"]}),
        alerts_opened=1,
        alerts_updated=0,
        alerts_unchanged=3,
        alerts_auto_resolved=0,
        lifecycle_counts={"acknowledged": 0, "resolved": 0, "suppressed": 0},
        detector_report={},
        artifact_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_anomaly_detection", lambda *_args, **_kwargs: fake
    )
    success = runner.invoke(
        cli_module.app,
        [
            "detect-anomalies",
            "--region",
            "PJM",
            "--targets",
            "demand_mw",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--no-mlflow",
        ],
    )
    assert success.exit_code == 0
    assert "Rows evaluated: 10" in success.output
    assert "opened=1; updated=0; unchanged=3" in success.output
    assert runner.invoke(cli_module.app, ["alerts", "--severity", "severe"]).exit_code != 0

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid detector")

    monkeypatch.setattr(cli_module, "run_anomaly_detection", fail)  # type: ignore[attr-defined]
    failed = runner.invoke(
        cli_module.app,
        [
            "detect-anomalies",
            "--region",
            "PJM",
            "--targets",
            "bad",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
        ],
    )
    assert failed.exit_code == 1
    assert "invalid detector" in failed.output


def test_backtest_listing_and_alert_update_cli_success_paths(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = Settings(DUCKDB_PATH=tmp_path / "grid.duckdb", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    fake_backtest = SimpleNamespace(
        injected_count=2,
        detected_count=1,
        metrics={
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
            "false_positives_per_day": 0.0,
            "mean_detection_delay_hours": 1.0,
        },
        artifact_dir=tmp_path / "backtest",
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_anomaly_backtest", lambda *_args, **_kwargs: fake_backtest
    )
    backtest = runner.invoke(
        cli_module.app,
        [
            "anomaly-backtest",
            "--region",
            "PJM",
            "--target",
            "demand_mw",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--no-mlflow",
        ],
    )
    assert backtest.exit_code == 0
    assert "Synthetic labels" in backtest.output

    event = pd.DataFrame(
        [
            make_anomaly(
                region="PJM",
                target="demand_mw",
                timestamp="2025-01-01T00:00:00Z",
                detector_name="rules",
                anomaly_type="demand_spike",
                anomaly_score=45,
                severity="warning",
                explanation="spike",
                detected_at="2025-01-01T00:00:00Z",
            )
        ]
    )
    AnomalyStorage(settings.duckdb_path).upsert(event)
    alerts = AlertStorage(settings.duckdb_path)
    from gridmind.alerts.lifecycle import AlertManager

    AlertManager(alerts).process(event)
    listed = runner.invoke(cli_module.app, ["anomalies", "--region", "PJM"])
    assert listed.exit_code == 0
    assert "2025-01-01T00:00:00Z" in listed.output
    csv_path = tmp_path / "alerts.csv"
    listed_alerts = runner.invoke(
        cli_module.app, ["alerts", "--status", "open", "--csv", str(csv_path)]
    )
    assert listed_alerts.exit_code == 0
    assert csv_path.exists()
    alert_id = str(alerts.read_alerts().iloc[0]["alert_id"])
    updated = runner.invoke(
        cli_module.app,
        ["alert-update", "--alert-id", alert_id, "--status", "acknowledged"],
    )
    assert updated.exit_code == 0
    assert "acknowledged" in updated.output
