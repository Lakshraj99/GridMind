"""Command-level tests for user-facing Typer behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

import gridmind.cli as cli_module
from gridmind.config import Settings
from gridmind.data.storage import write_processed_parquet
from gridmind.exceptions import ConfigurationError
from gridmind.pipelines.ingest import IngestionResult

runner = CliRunner()


def test_root_and_command_help() -> None:
    assert runner.invoke(cli_module.app, ["--help"]).exit_code == 0
    for command in (
        "ingest",
        "validate",
        "baseline",
        "inspect",
        "train",
        "predict",
        "leaderboard",
        "explain",
        "weather-ingest",
        "renewables-ingest",
        "train-target",
        "predict-target",
        "target-leaderboard",
    ):
        result = runner.invoke(cli_module.app, [command, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output


def test_ingest_success_and_domain_failure(tmp_path: Path, monkeypatch: object) -> None:
    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    ingestion_result = IngestionResult(
        rows=2,
        raw_path=tmp_path / "raw.json",
        processed_path=tmp_path / "processed",
        quality_report_path=tmp_path / "quality.json",
        duckdb_rows=2,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_ingestion", lambda *_args, **_kwargs: ingestion_result
    )
    result = runner.invoke(cli_module.app, ["ingest"])
    assert result.exit_code == 0
    assert "Ingested 2 rows" in result.output

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ConfigurationError("missing dates")

    monkeypatch.setattr(cli_module, "run_ingestion", fail)  # type: ignore[attr-defined]
    result = runner.invoke(cli_module.app, ["ingest"])
    assert result.exit_code == 1
    assert "missing dates" in result.output


def test_ingest_drop_policy_prints_quarantine_warning(tmp_path: Path, monkeypatch: object) -> None:
    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    captured: dict[str, object] = {}
    ingestion_result = IngestionResult(
        rows=2,
        raw_path=tmp_path / "raw.json",
        processed_path=tmp_path / "processed",
        quality_report_path=tmp_path / "quality.json",
        duckdb_rows=2,
        quarantined_rows=1,
        quarantine_path=tmp_path / "quarantine.parquet",
    )

    def run(*_args: object, **kwargs: object) -> IngestionResult:
        captured.update(kwargs)
        return ingestion_result

    monkeypatch.setattr(cli_module, "run_ingestion", run)  # type: ignore[attr-defined]
    result = runner.invoke(cli_module.app, ["ingest", "--missing-demand-policy", "drop"])
    assert result.exit_code == 0
    assert captured["missing_demand_policy"] == "drop"
    assert "WARNING: Quarantined and excluded 1 rows" in result.output
    invalid = runner.invoke(cli_module.app, ["ingest", "--missing-demand-policy", "interpolate"])
    assert invalid.exit_code != 0


def test_validate_success_and_failure(
    tmp_path: Path, hourly_frame: pd.DataFrame, monkeypatch: object
) -> None:
    settings = Settings(
        DATA_DIR=tmp_path / "data",
        DATA_QUALITY_DIR=tmp_path / "artifacts" / "data_quality",
        _env_file=None,
    )
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    directory = write_processed_parquet(hourly_frame.iloc[:24], tmp_path / "processed")
    result = runner.invoke(cli_module.app, ["validate", "--processed-dir", str(directory)])
    assert result.exit_code == 0
    assert "Validated 24 rows" in result.output
    assert not list(directory.rglob("*.json"))
    assert len(list(settings.data_quality_dir.glob("validation_data_quality_report_*.json"))) == 1

    result = runner.invoke(
        cli_module.app, ["validate", "--processed-dir", str(tmp_path / "missing")]
    )
    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_validate_wraps_invalid_parquet_without_internal_traceback(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = Settings(DATA_QUALITY_DIR=tmp_path / "artifacts" / "data_quality", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    directory = tmp_path / "processed"
    directory.mkdir()
    (directory / "broken.parquet").write_text("not parquet", encoding="utf-8")
    result = runner.invoke(cli_module.app, ["validate", "--processed-dir", str(directory)])
    assert result.exit_code == 1
    assert "Validation failed: Could not read processed Parquet data" in result.output
    assert "ArrowInvalid" not in result.output
    assert "Traceback" not in result.output


def test_baseline_and_inspect_commands(
    hourly_frame: pd.DataFrame, tmp_path: Path, monkeypatch: object
) -> None:
    settings = Settings(DUCKDB_PATH=tmp_path / "grid.duckdb", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]

    class FakeStorage:
        def __init__(self, _path: Path) -> None:
            pass

        def read_region(self, *_args: object) -> pd.DataFrame:
            return hourly_frame

        def inspect(self) -> dict[str, object]:
            return {
                "row_count": 240,
                "date_range": {
                    "start": "2023-01-01T05:30:00+05:30",
                    "end": "2026-01-01T04:30:00+05:30",
                },
                "missing_demand_count": 0,
                "regions": ["PJM"],
            }

    monkeypatch.setattr(cli_module, "DuckDBStorage", FakeStorage)  # type: ignore[attr-defined]
    pipeline_result = SimpleNamespace(
        leaderboard=pd.DataFrame([{"model_name": "last_value", "mae": 1.0}]),
        metrics_path=tmp_path / "metrics.json",
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_baseline_pipeline", lambda *_args, **_kwargs: pipeline_result
    )

    baseline = runner.invoke(cli_module.app, ["baseline", "--no-mlflow"])
    assert baseline.exit_code == 0
    assert "last_value" in baseline.output
    inspected = runner.invoke(cli_module.app, ["inspect"])
    assert inspected.exit_code == 0
    assert "Rows: 240" in inspected.output
    assert "Regions: PJM" in inspected.output
    assert "2023-01-01T00:00:00Z to 2025-12-31T23:00:00Z" in inspected.output
    assert "+05:30" not in inspected.output


def test_baseline_empty_data_returns_nonzero(tmp_path: Path, monkeypatch: object) -> None:
    settings = Settings(DUCKDB_PATH=tmp_path / "grid.duckdb", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]

    class EmptyStorage:
        def __init__(self, _path: Path) -> None:
            pass

        def read_region(self, *_args: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(cli_module, "DuckDBStorage", EmptyStorage)  # type: ignore[attr-defined]
    result = runner.invoke(cli_module.app, ["baseline"])
    assert result.exit_code == 1
    assert "No stored data" in result.output


def test_milestone_two_commands(tmp_path: Path, monkeypatch: object) -> None:
    settings = Settings(DATA_DIR=tmp_path / "data", _env_file=None)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)  # type: ignore[attr-defined]
    training = SimpleNamespace(
        leaderboard=pd.DataFrame([{"rank": 1, "model_name": "lightgbm_global"}]),
        selected_model="lightgbm_global",
        promotion=SimpleNamespace(champion_promoted=True, reason="passed"),
        bundle_path=tmp_path / "model.joblib",
        artifact_dir=tmp_path / "training",
        window_selection_path=tmp_path / "training" / "window_selection.json",
    )

    def train(*_args: object, **kwargs: object) -> SimpleNamespace:
        progress = kwargs["progress"]
        progress("Continuity: regions=1; timestamp gaps=2; segments=3; longest segment=100 hours")
        progress("Selected tuning origins: none")
        progress("Selected final evaluation origins: 2024-01-01T00:00:00Z")
        return training

    monkeypatch.setattr(cli_module, "run_training_pipeline", train)  # type: ignore[attr-defined]
    trained = runner.invoke(
        cli_module.app,
        ["train", "--region", "all", "--models", "lightgbm", "--no-mlflow"],
    )
    assert trained.exit_code == 0
    assert "Candidate model: lightgbm_global" in trained.output
    assert "timestamp gaps=2" in trained.output
    assert "Selected final evaluation origins" in trained.output
    assert "Window selection:" in trained.output

    prediction = SimpleNamespace(
        predictions=pd.DataFrame([{"predicted_demand_mw": 100.0}]),
        parquet_path=tmp_path / "forecast.parquet",
        duckdb_rows=1,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_prediction_pipeline", lambda *_args, **_kwargs: prediction
    )
    predicted = runner.invoke(cli_module.app, ["predict", "--bundle-path", "model.joblib"])
    assert predicted.exit_code == 0
    assert "DuckDB forecast rows: 1" in predicted.output

    explanation = SimpleNamespace(
        importance_csv=tmp_path / "importance.csv", summary_plot=tmp_path / "summary.png"
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli_module, "run_explain_pipeline", lambda *_args, **_kwargs: explanation
    )
    explained = runner.invoke(cli_module.app, ["explain", "--bundle-path", "model.joblib"])
    assert explained.exit_code == 0
    assert "SHAP importance" in explained.output


def test_leaderboard_reads_latest_local_file(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    directory = tmp_path / "artifacts" / "training" / "20240101"
    directory.mkdir(parents=True)
    pd.DataFrame([{"rank": 1, "model_name": "catboost_global", "wape": 0.1}]).to_csv(
        directory / "leaderboard.csv", index=False
    )
    output = tmp_path / "copy.csv"
    result = runner.invoke(cli_module.app, ["leaderboard", "--csv-output", str(output)])
    assert result.exit_code == 0
    assert "catboost_global" in result.output
    assert output.exists()
