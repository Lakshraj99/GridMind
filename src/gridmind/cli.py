"""Typer command-line interface for GridMind milestone one."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer
from mlflow.exceptions import MlflowException

from gridmind.anomalies.storage import AnomalyStorage
from gridmind.config import get_settings
from gridmind.data.processing import generate_quality_report
from gridmind.data.schemas import validate_processed_data
from gridmind.data.storage import (
    DuckDBStorage,
    read_processed_parquet,
    write_json_report,
)
from gridmind.exceptions import GridMindError, StorageError
from gridmind.logging_config import configure_logging
from gridmind.pipelines.backtest_anomalies import run_anomaly_backtest
from gridmind.pipelines.baseline import run_baseline_pipeline
from gridmind.pipelines.detect_anomalies import run_anomaly_detection
from gridmind.pipelines.explain import run_explain_pipeline
from gridmind.pipelines.ingest import run_ingestion
from gridmind.pipelines.manage_alerts import list_alerts, update_alert_status
from gridmind.pipelines.predict import run_prediction_pipeline
from gridmind.pipelines.predict_target import run_target_prediction
from gridmind.pipelines.renewable_ingest import run_renewable_ingestion
from gridmind.pipelines.train import run_training_pipeline
from gridmind.pipelines.train_target import run_target_training
from gridmind.pipelines.weather_ingest import run_weather_ingestion
from gridmind.time_utils import format_utc_timestamp

app = typer.Typer(
    name="gridmind",
    help="GridMind electricity-grid data and baseline forecasting CLI.",
    no_args_is_help=True,
)


class MissingDemandPolicyOption(StrEnum):
    """User-selectable missing actual-demand behavior."""

    error = "error"
    drop = "drop"


class SeverityOption(StrEnum):
    info = "info"
    warning = "warning"
    critical = "critical"


class AlertStatusOption(StrEnum):
    open = "open"
    acknowledged = "acknowledged"
    resolved = "resolved"
    suppressed = "suppressed"


@app.callback()
def main() -> None:
    """Configure application logging before a subcommand runs."""
    settings = get_settings()
    configure_logging(settings.log_level, eia_api_key=settings.eia_api_key)


@app.command()
def ingest(
    region: Annotated[str | None, typer.Option(help="EIA balancing-authority code.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Inclusive start timestamp/date.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Inclusive end timestamp/date.")] = None,
    missing_demand_policy: Annotated[
        MissingDemandPolicyOption | None,
        typer.Option(
            help="Missing actual-demand handling: 'error' fails after reporting; "
            "'drop' quarantines."
        ),
    ] = None,
) -> None:
    """Fetch, validate, and persist hourly EIA grid data."""
    try:
        result = run_ingestion(
            get_settings(),
            region=region,
            start_date=start_date,
            end_date=end_date,
            missing_demand_policy=(
                missing_demand_policy.value if missing_demand_policy is not None else None
            ),
        )
    except GridMindError as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.quarantined_rows:
        typer.echo(
            f"WARNING: Quarantined and excluded {result.quarantined_rows:,} rows with missing "
            f"actual demand: {result.quarantine_path}",
            err=True,
        )
    typer.echo(
        f"Ingested {result.rows:,} rows; DuckDB now has {result.duckdb_rows:,} rows. "
        f"Quality report: {result.quality_report_path}"
    )


@app.command("validate")
def validate_command(
    processed_dir: Annotated[
        Path | None, typer.Option(help="Partitioned Parquet directory.")
    ] = None,
) -> None:
    """Validate existing processed data and regenerate its quality report."""
    settings = get_settings()
    directory = processed_dir or settings.data_dir / "processed"
    try:
        frame = read_processed_parquet(directory)
        if frame.empty:
            raise StorageError(f"No processed Parquet files were found in {directory}.")
        validate_processed_data(frame)
        report = generate_quality_report(frame)
        report_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = write_json_report(
            report,
            settings.data_quality_dir / f"validation_data_quality_report_{report_stamp}.json",
        )
    except (GridMindError, OSError, ValueError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Validated {len(frame):,} rows. Quality report: {path}")


@app.command()
def baseline(
    region: Annotated[str | None, typer.Option(help="Region to evaluate.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Inclusive evaluation data start.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Inclusive evaluation data end.")] = None,
    horizon: Annotated[int, typer.Option(min=1, help="Forecast horizon in hours.")] = 24,
    windows: Annotated[int, typer.Option(min=1, help="Rolling validation window count.")] = 3,
    step_size: Annotated[int, typer.Option(min=1, help="Hours between validation origins.")] = 24,
    mlflow_enabled: Annotated[
        bool, typer.Option("--mlflow/--no-mlflow", help="Log MLflow runs.")
    ] = True,
) -> None:
    """Evaluate all demand baselines and print an MAE-sorted leaderboard."""
    settings = get_settings()
    selected_region = region or settings.grid_region
    selected_start = start_date or settings.data_start_date or "1900-01-01"
    selected_end = end_date or settings.data_end_date or "2100-01-01"
    try:
        frame = DuckDBStorage(settings.duckdb_path).read_region(
            selected_region, selected_start, selected_end
        )
        if frame.empty:
            raise ValueError(f"No stored data found for region {selected_region}.")
        result = run_baseline_pipeline(
            frame,
            settings,
            horizon=horizon,
            windows=windows,
            step_size=step_size,
            mlflow_enabled=mlflow_enabled,
        )
    except (GridMindError, OSError, ValueError) as exc:
        typer.echo(f"Baseline evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.leaderboard.to_string(index=False))
    typer.echo(f"Metrics: {result.metrics_path}")


@app.command("inspect")
def inspect_command() -> None:
    """Display stored row count, date range, missing demand, and regions."""
    settings = get_settings()
    try:
        summary = DuckDBStorage(settings.duckdb_path).inspect()
    except (OSError, RuntimeError) as exc:
        typer.echo(f"Inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Rows: {summary['row_count']:,}")
    start = format_utc_timestamp(summary["date_range"]["start"])
    end = format_utc_timestamp(summary["date_range"]["end"])
    typer.echo(f"Date range: {start} to {end}")
    typer.echo(f"Missing demand: {summary['missing_demand_count']:,}")
    typer.echo(f"Regions: {', '.join(summary['regions'])}")


@app.command()
def train(
    region: Annotated[
        str | None, typer.Option(help="Region code, or 'all' for a global multi-region model.")
    ] = None,
    models: Annotated[
        str, typer.Option(help="Comma-separated choices: lightgbm,catboost.")
    ] = "lightgbm,catboost",
    horizon: Annotated[int | None, typer.Option(min=1, help="Forecast horizon in hours.")] = None,
    validation_windows: Annotated[
        int | None, typer.Option(min=1, help="Untouched final rolling windows.")
    ] = None,
    step_size: Annotated[
        int | None, typer.Option(min=1, help="Hours between evaluation origins.")
    ] = None,
    tune: Annotated[
        bool, typer.Option("--tune/--no-tune", help="Run nested Optuna tuning.")
    ] = False,
    trials: Annotated[int | None, typer.Option(min=1, help="Optuna trial count.")] = None,
    mlflow_enabled: Annotated[
        bool, typer.Option("--mlflow/--no-mlflow", help="Track parent and child runs.")
    ] = True,
    register: Annotated[
        bool, typer.Option("--register/--no-register", help="Register candidate/champion.")
    ] = True,
    random_seed: Annotated[int | None, typer.Option(help="Deterministic model seed.")] = None,
) -> None:
    """Train, compare, explain, serialize, and optionally register global ML models."""
    settings = get_settings()
    selected_region = region or settings.grid_region
    selected_regions = None if selected_region.lower() == "all" else [selected_region]
    selected_models = [value.strip() for value in models.split(",") if value.strip()]
    typer.echo(
        f"Training regions: {selected_region}; models: {', '.join(selected_models)}; "
        f"tuning: {'enabled' if tune else 'disabled'}"
    )
    try:
        result = run_training_pipeline(
            settings,
            regions=selected_regions,
            model_names=selected_models,
            horizon=horizon,
            validation_windows=validation_windows,
            step_size=step_size,
            tune=tune,
            trials=trials,
            mlflow_enabled=mlflow_enabled,
            register_model=register,
            random_seed=random_seed,
            progress=typer.echo,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.leaderboard.to_string(index=False))
    typer.echo(f"Candidate model: {result.selected_model}")
    if result.promotion is not None:
        typer.echo(f"Champion promoted: {result.promotion.champion_promoted}")
        typer.echo(f"Promotion result: {result.promotion.reason}")
    typer.echo(f"Model bundle: {result.bundle_path}")
    typer.echo(f"Window selection: {result.window_selection_path}")
    typer.echo(f"Artifacts: {result.artifact_dir}")


@app.command()
def predict(
    region: Annotated[str | None, typer.Option(help="Region to forecast.")] = None,
    horizon: Annotated[int | None, typer.Option(min=1, help="Forecast horizon in hours.")] = None,
    model_alias: Annotated[str, typer.Option(help="Registry alias such as champion.")] = "champion",
    model_version: Annotated[str | None, typer.Option(help="Explicit registry version.")] = None,
    run_id: Annotated[str | None, typer.Option(help="MLflow run containing a bundle.")] = None,
    bundle_path: Annotated[Path | None, typer.Option(help="Local model bundle path.")] = None,
) -> None:
    """Generate and idempotently persist batch demand forecasts."""
    try:
        settings = get_settings()
        result = run_prediction_pipeline(
            settings,
            region=region or settings.grid_region,
            horizon=horizon,
            model_alias=model_alias,
            model_version=model_version,
            run_id=run_id,
            bundle_path=bundle_path,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Prediction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.predictions.to_string(index=False))
    typer.echo(f"Parquet: {result.parquet_path}")
    typer.echo(f"DuckDB forecast rows: {result.duckdb_rows:,}")


@app.command()
def leaderboard(
    run_id: Annotated[str | None, typer.Option(help="Parent MLflow run ID.")] = None,
    csv_output: Annotated[Path | None, typer.Option(help="Optional output CSV path.")] = None,
) -> None:
    """Display the latest local leaderboard or one stored in an MLflow run."""
    settings = get_settings()
    try:
        if run_id:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            path = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="training_artifacts/leaderboard.csv",
                )
            )
        else:
            candidates = sorted(Path("artifacts/training").glob("*/leaderboard.csv"))
            if not candidates:
                raise ValueError("No local training leaderboard is available.")
            path = candidates[-1]
        frame = pd.read_csv(path)
        if csv_output is not None:
            csv_output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(csv_output, index=False)
    except (MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Leaderboard failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(frame.to_string(index=False))


@app.command()
def explain(
    region: Annotated[
        str | None, typer.Option(help="Stored region used for SHAP sampling.")
    ] = None,
    model_alias: Annotated[str, typer.Option(help="Registry alias such as champion.")] = "champion",
    bundle_path: Annotated[Path | None, typer.Option(help="Local model bundle path.")] = None,
) -> None:
    """Generate deterministic global and local SHAP artifacts."""
    try:
        settings = get_settings()
        result = run_explain_pipeline(
            settings,
            region=region or settings.grid_region,
            model_alias=model_alias,
            bundle_path=bundle_path,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Explanation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"SHAP importance: {result.importance_csv}")
    typer.echo(f"SHAP summary: {result.summary_plot}")


@app.command("weather-ingest")
def weather_ingest(
    region: Annotated[str | None, typer.Option(help="Grid region mapping to ingest.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Inclusive YYYY-MM-DD start.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Inclusive YYYY-MM-DD end.")] = None,
    data_type: Annotated[
        str, typer.Option(help="Weather contract type: historical or forecast.")
    ] = "historical",
) -> None:
    """Fetch, validate, aggregate, cache, and persist hourly regional weather."""
    settings = get_settings()
    if data_type not in {"historical", "forecast"}:
        raise typer.BadParameter("data-type must be historical or forecast")
    try:
        result = run_weather_ingestion(
            settings,
            region=region or settings.grid_region,
            start_date=start_date or settings.data_start_date or "",
            end_date=end_date or settings.data_end_date or "",
            data_type=data_type,  # type: ignore[arg-type]
        )
    except (GridMindError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Weather ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Weather rows: {result.location_rows:,} location; {result.regional_rows:,} regional; "
        f"cache hits: {result.cache_hits:,}"
    )
    typer.echo(f"Quality report: {result.report_path}")


@app.command("renewables-ingest")
def renewables_ingest(
    region: Annotated[str | None, typer.Option(help="Grid region to ingest.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Inclusive YYYY-MM-DD start.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Inclusive YYYY-MM-DD end.")] = None,
) -> None:
    """Fetch, validate, quarantine, and persist EIA solar/wind generation."""
    settings = get_settings()
    try:
        result = run_renewable_ingestion(
            settings,
            region=region or settings.grid_region,
            start_date=start_date or settings.data_start_date or "",
            end_date=end_date or settings.data_end_date or "",
        )
    except (GridMindError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Renewable ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Renewable rows: {result.rows:,}; quarantined: {result.quarantined_rows:,}; "
        f"DuckDB rows: {result.duckdb_rows:,}"
    )
    typer.echo(f"Quality report: {result.report_path}")


@app.command("train-target")
def train_target(
    target: Annotated[str, typer.Option(help="Demand, renewable, or net-load target.")],
    region: Annotated[str | None, typer.Option(help="Grid region.")] = None,
    weather_mode: Annotated[
        str, typer.Option(help="realistic_forecast or historical_oracle.")
    ] = "realistic_forecast",
    models: Annotated[
        str, typer.Option(help="Comma-separated lightgbm,catboost.")
    ] = "lightgbm,catboost",
    horizon: Annotated[int, typer.Option(min=1)] = 24,
    validation_windows: Annotated[int, typer.Option(min=1)] = 5,
    tune: Annotated[bool, typer.Option("--tune/--no-tune")] = False,
    trials: Annotated[int, typer.Option(min=1)] = 10,
    net_load_method: Annotated[
        str, typer.Option(help="For net load: direct, component, or direct,component.")
    ] = "direct,component",
    mlflow_enabled: Annotated[bool, typer.Option("--mlflow/--no-mlflow")] = True,
    register: Annotated[bool, typer.Option("--register/--no-register")] = True,
) -> None:
    """Train and compare target-specific weather-aware models and baselines."""
    if weather_mode not in {"realistic_forecast", "historical_oracle"}:
        raise typer.BadParameter("Unsupported weather mode")
    if target == "net_load_mw" and not set(net_load_method.split(",")).issubset(
        {"direct", "component"}
    ):
        raise typer.BadParameter("Unsupported net-load method")
    settings = get_settings()
    try:
        result = run_target_training(
            settings,
            target=target,
            region=region or settings.grid_region,
            weather_mode=weather_mode,  # type: ignore[arg-type]
            model_names=[item.strip() for item in models.split(",") if item.strip()],
            horizon=horizon,
            validation_windows=validation_windows,
            mlflow_enabled=mlflow_enabled,
            register_model=register,
            tune=tune,
            trials=trials,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Target training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.leaderboard.to_string(index=False))
    typer.echo(f"Selected model: {result.selected_model}")
    typer.echo(f"Candidate assigned: {result.candidate_assigned}")
    typer.echo(f"Champion promoted: {result.champion_promoted}")
    typer.echo(f"Artifacts: {result.artifact_dir}")


@app.command("predict-target")
def predict_target(
    target: Annotated[str, typer.Option(help="Target to forecast.")],
    region: Annotated[str | None, typer.Option(help="Grid region.")] = None,
    horizon: Annotated[int, typer.Option(min=1)] = 24,
    model_alias: Annotated[str, typer.Option(help="Registry alias.")] = "champion",
    bundle_path: Annotated[Path | None, typer.Option(help="Optional local target bundle.")] = None,
) -> None:
    """Forecast and idempotently persist a registered Milestone 3 target."""
    settings = get_settings()
    try:
        result = run_target_prediction(
            settings,
            target=target,
            region=region or settings.grid_region,
            horizon=horizon,
            model_alias=model_alias,
            bundle_path=bundle_path,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Target prediction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    display = result.forecasts.copy()
    for column in ("forecast_origin", "timestamp_utc", "created_at_utc"):
        display[column] = display[column].map(format_utc_timestamp)
    typer.echo(display.to_string(index=False))
    typer.echo(f"Parquet: {result.parquet_path}")
    typer.echo(f"DuckDB rows: {result.duckdb_rows:,}")


@app.command("target-leaderboard")
def target_leaderboard(
    target: Annotated[str, typer.Option(help="Target leaderboard to display.")],
) -> None:
    """Display the latest local leaderboard for one Milestone 3 target."""
    paths = sorted(Path("artifacts/training_targets").glob(f"{target}/*/leaderboard.csv"))
    if not paths:
        typer.echo(f"No leaderboard is available for {target}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(pd.read_csv(paths[-1]).to_string(index=False))


@app.command("detect-anomalies")
def detect_anomalies_command(
    region: Annotated[str, typer.Option(help="Grid region to evaluate.")],
    targets: Annotated[str, typer.Option(help="Comma-separated supported targets.")],
    start_date: Annotated[str, typer.Option(help="Inclusive UTC start date/timestamp.")],
    end_date: Annotated[str, typer.Option(help="Inclusive UTC end date/timestamp.")],
    detectors: Annotated[
        str, typer.Option(help="Comma-separated rules,residual,isolation_forest.")
    ] = "rules,residual,isolation_forest",
    mlflow_enabled: Annotated[bool, typer.Option("--mlflow/--no-mlflow")] = True,
) -> None:
    """Detect, persist, ensemble, and alert on operational anomalies."""
    try:
        result = run_anomaly_detection(
            get_settings(),
            region=region,
            targets=tuple(item.strip() for item in targets.split(",") if item.strip()),
            start_date=start_date,
            end_date=end_date,
            detectors=tuple(item.strip() for item in detectors.split(",") if item.strip()),
            mlflow_enabled=mlflow_enabled,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Anomaly detection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    severity_counts = result.anomalies["severity"].value_counts().to_dict()
    detector_counts = result.anomalies["detector_name"].value_counts().to_dict()
    typer.echo(f"Rows evaluated: {result.rows_evaluated:,}")
    typer.echo(f"Anomalies found: {len(result.anomalies):,}")
    typer.echo(f"Severity counts: {severity_counts}")
    typer.echo(f"Detector counts: {detector_counts}")
    typer.echo(f"Alerts opened: {result.alerts_opened}; updated: {result.alerts_updated}")
    typer.echo(f"Artifacts: {result.artifact_dir}")


@app.command("anomaly-backtest")
def anomaly_backtest_command(
    region: Annotated[str, typer.Option(help="Grid region to evaluate.")],
    target: Annotated[str, typer.Option(help="Supported target to evaluate.")],
    start_date: Annotated[str, typer.Option(help="Inclusive UTC start date/timestamp.")],
    end_date: Annotated[str, typer.Option(help="Inclusive UTC end date/timestamp.")],
    inject: Annotated[bool, typer.Option("--inject/--no-inject")] = True,
    seed: Annotated[int, typer.Option(help="Deterministic injection seed.")] = 42,
    mlflow_enabled: Annotated[bool, typer.Option("--mlflow/--no-mlflow")] = True,
) -> None:
    """Evaluate anomaly rules using synthetic injection or unsupervised reporting."""
    try:
        result = run_anomaly_backtest(
            get_settings(),
            region=region,
            target=target,
            start_date=start_date,
            end_date=end_date,
            inject=inject,
            seed=seed,
            mlflow_enabled=mlflow_enabled,
        )
    except (GridMindError, MlflowException, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Anomaly backtest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Injected anomaly count: {result.injected_count}")
    typer.echo(f"Detected count: {result.detected_count}")
    for name in (
        "precision",
        "recall",
        "f1",
        "false_positives_per_day",
        "mean_detection_delay_hours",
    ):
        if name in result.metrics:
            typer.echo(f"{name}: {result.metrics[name]:.6f}")
    typer.echo("Synthetic labels are controlled test cases, not real grid incidents.")
    typer.echo(f"Artifacts: {result.artifact_dir}")


@app.command("anomalies")
def list_anomalies_command(
    region: Annotated[str | None, typer.Option(help="Optional grid region.")] = None,
    target: Annotated[str | None, typer.Option(help="Optional target.")] = None,
    severity: Annotated[SeverityOption | None, typer.Option(help="Optional severity.")] = None,
    detector: Annotated[str | None, typer.Option(help="Optional detector name.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Optional UTC start.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Optional UTC end.")] = None,
    csv_path: Annotated[
        Path | None, typer.Option("--csv", help="Optional CSV output path.")
    ] = None,
) -> None:
    """List persisted anomaly events with optional filters."""
    try:
        frame = AnomalyStorage(get_settings().duckdb_path).read(
            region=region,
            target=target,
            severity=severity.value if severity else None,
            detector=detector,
            start=start_date,
            end=end_date,
        )
        _write_or_display(frame, csv_path, ("timestamp_utc", "forecast_origin", "detected_at_utc"))
    except (GridMindError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Anomaly listing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("alerts")
def list_alerts_command(
    region: Annotated[str | None, typer.Option(help="Optional grid region.")] = None,
    target: Annotated[str | None, typer.Option(help="Optional target.")] = None,
    status: Annotated[AlertStatusOption | None, typer.Option(help="Optional status.")] = None,
    severity: Annotated[SeverityOption | None, typer.Option(help="Optional severity.")] = None,
    csv_path: Annotated[
        Path | None, typer.Option("--csv", help="Optional CSV output path.")
    ] = None,
) -> None:
    """List current alerts separately from raw anomaly events."""
    try:
        frame = list_alerts(
            get_settings(),
            region=region,
            target=target,
            status=status.value if status else None,
            severity=severity.value if severity else None,
        )
        _write_or_display(
            frame,
            csv_path,
            (
                "first_seen_utc",
                "last_seen_utc",
                "acknowledged_at_utc",
                "resolved_at_utc",
                "created_at_utc",
                "updated_at_utc",
            ),
        )
    except (GridMindError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Alert listing failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("alert-update")
def alert_update_command(
    alert_id: Annotated[str, typer.Option(help="Alert identifier.")],
    status: Annotated[AlertStatusOption, typer.Option(help="New lifecycle status.")],
) -> None:
    """Acknowledge, resolve, suppress, or reopen a persisted alert."""
    try:
        alert = update_alert_status(get_settings(), alert_id=alert_id, status=status.value)
    except (GridMindError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Alert update failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Updated alert {alert['alert_id']} to {alert['status']}.")


def _write_or_display(
    frame: pd.DataFrame, csv_path: Path | None, timestamp_columns: tuple[str, ...]
) -> None:
    display = frame.copy()
    for column in timestamp_columns:
        if column in display:
            display[column] = display[column].map(
                lambda value: format_utc_timestamp(value) if pd.notna(value) else ""
            )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        display.to_csv(csv_path, index=False)
        typer.echo(f"CSV: {csv_path}")
    else:
        typer.echo(display.to_string(index=False) if not display.empty else "No records found.")


if __name__ == "__main__":  # pragma: no cover
    app()
