"""Offline historical anomaly backtesting with controlled synthetic labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import mlflow
import pandas as pd

from gridmind.anomalies.evaluation import (
    EvaluationResult,
    InjectionResult,
    evaluate_detections,
    inject_synthetic_anomalies,
)
from gridmind.anomalies.rules import RuleConfig, RuleDetector
from gridmind.config import Settings
from gridmind.data.storage import write_json_report
from gridmind.mlflow_config import initialize_mlflow
from gridmind.pipelines.detect_anomalies import _load_target
from gridmind.time_utils import inclusive_hourly_range


@dataclass(frozen=True)
class AnomalyBacktestResult:
    injected_count: int
    detected_count: int
    metrics: dict[str, float]
    artifact_dir: Path
    mlflow_run_id: str | None
    detections: pd.DataFrame


def run_anomaly_backtest(
    settings: Settings,
    *,
    region: str,
    target: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    inject: bool = True,
    seed: int = 42,
    mlflow_enabled: bool | None = None,
    artifact_root: Path = Path("artifacts/anomaly_backtests"),
) -> AnomalyBacktestResult:
    date_range = inclusive_hourly_range(start_date, end_date)
    start, end = date_range[0], date_range[-1]
    original = _load_target(settings, region, target, start, end)
    if original.empty:
        raise ValueError("No historical rows are available for the requested anomaly backtest.")
    injection = (
        inject_synthetic_anomalies(
            original,
            target=target,
            seed=seed,
            anomaly_types=_target_injection_types(target),
        )
        if inject
        else InjectionResult(original.copy(deep=True), pd.DataFrame())
    )
    detections = RuleDetector(
        RuleConfig(
            demand_change_threshold=settings.demand_spike_pct_threshold,
            renewable_drop_threshold=settings.renewable_drop_pct_threshold,
            flatline_hours=settings.flatline_hours,
            missing_warning_count=settings.missing_hour_warning_count,
            missing_critical_count=settings.missing_hour_critical_count,
            stale_after_hours=settings.alert_auto_resolve_hours,
        )
    ).detect(injection.frame, target=target, now=end)
    evaluation = (
        evaluate_detections(
            detections, injection.labels, evaluation_start=start, evaluation_end=end
        )
        if inject
        else _unsupervised_evaluation(detections, start, end)
    )
    stamp = pd.Timestamp.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = artifact_root / stamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    injection.labels.to_parquet(artifact_dir / "injected_anomalies.parquet", index=False)
    detections.to_parquet(artifact_dir / "detected_anomalies.parquet", index=False)
    evaluation.match_results.to_parquet(artifact_dir / "match_results.parquet", index=False)
    write_json_report(evaluation.overall_metrics, artifact_dir / "overall_metrics.json")
    evaluation.per_type_metrics.to_csv(artifact_dir / "per_type_metrics.csv", index=False)
    evaluation.detection_delays.to_csv(artifact_dir / "detection_delay.csv", index=False)
    run_id = _log_backtest(
        settings,
        artifact_dir,
        region=region,
        target=target,
        start=start,
        end=end,
        seed=seed,
        metrics=evaluation.overall_metrics,
        enabled=settings.mlflow_enabled if mlflow_enabled is None else mlflow_enabled,
    )
    return AnomalyBacktestResult(
        len(injection.labels),
        len(detections),
        evaluation.overall_metrics,
        artifact_dir,
        run_id,
        detections,
    )


def _target_injection_types(target: str) -> tuple[str, ...]:
    common = ("flatline_sensor", "missing_hour_sequence", "gradual_drift")
    if target == "solar_generation_mw":
        return ("solar_generation_collapse", *common)
    if target == "wind_generation_mw":
        return ("wind_generation_spike", *common)
    return (
        "single_hour_demand_spike",
        "multi_hour_demand_drop",
        *common,
        "contextual_unusual_hour",
    )


def _unsupervised_evaluation(
    detections: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> EvaluationResult:
    days = max(float((end - start) / pd.Timedelta(days=1)), 1 / 24)
    metrics = {
        "anomaly_count": float(len(detections)),
        "anomalies_per_day": float(len(detections) / days),
    }
    per_type = (
        detections["anomaly_type"]
        .value_counts()
        .rename_axis("anomaly_type")
        .reset_index(name="count")
        if not detections.empty
        else pd.DataFrame(columns=["anomaly_type", "count"])
    )
    return EvaluationResult(metrics, per_type, pd.DataFrame(), pd.DataFrame())


def _log_backtest(
    settings: Settings,
    artifact_dir: Path,
    *,
    region: str,
    target: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int,
    metrics: dict[str, float],
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    setup = initialize_mlflow(settings, settings.anomaly_experiment_name)
    with mlflow.start_run(
        experiment_id=setup.experiment_id, run_name=f"backtest-{region}-{target}"
    ) as run:
        mlflow.log_params(
            {
                "region": region,
                "target": target,
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "injection_seed": seed,
                "labels": "synthetic_not_real_incidents",
            }
        )
        mlflow.log_metrics({key: value for key, value in metrics.items() if pd.notna(value)})
        mlflow.log_artifacts(str(artifact_dir), artifact_path="anomaly_backtest")
        return str(run.info.run_id)
