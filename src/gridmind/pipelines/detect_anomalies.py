"""Milestone 4 anomaly detection, persistence, alerting, artifacts, and MLflow."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from gridmind.alerts.lifecycle import AlertManager
from gridmind.alerts.storage import AlertStorage
from gridmind.anomalies.contracts import ANOMALY_COLUMNS, empty_anomaly_frame
from gridmind.anomalies.ensemble import combine_detector_events
from gridmind.anomalies.multivariate import IsolationForestConfig, MultivariateDetector
from gridmind.anomalies.residuals import ResidualConfig, ResidualDetector
from gridmind.anomalies.rules import RuleConfig, RuleDetector
from gridmind.anomalies.storage import AnomalyStorage
from gridmind.config import Settings
from gridmind.data.duckdb_connection import connect_duckdb
from gridmind.data.storage import DuckDBStorage, write_json_report
from gridmind.data.target_storage import TargetForecastStorage
from gridmind.exceptions import AnomalyDetectionError
from gridmind.mlflow_config import initialize_mlflow
from gridmind.renewables.storage import RenewableStorage
from gridmind.renewables.targets import SUPPORTED_TARGETS
from gridmind.time_utils import inclusive_hourly_range
from gridmind.weather.storage import WeatherStorage

SUPPORTED_DETECTORS = ("rules", "residual", "isolation_forest")


@dataclass(frozen=True)
class DetectionPipelineResult:
    rows_evaluated: int
    anomalies: pd.DataFrame
    anomaly_rows: int
    alerts_opened: int
    alerts_updated: int
    artifact_dir: Path
    mlflow_run_id: str | None
    detector_report: dict[str, Any]


def run_anomaly_detection(
    settings: Settings,
    *,
    region: str,
    targets: tuple[str, ...],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    detectors: tuple[str, ...] = SUPPORTED_DETECTORS,
    mlflow_enabled: bool | None = None,
    artifact_root: Path = Path("artifacts/anomalies"),
) -> DetectionPipelineResult:
    if not settings.anomaly_detection_enabled:
        raise AnomalyDetectionError("Anomaly detection is disabled by configuration.")
    unsupported_targets = set(targets).difference(SUPPORTED_TARGETS)
    unsupported_detectors = set(detectors).difference(SUPPORTED_DETECTORS)
    if unsupported_targets:
        raise AnomalyDetectionError(f"Unsupported anomaly targets: {sorted(unsupported_targets)}")
    if unsupported_detectors:
        raise AnomalyDetectionError(f"Unsupported detectors: {sorted(unsupported_detectors)}")
    requested = inclusive_hourly_range(start_date, end_date)
    start, end = requested[0], requested[-1]
    if end < start:
        raise AnomalyDetectionError("Anomaly end date must not precede its start date.")
    stamp = pd.Timestamp.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = artifact_root / stamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    event_parts: list[pd.DataFrame] = []
    detector_report: dict[str, Any] = {}
    rows_evaluated = 0
    detector_bundle: Path | None = None
    for target in targets:
        history_start = start - pd.Timedelta(hours=settings.anomaly_lookback_hours)
        source = _load_target(settings, region, target, history_start, end)
        evaluation = source.loc[source["timestamp_utc"].between(start, end)].copy()
        rows_evaluated += len(evaluation)
        if evaluation.empty:
            detector_report[target] = {"rows": 0, "condition": "no observations"}
            continue
        weather = _load_weather(settings, region, history_start, end)
        forecasts = _load_target_forecasts(settings, target)
        target_report: dict[str, Any] = {"rows": len(evaluation)}
        if "rules" in detectors:
            rule_detector = RuleDetector(
                RuleConfig(
                    demand_change_threshold=settings.demand_spike_pct_threshold,
                    renewable_drop_threshold=settings.renewable_drop_pct_threshold,
                    flatline_hours=settings.flatline_hours,
                    missing_warning_count=settings.missing_hour_warning_count,
                    missing_critical_count=settings.missing_hour_critical_count,
                    stale_after_hours=settings.alert_auto_resolve_hours,
                )
            )
            renewables = _load_renewable_coverage(settings, region, history_start, end)
            rules = rule_detector.detect(
                evaluation,
                target=target,
                weather=weather,
                renewables=renewables,
                now=end,
            )
            forecast_coverage = rule_detector.detect_forecast_weather_coverage(
                forecasts.loc[forecasts["timestamp_utc"].between(start, end)]
                if not forecasts.empty
                else forecasts,
                weather,
                target=target,
                now=end,
            )
            rules = _combine_event_frames([rules, forecast_coverage]).drop_duplicates(
                "anomaly_id", keep="last"
            )
            event_parts.append(rules)
            target_report["rule_anomalies"] = len(rules)
        if "residual" in detectors:
            residual = ResidualDetector(
                ResidualConfig(
                    min_history=min(24, settings.anomaly_min_training_rows),
                    zscore_warning=settings.residual_zscore_warning,
                    zscore_critical=settings.residual_zscore_critical,
                    mad_warning=settings.residual_mad_warning,
                    mad_critical=settings.residual_mad_critical,
                )
            ).detect(source, forecasts, target=target)
            residual_events = residual.anomalies.loc[
                residual.anomalies["timestamp_utc"].between(start, end)
            ]
            event_parts.append(residual_events)
            target_report.update(
                residual_anomalies=len(residual_events),
                residual_insufficient_history_rows=residual.insufficient_history_rows,
            )
        if "isolation_forest" in detectors:
            features = _multivariate_features(source, weather, target)
            training = features.loc[features["timestamp_utc"] < start]
            scoring = features.loc[features["timestamp_utc"].between(start, end)]
            feature_names = tuple(
                column for column in features.columns if column not in {"region", "timestamp_utc"}
            )
            if (
                len(training.dropna(subset=list(feature_names)))
                >= settings.anomaly_min_training_rows
            ):
                model = MultivariateDetector(
                    feature_names,
                    IsolationForestConfig(
                        contamination=settings.anomaly_contamination,
                        random_seed=settings.anomaly_random_seed,
                        min_training_rows=settings.anomaly_min_training_rows,
                    ),
                ).fit(training)
                scored = model.score(scoring)
                event_parts.append(scored.anomalies)
                detector_bundle = model.save(artifact_dir / f"{target}_detector.joblib")
                target_report.update(
                    isolation_forest_anomalies=len(scored.anomalies),
                    isolation_training_rows=scored.training_rows,
                    isolation_excluded_rows=scored.excluded_rows,
                    isolation_gap_count=scored.gap_count,
                )
            else:
                target_report["isolation_condition"] = "insufficient chronological training rows"
        detector_report[target] = target_report
    raw_events = _combine_event_frames(event_parts)
    raw_events = raw_events.drop_duplicates("anomaly_id", keep="last")
    ensemble = (
        combine_detector_events(raw_events) if not raw_events.empty else empty_anomaly_frame()
    )
    all_events = _combine_event_frames([raw_events, ensemble])
    anomaly_storage = AnomalyStorage(settings.duckdb_path)
    anomaly_rows = (
        anomaly_storage.upsert(all_events) if not all_events.empty else anomaly_storage.count()
    )
    alert_input = ensemble if not ensemble.empty else raw_events
    alert_manager = AlertManager(
        AlertStorage(settings.duckdb_path),
        dedup_hours=settings.alert_dedup_window_hours,
        auto_resolve_hours=settings.alert_auto_resolve_hours,
    )
    alert_counts = (
        alert_manager.process(alert_input) if not alert_input.empty else {"opened": 0, "updated": 0}
    )
    alert_manager.auto_resolve(now=end)
    _write_detection_artifacts(
        artifact_dir, all_events, detector_report, alert_counts, targets, detectors
    )
    run_id = _log_detection_mlflow(
        settings,
        artifact_dir,
        region=region,
        targets=targets,
        start=start,
        end=end,
        detectors=detectors,
        rows_evaluated=rows_evaluated,
        anomaly_count=len(all_events),
        detector_bundle=detector_bundle,
        enabled=settings.mlflow_enabled if mlflow_enabled is None else mlflow_enabled,
    )
    return DetectionPipelineResult(
        rows_evaluated,
        all_events,
        anomaly_rows,
        int(alert_counts["opened"]),
        int(alert_counts["updated"]),
        artifact_dir,
        run_id,
        detector_report,
    )


def _load_target(
    settings: Settings, region: str, target: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if target == "demand_mw":
        frame = DuckDBStorage(settings.duckdb_path).read_data(
            regions=[region], start_date=start.to_pydatetime(), end_date=end.to_pydatetime()
        )
        return frame[["region", "timestamp_utc", target]]
    if target in {"solar_generation_mw", "wind_generation_mw", "total_renewable_generation_mw"}:
        frame = RenewableStorage(settings.duckdb_path).read(region)
        return frame.loc[
            frame["timestamp_utc"].between(start, end), ["region", "timestamp_utc", target]
        ]
    with connect_duckdb(settings.duckdb_path, read_only=True) as connection:
        frame = connection.execute(
            "SELECT region, timestamp_utc, net_load_mw FROM target_net_load "
            "WHERE region = ? AND timestamp_utc BETWEEN ? AND ? ORDER BY timestamp_utc",
            [region, start, end],
        ).fetchdf()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


def _combine_event_frames(parts: list[pd.DataFrame]) -> pd.DataFrame:
    records = [record for part in parts for record in part.to_dict(orient="records")]
    return (
        pd.DataFrame.from_records(records, columns=ANOMALY_COLUMNS)
        if records
        else empty_anomaly_frame()
    )


def _load_weather(
    settings: Settings, region: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    try:
        frame = WeatherStorage(settings.duckdb_path).read_regions(region)
    except Exception:
        return pd.DataFrame(columns=["region", "timestamp_utc"])
    return frame.loc[frame["timestamp_utc"].between(start, end)].copy()


def _load_target_forecasts(settings: Settings, target: str) -> pd.DataFrame:
    with connect_duckdb(settings.duckdb_path, read_only=True) as connection:
        exists = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'target_forecasts'"
        ).fetchone()
    if not exists or not exists[0]:
        return pd.DataFrame()
    return TargetForecastStorage(settings.duckdb_path).read(target=target)


def _load_renewable_coverage(
    settings: Settings, region: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | None:
    with connect_duckdb(settings.duckdb_path, read_only=True) as connection:
        exists = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'hourly_renewable_generation'"
        ).fetchone()
    if not exists or not exists[0]:
        return None
    frame = RenewableStorage(settings.duckdb_path).read(region)
    return frame.loc[frame["timestamp_utc"].between(start, end)]


def _multivariate_features(
    target_frame: pd.DataFrame, weather: pd.DataFrame, target: str
) -> pd.DataFrame:
    result = target_frame[["region", "timestamp_utc", target]].copy()
    useful_weather = [
        column
        for column in (
            "temperature_c",
            "apparent_temperature_c",
            "relative_humidity_pct",
            "cloud_cover_pct",
            "wind_speed_10m_kph",
            "shortwave_radiation_wm2",
        )
        if column in weather
    ]
    if useful_weather:
        result = result.merge(
            weather[["region", "timestamp_utc", *useful_weather]].drop_duplicates(
                ["region", "timestamp_utc"], keep="last"
            ),
            on=["region", "timestamp_utc"],
            how="left",
        )
    hour = result["timestamp_utc"].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return result


def _write_detection_artifacts(
    artifact_dir: Path,
    events: pd.DataFrame,
    detector_report: dict[str, Any],
    alert_counts: dict[str, int],
    targets: tuple[str, ...],
    detectors: tuple[str, ...],
) -> None:
    events.to_parquet(artifact_dir / "anomaly_events.parquet", index=False)
    severity_counts = events["severity"].value_counts().to_dict() if not events.empty else {}
    write_json_report(
        {"anomaly_count": len(events), "severity_counts": severity_counts},
        artifact_dir / "anomaly_summary.json",
    )
    write_json_report(detector_report, artifact_dir / "detector_metrics.json")
    write_json_report(
        {"targets": list(targets), "detectors": list(detectors)},
        artifact_dir / "feature_schema.json",
    )
    write_json_report({"detector_report": detector_report}, artifact_dir / "threshold_report.json")
    write_json_report(alert_counts, artifact_dir / "alert_summary.json")


def _log_detection_mlflow(
    settings: Settings,
    artifact_dir: Path,
    *,
    region: str,
    targets: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    detectors: tuple[str, ...],
    rows_evaluated: int,
    anomaly_count: int,
    detector_bundle: Path | None,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    setup = initialize_mlflow(settings, settings.anomaly_experiment_name)
    with mlflow.start_run(experiment_id=setup.experiment_id, run_name=f"detect-{region}") as run:
        mlflow.log_params(
            {
                "region": region,
                "targets": ",".join(targets),
                "detectors": ",".join(detectors),
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "contamination": settings.anomaly_contamination,
                "random_seed": settings.anomaly_random_seed,
                "minimum_training_rows": settings.anomaly_min_training_rows,
                "demand_spike_threshold": settings.demand_spike_pct_threshold,
                "renewable_drop_threshold": settings.renewable_drop_pct_threshold,
                "residual_zscore_warning": settings.residual_zscore_warning,
                "residual_zscore_critical": settings.residual_zscore_critical,
                "residual_mad_warning": settings.residual_mad_warning,
                "residual_mad_critical": settings.residual_mad_critical,
                "git_commit": _git_commit(),
                "python_version": platform.python_version(),
                "package_versions": ",".join(
                    f"{name}={importlib.metadata.version(name)}"
                    for name in ("pandas", "numpy", "scikit-learn", "duckdb")
                ),
            }
        )
        mlflow.log_metrics({"rows_evaluated": rows_evaluated, "anomaly_count": anomaly_count})
        mlflow.log_artifacts(str(artifact_dir), artifact_path="anomaly_detection")
        if detector_bundle is not None:
            mlflow.log_artifact(str(detector_bundle), artifact_path="detector_bundle")
        return str(run.info.run_id)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
