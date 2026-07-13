"""Deterministic baseline-versus-ML model leaderboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gridmind.training.evaluator import EvaluationResult, relative_improvement


@dataclass(frozen=True)
class ModelEvaluationRecord:
    """One named evaluation and its optional MLflow run identifier."""

    model_name: str
    result: EvaluationResult
    run_id: str = ""


def create_leaderboard(
    records: list[ModelEvaluationRecord], *, primary_metric: str = "wape"
) -> pd.DataFrame:
    """Combine results, add baseline improvements, and rank deterministically."""
    metrics_by_name = {record.model_name: record.result.overall_metrics for record in records}
    seasonal_24 = metrics_by_name.get("seasonal_naive_24h", {}).get(primary_metric, float("nan"))
    seasonal_168 = metrics_by_name.get("seasonal_naive_168h", {}).get(primary_metric, float("nan"))
    rows = []
    for record in records:
        metrics = record.result.overall_metrics
        value = float(metrics[primary_metric])
        rows.append(
            {
                "model_name": record.model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wape": metrics["wape"],
                "mase": metrics["mase"],
                "forecast_bias": metrics["forecast_bias"],
                "improvement_vs_seasonal_24": relative_improvement(value, seasonal_24),
                "improvement_vs_seasonal_168": relative_improvement(value, seasonal_168),
                "training_seconds": record.result.training_seconds,
                "prediction_seconds": record.result.prediction_seconds,
                "run_id": record.run_id,
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        [primary_metric, "mae", "model_name"], ignore_index=True
    )
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    return leaderboard
