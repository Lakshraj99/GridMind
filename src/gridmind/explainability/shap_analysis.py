"""Deterministic SHAP analysis for the selected tree forecasting model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap

from gridmind.exceptions import ExplainabilityError
from gridmind.models.protocols import TrainableForecastModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShapArtifacts:
    """Paths to global, dependence, local, and metadata explanation artifacts."""

    importance_csv: Path
    summary_plot: Path
    dependence_plots: tuple[Path, ...]
    local_explanations: Path
    metadata: Path
    sample_rows: int


def generate_shap_artifacts(
    model: TrainableForecastModel,
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    sample_size: int = 2000,
    random_seed: int = 42,
    top_features: int = 3,
    local_rows: int = 5,
) -> ShapArtifacts:
    """Create sampled SHAP artifacts; explanations describe behavior, not causality."""
    if sample_size <= 0:
        raise ValueError("SHAP sample size must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    x, _target = model.training_features(frame)
    sampled = x.sample(n=min(sample_size, len(x)), random_state=random_seed).copy()
    try:
        explainer = shap.TreeExplainer(model.estimator)
        values = np.asarray(explainer.shap_values(sampled), dtype=float)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ExplainabilityError(f"SHAP could not explain model {model.name}: {exc}") from exc
    if values.ndim == 3:
        values = values[..., 0]
    importance = pd.DataFrame(
        {
            "feature": model.feature_names(),
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    importance_path = output_dir / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    summary_path = output_dir / "shap_summary.png"
    plot_frame = _plot_frame(sampled)
    shap.summary_plot(values, plot_frame, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140, bbox_inches="tight")
    plt.close()

    dependence_paths: list[Path] = []
    for feature in importance["feature"].head(top_features):
        feature_index = model.feature_names().index(str(feature))
        path = output_dir / f"shap_dependence_{feature}.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(plot_frame[str(feature)], values[:, feature_index], s=12, alpha=0.6)
        plt.xlabel(str(feature))
        plt.ylabel("SHAP value")
        plt.title(f"SHAP dependence: {feature}")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        dependence_paths.append(path)

    selected = min(local_rows, len(sampled))
    local_records: list[dict[str, Any]] = []
    for row_index in range(selected):
        contributions = {
            feature: float(values[row_index, index])
            for index, feature in enumerate(model.feature_names())
        }
        local_records.append(
            {
                "sample_index": str(sampled.index[row_index]),
                "top_contributions": dict(
                    sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
                ),
            }
        )
    local_path = output_dir / "shap_local_explanations.json"
    local_path.write_text(json.dumps(local_records, indent=2), encoding="utf-8")
    metadata_path = output_dir / "shap_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model.name,
                "model_version": model.model_version,
                "run_id": model.run_id,
                "sample_rows": len(sampled),
                "random_seed": random_seed,
                "statement": "SHAP describes model behaviour and does not establish causality.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ShapArtifacts(
        importance_csv=importance_path,
        summary_plot=summary_path,
        dependence_plots=tuple(dependence_paths),
        local_explanations=local_path,
        metadata=metadata_path,
        sample_rows=len(sampled),
    )


def _plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["category", "object", "string"]).columns:
        result[column] = pd.Categorical(result[column]).codes.astype(float)
    return result
