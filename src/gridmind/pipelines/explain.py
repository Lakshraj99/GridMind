"""Explain a local or registered GridMind tree model."""

from __future__ import annotations

from pathlib import Path

from gridmind.config import Settings
from gridmind.data.storage import DuckDBStorage
from gridmind.explainability.shap_analysis import ShapArtifacts, generate_shap_artifacts
from gridmind.pipelines.predict import load_prediction_bundle


def run_explain_pipeline(
    settings: Settings,
    *,
    region: str,
    model_alias: str = "champion",
    bundle_path: Path | None = None,
    output_dir: Path = Path("artifacts/explainability"),
) -> ShapArtifacts:
    """Load a selected model and create deterministic SHAP artifacts from stored history."""
    bundle = load_prediction_bundle(
        settings,
        model_alias=model_alias,
        model_version=None,
        run_id=None,
        bundle_path=bundle_path,
    )
    history = DuckDBStorage(settings.duckdb_path).read_region(region, "1900-01-01", "2100-01-01")
    return generate_shap_artifacts(
        bundle.model,
        history,
        output_dir,
        sample_size=settings.shap_sample_size,
        random_seed=settings.model_random_seed,
    )
