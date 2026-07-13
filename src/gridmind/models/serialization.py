"""Portable model-bundle serialization and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from gridmind.exceptions import ModelSerializationError
from gridmind.models.protocols import TrainableForecastModel


@dataclass
class ModelBundle:
    """Complete fitted model plus reproducibility and input-contract metadata."""

    model: TrainableForecastModel
    metadata: dict[str, Any]
    package_versions: dict[str, str]


class GridMindPythonModel(PythonModel):
    """MLflow pyfunc wrapper around GridMind's recursive history-to-forecast contract."""

    def __init__(self, model: TrainableForecastModel) -> None:
        self.model = model

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Forecast from canonical history; pass horizon as a pyfunc parameter."""
        del context
        horizon = int((params or {}).get("horizon", 24))
        return self.model.predict(model_input, horizon=horizon)


def save_model_bundle(
    model: TrainableForecastModel,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize a fitted model and write adjacent feature and metadata JSON files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    versions = {
        package: _package_version(package)
        for package in ("gridmind", "pandas", "lightgbm", "catboost", "mlforecast")
    }
    bundle_metadata = {
        "model_name": model.name,
        "required_history": model.specification.required_history,
        "target_name": model.specification.target_name,
        "frequency": model.specification.frequency,
        **(metadata or {}),
    }
    bundle = ModelBundle(model=model, metadata=bundle_metadata, package_versions=versions)
    try:
        joblib.dump(bundle, path)
    except (OSError, TypeError, ValueError) as exc:
        raise ModelSerializationError(f"Could not save model bundle to {path}: {exc}") from exc
    model.specification.save(path.parent / "feature_schema.json")
    (path.parent / "model_metadata.json").write_text(
        json.dumps({**bundle_metadata, "package_versions": versions}, indent=2),
        encoding="utf-8",
    )
    return path


def load_model_bundle(path: Path) -> ModelBundle:
    """Load and validate a complete GridMind model bundle."""
    if not path.exists():
        raise ModelSerializationError(f"Model bundle does not exist: {path}")
    try:
        bundle = joblib.load(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ModelSerializationError(f"Could not load model bundle {path}: {exc}") from exc
    if not isinstance(bundle, ModelBundle) or not isinstance(bundle.model, TrainableForecastModel):
        raise ModelSerializationError(f"File is not a valid GridMind model bundle: {path}")
    if bundle.model.feature_names() != list(bundle.model.specification.feature_names):
        raise ModelSerializationError("Serialized feature order does not match its specification.")
    return bundle


def log_mlflow_model(model: TrainableForecastModel) -> Any:
    """Log the complete forecasting object in MLflow's valid pyfunc model format."""
    return mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=GridMindPythonModel(model),
        pip_requirements=[
            "gridmind",
            "pandas>=2.2,<3",
            "lightgbm>=4.5,<5",
            "catboost>=1.2,<2",
        ],
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"
