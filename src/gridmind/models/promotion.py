"""MLflow registry creation and conservative candidate/champion promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from gridmind.exceptions import ModelPromotionError
from gridmind.models.serialization import load_model_bundle
from gridmind.training.evaluator import relative_improvement


@dataclass(frozen=True)
class PromotionDecision:
    """Aliases assigned and the explicit outcome of the champion gate."""

    candidate_assigned: bool
    champion_promoted: bool
    reason: str


def effective_registry_uri(tracking_uri: str, database_path: Path) -> str:
    """Use SQLite for local registry operations while preserving remote/database URIs."""
    if "://" in tracking_uri and not tracking_uri.startswith("file:"):
        return tracking_uri
    absolute = database_path.resolve()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{absolute}"


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create a registered model if it does not yet exist."""
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def create_model_version(client: MlflowClient, *, model_name: str, run_id: str, source: str) -> str:
    """Create and return a registry model version for a logged MLflow model."""
    ensure_registered_model(client, model_name)
    version = client.create_model_version(name=model_name, source=source, run_id=run_id)
    return str(version.version)


def apply_promotion_gate(
    client: MlflowClient,
    *,
    registered_model_name: str,
    version: str,
    metrics: dict[str, float],
    reference_metric: float,
    primary_metric: str,
    threshold: float,
    bundle_path: Path,
) -> PromotionDecision:
    """Always assign candidate; replace champion only after every safety gate passes."""
    try:
        load_model_bundle(bundle_path)
    except Exception as exc:
        raise ModelPromotionError(f"Candidate bundle failed reload validation: {exc}") from exc
    client.set_registered_model_alias(registered_model_name, "candidate", version)
    metric = float(metrics.get(primary_metric, float("nan")))
    bias = float(metrics.get("forecast_bias", float("nan")))
    improvement = relative_improvement(metric, reference_metric)
    if not math.isfinite(metric) or not math.isfinite(bias):
        return PromotionDecision(True, False, "candidate metrics or forecast bias are not finite")
    if not math.isfinite(improvement):
        return PromotionDecision(True, False, "reference baseline metric is not comparable")
    if improvement < threshold:
        return PromotionDecision(
            True,
            False,
            f"improvement {improvement:.6f} is below threshold {threshold:.6f}",
        )
    client.set_registered_model_alias(registered_model_name, "champion", version)
    return PromotionDecision(
        True, True, f"promotion gate passed with improvement {improvement:.6f}"
    )
