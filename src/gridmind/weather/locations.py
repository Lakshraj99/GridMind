"""Validated YAML-backed grid-region weather locations."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gridmind.exceptions import WeatherLocationError


class WeatherLocation(BaseModel):
    """One representative coordinate and normalized regional weight."""

    model_config = ConfigDict(frozen=True)
    name: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    weight: float = Field(gt=0)


class RegionLocationMapping(BaseModel):
    """Aggregation policy and representative locations for one grid region."""

    model_config = ConfigDict(frozen=True)
    region: str
    version: str
    source: str
    rationale: str
    aggregation: str = "weighted_mean"
    locations: tuple[WeatherLocation, ...]

    @field_validator("locations")
    @classmethod
    def _require_locations(cls, value: tuple[WeatherLocation, ...]) -> tuple[WeatherLocation, ...]:
        if not value:
            raise ValueError("At least one weather location is required.")
        names = [location.name for location in value]
        if len(names) != len(set(names)):
            raise ValueError("Weather location names must be unique within a region.")
        total = sum(location.weight for location in value)
        return tuple(
            location.model_copy(update={"weight": location.weight / total}) for location in value
        )

    @model_validator(mode="after")
    def _supported_aggregation(self) -> RegionLocationMapping:
        if self.aggregation != "weighted_mean":
            raise ValueError("Only weighted_mean weather aggregation is supported.")
        return self


def load_region_locations(path: Path, region: str) -> RegionLocationMapping:
    """Load and validate one region without hardcoded Python coordinates."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = payload["regions"][region]
        return RegionLocationMapping(
            region=region,
            version=str(payload.get("version", "unknown")),
            source=str(payload.get("source", "unspecified")),
            rationale=str(root.get("rationale", "unspecified")),
            aggregation=str(root.get("aggregation", "weighted_mean")),
            locations=tuple(WeatherLocation.model_validate(item) for item in root["locations"]),
        )
    except (OSError, KeyError, TypeError, ValidationError, yaml.YAMLError) as exc:
        raise WeatherLocationError(
            f"Could not load a valid weather-location mapping for region {region} from {path}."
        ) from exc
