"""Deterministic data-quality and operational anomaly rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

import numpy as np
import pandas as pd

from gridmind.anomalies.contracts import (
    Severity,
    empty_anomaly_frame,
    make_anomaly,
    validate_anomaly_frame,
)
from gridmind.anomalies.severity import severity_from_score, severity_score
from gridmind.time_utils import format_utc_timestamp


@dataclass(frozen=True)
class RuleConfig:
    demand_change_threshold: float = 0.20
    renewable_drop_threshold: float = 0.30
    flatline_hours: int = 4
    flatline_tolerance: float = 0.0
    solar_daylight_radiation_threshold_wm2: float = 25.0
    solar_min_expected_generation_mw: float = 100.0
    solar_min_absolute_drop_mw: float = 100.0
    solar_min_drop_duration_hours: int = 2
    missing_warning_count: int = 1
    missing_critical_count: int = 3
    stale_after_hours: int = 24


class RuleDetector:
    """Apply transparent rules without repairing or interpolating observations."""

    name = "rules"
    version = "1"

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()

    def detect(
        self,
        frame: pd.DataFrame,
        *,
        target: str,
        weather: pd.DataFrame | None = None,
        renewables: pd.DataFrame | None = None,
        now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        required = {"region", "timestamp_utc", target}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Rule input is missing columns: {sorted(missing)}")
        source = frame.copy()
        source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
        if weather is not None and target == "solar_generation_mw":
            context_columns = [
                column
                for column in ("shortwave_radiation_wm2", "daylight_indicator")
                if column in weather
            ]
            if context_columns:
                context = weather[["region", "timestamp_utc", *context_columns]].copy()
                context["timestamp_utc"] = pd.to_datetime(context["timestamp_utc"], utc=True)
                context = context.drop_duplicates(["region", "timestamp_utc"], keep="last")
                source = source.merge(
                    context, on=["region", "timestamp_utc"], how="left", validate="many_to_one"
                )
        detected_at = now or pd.Timestamp.now(tz=UTC)
        events: list[dict[str, object]] = []
        events.extend(self._ordering_events(source, target, detected_at))
        for region, group in source.groupby("region", sort=True, observed=True):
            events.extend(
                self._region_events(
                    group.sort_values("timestamp_utc"), str(region), target, detected_at
                )
            )
        if weather is not None:
            events.extend(
                self._coverage_events(
                    source,
                    weather,
                    target,
                    detected_at,
                    anomaly_type="weather_grid_mismatch",
                    explanation="Grid observation has no aligned weather row.",
                )
            )
        if renewables is not None and target == "demand_mw":
            events.extend(
                self._coverage_events(
                    source,
                    renewables,
                    target,
                    detected_at,
                    anomaly_type="coverage_mismatch",
                    explanation="Demand observation has no aligned renewable-generation row.",
                )
            )
        return validate_anomaly_frame(pd.DataFrame(events)) if events else empty_anomaly_frame()

    def detect_forecast_weather_coverage(
        self,
        forecasts: pd.DataFrame,
        weather: pd.DataFrame,
        *,
        target: str,
        now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Flag forecast target instants that have no weather required for scoring."""
        if forecasts.empty:
            return empty_anomaly_frame()
        source = forecasts.loc[forecasts["target"] == target, ["region", "timestamp_utc"]]
        events = self._coverage_events(
            source,
            weather,
            target,
            now or pd.Timestamp.now(tz=UTC),
            anomaly_type="weather_grid_mismatch",
            explanation="Stored forecast horizon has no aligned weather row.",
        )
        return validate_anomaly_frame(pd.DataFrame(events)) if events else empty_anomaly_frame()

    def _ordering_events(
        self, source: pd.DataFrame, target: str, detected_at: pd.Timestamp
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for region, group in source.groupby("region", sort=True, observed=True):
            timestamps = group["timestamp_utc"]
            duplicated = group.loc[timestamps.duplicated(keep=False)]
            for timestamp in duplicated["timestamp_utc"].drop_duplicates():
                events.append(
                    make_anomaly(
                        region=str(region),
                        target=target,
                        timestamp=timestamp,
                        detector_name=self.name,
                        anomaly_type="duplicate_timestamp",
                        anomaly_score=75,
                        severity="critical",
                        explanation=(
                            "Multiple observations share the same region and UTC timestamp."
                        ),
                        detected_at=detected_at,
                    )
                )
            if not timestamps.is_monotonic_increasing:
                bad = timestamps.loc[timestamps.diff() < pd.Timedelta(0)]
                for timestamp in bad:
                    events.append(
                        make_anomaly(
                            region=str(region),
                            target=target,
                            timestamp=timestamp,
                            detector_name=self.name,
                            anomaly_type="non_monotonic_timestamp",
                            anomaly_score=55,
                            severity="warning",
                            explanation="Source timestamp order moves backwards.",
                            detected_at=detected_at,
                        )
                    )
        return events

    def _region_events(
        self,
        group: pd.DataFrame,
        region: str,
        target: str,
        detected_at: pd.Timestamp,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        unique = group.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
        timestamps = pd.DatetimeIndex(unique["timestamp_utc"])
        if len(timestamps):
            deltas = pd.Series(timestamps).diff()
            for position in np.flatnonzero(
                (deltas.notna() & deltas.ne(pd.Timedelta(hours=1))).to_numpy()
            ):
                events.append(
                    make_anomaly(
                        region=region,
                        target=target,
                        timestamp=timestamps[position],
                        detector_name=self.name,
                        anomaly_type="unexpected_frequency",
                        anomaly_score=50,
                        severity="warning",
                        threshold=1.0,
                        explanation=(
                            f"Adjacent observations are separated by {deltas.iloc[position]}; "
                            "expected one hour."
                        ),
                        detected_at=detected_at,
                    )
                )
            expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h", tz="UTC")
            absent = expected.difference(timestamps)
            for timestamp in absent:
                gap = self._gap_size(timestamp, timestamps)
                severity: Severity = (
                    "critical" if gap >= self.config.missing_critical_count else "warning"
                )
                events.append(
                    make_anomaly(
                        region=region,
                        target=target,
                        timestamp=timestamp,
                        detector_name=self.name,
                        anomaly_type="missing_timestamp",
                        anomaly_score=80 if severity == "critical" else 45,
                        severity=severity,
                        threshold=float(self.config.missing_warning_count),
                        explanation=(
                            f"Expected hourly UTC observation is absent inside a {gap}-hour gap."
                        ),
                        metadata={"gap_hours": gap},
                        detected_at=detected_at,
                    )
                )
        numeric = pd.to_numeric(unique[target], errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric)
        invalid |= numeric < 0 if target != "net_load_mw" else False
        for index in unique.index[invalid]:
            events.append(
                make_anomaly(
                    region=region,
                    target=target,
                    timestamp=unique.at[index, "timestamp_utc"],
                    detector_name=self.name,
                    anomaly_type="invalid_value",
                    anomaly_score=90,
                    severity="critical",
                    observed_value=self._finite_or_none(numeric.loc[index]),
                    explanation=f"{target} is non-finite or outside its valid sign contract.",
                    detected_at=detected_at,
                )
            )
        events.extend(self._weather_range_events(unique, region, target, detected_at))
        events.extend(self._change_events(unique, numeric, region, target, detected_at))
        events.extend(self._flatline_events(unique, numeric, region, target, detected_at))
        if len(timestamps) and detected_at - timestamps.max() > pd.Timedelta(
            hours=self.config.stale_after_hours
        ):
            events.append(
                make_anomaly(
                    region=region,
                    target=target,
                    timestamp=timestamps.max(),
                    detector_name=self.name,
                    anomaly_type="stale_observation",
                    anomaly_score=40,
                    severity="warning",
                    threshold=float(self.config.stale_after_hours),
                    explanation="Latest observation is older than the configured freshness limit.",
                    detected_at=detected_at,
                )
            )
        return events

    def _change_events(
        self,
        group: pd.DataFrame,
        values: pd.Series,
        region: str,
        target: str,
        detected_at: pd.Timestamp,
    ) -> list[dict[str, object]]:
        if target == "solar_generation_mw":
            return self._solar_drop_events(group, values, region, target, detected_at)
        prior = values.shift(1)
        consecutive = group["timestamp_utc"].diff().eq(pd.Timedelta(hours=1))
        change = (values - prior) / prior.abs().replace(0, np.nan)
        events: list[dict[str, object]] = []
        for position in np.flatnonzero((consecutive & change.notna()).to_numpy()):
            amount = float(change.iloc[position])
            threshold = (
                self.config.renewable_drop_threshold
                if "generation" in target
                else self.config.demand_change_threshold
            )
            anomaly_type = ""
            if amount >= threshold and target == "demand_mw":
                anomaly_type = "demand_spike"
            elif amount <= -threshold and target == "demand_mw":
                anomaly_type = "demand_drop"
            elif amount <= -threshold and "generation" in target:
                anomaly_type = "renewable_drop"
            if not anomaly_type:
                continue
            score = severity_score(magnitude=abs(amount) / max(threshold, 1e-12))
            events.append(
                make_anomaly(
                    region=region,
                    target=target,
                    timestamp=group["timestamp_utc"].iloc[position],
                    detector_name=self.name,
                    anomaly_type=anomaly_type,
                    anomaly_score=score,
                    severity=severity_from_score(score),
                    observed_value=float(values.iloc[position]),
                    expected_value=float(prior.iloc[position]),
                    residual=float(values.iloc[position] - prior.iloc[position]),
                    threshold=threshold,
                    explanation=f"Consecutive-hour change of {amount:.1%} exceeds {threshold:.1%}.",
                    detected_at=detected_at,
                )
            )
        return events

    def _solar_drop_events(
        self,
        group: pd.DataFrame,
        values: pd.Series,
        region: str,
        target: str,
        detected_at: pd.Timestamp,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        position = 1
        while position < len(group):
            expected = float(values.iloc[position - 1])
            observed = float(values.iloc[position])
            relative_drop = (expected - observed) / abs(expected) if expected else 0.0
            qualifies = (
                group["timestamp_utc"].iloc[position] - group["timestamp_utc"].iloc[position - 1]
                == pd.Timedelta(hours=1)
                and self._solar_daylight(group.iloc[position])
                and expected >= self.config.solar_min_expected_generation_mw
                and expected - observed >= self.config.solar_min_absolute_drop_mw
                and relative_drop >= self.config.renewable_drop_threshold
            )
            if not qualifies:
                position += 1
                continue
            end_position = position
            while end_position + 1 < len(group):
                candidate = end_position + 1
                candidate_value = float(values.iloc[candidate])
                if not (
                    group["timestamp_utc"].iloc[candidate]
                    - group["timestamp_utc"].iloc[candidate - 1]
                    == pd.Timedelta(hours=1)
                    and self._solar_daylight(group.iloc[candidate])
                    and expected - candidate_value >= self.config.solar_min_absolute_drop_mw
                    and (expected - candidate_value) / abs(expected)
                    >= self.config.renewable_drop_threshold
                ):
                    break
                end_position = candidate
            duration = end_position - position + 1
            if duration >= self.config.solar_min_drop_duration_hours:
                score = severity_score(magnitude=relative_drop, duration_hours=duration)
                start_timestamp = group["timestamp_utc"].iloc[position]
                end_timestamp = group["timestamp_utc"].iloc[end_position]
                events.append(
                    make_anomaly(
                        region=region,
                        target=target,
                        timestamp=start_timestamp,
                        detector_name=self.name,
                        anomaly_type="renewable_drop",
                        anomaly_score=score,
                        severity=severity_from_score(score),
                        observed_value=observed,
                        expected_value=expected,
                        residual=observed - expected,
                        threshold=self.config.renewable_drop_threshold,
                        explanation=(
                            f"Daylight solar generation remained at least {relative_drop:.1%} "
                            f"below {expected:.1f} MW for {duration} consecutive hours."
                        ),
                        metadata={
                            "drop_start_utc": format_utc_timestamp(start_timestamp),
                            "drop_end_utc": format_utc_timestamp(end_timestamp),
                            "duration_hours": duration,
                            "supporting_observation_count": duration,
                            "radiation_threshold_wm2": (
                                self.config.solar_daylight_radiation_threshold_wm2
                            ),
                            "minimum_expected_generation_mw": (
                                self.config.solar_min_expected_generation_mw
                            ),
                            "minimum_absolute_drop_mw": self.config.solar_min_absolute_drop_mw,
                        },
                        detected_at=detected_at,
                    )
                )
            position = max(end_position + 1, position + 1)
        return events

    def _flatline_events(
        self,
        group: pd.DataFrame,
        values: pd.Series,
        region: str,
        target: str,
        detected_at: pd.Timestamp,
    ) -> list[dict[str, object]]:
        within_tolerance = values.diff().abs().le(self.config.flatline_tolerance)
        consecutive = group["timestamp_utc"].diff().eq(pd.Timedelta(hours=1))
        runs = (~(within_tolerance & consecutive)).cumsum()
        events: list[dict[str, object]] = []
        for _, positions in group.groupby(runs, observed=True).groups.items():
            if len(positions) < self.config.flatline_hours:
                continue
            position_list = [int(position) for position in positions]
            flatline = group.loc[position_list]
            if target == "solar_generation_mw":
                first_position = position_list[0]
                expected = (
                    float(values.iloc[first_position - 1])
                    if first_position > 0
                    else float(values.loc[position_list[0]])
                )
                if (
                    expected < self.config.solar_min_expected_generation_mw
                    or not flatline.apply(self._solar_daylight, axis=1).all()
                ):
                    continue
            last = positions[-1]
            first = positions[0]
            start_timestamp = group.loc[first, "timestamp_utc"]
            end_timestamp = group.loc[last, "timestamp_utc"]
            duration = len(positions)
            events.append(
                make_anomaly(
                    region=region,
                    target=target,
                    timestamp=group.loc[last, "timestamp_utc"],
                    detector_name=self.name,
                    anomaly_type="flatline",
                    anomaly_score=50,
                    severity="warning",
                    observed_value=self._finite_or_none(values.loc[last]),
                    threshold=float(self.config.flatline_hours),
                    explanation=(
                        f"Value remained within {self.config.flatline_tolerance:g} for "
                        f"{duration} consecutive hourly observations."
                    ),
                    metadata={
                        "flatline_start_utc": format_utc_timestamp(start_timestamp),
                        "flatline_end_utc": format_utc_timestamp(end_timestamp),
                        "duration_hours": duration,
                        "supporting_observation_count": duration,
                        "tolerance": self.config.flatline_tolerance,
                        "minimum_required_hours": self.config.flatline_hours,
                    },
                    detected_at=detected_at,
                )
            )
        return events

    def _solar_daylight(self, row: pd.Series) -> bool:
        if (
            "daylight_indicator" in row.index
            and pd.notna(row["daylight_indicator"])
            and not bool(row["daylight_indicator"])
        ):
            return False
        if "shortwave_radiation_wm2" not in row.index or pd.isna(row["shortwave_radiation_wm2"]):
            return False
        return (
            float(row["shortwave_radiation_wm2"])
            >= self.config.solar_daylight_radiation_threshold_wm2
        )

    def _weather_range_events(
        self, group: pd.DataFrame, region: str, target: str, detected_at: pd.Timestamp
    ) -> list[dict[str, object]]:
        ranges = {
            "relative_humidity_pct": (0.0, 100.0),
            "cloud_cover_pct": (0.0, 100.0),
            "shortwave_radiation_wm2": (0.0, 1500.0),
            "direct_radiation_wm2": (0.0, 1500.0),
            "diffuse_radiation_wm2": (0.0, 1500.0),
        }
        events: list[dict[str, object]] = []
        for column, (lower, upper) in ranges.items():
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            for index in group.index[values.notna() & ~values.between(lower, upper)]:
                events.append(
                    make_anomaly(
                        region=region,
                        target=target,
                        timestamp=group.at[index, "timestamp_utc"],
                        detector_name=self.name,
                        anomaly_type="invalid_value",
                        anomaly_score=80,
                        severity="critical",
                        observed_value=float(values.loc[index]),
                        explanation=f"{column} is outside [{lower}, {upper}].",
                        feature_summary={"feature": column},
                        detected_at=detected_at,
                    )
                )
        return events

    def _coverage_events(
        self,
        source: pd.DataFrame,
        counterpart: pd.DataFrame,
        target: str,
        detected_at: pd.Timestamp,
        *,
        anomaly_type: str,
        explanation: str,
    ) -> list[dict[str, object]]:
        counterpart_keys = counterpart[["region", "timestamp_utc"]].copy()
        counterpart_keys["timestamp_utc"] = pd.to_datetime(
            counterpart_keys["timestamp_utc"], utc=True
        )
        merged = (
            source[["region", "timestamp_utc"]]
            .drop_duplicates()
            .merge(
                counterpart_keys.drop_duplicates(),
                on=["region", "timestamp_utc"],
                how="left",
                indicator=True,
            )
        )
        return [
            make_anomaly(
                region=str(row.region),
                target=target,
                timestamp=row.timestamp_utc,
                detector_name=self.name,
                anomaly_type=anomaly_type,
                anomaly_score=45,
                severity="warning",
                explanation=explanation,
                detected_at=detected_at,
            )
            for row in merged.loc[merged["_merge"] == "left_only"].itertuples()
        ]

    @staticmethod
    def _gap_size(timestamp: pd.Timestamp, available: pd.DatetimeIndex) -> int:
        before = available[available < timestamp]
        after = available[available > timestamp]
        if before.empty or after.empty:
            return 1
        return max(int((after.min() - before.max()) / pd.Timedelta(hours=1)) - 1, 1)

    @staticmethod
    def _finite_or_none(value: object) -> float | None:
        numeric = float(np.asarray(value).item())
        return numeric if np.isfinite(numeric) else None
