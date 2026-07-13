"""UTC continuity detection and gap-aware forecast-window selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gridmind.exceptions import FeatureEngineeringError, InsufficientHistoryError
from gridmind.time_utils import format_utc_timestamp

HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class ContinuityResult:
    """Rows annotated with stable segment IDs plus segment-level metadata."""

    frame: pd.DataFrame
    segments: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        """Return machine-readable dataset continuity totals."""
        longest = self.segments.sort_values(
            ["row_count", "segment_start"], ascending=[False, False]
        ).iloc[0]
        return {
            "region_count": int(self.frame["region"].nunique()),
            "timestamp_gap_count": int(self.segments["missing_expected_hours_before"].sum()),
            "gap_event_count": int((self.segments["gap_before_hours"] > 1.0).sum()),
            "contiguous_segment_count": len(self.segments),
            "longest_contiguous_segment_rows": int(longest["row_count"]),
            "longest_contiguous_segment_hours": float(longest["duration_hours"]),
            "longest_contiguous_segment": str(longest["region_segment_id"]),
            "most_recent_contiguous_segment": _segment_record(
                self.segments.sort_values("segment_end").iloc[-1]
            ),
            "segments": [_segment_record(row) for _, row in self.segments.iterrows()],
        }


@dataclass(frozen=True)
class SelectedWindow:
    """One fully eligible rolling-origin validation window."""

    origin: pd.Timestamp
    validation_timestamps: pd.DatetimeIndex
    segments_by_region: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-safe window metadata."""
        return {
            "origin": format_utc_timestamp(self.origin),
            "validation_start": format_utc_timestamp(self.validation_timestamps[0]),
            "validation_end": format_utc_timestamp(self.validation_timestamps[-1]),
            "horizon": len(self.validation_timestamps),
            "segments_by_region": dict(self.segments_by_region),
        }


@dataclass(frozen=True)
class WindowSelection:
    """Accepted windows and the complete candidate-origin audit trail."""

    windows: tuple[SelectedWindow, ...]
    candidates: tuple[dict[str, Any], ...]
    requested_windows: int
    horizon: int
    required_history: int
    step_size: int

    @property
    def origins(self) -> tuple[pd.Timestamp, ...]:
        """Return accepted origins in chronological order."""
        return tuple(window.origin for window in self.windows)

    def to_dict(self) -> dict[str, Any]:
        """Return selected and rejected origins in machine-readable form."""
        accepted = [window.to_dict() for window in self.windows]
        rejected = [candidate for candidate in self.candidates if not candidate["accepted"]]
        return {
            "requested_windows": self.requested_windows,
            "available_valid_windows": len(self.windows),
            "horizon": self.horizon,
            "required_history": self.required_history,
            "step_size": self.step_size,
            "candidate_origins_considered": len(self.candidates),
            "accepted_origins": [item["origin"] for item in accepted],
            "rejected_origins": rejected,
            "selected_validation_windows": accepted,
        }


def detect_contiguous_segments(frame: pd.DataFrame) -> ContinuityResult:
    """Assign hourly segments independently per region using UTC timestamps."""
    required = {"region", "timestamp_utc"}
    missing = required.difference(frame.columns)
    if missing:
        raise FeatureEngineeringError(f"Continuity source is missing columns: {sorted(missing)}")
    annotated = frame.copy()
    annotated["timestamp_utc"] = pd.to_datetime(
        annotated["timestamp_utc"], utc=True, errors="raise"
    )
    annotated["region"] = annotated["region"].astype("string")
    annotated = annotated.sort_values(["region", "timestamp_utc"], ignore_index=True)
    if annotated.empty:
        raise InsufficientHistoryError("No timestamp observations are available.")
    if annotated.duplicated(["region", "timestamp_utc"]).any():
        raise FeatureEngineeringError("Continuity source contains duplicate region/timestamps.")

    differences = annotated.groupby("region", observed=True)["timestamp_utc"].diff()
    starts = differences.isna() | differences.ne(HOUR)
    ordinals = starts.groupby(annotated["region"], observed=True).cumsum().astype(int)
    annotated["region_segment_id"] = [
        f"{region}::{ordinal:04d}"
        for region, ordinal in zip(annotated["region"], ordinals, strict=True)
    ]

    rows: list[dict[str, Any]] = []
    previous_end: dict[str, pd.Timestamp] = {}
    for (region, segment_id), group in annotated.groupby(
        ["region", "region_segment_id"], sort=True, observed=True
    ):
        start = pd.Timestamp(group["timestamp_utc"].iloc[0])
        end = pd.Timestamp(group["timestamp_utc"].iloc[-1])
        prior = previous_end.get(str(region))
        gap_before = float((start - prior) / HOUR) if prior is not None else 0.0
        missing_hours = max(int(gap_before) - 1, 0) if gap_before > 1 else 0
        rows.append(
            {
                "region": str(region),
                "region_segment_id": str(segment_id),
                "segment_start": start,
                "segment_end": end,
                "row_count": len(group),
                "duration": end - start + HOUR,
                "duration_hours": float((end - start) / HOUR) + 1.0,
                "gap_before_segment": start - prior if prior is not None else pd.NaT,
                "gap_before_hours": gap_before,
                "missing_expected_hours_before": missing_hours,
            }
        )
        previous_end[str(region)] = end
    return ContinuityResult(annotated, pd.DataFrame(rows))


def select_gap_aware_windows(
    frame: pd.DataFrame,
    *,
    horizon: int,
    windows: int,
    step_size: int,
    required_history: int,
) -> WindowSelection:
    """Search backward for the requested number of fully contiguous shared windows."""
    if min(horizon, windows, step_size, required_history) <= 0:
        raise ValueError("Horizon, windows, step size, and required history must be positive.")
    continuity = detect_contiguous_segments(frame)
    annotated = continuity.frame
    regions = sorted(str(value) for value in annotated["region"].unique())
    timestamp_sets = {
        region: set(
            pd.DatetimeIndex(annotated.loc[annotated["region"] == region, "timestamp_utc"]).tolist()
        )
        for region in regions
    }
    segment_lookup = {
        region: annotated.loc[annotated["region"] == region].set_index("timestamp_utc")[
            "region_segment_id"
        ]
        for region in regions
    }
    maximums = annotated.groupby("region", observed=True)["timestamp_utc"].max()
    minimums = annotated.groupby("region", observed=True)["timestamp_utc"].min()
    cursor = pd.Timestamp(maximums.min()) - pd.Timedelta(hours=horizon)
    earliest = pd.Timestamp(minimums.max()) + pd.Timedelta(hours=required_history - 1)
    accepted_descending: list[SelectedWindow] = []
    candidates: list[dict[str, Any]] = []

    while cursor >= earliest and len(accepted_descending) < windows:
        validation = pd.date_range(cursor + HOUR, periods=horizon, freq="h")
        history = pd.date_range(end=cursor, periods=required_history, freq="h")
        reasons: list[str] = []
        segments: dict[str, str] = {}
        for region in regions:
            missing_history = sum(timestamp not in timestamp_sets[region] for timestamp in history)
            missing_validation = sum(
                timestamp not in timestamp_sets[region] for timestamp in validation
            )
            if missing_history:
                reasons.append(
                    f"{region}:insufficient_contiguous_history:{missing_history}_missing_hours"
                )
            if missing_validation:
                reasons.append(
                    f"{region}:incomplete_forecast_horizon:{missing_validation}_missing_hours"
                )
            if not missing_history and not missing_validation:
                segment = str(segment_lookup[region].loc[cursor])
                if str(segment_lookup[region].loc[validation[-1]]) != segment:
                    reasons.append(f"{region}:window_crosses_segment_boundary")
                else:
                    segments[region] = segment
        accepted = not reasons
        candidates.append(
            {
                "origin": format_utc_timestamp(cursor),
                "accepted": accepted,
                "rejection_reasons": reasons,
                "segments_by_region": segments,
            }
        )
        if accepted:
            accepted_descending.append(SelectedWindow(cursor, validation, segments))
            cursor -= pd.Timedelta(hours=step_size)
        else:
            cursor -= HOUR

    selected = tuple(reversed(accepted_descending))
    result = WindowSelection(
        windows=selected,
        candidates=tuple(candidates),
        requested_windows=windows,
        horizon=horizon,
        required_history=required_history,
        step_size=step_size,
    )
    if len(selected) != windows:
        summary = continuity.summary()
        recent = summary["most_recent_contiguous_segment"]
        raise InsufficientHistoryError(
            "Insufficient gap-safe validation windows: "
            f"requested windows={windows}, available valid windows={len(selected)}, "
            f"horizon={horizon}, required history={required_history}, "
            f"missing expected hours={summary['timestamp_gap_count']}, "
            f"most recent contiguous segment={recent['region_segment_id']} "
            f"({recent['segment_start']} to {recent['segment_end']}, "
            f"rows={recent['row_count']})."
        )
    return result


def _segment_record(row: pd.Series) -> dict[str, Any]:
    """Convert one segment metadata row into canonical JSON values."""
    return {
        "region": str(row["region"]),
        "region_segment_id": str(row["region_segment_id"]),
        "segment_start": format_utc_timestamp(row["segment_start"]),
        "segment_end": format_utc_timestamp(row["segment_end"]),
        "row_count": int(row["row_count"]),
        "duration": str(row["duration"]),
        "duration_hours": float(row["duration_hours"]),
        "gap_before_segment": (
            None if pd.isna(row["gap_before_segment"]) else str(row["gap_before_segment"])
        ),
        "gap_before_hours": float(row["gap_before_hours"]),
        "missing_expected_hours_before": int(row["missing_expected_hours_before"]),
    }
