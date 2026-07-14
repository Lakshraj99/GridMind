"""MILP physical-constraint matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from gridmind.optimization.contracts import BatterySpecification


def physical_constraints(
    frame: pd.DataFrame,
    spec: BatterySpecification,
    *,
    duration_hours: float,
    prior_daily_throughput_mwh: dict[str, float] | None = None,
) -> list[LinearConstraint]:
    count = len(frame)
    variables = 4 * count + 1
    charge = slice(0, count)
    discharge = slice(count, 2 * count)
    soc = slice(2 * count, 3 * count)
    binary = slice(3 * count, 4 * count)
    peak_index = 4 * count
    constraints: list[LinearConstraint] = []

    transition = np.zeros((count, variables))
    decay = (1.0 - spec.self_discharge_rate) ** duration_hours
    for index in range(count):
        transition[index, charge.start + index] = -spec.charge_efficiency * duration_hours
        transition[index, discharge.start + index] = duration_hours / spec.discharge_efficiency
        transition[index, soc.start + index] = 1.0
        if index:
            transition[index, soc.start + index - 1] = -decay
    rhs = np.zeros(count)
    rhs[0] = decay * spec.initial_soc_mwh
    constraints.append(LinearConstraint(transition, rhs, rhs))

    exclusivity = np.zeros((2 * count, variables))
    for index in range(count):
        exclusivity[index, charge.start + index] = 1.0
        exclusivity[index, binary.start + index] = -spec.max_charge_mw
        exclusivity[count + index, discharge.start + index] = 1.0
        exclusivity[count + index, binary.start + index] = spec.max_discharge_mw
    constraints.append(
        LinearConstraint(
            exclusivity,
            np.full(2 * count, -np.inf),
            np.concatenate([np.zeros(count), np.full(count, spec.max_discharge_mw)]),
        )
    )

    peak_rows = np.zeros((count, variables))
    net_load = frame["net_load_before_battery_mw"].to_numpy(dtype=float)
    for index in range(count):
        peak_rows[index, charge.start + index] = 1.0
        peak_rows[index, discharge.start + index] = -1.0
        peak_rows[index, peak_index] = -1.0
    constraints.append(LinearConstraint(peak_rows, np.full(count, -np.inf), -net_load))

    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    for _, positions in frame.groupby(timestamps.dt.floor("D"), sort=True).groups.items():
        throughput = np.zeros((1, variables))
        for position in positions:
            throughput[0, charge.start + int(position)] = duration_hours
            throughput[0, discharge.start + int(position)] = duration_hours
        day_key = pd.Timestamp(timestamps.iloc[int(next(iter(positions)))]).strftime("%Y-%m-%d")
        previously_used = (prior_daily_throughput_mwh or {}).get(day_key, 0.0)
        limit = 2.0 * spec.capacity_mwh * spec.maximum_daily_cycles - previously_used
        constraints.append(LinearConstraint(throughput, -np.inf, limit))

    terminal = np.zeros((1, variables))
    terminal[0, soc.stop - 1] = 1.0
    constraints.append(
        LinearConstraint(terminal, spec.terminal_soc_target_mwh, spec.terminal_soc_target_mwh)
    )
    return constraints
