# Model card

## Summary

GridMind forecasts hourly demand, solar generation, wind generation, total renewable generation,
and net load for configured electricity-grid regions. It also produces anomaly detections and
simulated battery-dispatch schedules from persisted forecast data.

## Models and targets

Forecasting uses LightGBM and CatBoost regressors with target-specific non-negativity and feature
contracts. MLflow records runs and supports conservative candidate/champion aliases. Battery
optimization uses SciPy MILP with HiGHS; it is not a learned control policy.

## Training and evaluation

- Chronological rolling validation keeps future observations out of feature construction.
- Lag and rolling features are past-only and do not bridge unexplained timestamp gaps.
- Optional Optuna tuning is confined to older inner windows.
- Final evaluation windows remain untouched until selection.
- Realistic future weather is required for weather-aware target prediction.

## Reported evaluation context

The README reports observed local experiment outputs, not universal performance guarantees. Demand
and net-load results are chronological forecast evaluation. Anomaly figures are synthetic-injection
backtests. Battery figures are a single 24-hour PJM simulation, not market-settlement or savings
claims.

## Intended use

Use outputs for exploratory analysis, model-development demonstrations, and decision support with
human review. Validate inputs, regional context, uncertainty, and operational constraints before
any real-world decision.

## Out-of-scope and risks

The project does not make causal claims, confirm incidents, control batteries, issue EMS/SCADA
commands, or model complete market, network, telemetry, regulatory, or electrochemical behavior.
EIA revisions, weather-provider availability, regional representativeness, recursive error, and
limited incident labels remain material limitations.
