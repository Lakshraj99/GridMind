# Data card

## Sources

- **EIA v2** supplies balancing-authority electricity measurements, including actual demand,
  forecast demand, net generation, and total interchange.
- **Open-Meteo** supplies historical and forecast weather features.

Use of either source remains subject to its provider terms, availability, revisions, and attribution
requirements. GridMind does not redistribute provider data as a committed repository artifact.

## Scope and resolution

The canonical observation contract is region-level, hourly, and UTC. Demonstrated PJM work uses
the inclusive 2023-01-01 to 2025-12-31 interval where available. Other configured regions and
periods may have different coverage and semantics.

## Processing safeguards

- Incoming data are normalized to UTC and checked with strict Pandera contracts.
- Pagination is verified at API boundaries; duplicate and conflicting records are reported.
- Data-quality reports reconcile source records, pivoted timestamps, duplicates, invalid rows, and
  retained rows.
- Missing actual demand is never fabricated. The `error` policy stops after producing a report;
  `drop` writes rows to quarantine and preserves resulting gaps.
- Weather and feature workflows detect gaps so lags and rolling windows do not imply continuity.

## Representativeness and limitations

These are regional aggregate measurements, not feeder-level telemetry or a complete operational
state estimate. Historical weather may differ from forecast-time availability. Source revisions and
changing reporting practices can alter future results. No committed dataset should be treated as a
benchmark license grant or a substitute for current source retrieval.
