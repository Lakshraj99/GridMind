# Local demo guide

This guide demonstrates persisted local results without retraining. It assumes an existing
`data/gridmind.duckdb`, artifact directories, and MLflow storage when model metadata is desired.

## 1. Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
```

For a read-only demo, leave `EIA_API_KEY` blank. To enable local API authentication, set both
`API_KEY_ENABLED=true` and a private `GRIDMIND_API_KEY` value in `.env`.

## 2. Confirm local data

```bash
gridmind inspect
gridmind dispatches --region PJM
```

If no persisted data exist, use the fresh-data path below; do not expect the dashboard to fabricate
results.

## 3. Start the API and dashboard

Use two terminals:

```bash
make api
```

```bash
make dashboard
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for OpenAPI.

## 4. Exercise the API

Liveness is public. When API keys are enabled, add the header shown below to `/api/v1` requests.

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl -H 'X-API-Key: example-local-key' \
  'http://localhost:8000/api/v1/forecasts/latest?region=PJM&target=demand_mw&horizon=24'
curl -H 'X-API-Key: example-local-key' \
  'http://localhost:8000/api/v1/alerts?status=open'
curl -H 'X-API-Key: example-local-key' \
  'http://localhost:8000/api/v1/dispatches?region=PJM'
curl -H 'X-API-Key: example-local-key' http://localhost:8000/api/v1/models
curl http://localhost:8000/metrics
```

The dashboard exposes forecast, anomaly, alert, battery-dispatch, and model views through this API
only. Anomalies require review; dispatches are simulations.

## 5. Fresh-data and full-training paths

Fresh EIA ingestion requires a local EIA key:

```bash
gridmind ingest --region PJM --start-date 2023-01-01 --end-date 2025-12-31 \
  --missing-demand-policy drop
```

Weather/renewable ingestion and model training are separate, potentially expensive workflows:

```bash
gridmind weather-ingest --region PJM --start-date 2023-01-01 --end-date 2025-12-31
gridmind renewables-ingest --region PJM --start-date 2023-01-01 --end-date 2025-12-31
gridmind train-target --target demand_mw --region PJM --no-mlflow
```

Run `gridmind --help` for the full command set and option names.

## 6. Docker-only serving

Docker Compose serves existing mounted data; it does not ingest or train models:

```bash
make docker-up
```

The API readiness check remains unhealthy until `target_forecasts` exists in the mounted DuckDB
database. Shut down cleanly with:

```bash
make docker-down
```

## 7. Capture portfolio screenshots

Use only sanitized, persisted real results. The dashboard will show an empty state when a result is
unavailable; never replace it with a fabricated value.

1. Start the API and dashboard, confirm `/health/ready` reports ready, and open the dashboard at a
   browser viewport near 1440 × 900 with 100% zoom.
2. Keep the sidebar visible, select a region that the API lists as available, and click **Refresh
   data** immediately before each capture.
3. For **Overview**, select `PJM` when available. Confirm the forecast, alert, anomaly, model, and
   battery panels are populated from the API; otherwise document the professional empty state.
4. For **Forecasts**, select `PJM`, `Demand MW`, `champion`, a 24-hour horizon, and **Latest complete
   horizon**. Confirm the origin, weather mode, model version, and run ID are visible.
5. For **Anomalies**, select `PJM`, then use `All` target/severity/detector/type filters. Select a
   representative persisted detection whose explanation and lineage are available. Do not label an
   informational IsolationForest detection as an incident.
6. For **Battery dispatch**, select `PJM` and the latest persisted successful run. Confirm all three
   charts, solver status, constraint validation, battery specification, and forecast lineage render.
7. Capture API documentation separately at `http://localhost:8000/docs`; it is intentionally not a
   Streamlit page.
8. Review every frame for credentials, local filesystem paths, browser-profile details, terminal
   output, or other private metadata before saving it under `docs/images/`.

Expected filenames remain documented in [`docs/images/README.md`](images/README.md). GridMind does
not create screenshots automatically.
