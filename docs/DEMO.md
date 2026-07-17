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

After using sanitized local data, capture the dashboard/API views listed in
[`docs/images/README.md`](images/README.md). Do not capture browser profiles, terminal output,
filesystem paths, or credentials.
