# Contributing

Thanks for improving GridMind. Keep changes narrow, offline-testable, UTC-aware, and explicit
about decision-support limitations.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make quality
```

Run `make check-secrets` before opening a pull request. Do not add local data, databases, MLflow
runs, trained artifacts, screenshots with sensitive information, or `.env` files.

## Pull requests

- Preserve existing milestones and public CLI/API contracts unless a migration is documented.
- Add offline tests for behavior changes.
- Keep timestamps timezone-aware UTC.
- Keep database queries in services/storage, not HTTP routes or dashboard pages.
- Do not add physical-control functionality.
- Update relevant documentation and run `make quality`.

Issues and pull requests are the project’s public collaboration channels. For security concerns,
follow [`docs/SECURITY.md`](docs/SECURITY.md).
