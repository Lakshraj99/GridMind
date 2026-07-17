# Security

## Scope

GridMind is local decision-support software. It has no physical-control, EMS, SCADA, payment,
multi-tenant, or cloud-control interface.

## Secrets

- Put `EIA_API_KEY` and `GRIDMIND_API_KEY` only in an ignored local `.env` or deployment secret
  store. Never commit them.
- The EIA client and application logging use redaction filters for `api_key` query parameters and
  configured credential values.
- API-key comparison uses a constant-time comparison. It is optional local authentication, not an
  enterprise identity or authorization system.
- Do not log `.env` contents, request authorization headers, or browser/terminal screenshots that
  expose credentials.

## Repository checks

Run:

```bash
make check-secrets
```

It inspects tracked text files for a small set of common signatures and does not print matched
secret values. It does **not** replace a professional secret scanner, pre-commit hook, hosted
secret scanning, or a scan of Git history.

## Dependency and deployment posture

Use a locked, reviewed environment for production. The repository does not currently provide a
security-support SLA. Deployments need TLS, secret rotation, external authentication, backups,
network restrictions, image scanning, and dependency monitoring.

## Reporting a concern

Do not post suspected credentials in a public issue. Open a minimal GitHub Issue requesting a
private disclosure path, without including the secret itself. Maintainers should acknowledge,
rotate exposed credentials, assess history, and coordinate remediation before public disclosure.
