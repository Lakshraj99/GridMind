"""Per-application Prometheus metrics without global registry collisions."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ApplicationMetrics:
    """GridMind HTTP, cache, and query instruments."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gridmind_http_requests_total",
            "HTTP requests",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.errors = Counter(
            "gridmind_http_errors_total",
            "HTTP error responses",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.latency = Histogram(
            "gridmind_http_request_duration_seconds",
            "HTTP request latency",
            ("method", "path"),
            registry=self.registry,
        )
        self.active = Gauge(
            "gridmind_http_active_requests", "Active HTTP requests", registry=self.registry
        )
        self.query_latency = Histogram(
            "gridmind_duckdb_query_duration_seconds",
            "DuckDB query duration",
            registry=self.registry,
        )
        self.cache = Counter(
            "gridmind_api_cache_total",
            "API cache outcomes",
            ("outcome",),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
