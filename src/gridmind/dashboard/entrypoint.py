"""Console entry point for the Streamlit dashboard process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gridmind.config import Settings


def main() -> None:
    settings = Settings()
    if not settings.dashboard_enabled:
        raise SystemExit("GridMind dashboard is disabled by DASHBOARD_ENABLED=false.")
    app_path = Path(__file__).with_name("app.py")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            settings.dashboard_host,
            "--server.port",
            str(settings.dashboard_port),
        ],
        check=False,
    )
    raise SystemExit(completed.returncode)
