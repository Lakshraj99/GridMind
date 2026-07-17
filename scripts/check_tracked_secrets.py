#!/usr/bin/env python3
"""Fail when Git-tracked text files contain common credential signatures.

This lightweight guard is intentionally conservative and is not a replacement for
repository-wide history scanning or a managed secret-scanning service.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
DOTENV_KEY_PATTERN = re.compile(r"(?im)^[ \t]*(?:EIA_API_KEY|GRIDMIND_API_KEY)[ \t]*=[ \t]*(?!#|$)")


def tracked_files() -> list[Path]:
    """Return repository files known to Git without trusting untracked inputs."""
    output = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(item) for item in output.decode().split("\0") if item]


def is_dotenv_file(path: Path) -> bool:
    """Identify dotenv-style files without treating Python settings as secrets."""
    return path.name == ".env" or path.name.startswith(".env.") or path.suffix == ".env"


def main() -> int:
    findings: list[tuple[Path, int, str]] = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((path, line_number, label))
            if is_dotenv_file(path) and DOTENV_KEY_PATTERN.search(line):
                findings.append((path, line_number, "non-empty GridMind/EIA key"))
    if not findings:
        print("No common credential signatures found in Git-tracked text files.")
        return 0
    print("Potential credentials found; review and rotate as appropriate:", file=sys.stderr)
    for path, line_number, label in findings:
        print(f"- {path}:{line_number} ({label})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
