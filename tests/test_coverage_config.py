"""Regression checks for the canonical branch-coverage configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_ci_erases_stale_data_and_uses_branch_coverage_config() -> None:
    """Keep CI on one explicit config and prevent mixed coverage databases."""
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ignore_patterns = (root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert configuration["tool"]["coverage"]["run"]["branch"] is True
    assert workflow.index("python -m coverage erase") < workflow.index("python -m pytest")
    assert workflow.index("find . -maxdepth 2") < workflow.index("python -m pytest")
    assert "--cov-config=pyproject.toml" in workflow
    assert {".coverage", ".coverage.*", "coverage.xml", "htmlcov/"}.issubset(ignore_patterns)
