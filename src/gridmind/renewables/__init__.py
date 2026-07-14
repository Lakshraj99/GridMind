"""Renewable-generation normalization, targets, and persistence."""

from gridmind.renewables.processing import process_renewable_records
from gridmind.renewables.targets import compute_net_load

__all__ = ["compute_net_load", "process_renewable_records"]
