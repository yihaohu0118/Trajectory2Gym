"""Thin tau-bench parser and official-environment integration facade."""

from .adapter import parse_tau_episode
from .official import TauBenchOfficialManager, build_manager

__all__ = ["TauBenchOfficialManager", "build_manager", "parse_tau_episode"]
