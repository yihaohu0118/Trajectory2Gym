"""Thin Tau2 parser and official-Gym integration facade."""

from .adapter import parse_tau2_episode
from .official import Tau2OfficialManager, build_manager

__all__ = ["Tau2OfficialManager", "build_manager", "parse_tau2_episode"]
