from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .test_case import TestCase


class Benchmark(Protocol):
    """Minimal benchmark loader contract used by runner-facing code."""

    name: str

    def load_cases(self, *, split: str = "train") -> list[TestCase]:
        ...


@dataclass(frozen=True)
class LoadedBenchmark:
    name: str
    cases: tuple[TestCase, ...]

    def load_cases(self, *, split: str = "train") -> list[TestCase]:
        del split
        return list(self.cases)
