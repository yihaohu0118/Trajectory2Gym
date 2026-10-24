from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .execution_result import ExecutionResult


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    def evaluate(self, result: ExecutionResult) -> EvaluationResult:
        ...
