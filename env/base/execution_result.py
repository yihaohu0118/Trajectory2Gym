from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    messages: tuple[dict[str, Any], ...] = ()
    reward: float | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
