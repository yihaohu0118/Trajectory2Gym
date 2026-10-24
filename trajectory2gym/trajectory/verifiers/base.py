from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trajectory2gym.trajectory.decision import Decision
from trajectory2gym.trajectory.ir import TrajectoryIR


class VerifierPlugin(ABC):
    """Common verifier interface for trajectory-evidence plugins."""

    name = "verifier"
    grounding = "unspecified"

    @abstractmethod
    def verify(self, action: Any, state: dict[str, Any], trajectory_ir: TrajectoryIR) -> Decision:
        raise NotImplementedError
