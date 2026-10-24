from __future__ import annotations

from typing import Any

from trajectory2gym.trajectory.compiler import Trajectory2GymCompiler
from trajectory2gym.trajectory.decision import Decision
from trajectory2gym.trajectory.ir import TrajectoryIR
from trajectory2gym.trajectory.schema import TrajectoryEpisode
from trajectory2gym.trajectory.verifiers.base import VerifierPlugin


class CompilerVerifier(VerifierPlugin):
    """Thin plugin facade over the shared trajectory compiler."""

    name = "CompilerVerifier"
    grounding = "trajectory_replay + observation_database"

    def __init__(self, episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...] = ()):
        self.episodes = tuple(episodes)

    def verify(self, action: Any, state: dict[str, Any], trajectory_ir: TrajectoryIR) -> Decision:
        step_index = int(state.get("step_index", 0))
        episodes = self.episodes or (trajectory_ir.to_episode(),)
        return Trajectory2GymCompiler(episodes).verify(trajectory_ir, step_index, action)
