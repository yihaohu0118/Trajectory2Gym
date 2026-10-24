from __future__ import annotations

from typing import Any

from .decision import Decision
from .ir import TrajectoryIR
from .recorded_state_graph import RecordedStateGraph
from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep
from .tool_semantics import DEFAULT_TOOL_SEMANTICS, ToolSemantics
from .verifiability import VerifiabilityAnalyzer, VerifiabilityDecision


class Trajectory2GymCompiler:
    """Shared compiler/runtime boundary for partial-environment decisions."""

    def __init__(
        self,
        episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
        *,
        accept_equivalent: bool = True,
        allow_step_replay: bool = True,
        allow_unsafe_hallucination: bool = False,
        tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
    ):
        self.episodes = tuple(episodes)
        self.analyzer = VerifiabilityAnalyzer.from_episodes(
            self.episodes,
            accept_equivalent=accept_equivalent,
            allow_step_replay=allow_step_replay,
            tool_semantics=tool_semantics,
        )
        # The graph is inert for ordinary corpora.  It contains only edges
        # whose converter explicitly declared pre/post state ids and a graph
        # scope; no state identity is inferred here.
        self.recorded_state_graph = RecordedStateGraph.from_episodes(
            self.episodes
        )
        self.allow_unsafe_hallucination = bool(allow_unsafe_hallucination)

    def verify(
        self,
        trajectory_ir: TrajectoryIR,
        step_index: int,
        action: dict[str, Any] | list[dict[str, Any]] | str,
    ) -> Decision:
        episode = trajectory_ir.to_episode()
        step = _step_at(episode, step_index)
        raw = self.analyzer.classify(episode, step, action)
        decision = decision_from_verifiability(raw)
        decision.assert_sound(allow_unsafe=self.allow_unsafe_hallucination)
        return decision


def decision_from_verifiability(raw: VerifiabilityDecision) -> Decision:
    evidence = {
        "reason": raw.reason,
        "match": raw.match.to_dict() if raw.match is not None else None,
        "observation_lookup": raw.observation_lookup.to_dict() if raw.observation_lookup is not None else None,
        "metadata": raw.metadata,
    }
    return Decision.from_subtype(
        raw.label,
        grounded_observation=raw.observation_messages,
        reward_signal=_reward_signal(raw),
        evidence=evidence,
        hallucinated_obs=_hallucinated(raw),
        verifier_name="VerifiabilityAnalyzer",
        typed_feedback=raw.reason if raw.is_invalid else "",
        metadata={"accepted": raw.accepted},
    )


def _step_at(episode: TrajectoryEpisode, step_index: int) -> TrajectoryStep:
    if step_index < 0 or step_index >= len(episode.steps):
        raise IndexError(f"step_index out of range: {step_index}")
    return episode.steps[step_index]


def _reward_signal(raw: VerifiabilityDecision) -> float:
    if raw.label in {"valid_replay_exact", "valid_terminal_stop"}:
        return 1.0
    if raw.label == "valid_replay_database":
        return 0.75
    if raw.label == "valid_replay_equivalent":
        return 0.5
    if raw.is_invalid:
        return -0.35
    return 0.0


def _hallucinated(raw: VerifiabilityDecision) -> bool:
    return raw.action_class in {"invalid", "ambiguous"} and bool(raw.observation_messages)


def canonical_action_key(action: ToolAction) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        action.name,
        tuple(sorted((str(key), str(value)) for key, value in action.arguments.items())),
    )
