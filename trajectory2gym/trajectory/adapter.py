from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from trajectory2gym.rewards.verified_failure import (
    VerifierRewardEvidence,
    apply_ranked_verified_failure_reward,
    lexicographic_verified_support_bonus,
)
from trajectory2gym.rewards.recovery_integrity import rejected_side_effect_failures

from .boundary import PARTIAL_EVIDENCE_BRANCH_LABELS, PARTIAL_EXTENSION_LABELS
from .matcher import ActionMatchResult, match_step_action
from .replay import ReplayDecision, replay_gold_observation
from .recovery import episode_explicit_error_count
from .schema import TrajectoryEpisode, TrajectoryStep
from .tool_semantics import DEFAULT_TOOL_SEMANTICS, ToolSemantics
from .external_request import external_action_catalog_from_episodes


@dataclass(frozen=True)
class StepDecision:
    """Adapter decision consumed by the neutral trajectory session runtime."""

    accepted: bool
    reward: float
    label: str
    observation_messages: tuple[dict[str, Any], ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reward": self.reward,
            "label": self.label,
            "observation_messages": copy.deepcopy(list(self.observation_messages)),
            "info": copy.deepcopy(self.info),
        }


class TrajectoryAdapter:
    """Policy hook interface for the neutral trajectory2gym runtime.

    The default adapter is intentionally conservative and policy-free. Dataset
    adapters can subclass this to add domain behavior without changing the
    session lifecycle.
    """

    source = "trajectory"
    equivalent_threshold = 0.78
    accept_equivalent = True
    tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS

    def is_side_effecting_tool(self, name: str) -> bool:
        return self.tool_semantics.is_side_effecting(name)

    def initial_messages(self, episode: TrajectoryEpisode) -> list[dict[str, Any]]:
        return copy.deepcopy(list(episode.initial_messages))

    def match(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        action: dict[str, Any] | str,
    ) -> ActionMatchResult:
        return match_step_action(
            step,
            action,
            tools=episode.tools,
            equivalent_threshold=self.equivalent_threshold,
            external_action_catalog=external_action_catalog_from_episodes((episode,)),
        )

    def replay(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        match: ActionMatchResult,
    ) -> ReplayDecision:
        return replay_gold_observation(step, match, accept_equivalent=self.accept_equivalent)

    def decide(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        match: ActionMatchResult,
    ) -> StepDecision:
        replay = self.replay(episode, step, match)
        if match.label == "terminal_stop" and not _terminal_stop_source_eligible(
            episode, step
        ):
            return StepDecision(
                accepted=False,
                reward=0.0,
                label="ambiguous_terminal_stop_without_success_evidence",
                observation_messages=(),
                info={
                    "branch_type": (
                        "ambiguous_terminal_stop_without_success_evidence"
                    ),
                    "verifier_status": "ambiguous",
                    "verified_support": False,
                    "progress_advanced": False,
                    "non_blocking_abstention": True,
                    "verified_terminal_stop": False,
                    "terminal_stop_source_eligible": False,
                    "terminal_stop_semantics_verified": False,
                    "match": match.to_dict(),
                    "replay": replay.to_dict(),
                    "actual_actions": [],
                    "expected_actions": [],
                },
            )
        reward = replay.reward_hint if replay.accepted else self.failure_reward(match)
        verifier_status = self._match_verifier_status(match, accepted=replay.accepted)
        decision_label = (
            "valid_terminal_stop"
            if replay.accepted and match.label == "terminal_stop"
            else match.label
        )
        return StepDecision(
            accepted=replay.accepted,
            reward=reward,
            label=decision_label,
            observation_messages=replay.observation_messages,
            info={
                "branch_type": decision_label,
                "verifier_status": verifier_status,
                "verified_support": False,
                "progress_advanced": replay.accepted,
                "verified_terminal_stop": bool(
                    replay.accepted and match.label == "terminal_stop"
                ),
                "terminal_stop_source_eligible": bool(
                    match.label == "terminal_stop"
                ),
                "terminal_stop_semantics_verified": False,
                "terminal_stop_provenance": str(
                    step.metadata.get("terminal_stop_provenance") or ""
                ),
                "terminal_stop_synthetic": bool(
                    step.metadata.get("terminal_stop_synthetic", False)
                ),
                "match": match.to_dict(),
                "replay": replay.to_dict(),
                "actual_actions": [action.to_dict() for action in match.actual_actions],
                "expected_actions": [action.to_dict() for action in match.expected_actions],
            },
        )

    @staticmethod
    def _match_verifier_status(match: ActionMatchResult, *, accepted: bool) -> str:
        if accepted:
            return "valid"
        if match.label in {
            "missing_tool_call",
            "schema_invalid",
            "spurious_tool_call",
            "unknown_tool",
        }:
            return "invalid"
        return "ambiguous"

    def failure_reward(self, match: ActionMatchResult) -> float:
        if match.label == "missing_tool_call":
            return -0.25
        if match.label == "schema_invalid":
            return -0.35
        if match.label == "wrong_args":
            return -0.4
        if match.label == "wrong_tool":
            return -0.45
        if match.label == "unknown_tool":
            return -0.6
        if match.label == "spurious_tool_call":
            return -0.4
        return 0.0

    def evaluate(
        self,
        episode: TrajectoryEpisode,
        *,
        completed: bool,
        step_index: int,
        step_rewards: list[float],
        step_infos: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
        sparse: bool = True,
    ) -> Any:
        eval_params = params or {}
        required_steps = tuple(step for step in episode.steps if step.kind != "external_event")
        completed_required_steps = sum(
            1 for step in episode.steps[:step_index] if step.kind != "external_event"
        )
        required_step_count = len(required_steps)
        progress = float(
            completed_required_steps / required_step_count if required_step_count else 1.0
        )
        local_success = bool(completed and step_index >= episode.step_count)
        rejected_infos = [
            info
            for info in step_infos
            if not bool(info.get("success", info.get("accepted", False)))
        ]
        abstention_infos = [
            info for info in rejected_infos if bool(info.get("non_blocking_abstention", False))
        ]
        failure_infos = [
            info for info in rejected_infos if not bool(info.get("non_blocking_abstention", False))
        ]
        integrity_failures = (
            # A schema-valid off-trajectory action is normally a retryable
            # abstention.  That does not make an attempted write safe: the
            # backend state after an unsupported side effect is unknowable.
            # Audit every rejected policy action here, including non-blocking
            # abstentions, so recovery can never turn such a session positive.
            rejected_side_effect_failures(
                rejected_infos, tool_semantics=self.tool_semantics
            )
            if bool(eval_params.get("recovery_integrity", False))
            else []
        )
        # Recovery remains useful for collecting later verified transitions,
        # but a trajectory containing any rejected decision is not a clean
        # end-to-end success and must not receive terminal success credit.
        source_explicit_error_count = episode_explicit_error_count(episode)
        runtime_explicit_error_count = sum(
            bool(info.get("database_observation_explicit_error", False))
            for info in step_infos
        )
        total_explicit_error_count = (
            source_explicit_error_count + runtime_explicit_error_count
        )
        # A compiled trajectory slice may be a valid local training state
        # without representing an end-to-end task from the original initial
        # state.  Local completion is still useful verified progress, but it
        # must never be reported as benchmark/session success.
        (
            policy_training_eligible,
            terminal_success_eligible,
        ) = _episode_runtime_eligibility(episode)
        requires_certified_stop = _episode_requires_certified_stop(episode)
        verified_terminal_stop_success = any(
            _is_runtime_verified_terminal_stop(info) for info in step_infos
        )
        arbitrary_window_truncated = bool(
            episode.metadata.get("decision_window_truncated", False)
        )
        recovery_success = bool(
            local_success and (rejected_infos or total_explicit_error_count)
        )
        success = bool(
            local_success
            and policy_training_eligible
            and terminal_success_eligible
            and (
                verified_terminal_stop_success or not requires_certified_stop
            )
            and not rejected_infos
            and not integrity_failures
            and total_explicit_error_count == 0
        )
        # A compiler-created decision window can be a clean, fully verified
        # local episode without claiming that the original benchmark task was
        # completed. This is a training objective, not an evaluation label.
        verified_window_success = bool(
            local_success
            and policy_training_eligible
            and not arbitrary_window_truncated
            and not rejected_infos
            and not integrity_failures
            and total_explicit_error_count == 0
        )
        # A state-complete suffix that reaches the recorded source end carries
        # stronger task-alignment evidence than an arbitrary completed local
        # slice.  This remains distinct from benchmark/session success: the
        # compiled window may start from a recorded prefix rather than from the
        # original initial state.  Full, non-window trajectories default to
        # their ordinary terminal eligibility.
        source_terminal_window = bool(
            episode.metadata.get(
                "covers_policy_end",
                episode.metadata.get("covers_source_end", terminal_success_eligible),
            )
        )
        terminal_frontier_success = bool(
            verified_window_success
            and source_terminal_window
            and verified_terminal_stop_success
        )
        accepted_partial_extensions = [
            info
            for info in step_infos
            if bool(info.get("success", info.get("accepted", False)))
            and str(info.get("branch_type") or info.get("label") or "")
            in PARTIAL_EXTENSION_LABELS
        ]
        accepted_partial_evidence = [
            info
            for info in accepted_partial_extensions
            if str(info.get("branch_type") or info.get("label") or "")
            in PARTIAL_EVIDENCE_BRANCH_LABELS
        ]
        partial_evidence_branch_types = sorted(
            {
                str(info.get("branch_type") or info.get("label") or "")
                for info in accepted_partial_evidence
            }
        )
        step_reward_total = float(sum(step_rewards))
        step_reward_mean = float(step_reward_total / len(step_rewards) if step_rewards else 0.0)
        negative_step_reward_total = float(sum(reward for reward in step_rewards if reward < 0.0))
        negative_step_reward_mean = float(
            negative_step_reward_total / len(step_rewards) if step_rewards else 0.0
        )
        verifier_evidence = VerifierRewardEvidence.from_step_infos(
            required_progress=progress,
            required_step_count=required_step_count,
            step_infos=step_infos,
        )
        verified_support_bonus = lexicographic_verified_support_bonus(verifier_evidence)
        reward_mode = str(eval_params.get("reward_mode") or "progress")
        if reward_mode in {
            "terminal_aware_verified_progress",
            "evidence_bounded_terminal_progress",
        }:
            reward_success = terminal_frontier_success
        else:
            reward_success = bool(
                verified_window_success
                if reward_mode
                in {
                "verified_window_progress",
                "verified_subtrajectory_progress",
                "conservative_verified_window_plus_step",
                "evidence_bounded_window",
                }
                else success
            )
        conservative_positive_cap_applied = False
        raw_reward_score: float | None = None
        conservative_raw_reward = 0.0
        if reward_mode in {"accuracy", "success"}:
            reward_score = float(1.0 if success else 0.0)
        elif reward_mode in {"progress", "trajectory_progress"}:
            reward_score = progress
        elif reward_mode in {
            "verified_progress",
            "verified_trajectory_progress",
            "verified_window_progress",
            "verified_subtrajectory_progress",
            "terminal_aware_verified_progress",
            "evidence_bounded_terminal_progress",
        }:
            # Clean completion receives 1. Every incomplete or recovered
            # trajectory stays non-positive. Required exact/equivalent
            # progress is the primary ordering; unique non-advancing verified
            # transitions provide only a bounded lexicographic tie-break.
            reward_score = (
                1.0
                if reward_success
                else min(0.0, progress - 1.0 + verified_support_bonus)
            )
        elif reward_mode in {"step_total", "cumulative_step"}:
            reward_score = step_reward_total
        elif reward_mode in {"step_mean", "mean_step"}:
            reward_score = step_reward_mean
        elif reward_mode in {
            "conservative_accuracy_plus_step",
            "conservative_dense",
            "conservative_verified_window_plus_step",
            "evidence_bounded_window",
        }:
            # A failed/recovered trajectory may still be useful for collecting
            # transitions, but it must not receive positive completion credit.
            # Keep the dense step signal only when it is non-positive; this is
            # the generic equivalent of the legacy conservative reward contract.
            weight = float(eval_params.get("step_reward_weight", 0.05) or 0.0)
            dense_delta = weight * step_reward_mean
            conservative_raw_reward = (1.0 if reward_success else 0.0) + dense_delta
            if reward_success:
                reward_score = conservative_raw_reward
            else:
                conservative_positive_cap_applied = dense_delta > 0.0
                reward_score = min(0.0, dense_delta)
        elif reward_mode in {"accuracy_plus_step", "dense", "progress_plus_step"}:
            weight = float(eval_params.get("step_reward_weight", 0.05) or 0.0)
            reward_score = progress + weight * step_reward_mean
        else:
            raise ValueError(f"unsupported trajectory2gym reward_mode: {reward_mode}")
        session_success_bonus = (
            float(eval_params.get("session_success_bonus", 0.0) or 0.0) if success else 0.0
        )
        reward_score += session_success_bonus
        if raw_reward_score is None:
            if reward_mode in {
                "conservative_accuracy_plus_step",
                "conservative_dense",
                "conservative_verified_window_plus_step",
                "evidence_bounded_window",
            }:
                raw_reward_score = float(conservative_raw_reward + session_success_bonus)
            else:
                raw_reward_score = float(reward_score)
        zero_positive_reward_on_failure = bool(eval_params.get("zero_positive_reward_on_failure", False))
        failure_positive_cap_applied = False
        # The contract is defined by the task/window success gate, not by the
        # verifier's typed reject taxonomy.  A retryable abstention still makes
        # this sampled session unsuccessful; allowing recovered progress to
        # remain positive here would recreate the same failed-session credit
        # leak later at the transition layer.
        if (
            not reward_success
            and zero_positive_reward_on_failure
            and reward_score > 0.0
        ):
            failure_positive_cap_applied = True
            reward_score = 0.0
        if integrity_failures and bool(eval_params.get("recovery_integrity_zero_positive_reward", True)):
            reward_score = min(0.0, reward_score)
        clip_min = eval_params.get("reward_clip_min")
        clip_max = eval_params.get("reward_clip_max")
        if clip_min is not None:
            reward_score = max(float(clip_min), reward_score)
        if clip_max is not None:
            reward_score = min(float(clip_max), reward_score)
        ranked_application = apply_ranked_verified_failure_reward(
            current_reward=reward_score,
            success=reward_success,
            evidence=verifier_evidence,
            params=eval_params,
        )
        reward_score = ranked_application.reward
        if total_explicit_error_count:
            reward_score = min(0.0, reward_score)
        if sparse:
            return reward_score
        return {
            "task_id": episode.id,
            "source": episode.source,
            "domain": episode.domain,
            "trajectory_group_id": str(
                episode.metadata.get("trajectory_group_id") or ""
            ),
            "trajectory_source_task_id": str(
                episode.metadata.get("source_task_id")
                or episode.metadata.get("task_id")
                or episode.metadata.get("trajectory_group_id")
                or episode.id
            ),
            "episode_mode": "trajectory_session",
            "success": success,
            "accuracy": float(success),
            "session_success": success,
            "local_session_success": local_success,
            "verified_window_success": verified_window_success,
            "source_terminal_window": source_terminal_window,
            "terminal_frontier_success": terminal_frontier_success,
            # Counterfactual is over the identical sampled action sequence:
            # every accepted branch below is outside Strict's exact cursor, so
            # Strict would stop/reject at its first occurrence.  This does not
            # claim that a separately sampled Strict policy could never solve
            # the task; it proves that this successful transition path is
            # causally enabled by Partial evidence.
            "partial_extension_accepted_count": len(accepted_partial_extensions),
            "partial_evidence_accepted_count": len(accepted_partial_evidence),
            "partial_evidence_branch_types": partial_evidence_branch_types,
            "strict_counterfactual_first_reject_turn": next(
                (
                    index
                    for index, info in enumerate(step_infos)
                    if bool(info.get("partial_extension_branch", False))
                ),
                -1,
            ),
            "partial_only_session_success": bool(success and accepted_partial_evidence),
            "partial_only_window_success": bool(
                verified_window_success and accepted_partial_evidence
            ),
            "partial_only_terminal_frontier_success": bool(
                terminal_frontier_success and accepted_partial_evidence
            ),
            "partial_only_reward_success": bool(
                reward_success and accepted_partial_evidence
            ),
            "verified_terminal_stop_success": verified_terminal_stop_success,
            "requires_certified_stop": requires_certified_stop,
            "reward_success": reward_success,
            "terminal_success_eligible": terminal_success_eligible,
            "training_eligible": policy_training_eligible,
            "arbitrary_window_truncated": arbitrary_window_truncated,
            "recovery_success": recovery_success,
            "source_explicit_error_count": source_explicit_error_count,
            "runtime_explicit_error_count": runtime_explicit_error_count,
            "total_explicit_error_count": total_explicit_error_count,
            "positive_credit_suppression_count": (
                len(rejected_infos) + total_explicit_error_count
            ),
            "recovery_integrity_applied": bool(eval_params.get("recovery_integrity", False)),
            "recovery_integrity_failure_count": len(integrity_failures),
            "failure_count": len(failure_infos),
            "abstention_count": len(abstention_infos),
            "zero_positive_reward_on_failure": zero_positive_reward_on_failure,
            "failure_positive_cap_applied": failure_positive_cap_applied,
            "conservative_positive_cap_applied": conservative_positive_cap_applied,
            "completed": completed,
            "session_completed": completed,
            "step": step_index,
            "session_task_count": required_step_count,
            "external_event_count": episode.step_count - required_step_count,
            "turn_progress": progress,
            "verified_support_count": verifier_evidence.verified_support_count,
            "verified_support_types": list(verifier_evidence.verified_support_types),
            "rewardable_support_count": verifier_evidence.rewardable_support_count,
            "rewardable_support_types": list(verifier_evidence.rewardable_support_types),
            "partial_support_count": verifier_evidence.partial_support_count,
            "partial_support_types": list(verifier_evidence.partial_support_types),
            "verified_support_bonus": (
                verified_support_bonus
                if reward_mode
                in {
                    "verified_progress",
                    "verified_trajectory_progress",
                    "verified_window_progress",
                    "verified_subtrajectory_progress",
                    "terminal_aware_verified_progress",
                    "evidence_bounded_terminal_progress",
                }
                else 0.0
            ),
            "reward_score": reward_score,
            "raw_reward_score": raw_reward_score,
            "session_success_bonus": session_success_bonus,
            "ranked_failure_reward_applied": ranked_application.applied,
            "ranked_failure_reward_score": (
                ranked_application.ranked.reward if ranked_application.ranked is not None else None
            ),
            "ranked_failure_required_progress": (
                ranked_application.ranked.required_progress
                if ranked_application.ranked is not None
                else None
            ),
            "ranked_failure_support_progress": (
                ranked_application.ranked.support_progress
                if ranked_application.ranked is not None
                else None
            ),
            "ranked_failure_verified_support_count": (
                verifier_evidence.rewardable_support_count
                if ranked_application.ranked is not None
                else None
            ),
            "ranked_failure_verified_support_types": (
                list(verifier_evidence.rewardable_support_types)
                if ranked_application.ranked is not None
                else []
            ),
            "ranked_failure_first_failure_status": (
                ranked_application.ranked.failure_status
                if ranked_application.ranked is not None
                else None
            ),
            "reward_mode": reward_mode,
            "step_rewards": list(step_rewards),
            "step_reward_total": step_reward_total,
            "step_reward_mean": step_reward_mean,
            "negative_step_reward_total": negative_step_reward_total,
            "negative_step_reward_mean": negative_step_reward_mean,
            "step_infos": copy.deepcopy(step_infos),
            "turn_infos": copy.deepcopy(step_infos),
            "failure_infos": copy.deepcopy(failure_infos),
        }


def _terminal_stop_source_eligible(
    episode: TrajectoryEpisode, step: TrajectoryStep
) -> bool:
    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    return bool(
        step.kind == "terminal_stop"
        and not step.expected_actions
        and not step.observations
        and step.metadata.get("verified_terminal_stop") is True
        and step.metadata.get("terminal_stop_provenance")
        == "recorded_successful_episode_boundary"
        and step.metadata.get("terminal_stop_semantics_verified") is False
        and metadata.get("source_training_eligible") is True
        and metadata.get("source_terminal_success_eligible") is True
    )


def _episode_runtime_eligibility(
    episode: TrajectoryEpisode,
) -> tuple[bool, bool]:
    """Resolve eligibility without letting legacy last-tool windows bypass stop.

    New decision windows carry explicit source and local flags. Both levels
    must be true. A window missing either declaration is ineligible even when
    its last recorded tool was replayed. Unsliced legacy episodes keep their
    historical default only outside this new compiled-window contract.
    """

    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    compiled_contract = bool(
        "decision_window_start" in metadata
        or "parent_trajectory_id" in metadata
        or "source_training_eligible" in metadata
        or "source_terminal_success_eligible" in metadata
        or any(step.kind == "terminal_stop" for step in episode.steps)
    )
    if not compiled_contract:
        policy_training_eligible = bool(
            metadata.get("training_eligible", True) is True
        )
        terminal_success_eligible = bool(
            policy_training_eligible
            and metadata.get("terminal_success_eligible", True) is True
        )
        return policy_training_eligible, terminal_success_eligible

    source_training_eligible = metadata.get("source_training_eligible") is True
    source_terminal_success_eligible = (
        metadata.get("source_terminal_success_eligible") is True
    )
    local_training_eligible = metadata.get("training_eligible") is True
    local_terminal_success_eligible = (
        metadata.get("terminal_success_eligible") is True
    )
    policy_training_eligible = bool(
        source_training_eligible and local_training_eligible
    )
    terminal_success_eligible = bool(
        policy_training_eligible
        and source_terminal_success_eligible
        and local_terminal_success_eligible
    )
    return policy_training_eligible, terminal_success_eligible


def _episode_requires_certified_stop(episode: TrajectoryEpisode) -> bool:
    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    return bool(
        metadata.get("terminal_stop_compiled") is True
        or "source_training_eligible" in metadata
        or "source_terminal_success_eligible" in metadata
        or any(step.kind == "terminal_stop" for step in episode.steps)
    )


def _is_runtime_verified_terminal_stop(info: dict[str, Any]) -> bool:
    return bool(
        isinstance(info, dict)
        and str(info.get("branch_type") or info.get("label") or "")
        == "valid_terminal_stop"
        and bool(info.get("success", info.get("accepted", False)))
        and info.get("verified_terminal_stop") is True
        and info.get("terminal_stop_source_eligible") is True
        and info.get("terminal_stop_semantics_verified") is False
    )


class DeclarativeTrajectoryAdapter(TrajectoryAdapter):
    """Thin adapter for a new trajectory dataset.

    Most datasets need no subclass. Give the adapter a source label and, only
    when defaults are insufficient, declarative tool-name semantics.
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
        equivalent_threshold: float = 0.78,
        accept_equivalent: bool = True,
    ) -> None:
        self.source = source or type(self).source
        self.tool_semantics = tool_semantics
        self.equivalent_threshold = equivalent_threshold
        self.accept_equivalent = accept_equivalent
