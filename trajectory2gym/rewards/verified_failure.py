from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

from trajectory2gym.trajectory.boundary import boundary_class


VERIFIER_STATUSES = {"valid", "invalid", "ambiguous", "unknown"}


@dataclass(frozen=True)
class VerifierRewardEvidence:
    """Benchmark-neutral evidence consumed by shared reward functions."""

    required_progress: float
    verified_support_count: int
    required_step_count: int
    first_failure_status: str = "unknown"
    hallucinated_observation_count: int = 0
    verified_support_types: tuple[str, ...] = ()
    # A verified DB observation is evidence coverage, but it is not task
    # progress by itself. Rewardable support is the subset eventually followed
    # by a verified required transition in the same session.
    rewardable_support_count: int = 0
    rewardable_support_types: tuple[str, ...] = ()
    # Rejected outputs can still contain a directly grounded subset of the
    # required evidence.  This never authorizes a transition, but it gives a
    # conservative ordering signal between otherwise equally failed samples.
    partial_support_count: int = 0
    partial_support_types: tuple[str, ...] = ()

    @classmethod
    def from_step_infos(
        cls,
        *,
        required_progress: float,
        required_step_count: int,
        step_infos: Iterable[Mapping[str, Any]],
        hallucinated_observation_count: int = 0,
    ) -> "VerifierRewardEvidence":
        first_failure_status = "unknown"
        verified_support_count = 0
        verified_support_types: list[str] = []
        rewardable_support_count = 0
        rewardable_support_types: list[str] = []
        seen_support_keys: set[str] = set()
        seen_accepted_action_keys: set[str] = set()
        pending_support: list[tuple[str, str]] = []
        partial_support_count = 0
        partial_support_types: list[str] = []
        seen_partial_support_keys: set[str] = set()
        for info in step_infos:
            status = verifier_status(info)
            accepted = _evidence_bool(
                info["success"]
                if "success" in info
                else info["accepted"]
                if "accepted" in info
                else status == "valid"
            )
            actual_actions = info.get("actual_actions") or []
            action_key = (
                json.dumps(
                    actual_actions,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if actual_actions
                else ""
            )
            if (
                accepted
                and status == "valid"
                and _evidence_bool(info.get("verified_support", False))
                and not _evidence_bool(
                    info.get("database_observation_explicit_error", False)
                )
                and not _evidence_bool(info.get("progress_advanced", False))
            ):
                support_type = str(
                    info.get("branch_type")
                    or info.get("runtime_label")
                    or info.get("boundary_label")
                    or "unknown"
                )
                support_key = json.dumps(
                    [support_type, actual_actions],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if (
                    support_key not in seen_support_keys
                    and (not action_key or action_key not in seen_accepted_action_keys)
                ):
                    seen_support_keys.add(support_key)
                    verified_support_count += 1
                    verified_support_types.append(support_type)
                    pending_support.append((support_key, support_type))
            elif accepted and status == "valid" and _evidence_bool(
                info.get("progress_advanced", False)
            ):
                # Delayed credit: support becomes useful for RL only when the
                # policy later returns to the verified trajectory. Recovery is
                # still non-successful; this only ranks evidence-backed progress.
                rewardable_support_count += len(pending_support)
                rewardable_support_types.extend(
                    support_type for _, support_type in pending_support
                )
                pending_support.clear()
            if accepted and status == "valid" and action_key:
                seen_accepted_action_keys.add(action_key)
            if not accepted and _evidence_bool(info.get("partial_verified_support", False)):
                support_type = str(
                    info.get("partial_verified_support_type")
                    or info.get("branch_type")
                    or "unknown"
                )
                support_key = str(
                    info.get("partial_verified_support_key")
                    or json.dumps(
                        [support_type, info.get("expected_actions") or []],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                if support_key not in seen_partial_support_keys:
                    seen_partial_support_keys.add(support_key)
                    partial_support_count += 1
                    partial_support_types.append(support_type)
            if first_failure_status == "unknown" and not accepted and status in {"invalid", "ambiguous"}:
                first_failure_status = status
        return cls(
            required_progress=float(required_progress),
            verified_support_count=verified_support_count,
            required_step_count=int(required_step_count),
            first_failure_status=first_failure_status,
            hallucinated_observation_count=int(hallucinated_observation_count),
            verified_support_types=tuple(verified_support_types),
            rewardable_support_count=rewardable_support_count,
            rewardable_support_types=tuple(rewardable_support_types),
            partial_support_count=partial_support_count,
            partial_support_types=tuple(partial_support_types),
        )


@dataclass(frozen=True)
class VerifiedFailureRewardConfig:
    scale: float = 0.2
    support_weight: float = 0.05
    invalid_penalty: float = 0.05
    floor: float = -0.3
    ceiling: float = -1e-4

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "VerifiedFailureRewardConfig":
        return cls(
            scale=_float_param(params, "ranked_failure_reward_scale", 0.2),
            support_weight=_float_param(params, "ranked_failure_l2_support_weight", 0.05),
            invalid_penalty=_float_param(params, "ranked_failure_invalid_penalty", 0.05),
            floor=_float_param(params, "ranked_failure_reward_floor", -0.3),
            ceiling=_float_param(params, "ranked_failure_reward_ceiling", -1e-4),
        )


@dataclass(frozen=True)
class VerifiedFailureReward:
    """A non-positive failure reward ranked by verifier-grounded progress."""

    reward: float
    required_progress: float
    support_progress: float
    effective_progress: float
    failure_status: str


@dataclass(frozen=True)
class VerifiedFailureRewardApplication:
    reward: float
    applied: bool
    ranked: VerifiedFailureReward | None = None


def lexicographic_verified_support_bonus(evidence: VerifierRewardEvidence) -> float:
    """Return a bounded tie-break for support converted into verified progress.

    Required trajectory progress remains the primary objective.  For a
    trajectory with ``T`` required transitions, one required transition is
    worth ``1 / T`` while *all* off-trajectory support combined is worth at
    most ``1 / (T + 1)``.  Consequently no collection of database lookups can
    outrank one additional required transition, and no benchmark-specific
    weight is needed.
    """

    total = max(1, int(evidence.required_step_count))
    full_support = max(0, int(evidence.rewardable_support_count))
    partial_support = max(0, int(evidence.partial_support_count))
    support_units = min(float(total), float(full_support) + 0.5 * float(partial_support))
    return float(support_units / (total * (total + 1)))


def verifier_status(info: Mapping[str, Any]) -> str:
    """Read the normalized verifier status, with generic boundary fallbacks."""

    for key in ("verifier_status", "decision_status", "action_class", "status"):
        value = str(info.get(key) or "").strip().lower()
        if value in VERIFIER_STATUSES:
            return value
    for key in ("runtime_label", "boundary_label", "label"):
        value = str(info.get(key) or "")
        status = boundary_class(value)
        if status != "unknown":
            return status
    if _evidence_bool(info.get("hallucinated_obs", False)):
        return "unknown"
    accepted = info["success"] if "success" in info else info.get("accepted", False)
    return "valid" if _evidence_bool(accepted) else "unknown"


def apply_ranked_verified_failure_reward(
    *,
    current_reward: float,
    success: bool,
    evidence: VerifierRewardEvidence,
    params: Mapping[str, Any],
) -> VerifiedFailureRewardApplication:
    """Apply the shared failure ranking contract used by every benchmark."""

    if success or not _evidence_bool(params.get("ranked_failure_reward_on_failure", False)):
        return VerifiedFailureRewardApplication(reward=float(current_reward), applied=False)
    ranked = ranked_verified_failure_reward(
        evidence=evidence,
        config=VerifiedFailureRewardConfig.from_params(params),
    )
    return VerifiedFailureRewardApplication(
        reward=float(min(float(current_reward), ranked.reward)),
        applied=True,
        ranked=ranked,
    )


def ranked_verified_failure_reward(
    *,
    evidence: VerifierRewardEvidence,
    config: VerifiedFailureRewardConfig | None = None,
) -> VerifiedFailureReward:
    """Rank failed trajectories without ever making a failure positive."""

    config = config or VerifiedFailureRewardConfig()
    if evidence.hallucinated_observation_count:
        raise ValueError("verified failure rewards cannot consume hallucinated observations")
    if evidence.first_failure_status not in VERIFIER_STATUSES:
        raise ValueError(f"unsupported verifier status: {evidence.first_failure_status}")
    if config.scale <= 0.0:
        raise ValueError("scale must be positive")
    if config.support_weight < 0.0:
        raise ValueError("support_weight must be non-negative")
    if config.invalid_penalty < 0.0:
        raise ValueError("invalid_penalty must be non-negative")
    if config.floor > config.ceiling:
        raise ValueError("floor must not exceed ceiling")
    if config.ceiling > 0.0:
        raise ValueError("ceiling must be non-positive")

    progress = min(1.0, max(0.0, float(evidence.required_progress)))
    total = max(1, int(evidence.required_step_count))
    support = min(1.0, max(0, int(evidence.rewardable_support_count)) / total)
    effective = min(1.0, progress + float(config.support_weight) * support)
    invalid_penalty = config.invalid_penalty if evidence.first_failure_status == "invalid" else 0.0
    reward = -float(config.scale) * (1.0 - effective) - float(invalid_penalty)
    reward = max(float(config.floor), min(float(config.ceiling), reward))
    return VerifiedFailureReward(
        reward=float(reward),
        required_progress=progress,
        support_progress=support,
        effective_progress=effective,
        failure_status=evidence.first_failure_status,
    )


def _float_param(params: Mapping[str, Any], key: str, default: float) -> float:
    value = params.get(key)
    return float(default if value in (None, "") else value)


def _evidence_bool(value: Any) -> bool:
    """Parse verifier booleans conservatively across JSON and env boundaries."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False
