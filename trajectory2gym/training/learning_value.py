"""Shared learning-value contract for trajectory post-training views.

The same authentic trajectory corpus can have different *permissions* in
different stages:

* evidence may be retained when it is authentic and certified;
* SFT may imitate only a success-eligible assistant decision;
* RL may sample only a policy-reachable certified episode;
* Partial Gym may execute only a uniquely certified edge.

Those masks are deliberately separate from the ranking features.  A failed
trajectory can therefore improve graph coverage or frontier contrast without
silently becoming a positive policy target.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


JsonObject = dict[str, Any]

LEARNING_VALUE_CONTRACT_FORMAT = "trajectory2gym_learning_value_contract_v1"
LEARNING_VALUE_EVALUATION_FORMAT = "trajectory2gym_learning_value_v1"
LEARNING_VALUE_VIEWS = frozenset({"evidence", "sft", "rl", "partial"})


# The SFT profile preserves the original canonical learning-value ordering.
# RL uses the same factors but places more mass on policy-reachable outcome
# contrast.  These are method-level profiles, not benchmark/domain knobs.
LEARNING_VALUE_PROFILES: dict[str, dict[str, float]] = {
    "sft": {
        "marginal_certified_coverage": 4.0,
        "contrastive_frontier": 3.0,
        "policy_gap": 4.0,
        "branch_coverage": 1.5,
        "authentic_recovery": 2.0,
        "path_diversity": 1.0,
        "terminal_coverage": 0.0,
        "cost_efficiency": 1.0,
        "partial_only_exposure": 0.0,
        "unsupported_reference_risk": -2.0,
        "loop_redundancy": -2.0,
    },
    "rl": {
        "marginal_certified_coverage": 1.0,
        "contrastive_frontier": 4.0,
        "policy_gap": 2.0,
        "branch_coverage": 2.0,
        "authentic_recovery": 1.0,
        "path_diversity": 1.0,
        "terminal_coverage": 1.0,
        "cost_efficiency": 1.0,
        "partial_only_exposure": 1.0,
        "unsupported_reference_risk": -2.0,
        "loop_redundancy": -2.0,
    },
}


@dataclass(frozen=True)
class LearningValueFeatures:
    """Benchmark-neutral factors measured on one trajectory or decision.

    Continuous features are normalized to ``[0, 1]`` before construction.
    Certification and stage permissions remain explicit booleans so a large
    heuristic score can never override an evidence boundary.
    """

    evidence_certified: bool
    success_eligible: bool = False
    policy_reachable: bool = False
    partial_executable: bool = False
    marginal_certified_coverage: float = 0.0
    contrastive_frontier: float = 0.0
    policy_gap: float = 0.0
    branch_coverage: float = 0.0
    authentic_recovery: float = 0.0
    path_diversity: float = 0.0
    terminal_coverage: float = 0.0
    cost_efficiency: float = 0.0
    partial_only_exposure: float = 0.0
    unsupported_reference_risk: float = 0.0
    loop_redundancy: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class LearningValueEvaluation:
    """One auditable stage-specific projection of shared trajectory value."""

    view: str
    eligible: bool
    raw_score: float
    normalized_score: float
    features: LearningValueFeatures
    weighted_components: Mapping[str, float]

    def to_dict(self) -> JsonObject:
        return {
            "format": LEARNING_VALUE_EVALUATION_FORMAT,
            "contract_id": learning_value_contract_id(),
            "view": self.view,
            "eligible": self.eligible,
            "raw_score": round(self.raw_score, 8),
            "normalized_score": round(self.normalized_score, 8),
            "features": asdict(self.features),
            "weighted_components": {
                key: round(float(value), 8)
                for key, value in sorted(self.weighted_components.items())
            },
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _contract_payload() -> JsonObject:
    return {
        "format": LEARNING_VALUE_CONTRACT_FORMAT,
        "principle": (
            "evidence validity gates every view; stage permissions are masks, "
            "while novelty, policy gap, frontier contrast, recovery, diversity "
            "and cost only rank already-permitted data"
        ),
        "views": {
            "evidence": {
                "eligibility": ["evidence_certified"],
                "role": "retain authentic environment evidence",
            },
            "sft": {
                "eligibility": ["evidence_certified", "success_eligible"],
                "profile": "sft",
                "role": "imitate verified successful assistant decisions",
            },
            "rl": {
                "eligibility": ["evidence_certified", "policy_reachable"],
                "profile": "rl",
                "role": "prioritize learnable policy-reachable frontiers",
            },
            "partial": {
                "eligibility": ["evidence_certified", "partial_executable"],
                "role": "execute only uniquely certified graph edges",
            },
        },
        "profiles": LEARNING_VALUE_PROFILES,
        "invariants": {
            "failed_evidence_can_be_sft_target": False,
            "heuristic_score_can_override_evidence_boundary": False,
            "unsupported_observation_can_be_generated": False,
            "benchmark_specific_weights": False,
        },
    }


def learning_value_contract_id() -> str:
    return hashlib.sha256(
        _canonical_json(_contract_payload()).encode("utf-8")
    ).hexdigest()


def learning_value_contract() -> JsonObject:
    payload = _contract_payload()
    payload["contract_id"] = learning_value_contract_id()
    return payload


def _eligible(view: str, features: LearningValueFeatures) -> bool:
    if view == "evidence":
        return features.evidence_certified
    if view == "sft":
        return features.evidence_certified and features.success_eligible
    if view == "rl":
        return features.evidence_certified and features.policy_reachable
    if view == "partial":
        return features.evidence_certified and features.partial_executable
    raise ValueError(
        "learning-value view must be one of "
        + ", ".join(sorted(LEARNING_VALUE_VIEWS))
    )


def evaluate_learning_value(
    view: str,
    features: LearningValueFeatures,
    *,
    profile: Mapping[str, float] | None = None,
) -> LearningValueEvaluation:
    """Apply one stage mask and rank already-permitted evidence.

    ``evidence`` and ``partial`` are permissions rather than optimization
    targets, so their score is the binary eligibility value.  SFT and RL use
    the shared factor vocabulary with their fixed method-level profiles.
    """

    normalized_view = str(view).strip().lower()
    eligible = _eligible(normalized_view, features)
    if normalized_view in {"evidence", "partial"}:
        score = float(eligible)
        return LearningValueEvaluation(
            view=normalized_view,
            eligible=eligible,
            raw_score=score,
            normalized_score=score,
            features=features,
            weighted_components={},
        )

    weights = dict(profile or LEARNING_VALUE_PROFILES[normalized_view])
    unknown = set(weights) - {
        key
        for key, value in asdict(features).items()
        if not isinstance(value, bool)
    }
    if unknown:
        raise ValueError(
            "unknown learning-value features: " + ", ".join(sorted(unknown))
        )
    weighted = {
        key: float(weight) * float(getattr(features, key))
        for key, weight in weights.items()
    }
    raw_score = sum(weighted.values()) if eligible else 0.0
    positive_budget = sum(max(0.0, float(weight)) for weight in weights.values())
    normalized_score = (
        max(0.0, min(1.0, raw_score / positive_budget))
        if eligible and positive_budget
        else 0.0
    )
    return LearningValueEvaluation(
        view=normalized_view,
        eligible=eligible,
        raw_score=raw_score,
        normalized_score=normalized_score,
        features=features,
        weighted_components=weighted,
    )
