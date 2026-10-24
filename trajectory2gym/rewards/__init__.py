"""Reward signal exports for Trajectory2Gym.

Reward shaping is carried on :class:`trajectory2gym.trajectory.decision.Decision`
objects. The verifier evidence and reward ordering are shared by every dataset;
benchmark integrations do not define independent reward formulas.
"""

from trajectory2gym.trajectory.compiler import decision_from_verifiability
from trajectory2gym.trajectory.decision import Decision
from trajectory2gym.trajectory.verifiability import VerifiabilityDecision
from trajectory2gym.rewards.verified_failure import (
    VerifierRewardEvidence,
    VerifiedFailureReward,
    VerifiedFailureRewardApplication,
    VerifiedFailureRewardConfig,
    apply_ranked_verified_failure_reward,
    lexicographic_verified_support_bonus,
    ranked_verified_failure_reward,
    verifier_status,
)

__all__ = [
    "Decision",
    "VerifierRewardEvidence",
    "VerifiedFailureReward",
    "VerifiedFailureRewardApplication",
    "VerifiedFailureRewardConfig",
    "VerifiabilityDecision",
    "apply_ranked_verified_failure_reward",
    "decision_from_verifiability",
    "lexicographic_verified_support_bonus",
    "ranked_verified_failure_reward",
    "verifier_status",
]
