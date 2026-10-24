"""Compatibility hash helper for recorded evidence; no RL objectives are included."""
from __future__ import annotations

import hashlib
import json

FULL_SUCCESSOR_BINDING_VERSION = "trajectory2gym_full_successor_v1"


def full_successor_binding_sha256(
    *,
    pre_state_id: str,
    post_state_id: str,
    entry_context_id: str,
    next_policy_context_id: str,
    action_key: str,
    observation_key: str,
    version: str = FULL_SUCCESSOR_BINDING_VERSION,
) -> str:
    """Bind canonical context/action/observation/successor evidence."""
    payload = {
        "action_key": str(action_key),
        "entry_context_id": str(entry_context_id),
        "next_policy_context_id": str(next_policy_context_id),
        "observation_key": str(observation_key),
        "post_state_id": str(post_state_id),
        "pre_state_id": str(pre_state_id),
        "version": str(version),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
