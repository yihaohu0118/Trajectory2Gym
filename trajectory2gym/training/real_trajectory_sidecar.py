"""Trajectory-derived local credit for an otherwise authoritative real Gym.

The sidecar never executes an action, supplies an observation, or changes the
terminal reward.  It indexes exact action--observation successors from static
trajectories and signs a real transition only after the authoritative backend
has returned the same successor at the same task-local interaction prefix.

State identity uses a stable reset identity plus an evidence-bounded set of
visible trajectory states. Exact real observations may neutrally re-anchor
that set after a path deviation; the deviating action never earns credit, and
only a later exact action--successor match can be positive. This makes the
contract portable without adding benchmark workflow rules to core.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajectory2gym.tool_calls import actions_from_assistant_message
from trajectory2gym.training.probe_contract import (
    FULL_SUCCESSOR_BINDING_VERSION,
    full_successor_binding_sha256,
)
from trajectory2gym.training.verified_prefix import VerifiedPrefixIndex
from trajectory2gym.trajectory import (
    TrajectoryEpisode,
    load_trajectory_episodes,
    observation_is_explicit_error,
)
from trajectory2gym.trajectory.recorded_state_graph import (
    recorded_action_key as _recorded_action_key,
)
from trajectory2gym.trajectory.schema import ToolAction
from trajectory2gym.text import strip_visible_thinking

REAL_TRAJECTORY_SIDECAR_CERTIFICATE = (
    "real_observation_anchored_context_action_observation_match_v2"
)
REAL_TRAJECTORY_STATE_HASH_VERSION = (
    "real_policy_visible_initial_context.v2"
)
REAL_TRAJECTORY_TASK_ANCHORED_PROPOSAL_CERTIFICATE = (
    "task_anchored_action_proposal_after_uncredited_external_event_v1"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_content(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        for transport_role in ("user", "tool", "environment"):
            prefix = f"{transport_role}:"
            if lowered.startswith(prefix):
                stripped = stripped[len(prefix) :].lstrip()
                break
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        return stripped
    return value


def observation_key(observations: Sequence[Mapping[str, Any]]) -> str:
    """Canonicalize backend contents without depending on transport roles.

    Some official adapters expose a tool observation as ``role=tool`` while
    others transport ``tool: ...`` inside a user message. The authentic
    content is the environment evidence; these transport role wrappers are
    not environment state.
    """

    return _canonical_json(
        [_normalized_content(item.get("content")) for item in observations]
    )


def policy_context_key(messages: Sequence[Mapping[str, Any]]) -> str:
    """Canonicalize the messages visible before the first policy decision.

    Unlike an observation key, context identity retains message roles and the
    exact visible text. Transport-only tool-call ids are excluded, while
    assistant actions use the same semantic action key as graph edges.
    """

    canonical: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        item: dict[str, Any] = {"role": role}
        if role == "assistant":
            item["action"] = assistant_action_key(message)
        else:
            item["content"] = _normalized_content(message.get("content"))
            if role == "tool" and message.get("name"):
                item["name"] = str(message.get("name"))
            if role == "user" and message.get("tool_calls"):
                item["tool_calls"] = message.get("tool_calls")
        canonical.append(item)
    return _canonical_json(canonical)


def assistant_action_key(message: Mapping[str, Any]) -> str:
    parsed = actions_from_assistant_message(dict(message))
    if not parsed:
        return _canonical_json(
            {
                "type": "assistant_text",
                # Qwen completion adapters may transport a native empty
                # <think></think> prefix even with thinking disabled.  It is
                # not a policy-visible action and must not break exact text
                # proposal identity.  Non-empty reasoning remains a separate
                # protocol violation in the non-thinking audit.
                "content": strip_visible_thinking(
                    str(message.get("content") or "")
                ),
            }
        )
    actions = tuple(
        ToolAction(
            name=str(item.get("name") or ""),
            arguments=dict(item.get("arguments") or {}),
        )
        for item in parsed
    )
    return _recorded_action_key(actions)


def recorded_action_key(episode_step: Any) -> str:
    actions = tuple(episode_step.expected_actions or ())
    if actions:
        return _recorded_action_key(actions)
    return _canonical_json(
        {
            "type": "assistant_text",
            "content": str(
                episode_step.metadata.get("expected_assistant_message") or ""
            ).strip(),
        }
    )


def _source_task_ids(episode: TrajectoryEpisode) -> tuple[str, ...]:
    """Return the declared reset identity and safe serialization aliases.

    Dataset compilers commonly namespace a task identity by its already
    declared domain (for example ``domain_a:7``), while a live environment uses
    the domain-local reset identity (``7``).  The sidecar is already scoped by
    ``episode.domain``; indexing both spellings is therefore unambiguous and
    avoids making an otherwise generic real-environment adapter understand a
    dataset's serialization convention.

    Explicit reset/environment identities take priority.  The domain-prefix
    alias is additive: the original provenance identity is always retained.
    """

    metadata = episode.metadata
    task = episode.task if isinstance(episode.task, Mapping) else {}
    candidates = [
        metadata.get("environment_task_id"),
        metadata.get("reset_id"),
        metadata.get("source_task_id"),
        task.get("task_id"),
        task.get("id"),
    ]
    identities: list[str] = []
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value and value not in identities:
            identities.append(value)

    domain = str(episode.domain or "").strip()
    prefix = f"{domain}:" if domain else ""
    for value in tuple(identities):
        if prefix and value.startswith(prefix):
            alias = value[len(prefix) :].strip()
            if alias and alias not in identities:
                identities.append(alias)
    return tuple(identities)


def _successful_episode(episode: TrajectoryEpisode) -> bool:
    metadata = episode.metadata
    return bool(
        metadata.get("training_eligible") is True
        and metadata.get(
            "source_session_success",
            metadata.get("terminal_success_eligible", False),
        )
        is True
    )


def _episode_has_explicit_error(episode: TrajectoryEpisode) -> bool:
    return any(
        observation_is_explicit_error(dict(observation))
        for step in episode.steps
        for observation in step.observations
    )


def _state_id(
    *,
    domain: str,
    reset_id: str,
    executed_prefix: Sequence[Mapping[str, str]],
    initial_context_key: str = "",
) -> str:
    return _stable_hash(
        {
            "domain": str(domain),
            "reset_id": str(reset_id),
            "initial_context": str(initial_context_key),
            "executed_prefix": list(executed_prefix),
        }
    )


@dataclass(frozen=True)
class RealSuccessorEvidence:
    successful_observations: frozenset[str] = frozenset()
    explicit_error_observations: frozenset[str] = frozenset()
    failed_nonerror_observations: frozenset[str] = frozenset()

    @property
    def conflict(self) -> bool:
        return bool(
            self.successful_observations
            and self.explicit_error_observations
        )


@dataclass(frozen=True)
class RealTrajectoryActionProposal:
    """A successful trajectory action proposed at an exact visible state.

    A proposal is behavior guidance, not an environment transition.  The live
    backend must still execute the message and remains the sole source of the
    observation and terminal reward.
    """

    action_key: str
    assistant_message: dict[str, Any]
    evidence_count: int
    pre_context_id: str = ""
    certificate: str = REAL_TRAJECTORY_SIDECAR_CERTIFICATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_key": self.action_key,
            "assistant_message": dict(self.assistant_message),
            "evidence_count": self.evidence_count,
            "certificate": self.certificate,
            "entry_context_id": self.pre_context_id,
            "context_certified": (
                self.certificate == REAL_TRAJECTORY_SIDECAR_CERTIFICATE
                and bool(self.pre_context_id)
            ),
            "observation_supplied": False,
            "reward_supplied": False,
        }


def recorded_assistant_message(episode_step: Any) -> dict[str, Any] | None:
    """Materialize one portable assistant action from a recorded step.

    The authoritative Tau adapters execute at most one tool call per assistant
    turn.  Multi-action records are therefore not proposed: replaying only a
    prefix of such a record would silently change its semantics.
    """

    actions = tuple(episode_step.expected_actions or ())
    if len(actions) > 1:
        return None
    if actions:
        action = actions[0]
        payload = _canonical_json(
            {"name": action.name, "arguments": dict(action.arguments)}
        )
        return {
            "role": "assistant",
            "content": f"<tool_call>\n{payload}\n</tool_call>",
        }
    content = str(
        episode_step.metadata.get("expected_assistant_message") or ""
    ).strip()
    if not content:
        return None
    return {"role": "assistant", "content": content}


def _group_action_proposals(
    sources: Mapping[tuple[str, str, str, str], set[str]],
    messages: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[RealTrajectoryActionProposal]]:
    grouped: dict[
        tuple[str, str, str], list[RealTrajectoryActionProposal]
    ] = defaultdict(list)
    for (domain, reset_id, state_id, action_key), episode_ids in sources.items():
        message = messages.get((domain, reset_id, state_id, action_key))
        if message is None:
            continue
        grouped[(domain, reset_id, state_id)].append(
            RealTrajectoryActionProposal(
                action_key=action_key,
                assistant_message=dict(message),
                evidence_count=len(episode_ids),
                pre_context_id=state_id,
            )
        )
    return grouped


@dataclass(frozen=True)
class RealTrajectoryTurnCertificate:
    label: float
    actor_eligible: bool
    group_id: str
    state_id: str
    action_key: str
    observed_successor_key: str
    branch_type: str
    reason: str
    evidence_count: int
    reanchored: bool = False
    candidate_state_count: int = 0
    proposal_candidate_state_count: int = 0
    proposal_external_event_advanced: bool = False
    pre_state_id: str = ""
    post_state_id: str = ""
    entry_context_id: str = ""
    next_policy_context_id: str = ""
    action_context_certified: bool = False
    successor_context_certified: bool = False
    external_event_chain_certified: bool = False
    external_event_chain_length: int = 0
    scope_certified: bool = False
    schema_certified: bool = False
    unique_pre_context: bool = False
    unique_success_observation: bool = False
    unique_post_context: bool = False
    fresh_next_context_exact_match: bool = False
    pre_context_candidate_count: int = 0
    canonical_success_observation_count: int = 0
    post_context_candidate_count: int = 0
    context_matched_record_count: int = 0
    context_mismatched_record_count: int = 0
    successor_binding_version: str = ""
    successor_binding_sha256: str = ""
    certificate: str = REAL_TRAJECTORY_SIDECAR_CERTIFICATE
    state_hash_version: str = REAL_TRAJECTORY_STATE_HASH_VERSION

    def to_turn_info(self) -> dict[str, Any]:
        return {
            "accepted": True,
            "success": self.label > 0,
            "verifier_status": (
                "valid"
                if self.label > 0
                else "invalid" if self.label < 0 else "ambiguous"
            ),
            "branch_type": self.branch_type,
            "policy_context_id": self.entry_context_id or self.group_id,
            "trajectory_sidecar_credit_label": self.label,
            "trajectory_sidecar_policy_eligible": self.actor_eligible,
            "trajectory_sidecar_group_id": self.group_id,
            "trajectory_sidecar_state_id": self.state_id,
            "trajectory_sidecar_pre_state_id": self.pre_state_id,
            "trajectory_sidecar_post_state_id": self.post_state_id,
            "trajectory_sidecar_entry_context_id": self.entry_context_id,
            "trajectory_sidecar_next_policy_context_id": (
                self.next_policy_context_id
            ),
            "trajectory_sidecar_action_key": self.action_key,
            "trajectory_sidecar_observation_key": self.observed_successor_key,
            "trajectory_sidecar_evidence_count": self.evidence_count,
            "trajectory_sidecar_reanchored": self.reanchored,
            "trajectory_sidecar_candidate_state_count": (
                self.candidate_state_count
            ),
            "trajectory_proposal_candidate_state_count": (
                self.proposal_candidate_state_count
            ),
            "trajectory_proposal_external_event_advanced": (
                self.proposal_external_event_advanced
            ),
            "trajectory_sidecar_certificate": self.certificate,
            "trajectory_sidecar_state_hash_version": self.state_hash_version,
            "trajectory_sidecar_successor_binding_version": (
                self.successor_binding_version
            ),
            "trajectory_sidecar_successor_binding_sha256": (
                self.successor_binding_sha256
            ),
            "recorded_graph_pre_state_id": self.pre_state_id,
            "recorded_graph_post_state_id": self.post_state_id,
            "recorded_graph_entry_context_id": self.entry_context_id,
            "recorded_graph_next_policy_context_id": (
                self.next_policy_context_id
            ),
            "recorded_graph_action_key": self.action_key,
            "recorded_graph_observation_key": self.observed_successor_key,
            "recorded_graph_action_context_certified": (
                self.action_context_certified
            ),
            "recorded_graph_successor_context_certified": (
                self.successor_context_certified
            ),
            "recorded_graph_external_event_chain_certified": (
                self.external_event_chain_certified
            ),
            "recorded_graph_external_event_chain_length": (
                self.external_event_chain_length
            ),
            "recorded_graph_scope_certified": self.scope_certified,
            "recorded_graph_schema_certified": self.schema_certified,
            "recorded_graph_unique_pre_context": self.unique_pre_context,
            "recorded_graph_unique_success_observation": (
                self.unique_success_observation
            ),
            "recorded_graph_unique_post_context": self.unique_post_context,
            "recorded_graph_fresh_next_context_exact_match": (
                self.fresh_next_context_exact_match
            ),
            "recorded_graph_pre_context_candidate_count": (
                self.pre_context_candidate_count
            ),
            "recorded_graph_canonical_success_observation_count": (
                self.canonical_success_observation_count
            ),
            "recorded_graph_post_context_candidate_count": (
                self.post_context_candidate_count
            ),
            "recorded_graph_context_matched_record_count": (
                self.context_matched_record_count
            ),
            "recorded_graph_context_mismatched_record_count": (
                self.context_mismatched_record_count
            ),
            "recorded_graph_successor_binding_version": (
                self.successor_binding_version
            ),
            "recorded_graph_successor_binding_sha256": (
                self.successor_binding_sha256
            ),
            "recorded_state_graph_transition": bool(
                self.successor_binding_sha256
            ),
            "recorded_state_graph_certificate": (
                self.certificate if self.successor_binding_sha256 else ""
            ),
            "recorded_graph_state_hash_version": self.state_hash_version,
            "observation_generated": False,
            "reason": self.reason,
        }


class RealTrajectoryCreditIndex:
    """Exact task-local successor evidence compiled from trajectories."""

    def __init__(
        self,
        evidence: Mapping[tuple[str, str, str, str], RealSuccessorEvidence],
        *,
        successors: Mapping[
            tuple[str, str, str, str, str], frozenset[str]
        ] | None = None,
        observation_anchors: Mapping[
            tuple[str, str, str], frozenset[str]
        ] | None = None,
        action_proposals: Mapping[
            tuple[str, str, str], tuple[RealTrajectoryActionProposal, ...]
        ] | None = None,
        initial_contexts: Mapping[
            tuple[str, str], frozenset[str]
        ] | None = None,
    ) -> None:
        self._evidence = dict(evidence)
        self._successors = dict(successors or {})
        self._observation_anchors = dict(observation_anchors or {})
        self._action_proposals = dict(action_proposals or {})
        self._initial_contexts = dict(initial_contexts or {})
        self._scopes = frozenset(
            (str(domain), str(reset_id))
            for domain, reset_id, _state_id_value, _action in self._evidence
        )

    @classmethod
    def from_episodes(
        cls, episodes: Iterable[TrajectoryEpisode]
    ) -> RealTrajectoryCreditIndex:
        buckets: dict[tuple[str, str, str, str], dict[str, set[str]]] = defaultdict(
            lambda: {
                "successful": set(),
                "explicit_error": set(),
                "failed_nonerror": set(),
            }
        )
        successors: dict[
            tuple[str, str, str, str, str], set[str]
        ] = defaultdict(set)
        observation_anchors: dict[tuple[str, str, str], set[str]] = defaultdict(
            set
        )
        proposal_sources: dict[
            tuple[str, str, str, str], set[str]
        ] = defaultdict(set)
        proposal_messages: dict[
            tuple[str, str, str, str], dict[str, Any]
        ] = {}
        initial_contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
        for episode in episodes:
            reset_ids = _source_task_ids(episode)
            if not reset_ids:
                continue
            domain = str(episode.domain)
            initial_context = policy_context_key(episode.initial_messages)
            successful = _successful_episode(episode)
            error_tainted = _episode_has_explicit_error(episode)
            for reset_id in reset_ids:
                initial_contexts[(domain, reset_id)].add(initial_context)
                prefix: list[dict[str, str]] = []
                for step in episode.steps:
                    if not step.observations:
                        continue
                    state_id = _state_id(
                        domain=domain,
                        reset_id=reset_id,
                        executed_prefix=prefix,
                        initial_context_key=initial_context,
                    )
                    step_action_key = recorded_action_key(step)
                    step_observation_key = observation_key(step.observations)
                    next_prefix = [
                        *prefix,
                        {
                            "action": step_action_key,
                            "observation": step_observation_key,
                        },
                    ]
                    next_state_id = _state_id(
                        domain=domain,
                        reset_id=reset_id,
                        executed_prefix=next_prefix,
                        initial_context_key=initial_context,
                    )
                    bucket = buckets[
                        (domain, reset_id, state_id, step_action_key)
                    ]
                    if any(
                        observation_is_explicit_error(dict(item))
                        for item in step.observations
                    ):
                        bucket["explicit_error"].add(step_observation_key)
                    elif successful and not error_tainted:
                        bucket["successful"].add(step_observation_key)
                        proposal_message = recorded_assistant_message(step)
                        if proposal_message is not None:
                            proposal_key = (
                                domain,
                                reset_id,
                                state_id,
                                step_action_key,
                            )
                            proposal_sources[proposal_key].add(str(episode.id))
                            proposal_messages[proposal_key] = proposal_message
                    else:
                        bucket["failed_nonerror"].add(step_observation_key)
                    successors[
                        (
                            domain,
                            reset_id,
                            state_id,
                            step_action_key,
                            step_observation_key,
                        )
                    ].add(next_state_id)
                    if not any(
                        observation_is_explicit_error(dict(item))
                        for item in step.observations
                    ):
                        observation_anchors[
                            (domain, reset_id, step_observation_key)
                        ].add(next_state_id)
                    prefix = next_prefix
        return cls(
            {
                key: RealSuccessorEvidence(
                    successful_observations=frozenset(value["successful"]),
                    explicit_error_observations=frozenset(
                        value["explicit_error"]
                    ),
                    failed_nonerror_observations=frozenset(
                        value["failed_nonerror"]
                    ),
                )
                for key, value in buckets.items()
            },
            successors={
                key: frozenset(value) for key, value in successors.items()
            },
            observation_anchors={
                key: frozenset(value)
                for key, value in observation_anchors.items()
            },
            action_proposals={
                state_key: tuple(
                    sorted(
                        proposals,
                        key=lambda proposal: proposal.action_key,
                    )
                )
                for state_key, proposals in _group_action_proposals(
                    {
                        key: value
                        for key, value in proposal_sources.items()
                        if not buckets[key]["explicit_error"]
                        and len(buckets[key]["successful"]) == 1
                        and len(
                            buckets[key]["successful"]
                            | buckets[key]["failed_nonerror"]
                        )
                        == 1
                        and len(
                            successors[
                                (
                                    *key,
                                    next(iter(buckets[key]["successful"])),
                                )
                            ]
                        )
                        == 1
                    },
                    proposal_messages,
                ).items()
            },
            initial_contexts={
                key: frozenset(value)
                for key, value in initial_contexts.items()
            },
        )

    @classmethod
    def from_paths(
        cls, paths: Iterable[str | Path]
    ) -> RealTrajectoryCreditIndex:
        episodes: list[TrajectoryEpisode] = []
        for path in paths:
            episodes.extend(
                load_trajectory_episodes(Path(path), source="auto")
            )
        return cls.from_episodes(episodes)

    @property
    def edge_count(self) -> int:
        return len(self._evidence)

    def start(
        self,
        *,
        domain: str,
        reset_id: str,
        legacy_mode: bool = False,
        initial_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> RealTrajectorySession:
        scope = (str(domain), str(reset_id))
        compiled_contexts = self._initial_contexts.get(scope, frozenset())
        if initial_messages is not None:
            initial_context = policy_context_key(initial_messages)
            context_matched = (
                not compiled_contexts or initial_context in compiled_contexts
            )
        elif len(compiled_contexts) == 1:
            initial_context = next(iter(compiled_contexts))
            context_matched = True
        elif not compiled_contexts:
            # Manually constructed unit indexes predate explicit initial
            # context metadata. Preserve their isolated test semantics.
            initial_context = ""
            context_matched = True
        else:
            initial_context = ""
            context_matched = False
        return RealTrajectorySession(
            index=self,
            domain=str(domain),
            reset_id=str(reset_id),
            legacy_mode=legacy_mode,
            initial_context_key=initial_context,
            initial_context_matched=context_matched,
        )

    def has_scope(self, *, domain: str, reset_id: str) -> bool:
        return (str(domain), str(reset_id)) in self._scopes

    def evidence_for(
        self,
        *,
        domain: str,
        reset_id: str,
        state_id: str,
        action: str,
    ) -> RealSuccessorEvidence | None:
        return self._evidence.get((domain, reset_id, state_id, action))

    def successors_for(
        self,
        *,
        domain: str,
        reset_id: str,
        state_id: str,
        action: str,
        observation: str,
    ) -> frozenset[str]:
        return self._successors.get(
            (domain, reset_id, state_id, action, observation), frozenset()
        )

    def anchors_for_observation(
        self, *, domain: str, reset_id: str, observation: str
    ) -> frozenset[str]:
        return self._observation_anchors.get(
            (domain, reset_id, observation), frozenset()
        )

    def proposals_for_state(
        self, *, domain: str, reset_id: str, state_id: str
    ) -> tuple[RealTrajectoryActionProposal, ...]:
        return self._action_proposals.get(
            (domain, reset_id, state_id), ()
        )


class RealTrajectorySession:
    """Track one real rollout while signing, but never executing, its turns."""

    def __init__(
        self,
        *,
        index: RealTrajectoryCreditIndex,
        domain: str,
        reset_id: str,
        legacy_mode: bool = False,
        initial_context_key: str = "",
        initial_context_matched: bool = True,
    ) -> None:
        self.index = index
        self.domain = domain
        self.reset_id = reset_id
        self.legacy_mode = bool(legacy_mode)
        self.initial_context_key = str(initial_context_key)
        self.executed_prefix: list[dict[str, str]] = []
        initial_state_id = _state_id(
            domain=domain,
            reset_id=reset_id,
            executed_prefix=[],
            initial_context_key=self.initial_context_key,
        )
        self.candidate_state_ids: set[str] = (
            {initial_state_id} if initial_context_matched else set()
        )
        self.proposal_candidate_state_ids = set(self.candidate_state_ids)

    def action_proposals(self) -> tuple[RealTrajectoryActionProposal, ...]:
        """Return actions shared by every current exact or proposal-only state."""

        runtime_context_id = _state_id(
            domain=self.domain,
            reset_id=self.reset_id,
            executed_prefix=self.executed_prefix,
            initial_context_key=self.initial_context_key,
        )
        if self.legacy_mode:
            proposal_states = (
                self.candidate_state_ids
                if self.candidate_state_ids
                else self.proposal_candidate_state_ids
            )
        else:
            # Probe proposals must be attached to the one fresh runtime
            # context.  Observation-only reanchors and task-level fallback
            # paths remain available exclusively in explicit legacy mode.
            if self.candidate_state_ids != {runtime_context_id}:
                return ()
            proposal_states = self.candidate_state_ids
        if not proposal_states:
            return ()
        proposals_by_state = []
        for state_id in sorted(proposal_states):
            proposals_by_state.append(
                {
                    proposal.action_key: proposal
                    for proposal in self.index.proposals_for_state(
                        domain=self.domain,
                        reset_id=self.reset_id,
                        state_id=state_id,
                    )
                }
            )
        common_keys = set(proposals_by_state[0])
        for proposals in proposals_by_state[1:]:
            common_keys.intersection_update(proposals)
        emitted: list[RealTrajectoryActionProposal] = []
        for action_key in sorted(common_keys):
            assistant_message = dict(
                proposals_by_state[0][action_key].assistant_message
            )
            # Imported historical rows can occasionally declare an
            # assistant-text step whose content is actually a serialized
            # <tool_call>.  Executing that message makes the live adapter
            # canonicalize it as a tool action, so the stored text key can
            # never satisfy the exact intervention binding.  Keep the row as
            # evidence, but never expose an internally inconsistent proposal.
            if assistant_action_key(assistant_message) != action_key:
                continue
            emitted.append(
                RealTrajectoryActionProposal(
                action_key=action_key,
                assistant_message=assistant_message,
                evidence_count=sum(
                    proposals[action_key].evidence_count
                    for proposals in proposals_by_state
                ),
                pre_context_id=(
                    next(iter(proposal_states))
                    if len(proposal_states) == 1
                    else ""
                ),
                certificate=(
                    REAL_TRAJECTORY_SIDECAR_CERTIFICATE
                    if self.candidate_state_ids
                    else REAL_TRAJECTORY_TASK_ANCHORED_PROPOSAL_CERTIFICATE
                ),
            )
            )
        return tuple(emitted)

    def observe(
        self,
        *,
        assistant_message: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
    ) -> RealTrajectoryTurnCertificate:
        prior_candidate_state_ids = set(self.candidate_state_ids)
        prior_proposal_candidate_state_ids = set(
            self.proposal_candidate_state_ids
        )
        runtime_state_id = _state_id(
            domain=self.domain,
            reset_id=self.reset_id,
            executed_prefix=self.executed_prefix,
            initial_context_key=self.initial_context_key,
        )
        schema_certified = bool(
            isinstance(assistant_message, Mapping)
            and observations
            and all(isinstance(item, Mapping) for item in observations)
        )
        try:
            if schema_certified and len(
                actions_from_assistant_message(dict(assistant_message))
            ) > 1:
                schema_certified = False
            sampled_action_key = (
                assistant_action_key(assistant_message)
                if schema_certified
                else _canonical_json({"type": "invalid_assistant_schema"})
            )
            sampled_observation_key = (
                observation_key(observations)
                if schema_certified
                else _canonical_json({"type": "invalid_observation_schema"})
            )
        except (TypeError, ValueError, AttributeError):
            schema_certified = False
            sampled_action_key = _canonical_json(
                {"type": "invalid_assistant_schema"}
            )
            sampled_observation_key = _canonical_json(
                {"type": "invalid_observation_schema"}
            )
        runtime_next_prefix = [
            *self.executed_prefix,
            {
                "action": sampled_action_key,
                "observation": sampled_observation_key,
            },
        ]
        runtime_next_context_id = _state_id(
            domain=self.domain,
            reset_id=self.reset_id,
            executed_prefix=runtime_next_prefix,
            initial_context_key=self.initial_context_key,
        )
        state_id = (
            next(iter(prior_candidate_state_ids))
            if len(prior_candidate_state_ids) == 1
            else _stable_hash(
                {
                    "domain": self.domain,
                    "reset_id": self.reset_id,
                    "candidate_states": sorted(prior_candidate_state_ids),
                    "runtime_state_id": runtime_state_id,
                }
            )
        )
        group_id = f"{self.domain}:{self.reset_id}:{state_id}"

        scope_certified = self.index.has_scope(
            domain=self.domain,
            reset_id=self.reset_id,
        )
        unique_pre_context = bool(
            scope_certified
            and schema_certified
            and prior_candidate_state_ids == {runtime_state_id}
        )
        exact_evidence = (
            self.index.evidence_for(
                domain=self.domain,
                reset_id=self.reset_id,
                state_id=runtime_state_id,
                action=sampled_action_key,
            )
            if unique_pre_context
            else None
        )

        successful_edges: set[tuple[str, str]] = set()
        if exact_evidence is not None:
            for successful_observation in (
                exact_evidence.successful_observations
            ):
                post_states = self.index.successors_for(
                    domain=self.domain,
                    reset_id=self.reset_id,
                    state_id=runtime_state_id,
                    action=sampled_action_key,
                    observation=successful_observation,
                )
                if post_states:
                    successful_edges.update(
                        (successful_observation, post_state)
                        for post_state in post_states
                    )
                else:
                    successful_edges.add((successful_observation, ""))
        matched_success_edges = {
            edge
            for edge in successful_edges
            if edge[0] == sampled_observation_key and edge[1]
        }
        mismatched_success_edges = successful_edges - matched_success_edges
        successful_post_contexts = {
            post_state for _observation, post_state in matched_success_edges
        }
        canonical_success_observation_count = (
            len(exact_evidence.successful_observations)
            if exact_evidence is not None
            else 0
        )
        authentic_observations = (
            exact_evidence.successful_observations
            | exact_evidence.explicit_error_observations
            | exact_evidence.failed_nonerror_observations
            if exact_evidence is not None
            else frozenset()
        )
        strict_evidence_conflict = bool(
            exact_evidence is not None
            and exact_evidence.successful_observations
            and (
                exact_evidence.conflict
                or len(authentic_observations) != 1
            )
        )
        unique_success_observation = bool(
            exact_evidence is not None
            and not strict_evidence_conflict
            and canonical_success_observation_count == 1
        )
        unique_post_context = len(successful_post_contexts) == 1
        post_state_id = (
            next(iter(successful_post_contexts))
            if unique_post_context
            else ""
        )
        fresh_next_context_exact_match = bool(
            unique_post_context
            and post_state_id == runtime_next_context_id
        )
        action_context_certified = bool(
            unique_pre_context and exact_evidence is not None
        )
        successor_context_certified = bool(
            action_context_certified
            and not strict_evidence_conflict
            and unique_success_observation
            and sampled_observation_key
            in exact_evidence.successful_observations
            and unique_post_context
            and fresh_next_context_exact_match
            and len(matched_success_edges) == 1
            and not mismatched_success_edges
        )
        successor_binding_version = ""
        successor_binding_sha256 = ""
        if successor_context_certified:
            successor_binding_version = FULL_SUCCESSOR_BINDING_VERSION
            successor_binding_sha256 = full_successor_binding_sha256(
                pre_state_id=runtime_state_id,
                post_state_id=post_state_id,
                entry_context_id=runtime_state_id,
                next_policy_context_id=runtime_next_context_id,
                action_key=sampled_action_key,
                observation_key=sampled_observation_key,
                version=successor_binding_version,
            )

        # Pre-Probe artifacts merged all observation-anchored candidates. Keep
        # that behavior only for an explicitly requested legacy session.
        candidate_evidence = [
            evidence
            for candidate_state_id in prior_candidate_state_ids
            if (
                evidence := self.index.evidence_for(
                    domain=self.domain,
                    reset_id=self.reset_id,
                    state_id=candidate_state_id,
                    action=sampled_action_key,
                )
            )
            is not None
        ]
        legacy_evidence = (
            RealSuccessorEvidence(
                successful_observations=frozenset().union(
                    *(item.successful_observations for item in candidate_evidence)
                ),
                explicit_error_observations=frozenset().union(
                    *(item.explicit_error_observations for item in candidate_evidence)
                ),
                failed_nonerror_observations=frozenset().union(
                    *(item.failed_nonerror_observations for item in candidate_evidence)
                ),
            )
            if candidate_evidence
            else None
        )
        label_evidence = legacy_evidence if self.legacy_mode else exact_evidence
        evidence_count = (
            sum(
                len(values)
                for values in (
                    label_evidence.successful_observations,
                    label_evidence.explicit_error_observations,
                    label_evidence.failed_nonerror_observations,
                )
            )
            if label_evidence is not None
            else 0
        )
        label = 0.0
        branch_type = "ambiguous_real_successor_not_in_trajectory_evidence"
        reason = (
            "the real environment executed this transition, but no exact "
            "trajectory successor certifies local policy direction"
        )
        if self.legacy_mode and label_evidence is not None:
            if label_evidence.conflict:
                branch_type = "ambiguous_conflicting_trajectory_successor"
                reason = "successful and explicit-error evidence conflict"
            elif sampled_observation_key in label_evidence.successful_observations:
                label = 1.0
                branch_type = "valid_real_trajectory_successor"
                reason = "legacy observation-anchored successful successor match"
            elif sampled_observation_key in label_evidence.explicit_error_observations:
                label = -1.0
                branch_type = "invalid_real_recorded_error_successor"
                reason = "legacy exact recorded explicit backend error match"
            else:
                branch_type = "ambiguous_real_successor_mismatch"
                reason = "legacy trajectory successor observation mismatch"
        elif not schema_certified:
            branch_type = "ambiguous_real_successor_schema_mismatch"
            reason = "assistant action or real observation has an invalid schema"
        elif not scope_certified:
            branch_type = "ambiguous_real_successor_scope_mismatch"
            reason = "no trajectory evidence exists for this exact task scope"
        elif not unique_pre_context:
            branch_type = "ambiguous_nonunique_real_pre_context"
            reason = "the fresh runtime pre-context is not uniquely anchored"
        elif exact_evidence is None:
            pass
        elif strict_evidence_conflict:
            branch_type = "ambiguous_conflicting_trajectory_successor"
            reason = "the context-action has conflicting authentic successors"
        elif sampled_observation_key in exact_evidence.explicit_error_observations:
            label = -1.0
            branch_type = "invalid_real_recorded_error_successor"
            reason = "the real observation exactly matches a recorded backend error"
        elif canonical_success_observation_count != 1:
            branch_type = "ambiguous_nonunique_canonical_success_observation"
            reason = "the exact context-action has no unique success observation"
        elif sampled_observation_key not in exact_evidence.successful_observations:
            branch_type = "ambiguous_real_successor_mismatch"
            reason = "the real observation differs from the unique success observation"
        elif not unique_post_context:
            branch_type = "ambiguous_nonunique_real_post_context"
            reason = "the exact context-action-observation has no unique successor"
        elif not fresh_next_context_exact_match:
            branch_type = "ambiguous_fresh_next_context_mismatch"
            reason = "the freshly reconstructed next context differs from evidence"
        elif successor_context_certified:
            label = 1.0
            branch_type = "valid_real_trajectory_successor"
            reason = "the unique bound (context, action, observation, next-context) matches"

        exact_successors: set[str] = set()
        transition_source_states = prior_candidate_state_ids
        if self.legacy_mode and not transition_source_states:
            transition_source_states = prior_proposal_candidate_state_ids
        if schema_certified:
            for candidate_state_id in transition_source_states:
                exact_successors.update(
                    self.index.successors_for(
                        domain=self.domain,
                        reset_id=self.reset_id,
                        state_id=candidate_state_id,
                        action=sampled_action_key,
                        observation=sampled_observation_key,
                    )
                )
        reanchored = False
        proposal_external_event_advanced = False
        if exact_successors:
            self.candidate_state_ids = exact_successors
            self.proposal_candidate_state_ids = set(exact_successors)
        elif self.legacy_mode:
            anchors = self.index.anchors_for_observation(
                domain=self.domain,
                reset_id=self.reset_id,
                observation=sampled_observation_key,
            )
            self.candidate_state_ids = set(anchors)
            reanchored = bool(anchors)
            if anchors:
                self.proposal_candidate_state_ids = set(anchors)
            else:
                parsed_actions = actions_from_assistant_message(
                    dict(assistant_message)
                )
                explicit_error = any(
                    observation_is_explicit_error(dict(item))
                    for item in observations
                )
                proposal_sources = (
                    prior_candidate_state_ids
                    or prior_proposal_candidate_state_ids
                )
                proposal_successors: set[str] = set()
                if not parsed_actions and not explicit_error:
                    for candidate_state_id in proposal_sources:
                        proposal_evidence = self.index.evidence_for(
                            domain=self.domain,
                            reset_id=self.reset_id,
                            state_id=candidate_state_id,
                            action=sampled_action_key,
                        )
                        if (
                            proposal_evidence is None
                            or proposal_evidence.conflict
                            or len(proposal_evidence.successful_observations) != 1
                        ):
                            continue
                        expected_observation = next(
                            iter(proposal_evidence.successful_observations)
                        )
                        proposal_successors.update(
                            self.index.successors_for(
                                domain=self.domain,
                                reset_id=self.reset_id,
                                state_id=candidate_state_id,
                                action=sampled_action_key,
                                observation=expected_observation,
                            )
                        )
                self.proposal_candidate_state_ids = proposal_successors
                proposal_external_event_advanced = bool(proposal_successors)
        else:
            self.candidate_state_ids = set()
            self.proposal_candidate_state_ids = set()

        self.executed_prefix[:] = runtime_next_prefix
        return RealTrajectoryTurnCertificate(
            label=label,
            actor_eligible=label != 0.0,
            group_id=group_id,
            state_id=state_id,
            action_key=sampled_action_key,
            observed_successor_key=sampled_observation_key,
            branch_type=branch_type,
            reason=reason,
            evidence_count=evidence_count,
            reanchored=reanchored,
            candidate_state_count=len(self.candidate_state_ids),
            proposal_candidate_state_count=len(self.proposal_candidate_state_ids),
            proposal_external_event_advanced=proposal_external_event_advanced,
            pre_state_id=runtime_state_id if unique_pre_context else "",
            post_state_id=post_state_id,
            entry_context_id=runtime_state_id,
            next_policy_context_id=runtime_next_context_id,
            action_context_certified=action_context_certified,
            successor_context_certified=successor_context_certified,
            external_event_chain_certified=successor_context_certified,
            external_event_chain_length=(
                max(len(observations) - 1, 0) if schema_certified else 0
            ),
            scope_certified=scope_certified,
            schema_certified=schema_certified,
            unique_pre_context=unique_pre_context,
            unique_success_observation=unique_success_observation,
            unique_post_context=unique_post_context,
            fresh_next_context_exact_match=fresh_next_context_exact_match,
            pre_context_candidate_count=len(prior_candidate_state_ids),
            canonical_success_observation_count=(
                canonical_success_observation_count
            ),
            post_context_candidate_count=len(successful_post_contexts),
            context_matched_record_count=len(matched_success_edges),
            context_mismatched_record_count=len(mismatched_success_edges),
            successor_binding_version=successor_binding_version,
            successor_binding_sha256=successor_binding_sha256,
        )


def audit_verified_prefix_alignment(
    *,
    index: RealTrajectoryCreditIndex,
    prefixes: VerifiedPrefixIndex,
) -> dict[str, Any]:
    """Prove every reset prefix is connected to the local-credit graph.

    Prefix replay and trajectory credit intentionally consume different data
    views: the former needs the original dialogue, while the latter needs
    compiler-normalized evidence.  A syntactically valid dialogue file can
    therefore build reset plans yet compile to zero sidecar transitions.  A
    health check cannot detect that stage mismatch.  Replaying every expected
    prefix transition through the read-only sidecar does, without calling or
    simulating an environment.
    """

    failures: list[dict[str, Any]] = []
    checked_turns = 0
    positive_turns = 0
    for reference in prefixes.references:
        plan = prefixes.get(reference)
        session = index.start(
            domain=plan.domain,
            reset_id=plan.task_id,
            initial_messages=[
                plan.system_message,
                plan.initial_user_message,
            ],
        )
        for turn_index, turn in enumerate(plan.turns):
            certificate = session.observe(
                assistant_message=turn.assistant,
                observations=[turn.expected],
            )
            checked_turns += 1
            positive_turns += certificate.label > 0.0
            if certificate.candidate_state_count > 0:
                continue
            failures.append(
                {
                    "reference": reference,
                    "domain": plan.domain,
                    "task_id": plan.task_id,
                    "turn_index": turn_index,
                    "branch_type": certificate.branch_type,
                    "reason": certificate.reason,
                }
            )
            break
    return {
        "format": "trajectory2gym_verified_prefix_sidecar_alignment_v1",
        "status": "pass" if not failures else "fail",
        "prefixes": len(prefixes),
        "checked_turns": checked_turns,
        "positive_turns": positive_turns,
        "failed_prefixes": len(failures),
        "failures": failures,
    }


__all__ = [
    "REAL_TRAJECTORY_SIDECAR_CERTIFICATE",
    "REAL_TRAJECTORY_TASK_ANCHORED_PROPOSAL_CERTIFICATE",
    "RealSuccessorEvidence",
    "RealTrajectoryActionProposal",
    "RealTrajectoryCreditIndex",
    "RealTrajectorySession",
    "RealTrajectoryTurnCertificate",
    "assistant_action_key",
    "audit_verified_prefix_alignment",
    "observation_key",
    "recorded_assistant_message",
]
