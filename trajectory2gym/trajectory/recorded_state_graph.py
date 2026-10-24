from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .observation_db import action_key, canonical_visible_message
from .recovery import episode_explicit_error_count
from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep


@dataclass(frozen=True)
class RecordedStateScope:
    """Explicit namespace for one recovered state graph.

    State ids are never treated as globally meaningful.  A graph edge is
    reusable only inside the same declared trajectory group, source, domain,
    and tool-schema namespace.
    """

    source: str
    domain: str
    trajectory_group_id: str
    tool_signature: str


@dataclass(frozen=True)
class RecordedTransitionRecord:
    """One authentic state transition copied from a static trajectory."""

    scope: RecordedStateScope
    pre_state_id: str
    post_state_id: str
    action_key: str
    actions: tuple[ToolAction, ...]
    observations: tuple[dict[str, Any], ...]
    episode: TrajectoryEpisode
    step: TrajectoryStep
    step_position: int
    training_eligible: bool
    terminal_certificate: bool
    local_completion_certificate: bool
    pre_context_id: str
    post_context_id: str
    next_policy_context_id: str
    pre_environment_context_id: str
    next_policy_environment_context_id: str
    external_event_chain_length: int

    @property
    def episode_id(self) -> str:
        return self.episode.id


@dataclass(frozen=True)
class RecordedTransitionLookup:
    """Resolution of an action at one exact recorded state node."""

    status: str
    records: tuple[RecordedTransitionRecord, ...] = ()
    selected_record: RecordedTransitionRecord | None = None
    observations: tuple[dict[str, Any], ...] = ()
    post_state_id: str = ""
    progress_eligible: bool = False
    terminal: bool = False
    local_completion: bool = False
    context_certified: bool = False
    context_id: str = ""
    context_matched_record_count: int = 0
    context_mismatched_record_count: int = 0
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.status != "miss"


class RecordedStateGraph:
    """Evidence-bounded graph compiled from explicitly labelled transitions.

    The graph does not infer state identity.  Every edge must declare both a
    pre-state and a post-state.  Lookup is keyed by the exact current node and
    canonical action.  A transition advances only when all records at that
    key agree on both observation and post-state and at least one successful
    trajectory supplies a recorded continuation, complete local block, or
    terminal certificate.

    Failed trajectories remain useful observation evidence.  They can make a
    lookup conflict or provide an evidence-only hit, but can never supply a
    positive continuation, local completion, or terminal certificate.
    """

    def __init__(
        self,
        records: Iterable[RecordedTransitionRecord] = (),
    ) -> None:
        self.records = tuple(records)
        self._by_edge: dict[
            tuple[RecordedStateScope, str, str],
            list[RecordedTransitionRecord],
        ] = {}
        for record in self.records:
            self._by_edge.setdefault(
                (record.scope, record.pre_state_id, record.action_key), []
            ).append(record)

    @classmethod
    def from_episodes(
        cls,
        episodes: Iterable[TrajectoryEpisode],
    ) -> "RecordedStateGraph":
        records: list[RecordedTransitionRecord] = []
        for episode in episodes:
            scope = recorded_state_scope(episode)
            if scope is None:
                continue
            # A recovered trajectory remains authentic observation evidence,
            # but an episode containing an explicit backend error cannot
            # certify positive graph progress.  This is intentionally an
            # episode-level decision: compiler-created windows may move the
            # failing observation into ``initial_messages``, and a later
            # successful suffix must not erase that recovery history.
            eligible = bool(
                _training_eligible(episode)
                and episode_explicit_error_count(episode) == 0
            )
            context_path = recorded_context_path(episode)
            environment_context_path = recorded_environment_context_path(episode)
            for position, step in enumerate(episode.steps):
                pre_state_id = _state_id(step.metadata, "pre_state_id")
                post_state_id = _state_id(step.metadata, "post_state_id")
                if (
                    not pre_state_id
                    or not post_state_id
                    or not step.expected_actions
                    or not step.observations
                ):
                    continue
                records.append(
                    RecordedTransitionRecord(
                        scope=scope,
                        pre_state_id=pre_state_id,
                        post_state_id=post_state_id,
                        action_key=recorded_action_key(step.expected_actions),
                        actions=step.expected_actions,
                        observations=tuple(copy.deepcopy(list(step.observations))),
                        episode=episode,
                        step=step,
                        step_position=position,
                        training_eligible=eligible,
                        terminal_certificate=bool(
                            eligible
                            and step.metadata.get("recorded_graph_terminal") is True
                            and _no_later_policy_step(episode, position)
                        ),
                        local_completion_certificate=bool(
                            eligible
                            and episode.metadata.get(
                                "complete_verifiable_decision_block"
                            )
                            is True
                            and episode.metadata.get("decision_window_truncated")
                            is not True
                            and _no_later_policy_step(episode, position)
                        ),
                        pre_context_id=context_path[position],
                        post_context_id=context_path[position + 1],
                        next_policy_context_id=_next_policy_context_id(
                            episode,
                            position,
                            context_path,
                        ),
                        pre_environment_context_id=(
                            environment_context_path[position]
                        ),
                        next_policy_environment_context_id=(
                            _next_policy_context_id(
                                episode,
                                position,
                                environment_context_path,
                            )
                        ),
                        external_event_chain_length=_external_event_chain_length(
                            episode, position
                        ),
                    )
                )
        return cls(records)

    def scope_for_episode(
        self, episode: TrajectoryEpisode
    ) -> RecordedStateScope | None:
        return recorded_state_scope(episode)

    def initial_state_id(self, episode: TrajectoryEpisode) -> str:
        return str(episode.metadata.get("trajectory_state_id") or "").strip()

    def lookup(
        self,
        *,
        scope: RecordedStateScope,
        pre_state_id: str,
        actions: Iterable[ToolAction],
        context_id: str | None = None,
    ) -> RecordedTransitionLookup:
        canonical = recorded_action_key(tuple(actions))
        all_records = tuple(
            self._by_edge.get((scope, str(pre_state_id).strip(), canonical), ())
        )
        if not all_records:
            return RecordedTransitionLookup(
                status="miss",
                reason="no recorded edge for exact scope, pre-state, and action",
            )

        normalized_context_id = str(context_id or "").strip()
        if normalized_context_id:
            records = tuple(
                record
                for record in all_records
                if record.pre_context_id == normalized_context_id
            )
            if not records:
                return RecordedTransitionLookup(
                    status="context_mismatch",
                    records=all_records,
                    context_id=normalized_context_id,
                    context_matched_record_count=0,
                    context_mismatched_record_count=len(all_records),
                    reason=(
                        "recorded state/action evidence exists, but no edge has "
                        "the exact canonical policy-visible entry context"
                    ),
                )
        else:
            # Context-free lookup remains available to offline corpus audits.
            # Runtime callers always provide a context id and must check
            # ``context_certified`` before advancing.
            records = all_records

        lookup_context = {
            "context_certified": bool(normalized_context_id),
            "context_id": normalized_context_id,
            "context_matched_record_count": len(records),
            "context_mismatched_record_count": len(all_records) - len(records),
        }

        observation_signatures = {
            _observation_signature(record.observations) for record in records
        }
        post_states = {record.post_state_id for record in records}
        if len(observation_signatures) != 1 or len(post_states) != 1:
            return RecordedTransitionLookup(
                status="conflict",
                records=records,
                **lookup_context,
                reason=(
                    "recorded edge has conflicting authentic observations"
                    if len(observation_signatures) != 1
                    else "recorded edge has conflicting post-state ids"
                ),
            )

        post_state_id = next(iter(post_states))
        eligible_records = tuple(
            record for record in records if record.training_eligible
        )
        selected = self._select_progress_record(
            eligible_records, post_state_id=post_state_id
        )
        if selected is None:
            # The observation is authentic, but without a successful recorded
            # suffix it cannot become positive policy progress.
            return RecordedTransitionLookup(
                status="evidence_only",
                records=records,
                observations=records[0].observations,
                post_state_id=post_state_id,
                **lookup_context,
                reason=(
                    "edge is supported only by failed trajectories"
                    if not eligible_records
                    else (
                        "edge has no recorded successful continuation, complete "
                        "verifiable block, or terminal certificate"
                    )
                ),
            )

        return RecordedTransitionLookup(
            status="hit",
            records=records,
            selected_record=selected,
            observations=selected.observations,
            post_state_id=post_state_id,
            progress_eligible=True,
            terminal=selected.terminal_certificate,
            local_completion=selected.local_completion_certificate,
            **lookup_context,
            reason=(
                "unique authentic edge reaches a certified terminal state"
                if selected.terminal_certificate
                else (
                    "unique authentic edge completes a recorded successful "
                    "verifiable decision block"
                    if selected.local_completion_certificate
                    else "unique authentic edge has a successful recorded continuation"
                )
            ),
        )

    def continuation_cursor(
        self,
        *,
        scope: RecordedStateScope,
        state_id: str,
    ) -> RecordedTransitionRecord | None:
        """Return a deterministic successful cursor rooted at ``state_id``."""

        records = self._unambiguous_eligible_outgoing(
            scope=scope, state_id=state_id
        )
        if not records:
            return None
        return min(records, key=_record_sort_key)

    def has_eligible_outgoing(
        self,
        *,
        scope: RecordedStateScope,
        state_id: str,
    ) -> bool:
        return bool(
            self._unambiguous_eligible_outgoing(
                scope=scope, state_id=state_id
            )
        )

    def _unambiguous_eligible_outgoing(
        self,
        *,
        scope: RecordedStateScope,
        state_id: str,
    ) -> tuple[RecordedTransitionRecord, ...]:
        """Return only outgoing records with unique observation and post-state."""

        state = str(state_id).strip()
        result: list[RecordedTransitionRecord] = []
        for (edge_scope, pre_state, _), records in self._by_edge.items():
            if edge_scope != scope or pre_state != state:
                continue
            if len({_observation_signature(item.observations) for item in records}) != 1:
                continue
            if len({item.post_state_id for item in records}) != 1:
                continue
            result.extend(item for item in records if item.training_eligible)
        return tuple(result)

    def _select_progress_record(
        self,
        records: tuple[RecordedTransitionRecord, ...],
        *,
        post_state_id: str,
    ) -> RecordedTransitionRecord | None:
        ordered = sorted(records, key=_record_sort_key)
        local_paths = [
            (record, *_local_recorded_continuation(record, post_state_id))
            for record in ordered
        ]
        terminal_records = [
            record for record in ordered if record.terminal_certificate
        ]
        # A terminal certificate is usable only when the declared post-state
        # has no successful outgoing edge and no copy of the same transition
        # continues to another policy decision.  This prevents inconsistent
        # terminal annotations from beating a real recorded continuation.
        if (
            terminal_records
            and all(
                valid_chain
                for record, _, _, valid_chain in local_paths
                if record.terminal_certificate
            )
            and not any(
                valid_chain
                and self.has_eligible_outgoing(
                    scope=record.scope, state_id=final_state_id
                )
                for record, _, final_state_id, valid_chain in local_paths
                if record.terminal_certificate
            )
            and not any(has_policy for _, has_policy, _, _ in local_paths)
        ):
            return terminal_records[0]
        # A compiler-declared complete decision block is a valid *local*
        # outcome even when ordinary dialogue prevents static verification of
        # the rest of the benchmark task.  It is never promoted to terminal
        # task success.  The source must be successful and the window must be
        # untruncated, so an arbitrary cutoff cannot create this certificate.
        #
        # Prefer this finite certificate over another copy of the same
        # state/action/observation edge that merely has a longer recorded
        # continuation.  Both copies are authentic and agree on the successor
        # state, but the complete block is the stronger *bounded* claim: it can
        # close the current partial-environment episode without asking the
        # policy to reproduce an unrelated suffix.  Choosing a continuation
        # first made selection depend on lexicographic episode ids and could
        # hide an otherwise available local completion certificate.
        completion_records = [
            record
            for record, _, _, valid_chain in local_paths
            if valid_chain and record.local_completion_certificate
        ]
        if completion_records:
            return completion_records[0]
        for record, has_policy, _, _ in local_paths:
            if has_policy:
                return record
        # Do not stitch a completed decision window to an arbitrary outgoing
        # record merely because their backend state ids agree. The next
        # window may contain ordinary assistant/user dialogue in its visible
        # prefix; injecting or omitting that bridge would both be unsound.
        # A graph edge is positive progress only when its own compiled episode
        # already retains the next policy decision (including terminal_stop).
        return None


def recorded_state_scope(
    episode: TrajectoryEpisode,
) -> RecordedStateScope | None:
    group_id = str(episode.metadata.get("trajectory_group_id") or "").strip()
    if not group_id:
        return None
    tool_signature = json.dumps(
        list(episode.tools),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return RecordedStateScope(
        source=str(episode.source),
        domain=str(episode.domain or ""),
        trajectory_group_id=group_id,
        tool_signature=tool_signature,
    )


def recorded_action_key(actions: Iterable[ToolAction]) -> str:
    """Canonical action-sequence key shared by graph compile and lookup."""

    return json.dumps(
        [json.loads(action_key(action)) for action in actions],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def recorded_context_id(messages: Iterable[dict[str, Any]]) -> str:
    """Hash one complete policy-visible transcript after transport cleanup."""

    payload = json.dumps(
        [canonical_visible_message(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_environment_context_message(
    message: dict[str, Any],
) -> dict[str, Any] | None:
    """Project a transcript message to environment-relevant context.

    Free-form assistant narration is a policy artifact, not backend state.
    Likewise, text emitted alongside a structured tool call is not sent to the
    tool executor. User/system messages, structured calls, tool results, and
    external observations remain exact. This projection is used only for a
    one-transition local certificate; multistep continuation still requires
    the complete policy-visible context id above.
    """

    canonical = canonical_visible_message(message)
    role = str(canonical.get("role") or "").strip().lower()
    if role != "assistant":
        return canonical
    tool_calls = canonical.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    canonical["content"] = None
    return canonical


def recorded_environment_context_id(
    messages: Iterable[dict[str, Any]],
) -> str:
    payload = json.dumps(
        [
            projected
            for message in messages
            if (
                projected := canonical_environment_context_message(message)
            )
            is not None
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recorded_context_path(episode: TrajectoryEpisode) -> tuple[str, ...]:
    """Return canonical visible context before/after every recorded step.

    The reconstruction uses only messages and actions already present in the
    trajectory.  Tool-call ids and JSON serialization are normalized by
    ``canonical_visible_message``; ordinary text is preserved byte-for-byte.
    """

    messages = copy.deepcopy(list(episode.initial_messages))
    path = [recorded_context_id(messages)]
    for step in episode.steps:
        _append_recorded_visible_step(messages, step)
        path.append(recorded_context_id(messages))
    return tuple(path)


def recorded_environment_context_path(
    episode: TrajectoryEpisode,
) -> tuple[str, ...]:
    """Return the environment-relevant context around every recorded step."""

    messages = copy.deepcopy(list(episode.initial_messages))
    path = [recorded_environment_context_id(messages)]
    for step in episode.steps:
        _append_recorded_visible_step(messages, step)
        path.append(recorded_environment_context_id(messages))
    return tuple(path)


def recorded_context_messages_at(
    episode: TrajectoryEpisode,
    step_position: int,
) -> tuple[dict[str, Any], ...]:
    """Reconstruct the exact visible transcript before one recorded step.

    This is deliberately reconstructed from messages rather than inferred
    from a state id.  Offline certification must start from the entry
    episode's real visible context, including explicit external events that
    precede its first policy decision.
    """

    if step_position < 0 or step_position > len(episode.steps):
        raise IndexError("recorded context step position is out of range")
    messages = copy.deepcopy(list(episode.initial_messages))
    for step in episode.steps[:step_position]:
        _append_recorded_visible_step(messages, step)
    return tuple(messages)


def recorded_step_action_message(step: TrajectoryStep) -> dict[str, Any] | None:
    """Build the canonical recorded assistant action for context validation."""

    if step.expected_actions:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": copy.deepcopy(action.arguments),
                    },
                }
                for action in step.expected_actions
            ],
        }
    if step.kind in {"assistant_text", "external_request", "terminal_stop"}:
        return {
            "role": "assistant",
            "content": str(step.metadata.get("expected_assistant_message") or ""),
        }
    return None


def _append_recorded_visible_step(
    messages: list[dict[str, Any]], step: TrajectoryStep
) -> None:
    if step.kind != "external_event":
        assistant = recorded_step_action_message(step)
        if assistant is not None:
            messages.append(assistant)
    messages.extend(copy.deepcopy(list(step.observations)))


def _next_policy_context_id(
    episode: TrajectoryEpisode,
    step_position: int,
    context_path: tuple[str, ...],
) -> str:
    position = step_position + 1
    while position < len(episode.steps):
        if episode.steps[position].kind != "external_event":
            break
        position += 1
    return context_path[position]


def _external_event_chain_length(
    episode: TrajectoryEpisode, step_position: int
) -> int:
    count = 0
    for step in episode.steps[step_position + 1 :]:
        if step.kind != "external_event":
            break
        count += 1
    return count


def _state_id(metadata: dict[str, Any], field: str) -> str:
    value = metadata.get(field)
    if value is None:
        # Accept the longer names as a serialization-compatible alias while
        # keeping the public core contract concise.
        value = metadata.get(f"recorded_graph_{field}")
    return str(value or "").strip()


def _training_eligible(episode: TrajectoryEpisode) -> bool:
    value = episode.metadata.get("training_eligible", True)
    # Malformed declarations fail closed instead of turning string values such
    # as ``"false"`` into successful graph continuations.
    return value is True


def _no_later_policy_step(
    episode: TrajectoryEpisode, step_position: int
) -> bool:
    return not any(
        step.kind != "external_event"
        for step in episode.steps[step_position + 1 :]
    )


def _local_recorded_continuation(
    record: RecordedTransitionRecord,
    post_state_id: str,
) -> tuple[bool, str, bool]:
    """Check an explicit state chain after one recorded edge.

    External events may occur before the next policy decision.  They are
    accepted only when each event explicitly connects the current post-state
    to another recorded state node.
    """

    state_id = post_state_id
    for step in record.episode.steps[record.step_position + 1 :]:
        pre_state = _state_id(step.metadata, "pre_state_id")
        next_state = _state_id(step.metadata, "post_state_id")
        if pre_state != state_id or not next_state:
            return False, state_id, False
        if step.kind != "external_event":
            return True, state_id, True
        state_id = next_state
    return False, state_id, True


def _record_sort_key(record: RecordedTransitionRecord) -> tuple[str, int, str]:
    return (record.episode.id, record.step_position, record.action_key)


def _observation_signature(
    observations: tuple[dict[str, Any], ...],
) -> str:
    return json.dumps(
        _normalize_value(copy.deepcopy(list(observations))),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value
