from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from .adapter import StepDecision, TrajectoryAdapter
from .boundary import (
    PARTIAL_EVIDENCE_BRANCH_LABELS,
    PARTIAL_EXTENSION_LABELS,
    STRICT_REPLAY_ACCEPTED_LABELS,
)
from .observation_db import canonical_visible_message
from .recorded_state_graph import (
    RecordedStateGraph,
    RecordedStateScope,
    recorded_context_id,
    recorded_step_action_message,
)
from .schema import ToolAction, TrajectoryEpisode
from .verifiability import VerifiabilityAnalyzer, VerifiabilityDecision


class TrajectoryGymSession:
    """Benchmark-agnostic in-memory trajectory lifecycle."""

    def __init__(
        self,
        episode: TrajectoryEpisode,
        *,
        candidate_episodes: tuple[TrajectoryEpisode, ...] = (),
        adapter: TrajectoryAdapter | None = None,
        verifiability_analyzer: VerifiabilityAnalyzer | None = None,
        recorded_state_graph: RecordedStateGraph | None = None,
        terminate_on_rejected_side_effect: bool = False,
        max_rejected_attempts_per_step: int | None = None,
    ):
        self.root_episode = episode
        self.episode = episode
        self.candidate_episodes = _deduplicate_candidates(
            episode, candidate_episodes
        )
        self.adapter = adapter or TrajectoryAdapter()
        self.verifiability_analyzer = verifiability_analyzer
        self.recorded_state_graph = recorded_state_graph
        self.recorded_graph_scope: RecordedStateScope | None = None
        self.recorded_graph_state_id = ""
        self.recorded_graph_composed = False
        self.terminate_on_rejected_side_effect = terminate_on_rejected_side_effect
        self.max_rejected_attempts_per_step = max_rejected_attempts_per_step
        self.messages: list[dict[str, Any]] = []
        self.step_index = 0
        self.done = False
        self.step_rewards: list[float] = []
        self.step_infos: list[dict[str, Any]] = []
        self.rejected_attempts_at_step = 0
        # Canonical actions already attempted in this episode.  A later
        # off-trajectory DB lookup for an action whose observation has already
        # been returned is safe, but it is not new verifier evidence and must
        # not earn support credit.
        self.seen_action_keys: set[str] = set()

    def reset(self) -> dict[str, Any]:
        self.episode = self.root_episode
        self.messages = self.adapter.initial_messages(self.episode)
        self.step_index = 0
        self.done = False
        self.step_rewards = []
        self.step_infos = []
        self.rejected_attempts_at_step = 0
        self.seen_action_keys = set()
        self.recorded_graph_scope = (
            self.recorded_state_graph.scope_for_episode(self.episode)
            if self.recorded_state_graph is not None
            else None
        )
        self.recorded_graph_state_id = (
            self.recorded_state_graph.initial_state_id(self.episode)
            if self.recorded_state_graph is not None
            else ""
        )
        self.recorded_graph_composed = False
        self._drain_external_events()
        return {"state": self.state_messages(), "info": self.info()}

    def step(self, action: dict[str, Any] | str) -> dict[str, Any]:
        if self.done:
            return {
                "state": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}],
                "reward": 0.0,
                "is_terminated": True,
                "done": True,
                "info": self.info(step_info={"branch_type": "already_done"}),
            }

        # Capture the actual runtime cursor around this policy decision.  The
        # post cursor is recorded only after any explicit external events are
        # drained, so a later policy turn can prove exact state continuity
        # without benchmark-specific workflow rules.
        local_credit_runtime_pre_state_id = self._local_runtime_state_id()

        # The signature contains only the policy-visible transcript before the
        # action.  It is used to form conservative transition-GRPO comparison
        # groups and is never exposed as a model input or success label.
        policy_context_id = _visible_context_signature(self.messages)
        assistant = self._assistant_message(action)
        self.messages.append(assistant)
        # ``state`` remains the complete session transcript for the public
        # in-memory API.  Trainer integrations, however, must append only the
        # messages emitted by this transition.  Record the boundary after the
        # policy action so a verified tool result followed by assistant/user
        # bridge messages can be forwarded without duplicating the history or
        # dropping all but the final user message.
        emitted_state_start = len(self.messages)
        terminated_on_rejected_side_effect = False
        terminated_on_retry_limit = False
        terminated_on_nonprogress_cycle = False
        terminated_on_evidence_only = False
        recorded_graph_progress = False
        recorded_graph_terminal = False

        if self.step_index >= self.episode.step_count:
            decision = StepDecision(
                accepted=True,
                reward=0.0,
                label="already_completed",
                info={"branch_type": "already_completed"},
            )
            self.done = True
        else:
            step = self.episode.steps[self.step_index]
            if self.verifiability_analyzer is not None:
                decision = self._verifiability_decision(step, assistant)
                graph_alternative = self._recorded_state_graph_decision(
                    decision,
                    policy_context_id=policy_context_id,
                )
                if graph_alternative is not None:
                    step, decision = graph_alternative
                else:
                    alternative = self._alternative_trajectory_decision(
                        assistant, decision
                    )
                    if alternative is not None:
                        step, decision = alternative
            else:
                match = self.adapter.match(self.episode, step, assistant)
                decision = self.adapter.decide(self.episode, step, match)
            # A database replay is an accepted, evidence-backed observation,
            # but it is intentionally off-trajectory.  It must not consume
            # the next expected trajectory transition; otherwise an agent can
            # repeat one read-only DB hit and reach ``session_success`` without
            # completing the recorded trajectory.  Exact/equivalent replay
            # carries ``progress_advanced=True`` from the verifier.
            progress_advanced = bool(
                decision.info.get("progress_advanced", decision.accepted)
            )
            if decision.accepted and progress_advanced:
                if bool(decision.info.get("recorded_state_graph_transition", False)):
                    recorded_graph_progress = True
                    recorded_graph_terminal = bool(
                        decision.info.get("recorded_graph_terminal", False)
                    )
                else:
                    recorded_graph_progress, recorded_graph_terminal = (
                        self._advance_recorded_graph_for_step(step)
                    )
            action_key = _nonprogress_action_key(decision)
            if not progress_advanced:
                if action_key and action_key in self.seen_action_keys:
                    decision = _repeated_nonprogress_decision(decision)
                    terminated_on_nonprogress_cycle = True
                elif action_key:
                    self.seen_action_keys.add(action_key)
            elif action_key:
                self.seen_action_keys.add(action_key)
            if decision.accepted and progress_advanced:
                self.messages.extend(copy.deepcopy(list(decision.observation_messages)))
                self.step_index += 1
                external_events = self._drain_external_events()
                if external_events:
                    decision.info["external_events_applied"] = external_events
                    recorded_graph_terminal = bool(
                        recorded_graph_terminal
                        or any(
                            event.get("recorded_graph_terminal", False)
                            for event in external_events
                        )
                    )
                self.rejected_attempts_at_step = 0
                self.done = self.step_index >= self.episode.step_count
                # Never jump to a separate decision window after exhausting
                # the active suffix. Equal backend state ids do not prove the
                # policy-visible transcript is equal: an omitted ordinary
                # assistant/user bridge would make the next action hidden-
                # context supervision. Safe graph alternatives therefore
                # retain their continuation/terminal_stop locally.
            elif decision.accepted:
                self.messages.extend(copy.deepcopy(list(decision.observation_messages)))
                self.rejected_attempts_at_step = 0
                terminated_on_evidence_only = bool(
                    decision.info.get("terminate_after_observation", False)
                )
                if terminated_on_evidence_only:
                    self.done = True
            elif decision.info.get("verifiability") is not None:
                self.messages.append(_typed_feedback_message(decision.info["verifiability"]))
            if not decision.accepted:
                self.rejected_attempts_at_step += 1
            if not decision.accepted and self.terminate_on_rejected_side_effect:
                terminated_on_rejected_side_effect = any(
                    self.adapter.is_side_effecting_tool(str(action.get("name") or ""))
                    for action in decision.info.get("actual_actions", ())
                    if isinstance(action, dict)
                )
                if terminated_on_rejected_side_effect:
                    self.done = True
            terminated_on_retry_limit = bool(
                not decision.accepted
                and self.max_rejected_attempts_per_step is not None
                and self.rejected_attempts_at_step >= self.max_rejected_attempts_per_step
            )
            if terminated_on_retry_limit:
                self.done = True
            if terminated_on_nonprogress_cycle:
                self.done = True

        step_info = copy.deepcopy(decision.info)
        step_info.setdefault("accepted", bool(decision.accepted))
        step_info.setdefault("success", bool(decision.accepted))
        step_info.setdefault("label", decision.label)
        step_info.setdefault("reward", float(decision.reward))
        step_info.setdefault("policy_context_id", policy_context_id)
        branch_type = str(step_info.get("branch_type") or decision.label or "")
        accepted = bool(step_info.get("accepted", decision.accepted))
        verifier_status = str(step_info.get("verifier_status") or "").strip().lower()
        explicit_rejection = bool(
            verifier_status in {"invalid", "rejected"}
            or branch_type.startswith(("invalid_", "rejected_"))
        )
        # Some Partial decisions are derived from an initially ambiguous
        # verifier result (for example, a context-certified state-graph hit).
        # Those decisions copy the original audit payload before replacing
        # the final branch.  Canonicalize this flag from the *final* decision
        # so a sound accepted graph/DB transition cannot retain a stale
        # ``non_blocking_abstention=True`` marker and be masked from actor
        # credit.  Conversely, explicit rejections are never abstentions.
        step_info["non_blocking_abstention"] = bool(
            not accepted
            and not explicit_rejection
            and (
                step_info.get("non_blocking_abstention", False)
                or verifier_status == "ambiguous"
                or branch_type.startswith("ambiguous_")
            )
        )
        step_info["strict_replay_would_accept"] = bool(
            accepted and branch_type in STRICT_REPLAY_ACCEPTED_LABELS
        )
        step_info["partial_extension_branch"] = bool(
            accepted and branch_type in PARTIAL_EXTENSION_LABELS
        )
        step_info["partial_evidence_branch"] = bool(
            accepted and branch_type in PARTIAL_EVIDENCE_BRANCH_LABELS
        )
        # Override branch-local edge fields with the session's actual cursor
        # before/after the full transition.  In particular, a DB hit keeps the
        # same cursor, while Graph/Alternative may drain one or more explicit
        # external events before the next policy decision.
        step_info["local_credit_pre_state_id"] = (
            local_credit_runtime_pre_state_id
        )
        step_info["local_credit_post_state_id"] = (
            self._local_runtime_state_id()
        )
        step_info["terminated_on_rejected_side_effect"] = terminated_on_rejected_side_effect
        step_info["terminated_on_retry_limit"] = terminated_on_retry_limit
        step_info["terminated_on_nonprogress_cycle"] = terminated_on_nonprogress_cycle
        step_info["terminated_on_evidence_only"] = terminated_on_evidence_only
        step_info["rejected_attempts_at_step"] = self.rejected_attempts_at_step
        self.step_rewards.append(float(decision.reward))
        self.step_infos.append(step_info)
        emitted_state = copy.deepcopy(self.messages[emitted_state_start:])
        return {
            "state": self.state_messages(done=self.done),
            "emitted_state": emitted_state,
            "reward": float(decision.reward),
            "is_terminated": bool(self.done),
            "done": bool(self.done),
            "info": self.info(step_info=step_info),
        }

    def evaluate(self, *, sparse: bool = True, params: dict[str, Any] | None = None) -> Any:
        return self.adapter.evaluate(
            self.episode,
            completed=self.done,
            step_index=self.step_index,
            step_rewards=self.step_rewards,
            step_infos=self.step_infos,
            params=params,
            sparse=sparse,
        )

    def state_messages(self, *, done: bool = False) -> list[dict[str, Any]]:
        if done:
            return [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]
        return copy.deepcopy(self.messages)

    def info(self, *, step_info: dict[str, Any] | None = None) -> dict[str, Any]:
        current_step = self.episode.steps[self.step_index] if self.step_index < self.episode.step_count else None
        info = {
            "task_id": self.root_episode.id,
            "active_trajectory_id": self.episode.id,
            "candidate_trajectory_count": len(self.candidate_episodes),
            "recorded_state_graph_enabled": self.recorded_state_graph is not None,
            "recorded_graph_state_id": self.recorded_graph_state_id,
            "source": self.episode.source,
            "domain": self.episode.domain,
            "step": self.step_index,
            "step_count": self.episode.step_count,
            "done": self.done,
            "adapter": self.adapter.source,
        }
        if step_info is not None:
            info["step_info"] = copy.deepcopy(step_info)
        return info

    def _local_runtime_state_id(self) -> str:
        """Return a benchmark-neutral, policy-hidden runtime cursor identity."""

        if self.recorded_graph_state_id:
            scope = self.recorded_graph_scope
            payload = [
                "recorded_graph",
                {
                    "source": scope.source if scope is not None else "",
                    "domain": scope.domain if scope is not None else "",
                    "trajectory_group_id": (
                        scope.trajectory_group_id if scope is not None else ""
                    ),
                    "tool_signature": (
                        scope.tool_signature if scope is not None else ""
                    ),
                },
                self.episode.id,
                self.step_index,
                self.recorded_graph_state_id,
            ]
        else:
            # Some pure-trajectory integrations do not compile a state graph.
            # Episode/cursor identity still proves same-session DB→Exact
            # continuity; hash it so no source identifier becomes policy data.
            payload = ["trajectory_cursor", self.episode.id, self.step_index]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _drain_external_events(self) -> list[dict[str, Any]]:
        """Apply observed non-agent transitions without requesting a policy action."""

        applied: list[dict[str, Any]] = []
        while self.step_index < self.episode.step_count:
            step = self.episode.steps[self.step_index]
            if step.kind != "external_event":
                break
            graph_progress, graph_terminal = self._advance_recorded_graph_for_step(
                step
            )
            self.messages.extend(copy.deepcopy(list(step.observations)))
            applied.append(
                {
                    "step_index": step.index,
                    "external_actions": copy.deepcopy(step.metadata.get("external_actions", [])),
                    "observation_count": len(step.observations),
                    "recorded_graph_progress": graph_progress,
                    "recorded_graph_terminal": graph_terminal,
                }
            )
            self.step_index += 1
        self.done = self.step_index >= self.episode.step_count
        return applied

    def _recorded_state_graph_decision(
        self,
        current_decision: StepDecision,
        *,
        policy_context_id: str,
    ) -> tuple[Any, StepDecision] | None:
        """Resolve an off-path action at the exact current recorded node.

        Global observation-DB evidence is deliberately insufficient here.  A
        graph branch is considered only when the runtime has an explicit
        current state id and the candidate edge lives in the same graph scope.
        """

        if (
            self.recorded_state_graph is None
            or self.recorded_graph_scope is None
            or not self.recorded_graph_state_id
            or current_decision.info.get("verifier_status") == "invalid"
        ):
            return None
        current_progress = bool(
            current_decision.info.get("progress_advanced", False)
        )
        uncomposed_exact_progress = bool(
            current_progress and not self.recorded_graph_composed
        )
        actual_actions = current_decision.info.get("actual_actions") or ()
        if not isinstance(actual_actions, list) or not actual_actions:
            return None
        actions = tuple(
            ToolAction.from_raw(action)
            for action in actual_actions
            if isinstance(action, dict) and action.get("name")
        )
        if not actions:
            return None
        lookup = self.recorded_state_graph.lookup(
            scope=self.recorded_graph_scope,
            pre_state_id=self.recorded_graph_state_id,
            actions=actions,
            context_id=policy_context_id,
        )
        if uncomposed_exact_progress and lookup.status != "hit":
            # Exact replay on the active trajectory has stronger contextual
            # evidence than a conflicting or incomplete cross-trajectory
            # declaration.  A graph lookup may enrich that exact transition
            # only when it independently proves one unique, context-matched
            # edge; it can never turn an otherwise accepted exact action into
            # a rejection.
            return None
        if lookup.status in {"miss", "context_mismatch"}:
            if current_progress and self.recorded_graph_composed:
                label = "recorded_state_graph_unlinked_evidence"
                info = copy.deepcopy(current_decision.info)
                verifiability = copy.deepcopy(info.get("verifiability") or {})
                reason = (
                    "current composed state has no explicit recorded edge; "
                    "returning exact observation without positive continuation"
                )
                verifiability.update(
                    {
                        "label": label,
                        "action_class": "valid",
                        "accepted": True,
                        "observation_messages": copy.deepcopy(
                            list(current_decision.observation_messages)
                        ),
                        "reason": reason,
                    }
                )
                info.update(
                    {
                        "branch_type": label,
                        "verified_support": False,
                        "observation_evidence": True,
                        "positive_continuation_eligible": False,
                        "progress_advanced": False,
                        "terminate_after_observation": True,
                        "verifiability": verifiability,
                    }
                )
                return self.episode.steps[self.step_index], StepDecision(
                    accepted=True,
                    reward=0.0,
                    label=label,
                    observation_messages=current_decision.observation_messages,
                    info=info,
                )
            return None

        evidence_ids = sorted({record.episode_id for record in lookup.records})
        info = copy.deepcopy(current_decision.info)
        verifiability = copy.deepcopy(info.get("verifiability") or {})
        if lookup.status == "conflict":
            label = "ambiguous_recorded_state_graph_conflict"
            verifiability.update(
                {
                    "label": label,
                    "action_class": "ambiguous",
                    "accepted": False,
                    "observation_messages": [],
                    "reason": lookup.reason,
                }
            )
            info.update(
                {
                    "branch_type": label,
                    "verifier_status": "ambiguous",
                    "verified_support": False,
                    "progress_advanced": False,
                    "non_blocking_abstention": True,
                    "recorded_state_graph_transition": False,
                    "recorded_graph_pre_state_id": self.recorded_graph_state_id,
                    "recorded_graph_evidence_episode_ids": evidence_ids,
                    "verifiability": verifiability,
                }
            )
            return lookup.records[0].step, StepDecision(
                accepted=False,
                reward=0.0,
                label=label,
                observation_messages=(),
                info=info,
            )

        if lookup.status == "evidence_only":
            label = "recorded_state_graph_evidence_only"
            verifiability.update(
                {
                    "label": label,
                    "action_class": "valid",
                    "accepted": True,
                    "observation_messages": copy.deepcopy(
                        list(lookup.observations)
                    ),
                    "reason": lookup.reason,
                }
            )
            info.update(
                {
                    "branch_type": label,
                    "verifier_status": "valid",
                    "verified_support": False,
                    "observation_evidence": True,
                    "positive_continuation_eligible": False,
                    "progress_advanced": False,
                    # The observed result is useful evidence, but without a
                    # successful recorded suffix the backend state cannot be
                    # used for positive policy continuation.
                    "terminate_after_observation": True,
                    "recorded_state_graph_transition": False,
                    "recorded_graph_pre_state_id": self.recorded_graph_state_id,
                    "recorded_graph_post_state_id": lookup.post_state_id,
                    "recorded_graph_evidence_episode_ids": evidence_ids,
                    "verifiability": verifiability,
                }
            )
            return lookup.records[0].step, StepDecision(
                accepted=True,
                reward=0.0,
                label=label,
                observation_messages=lookup.observations,
                info=info,
            )

        if current_progress:
            # Preserve the ordinary Exact/Equivalent branch: this action is not
            # a Partial-only extension.  Still attach a local completion
            # certificate when the *active* recorded edge (rather than merely
            # another trajectory sharing its backend ids) proves the same
            # context, action, observation, successor, and external-event
            # chain.  Without this enrichment, a fully verified frontier reset
            # is accepted but silently loses its actor credit.
            active_record = next(
                (
                    record
                    for record in lookup.records
                    if record.episode_id == self.episode.id
                    and record.step_position == self.step_index
                    and record.training_eligible
                ),
                None,
            ) if uncomposed_exact_progress else None
            if active_record is not None:
                context_certificate = self._recorded_graph_context_certificate(
                    active_record,
                    observations=tuple(current_decision.observation_messages),
                )
                locally_complete = bool(
                    active_record.local_completion_certificate
                    or active_record.terminal_certificate
                )
                if context_certificate is not None and locally_complete:
                    current_decision.info.update(
                        {
                            "verified_support": True,
                            "recorded_graph_pre_state_id": (
                                active_record.pre_state_id
                            ),
                            "recorded_graph_post_state_id": (
                                active_record.post_state_id
                            ),
                            "recorded_graph_terminal": (
                                active_record.terminal_certificate
                            ),
                            "recorded_graph_local_completion": (
                                active_record.local_completion_certificate
                            ),
                            "recorded_graph_evidence_episode_ids": evidence_ids,
                            "recorded_state_graph_certificate": (
                                "context_certified_state_action_observation_successor_v1"
                            ),
                            "recorded_graph_context_matched_record_count": (
                                lookup.context_matched_record_count
                            ),
                            "recorded_graph_context_mismatched_record_count": (
                                lookup.context_mismatched_record_count
                            ),
                            **context_certificate,
                            "positive_continuation_eligible": True,
                            "local_continuation_contract": (
                                "context_certified_state_graph_continuation_v1"
                            ),
                        }
                    )
            # The common path below advances the explicit cursor.  Do not mark
            # this as ``recorded_state_graph_transition``: that flag is reserved
            # for a Partial-only branch that already moved the graph cursor.
            return None

        selected = lookup.selected_record
        if selected is None or not lookup.context_certified:
            return None
        context_certificate = self._recorded_graph_context_certificate(
            selected,
            observations=lookup.observations,
        )
        if context_certificate is None:
            # The backend node/action may be known, but the exact runtime
            # transcript cannot be reconciled with the recorded successor.
            # Preserve the ordinary DB observation/abstention semantics and do
            # not issue graph progress.
            return None
        previous_episode = self.episode
        self.episode = selected.episode
        self.step_index = selected.step_position
        self.recorded_graph_scope = selected.scope
        self.recorded_graph_state_id = lookup.post_state_id
        self.recorded_graph_composed = True
        label = "valid_replay_recorded_state_graph"
        verifiability.update(
            {
                "label": label,
                "action_class": "valid",
                "accepted": True,
                "observation_messages": copy.deepcopy(list(lookup.observations)),
                "reason": lookup.reason,
            }
        )
        info.update(
            {
                "branch_type": label,
                "verifier_status": "valid",
                "verified_support": True,
                "progress_advanced": True,
                "recorded_state_graph_transition": True,
                "recorded_graph_pre_state_id": selected.pre_state_id,
                "recorded_graph_post_state_id": selected.post_state_id,
                "local_credit_pre_state_id": selected.pre_state_id,
                "local_credit_post_state_id": selected.post_state_id,
                "recorded_graph_terminal": lookup.terminal,
                "recorded_graph_local_completion": lookup.local_completion,
                "recorded_graph_evidence_episode_ids": evidence_ids,
                "trajectory_branch_from": previous_episode.id,
                "trajectory_branch_to": selected.episode.id,
                "trajectory_branch_evidence": label,
                # This certificate contains no benchmark rule.  It states the
                # exact generic facts already checked by RecordedStateGraph:
                # same declared scope/node, canonical action, unique authentic
                # observation and successor, and a recorded successful suffix.
                "recorded_state_graph_certificate": (
                    "context_certified_state_action_observation_successor_v1"
                ),
                "recorded_graph_context_matched_record_count": (
                    lookup.context_matched_record_count
                ),
                "recorded_graph_context_mismatched_record_count": (
                    lookup.context_mismatched_record_count
                ),
                **context_certificate,
                # The branch is actor-eligible only because the certificate
                # above proves equal policy-visible entry context and the
                # complete authentic successor/external-event chain.  A raw
                # state-id match without that certificate never reaches here.
                "positive_continuation_eligible": True,
                "local_continuation_contract": (
                    "context_certified_state_graph_continuation_v1"
                ),
                "verifiability": verifiability,
            }
        )
        return selected.step, StepDecision(
            accepted=True,
            reward=1.0,
            label=label,
            observation_messages=lookup.observations,
            info=info,
        )

    def _recorded_graph_context_certificate(
        self,
        selected: Any,
        *,
        observations: tuple[dict[str, Any], ...],
    ) -> dict[str, Any] | None:
        """Prove entry, action, observation, and event-chain compatibility.

        ``self.messages`` already contains the just-sampled assistant action.
        The certificate reconstructs the candidate context through every
        immediately following external event and requires it to equal the
        compiler-recorded context at the next policy decision.  No semantic
        similarity or benchmark workflow rule is used.
        """

        if not self.messages:
            return None
        expected_action = recorded_step_action_message(selected.step)
        if expected_action is None:
            return None
        if canonical_visible_message(self.messages[-1]) != canonical_visible_message(
            expected_action
        ):
            return None

        projected_messages = copy.deepcopy(self.messages)
        projected_messages.extend(copy.deepcopy(list(observations)))
        event_count = 0
        for step in selected.episode.steps[selected.step_position + 1 :]:
            if step.kind != "external_event":
                break
            projected_messages.extend(copy.deepcopy(list(step.observations)))
            event_count += 1
        projected_context_id = recorded_context_id(projected_messages)
        if projected_context_id != selected.next_policy_context_id:
            return None
        if event_count != selected.external_event_chain_length:
            return None
        return {
            "recorded_graph_entry_context_id": selected.pre_context_id,
            "recorded_graph_next_policy_context_id": (
                selected.next_policy_context_id
            ),
            "recorded_graph_action_context_certified": True,
            "recorded_graph_successor_context_certified": True,
            "recorded_graph_external_event_chain_certified": True,
            "recorded_graph_external_event_chain_length": event_count,
        }

    def _advance_recorded_graph_for_step(
        self, step: Any
    ) -> tuple[bool, bool]:
        """Advance graph state only across one explicitly connected edge."""

        if (
            self.recorded_state_graph is None
            or self.recorded_graph_scope is None
            or not self.recorded_graph_state_id
        ):
            return False, False
        episode_scope = self.recorded_state_graph.scope_for_episode(self.episode)
        if episode_scope != self.recorded_graph_scope:
            self.recorded_graph_state_id = ""
            return False, False
        pre_state_id = _recorded_step_state_id(step, "pre_state_id")
        post_state_id = _recorded_step_state_id(step, "post_state_id")
        if pre_state_id != self.recorded_graph_state_id or not post_state_id:
            # Never guess across a missing or mismatched state declaration.
            self.recorded_graph_state_id = ""
            return False, False
        self.recorded_graph_state_id = post_state_id
        eligible = self.episode.metadata.get("training_eligible", True) is True
        terminal = bool(
            eligible
            and step.metadata.get("recorded_graph_terminal") is True
            and not any(
                candidate.kind != "external_event"
                for candidate in self.episode.steps[self.step_index + 1 :]
            )
        )
        return True, terminal

    @staticmethod
    def _assistant_message(action: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(action, str):
            return {"role": "assistant", "content": action}
        message = copy.deepcopy(action)
        message.setdefault("role", "assistant")
        return message

    def _verifiability_decision(self, step: Any, assistant: dict[str, Any]) -> StepDecision:
        assert self.verifiability_analyzer is not None
        decision = self.verifiability_analyzer.classify(self.episode, step, assistant)
        match = decision.match
        grounding = (
            match.metadata.get("external_request_grounding", {})
            if match is not None and isinstance(match.metadata, dict)
            else {}
        )
        partial_external_support = bool(
            not decision.accepted
            and grounding.get("expected_actions", 0)
            and grounding.get("matched_actions") == grounding.get("expected_actions")
        )
        database_observation_explicit_error = bool(
            decision.label == "valid_replay_database"
            and decision.observation_lookup is not None
            and any(
                bool(record.metadata.get("explicit_error_observation", False))
                for record in decision.observation_lookup.records
            )
        )
        verified_terminal_stop = bool(
            decision.accepted and decision.label == "valid_terminal_stop"
        )
        return StepDecision(
            accepted=decision.accepted,
            reward=_verifiability_reward(decision),
            label=decision.label,
            observation_messages=decision.observation_messages,
            info={
                "branch_type": decision.label,
                "verifier_status": decision.action_class,
                # A conflict-free observation DB hit is verified evidence even
                # though it deliberately does not consume the expected
                # trajectory transition.  Ranked failure rewards use this bit
                # to distinguish useful off-path reads from unsupported calls.
                "verified_support": bool(
                    decision.label == "valid_replay_database"
                    and not database_observation_explicit_error
                ),
                # Backend errors are still authentic observations and remain
                # available for recovery, but consuming one must suppress any
                # later positive completion credit in this rollout.
                "database_observation_explicit_error": (
                    database_observation_explicit_error
                ),
                "partial_verified_support": partial_external_support,
                "partial_verified_support_type": (
                    "external_request_target_grounded" if partial_external_support else None
                ),
                "partial_verified_support_key": (
                    f"{self.episode.id}:{step.index}:external_request_target_grounded"
                    if partial_external_support
                    else None
                ),
                # Only replay of the current expected transition advances the
                # trajectory cursor.  A conflict-free observation-DB hit is
                # useful evidence, but remains an off-trajectory query.
                "progress_advanced": bool(
                    decision.accepted and decision.label != "valid_replay_database"
                ),
                # Generic state linkage used only to audit local credit.  A
                # non-advancing DB query remains at ``pre_state_id``; an
                # exact/equivalent decision consumes the declared edge.
                "local_credit_pre_state_id": _recorded_step_state_id(
                    step, "pre_state_id"
                ),
                "local_credit_post_state_id": _recorded_step_state_id(
                    step, "post_state_id"
                ),
                "verified_terminal_stop": verified_terminal_stop,
                "terminal_stop_source_eligible": bool(
                    decision.metadata.get("terminal_stop_source_eligible", False)
                ),
                "terminal_stop_semantics_verified": bool(
                    decision.metadata.get(
                        "terminal_stop_semantics_verified", False
                    )
                ),
                "terminal_stop_provenance": str(
                    decision.metadata.get("terminal_stop_provenance") or ""
                ),
                "terminal_stop_synthetic": bool(
                    decision.metadata.get("terminal_stop_synthetic", False)
                ),
                "verifiability": decision.to_dict(),
                "partial_environment": True,
                # Abstaining on an unsupported assistant message is not the
                # same as rejecting an invalid tool action, so it may be
                # retried once without terminating the rollout. Evaluation
                # still marks the resulting path as recovered and suppresses
                # positive credit; retryability is not reward eligibility.
                "non_blocking_abstention": decision.action_class == "ambiguous",
                "actual_actions": (
                    [action.to_dict() for action in match.actual_actions] if match is not None else []
                ),
                "expected_actions": (
                    [action.to_dict() for action in match.expected_actions] if match is not None else []
                ),
            },
        )

    def _alternative_trajectory_decision(
        self,
        assistant: dict[str, Any],
        current_decision: StepDecision,
    ) -> tuple[Any, StepDecision] | None:
        """Advance along another recorded branch with the same visible prefix.

        Observation-DB hits alone remain non-progressing.  A branch advances
        only when the action is the exact/equivalent next transition of a
        trajectory in the same declarative group, and all matching branches
        agree on the returned observation.  This is a finite trajectory trie,
        not a simulator or a fuzzy workflow heuristic.
        """

        if len(self.candidate_episodes) <= 1:
            return None
        if bool(current_decision.info.get("progress_advanced", False)):
            return None
        assert self.verifiability_analyzer is not None

        matches: list[tuple[TrajectoryEpisode, Any, VerifiabilityDecision]] = []
        for candidate in self.candidate_episodes:
            if candidate.id == self.episode.id:
                continue
            if self.step_index >= candidate.step_count:
                continue
            if not _same_recorded_prefix(
                self.episode, candidate, self.step_index
            ):
                continue
            candidate_step = candidate.steps[self.step_index]
            classified = self.verifiability_analyzer.classify(
                candidate, candidate_step, assistant
            )
            if not classified.accepted:
                continue
            if classified.label == "valid_replay_database":
                continue
            matches.append((candidate, candidate_step, classified))

        if not matches:
            return None
        signatures = {
            _messages_signature(classified.observation_messages)
            for _, _, classified in matches
        }
        if len(signatures) != 1:
            return None

        candidate, candidate_step, classified = min(
            matches, key=lambda item: item[0].id
        )
        previous_episode = self.episode
        self.episode = candidate
        decision = self._verifiability_decision(candidate_step, assistant)
        info = copy.deepcopy(decision.info)
        # A trajectory switch describes where the recorded evidence came
        # from; it must not erase the semantic class of a certified terminal
        # boundary.  In particular, overwriting ``valid_terminal_stop`` with
        # ``valid_replay_alternative`` leaves a stop certificate attached to
        # a non-stop branch.  The evaluator then cannot close the otherwise
        # valid local suffix, and the audit quite correctly reports malformed
        # transition metadata.  Preserve the structural stop label while
        # retaining the branch provenance in orthogonal fields.  Ordinary
        # tool transitions keep the alternative label below.
        branch_label = (
            "valid_terminal_stop"
            if decision.label == "valid_terminal_stop"
            else "valid_replay_alternative"
        )
        local_continuation_eligible = False
        info.update(
            {
                "branch_type": branch_label,
                "progress_advanced": True,
                "trajectory_branch_from": previous_episode.id,
                "trajectory_branch_to": candidate.id,
                "trajectory_branch_evidence": classified.label,
                "recorded_graph_pre_state_id": _recorded_step_state_id(
                    candidate_step, "pre_state_id"
                ),
                "recorded_graph_post_state_id": _recorded_step_state_id(
                    candidate_step, "post_state_id"
                ),
                "local_credit_pre_state_id": _recorded_step_state_id(
                    candidate_step, "pre_state_id"
                ),
                "local_credit_post_state_id": _recorded_step_state_id(
                    candidate_step, "post_state_id"
                ),
                "positive_continuation_eligible": (
                    local_continuation_eligible
                ),
                "local_continuation_contract": (
                    "trajectory_alternative_local_continuation_v1"
                    if local_continuation_eligible
                    else ""
                ),
            }
        )
        return candidate_step, StepDecision(
            accepted=True,
            reward=decision.reward,
            label=branch_label,
            observation_messages=decision.observation_messages,
            info=info,
        )


def _verifiability_reward(decision: VerifiabilityDecision) -> float:
    if decision.label in {
        "valid_replay_exact",
        "valid_external_request_grounded",
        "valid_terminal_stop",
    }:
        return 1.0
    if decision.label == "valid_replay_database":
        # A DB hit expands verified environment coverage but does not advance
        # the required trajectory. Session-level delayed credit is assigned
        # only if the policy subsequently returns to verified progress.
        return 0.0
    if decision.label == "valid_replay_equivalent":
        return 0.5
    if decision.is_invalid:
        return -0.35
    return 0.0


def _deduplicate_candidates(
    episode: TrajectoryEpisode,
    candidates: tuple[TrajectoryEpisode, ...],
) -> tuple[TrajectoryEpisode, ...]:
    by_id = {candidate.id: candidate for candidate in candidates}
    by_id[episode.id] = episode
    return tuple(by_id[key] for key in sorted(by_id))


def _same_recorded_prefix(
    left: TrajectoryEpisode,
    right: TrajectoryEpisode,
    length: int,
) -> bool:
    if left.source != right.source or left.domain != right.domain:
        return False
    left_state = str(left.metadata.get("trajectory_state_id") or "").strip()
    right_state = str(right.metadata.get("trajectory_state_id") or "").strip()
    if not left_state or left_state != right_state:
        return False
    if length > left.step_count or length > right.step_count:
        return False
    return all(
        _step_signature(left.steps[index]) == _step_signature(right.steps[index])
        for index in range(length)
    )


def _step_signature(step: Any) -> str:
    return json.dumps(
        {
            "kind": step.kind,
            "actions": [action.to_dict() for action in step.expected_actions],
            "observations": list(step.observations),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _visible_context_signature(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [canonical_visible_message(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _messages_signature(messages: Any) -> str:
    return json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _recorded_step_state_id(step: Any, field: str) -> str:
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    value = metadata.get(field)
    if value is None:
        value = metadata.get(f"recorded_graph_{field}")
    return str(value or "").strip()


def _typed_feedback_message(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": json_dumps(
            {
                "type": "trajectory2gym_feedback",
                "classification": decision.get("action_class"),
                "label": decision.get("label"),
                "reason": decision.get("reason"),
            }
        ),
    }


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nonprogress_action_key(decision: StepDecision) -> str:
    actions = decision.info.get("actual_actions") or []
    if isinstance(actions, list) and actions:
        return json.dumps(
            actions,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    # Parser failures and missing calls have no canonical action. Repeating
    # the same typed branch at the same cursor still returns no new evidence,
    # so use the verifier label as the finite-cycle key.
    branch = str(decision.info.get("branch_type") or decision.label or "").strip()
    if branch:
        return json.dumps(["no_action_branch", branch], separators=(",", ":"))
    return ""


def _repeated_nonprogress_decision(decision: StepDecision) -> StepDecision:
    """Abstain when the same evidence-backed query cannot change the state."""

    reason = "repeated action returns no new evidence and does not advance the trajectory"
    info = copy.deepcopy(decision.info)
    verifiability = copy.deepcopy(info.get("verifiability") or {})
    verifiability.update(
        {
            "label": "ambiguous_repeated_action",
            "action_class": "ambiguous",
            "accepted": False,
            "observation_messages": [],
            "reason": reason,
        }
    )
    info.update(
        {
            "branch_type": "ambiguous_repeated_action",
            "verifier_status": "ambiguous",
            "verified_support": False,
            "progress_advanced": False,
            "accepted": False,
            "success": False,
            "repeated_nonprogress_action": True,
            "verifiability": verifiability,
        }
    )
    return StepDecision(
        accepted=False,
        reward=0.0,
        label="ambiguous_repeated_action",
        observation_messages=(),
        info=info,
    )
