"""Thin bridge from the official tau-bench Env to the shared RL lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from trajectory2gym.tool_calls import actions_from_assistant_message
from trajectory2gym.training.real_trajectory_sidecar import (
    RealTrajectoryCreditIndex,
    RealTrajectorySession,
)
from trajectory2gym.training.verified_prefix import (
    VERIFIED_PREFIX_CERTIFICATE,
    VerifiedPrefixIndex,
    VerifiedPrefixPlan,
    observation_matches,
)


@dataclass
class _Instance:
    env: Any
    domain: str
    task_id: str
    reward: float = 0.0
    terminated: bool = False
    evaluation: dict[str, Any] | None = None
    trajectory_sidecar: RealTrajectorySession | None = None
    step_infos: list[dict[str, Any]] | None = None


class TauBenchOfficialManager:
    """Transport adapter; tau-bench remains the sole transition oracle."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = dict(config or {})
        tau_root = config.pop("tau_root", None)
        if tau_root:
            root = Path(str(tau_root)).expanduser().resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

        injected_get_env = config.pop("_get_env", None)
        injected_action = config.pop("_action_class", None)
        injected_sidecar = config.pop("_trajectory_sidecar_index", None)
        injected_prefix_index = config.pop("_verified_prefix_index", None)
        self._uses_injected_env = bool(
            callable(injected_get_env) and injected_action is not None
        )
        if self._uses_injected_env:
            self._get_env: Callable[..., Any] = injected_get_env
            self._action_class = injected_action
            self._respond_action_name = str(
                config.pop("_respond_action_name", "respond")
            )
        else:
            from tau_bench.envs import get_env
            from tau_bench.types import Action, RESPOND_ACTION_NAME

            self._get_env = get_env
            self._action_class = Action
            self._respond_action_name = RESPOND_ACTION_NAME

        self.domains = tuple(
            str(value)
            for value in config.pop("domains", ("retail", "airline"))
        )
        self.task_split = str(config.pop("task_split", "train"))
        self.user_strategy = str(config.pop("user_strategy", "llm"))
        self.user_model = str(config.pop("user_model", "gpt-4o"))
        self.user_provider = config.pop("user_provider", None)
        self.user_temperature = float(config.pop("user_temperature", 0.0))
        self.user_enable_thinking = _as_bool(
            config.pop("user_enable_thinking", False)
        )
        self.same_start_initial_user_cache = _as_bool(
            config.pop("same_start_initial_user_cache", False)
        )
        self._initial_user_messages: dict[tuple[str, str], str] = {}
        self._initial_user_locks: dict[tuple[str, str], threading.Lock] = {}
        self._initial_user_lock_guard = threading.Lock()
        self.user_max_retries = int(config.pop("user_max_retries", 60))
        self.user_request_timeout_seconds = float(
            config.pop("user_request_timeout_seconds", 180.0)
        )
        self.user_retry_initial_seconds = float(
            config.pop("user_retry_initial_seconds", 2.0)
        )
        self.user_retry_max_seconds = float(
            config.pop("user_retry_max_seconds", 30.0)
        )
        if self.user_max_retries < 0:
            raise ValueError("user_max_retries must be non-negative")
        if self.user_request_timeout_seconds <= 0:
            raise ValueError("user_request_timeout_seconds must be positive")
        if self.user_retry_initial_seconds < 0:
            raise ValueError("user_retry_initial_seconds must be non-negative")
        if self.user_retry_max_seconds < self.user_retry_initial_seconds:
            raise ValueError(
                "user_retry_max_seconds must be >= user_retry_initial_seconds"
            )
        self.user_api_base = str(
            config.pop("user_api_base", None)
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).strip()
        sidecar_paths = config.pop("trajectory_sidecar_evidence", ())
        if isinstance(sidecar_paths, (str, Path)):
            sidecar_paths = [sidecar_paths]
        self.trajectory_sidecar_index = (
            injected_sidecar
            if injected_sidecar is not None
            else (
                RealTrajectoryCreditIndex.from_paths(sidecar_paths)
                if sidecar_paths
                else None
            )
        )
        prefix_paths = config.pop("verified_prefix_evidence", ())
        if isinstance(prefix_paths, (str, Path)):
            prefix_paths = [prefix_paths]
        self.verified_prefix_index = (
            injected_prefix_index
            if injected_prefix_index is not None
            else (
                VerifiedPrefixIndex.from_paths(prefix_paths)
                if prefix_paths
                else None
            )
        )
        if config:
            raise ValueError(
                "unsupported tau-bench manager config: "
                + ", ".join(sorted(config))
            )
        self.instances: dict[str, _Instance] = {}

    def get_env_profile(
        self,
        *,
        env_type: str = "tau_bench",
        split: str = "train",
        params: dict[str, Any] | None = None,
    ) -> list[str]:
        _require_env_type(env_type)
        # Task IDs are carried by the compiled guidance rows.  Avoid creating
        # an LLM user merely to enumerate official task arrays.
        return []

    def create_instance(
        self,
        *,
        env_type: str,
        task_id: str,
        instance_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _require_env_type(env_type)
        guidance = _real_guidance(params)
        reset_strategy = str(guidance.get("guidance_reset_strategy") or "")
        prefix_plan = self._prefix_plan(guidance)
        domain = str(guidance.get("environment_domain") or "")
        if domain not in self.domains:
            raise ValueError(
                f"tau-bench domain {domain!r} is not enabled; expected one of "
                + ", ".join(self.domains)
            )
        try:
            task_index = int(task_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tau-bench environment_task_id must be the official numeric "
                f"task index, got {task_id!r}"
            ) from exc

        instance_id = instance_id or f"tau1_{uuid.uuid4().hex[:12]}"
        if instance_id in self.instances:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        if not self._uses_injected_env and self.user_strategy == "llm":
            # tau-bench's LLM user constructor calls reset() before the task
            # instruction exists, then Env.reset() calls it a second time.
            # Constructing with its Human user avoids that discarded LLM call;
            # replacing it before the only observable reset preserves the
            # official transition semantics while matching evaluation-time
            # temperature/thinking settings explicitly.
            env = self._get_env(
                domain,
                user_strategy="human",
                user_model=self.user_model,
                user_provider=self.user_provider,
                task_split=self.task_split,
                task_index=task_index,
            )
            live_user = _configured_llm_user(
                model=self.user_model,
                provider=self.user_provider,
                temperature=self.user_temperature,
                enable_thinking=self.user_enable_thinking,
                api_base=self.user_api_base,
                max_retries=self.user_max_retries,
                request_timeout_seconds=self.user_request_timeout_seconds,
                retry_initial_seconds=self.user_retry_initial_seconds,
                retry_max_seconds=self.user_retry_max_seconds,
            )
            if self.same_start_initial_user_cache and prefix_plan is None:
                key = (domain, str(task_id))
                with self._initial_user_lock_guard:
                    task_lock = self._initial_user_locks.setdefault(
                        key, threading.Lock()
                    )
                live_user = _same_start_initial_user(
                    live_user,
                    cache=self._initial_user_messages,
                    key=key,
                    lock=task_lock,
                )
            env.user = (
                _prefix_replay_user(live_user, prefix_plan)
                if prefix_plan is not None
                else live_user
            )
        else:
            env = self._get_env(
                domain,
                user_strategy=self.user_strategy,
                user_model=self.user_model,
                user_provider=self.user_provider,
                task_split=self.task_split,
                task_index=task_index,
            )
        reset = env.reset(task_index=task_index)
        base_initial_state = _initial_state(env, str(reset.observation))
        # The trajectory sidecar is a stateful verifier.  It must observe the
        # same real transitions used to restore a verified prefix; creating it
        # only after replay leaves it incorrectly anchored at the task's
        # initial state and silently neutralizes credit at the frontier.
        trajectory_sidecar = (
            self.trajectory_sidecar_index.start(
                domain=domain,
                reset_id=str(task_id),
                initial_messages=(
                    [
                        copy.deepcopy(prefix_plan.system_message),
                        copy.deepcopy(prefix_plan.initial_user_message),
                    ]
                    if prefix_plan is not None
                    else base_initial_state
                ),
            )
            if self.trajectory_sidecar_index is not None
            else None
        )
        if prefix_plan is not None:
            _validate_prefix_identity(
                prefix_plan,
                domain=domain,
                task_id=str(task_index),
                wiki=str(getattr(env, "wiki", "") or ""),
                initial_observation=str(reset.observation),
            )
            initial_state = self._replay_verified_prefix(
                env=env,
                plan=prefix_plan,
                instance_id=instance_id,
                trajectory_sidecar=trajectory_sidecar,
            )
        else:
            initial_state = base_initial_state
        get_data_hash = getattr(env, "get_data_hash", None)
        if callable(get_data_hash):
            backend_state = str(get_data_hash())
            state_hash_version = "tau_bench_data_hash.v1"
        else:
            backend_state = _canonical_digest(
                {
                    "domain": domain,
                    "task_id": str(task_id),
                    "initial_state": initial_state,
                }
            )
            state_hash_version = "tau_bench_reset_recipe_fallback.v1"
        initial_context_sha256 = _canonical_digest(initial_state)
        state_fingerprint = _canonical_digest(
            {
                "backend_state": backend_state,
                "domain": domain,
                "reset_strategy": reset_strategy or "initial",
                "task_id": str(task_id),
            }
        )
        reset_identity_sha256 = _canonical_digest(
            {
                "initial_context_sha256": initial_context_sha256,
                "state_fingerprint": state_fingerprint,
                "state_hash_version": state_hash_version,
            }
        )
        self.instances[instance_id] = _Instance(
            env=env,
            domain=domain,
            task_id=str(task_id),
            trajectory_sidecar=trajectory_sidecar,
            step_infos=[],
        )
        # tau-bench's official ToolCallingAgent receives ``env.wiki`` as the
        # system policy before the simulated user's first message.  Omitting
        # it here makes RL optimize under a different context than official
        # evaluation (and removes consequential-action/confirmation rules).
        # The adapter must expose the oracle's policy verbatim; it does not
        # synthesize benchmark feedback or alter transitions/reward.
        return {
            "state": initial_state,
            "info": {
                "tools": list(env.tools_info),
                "domain": domain,
                "environment_task_id": str(task_id),
                "state_fingerprint": state_fingerprint,
                "state_hash_version": state_hash_version,
                "initial_visible_context_sha256": initial_context_sha256,
                "reset_identity_sha256": reset_identity_sha256,
                "same_start_initial_user_cache": (
                    self.same_start_initial_user_cache
                ),
                "oracle_source": "real_environment",
                "trajectory_sidecar_enabled": (
                    self.trajectory_sidecar_index is not None
                ),
                "trajectory_action_proposals": _trajectory_action_proposals(
                    trajectory_sidecar
                ),
                "reset_strategy": reset_strategy,
                "verified_prefix_certificate": (
                    VERIFIED_PREFIX_CERTIFICATE
                    if prefix_plan is not None
                    else None
                ),
                "verified_prefix_turns": (
                    len(prefix_plan.turns) if prefix_plan is not None else 0
                ),
            },
        }

    def step(
        self,
        *,
        instance_id: str,
        action: dict[str, Any] | list[dict[str, Any]] | str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        message = _assistant_message(action)
        parsed = actions_from_assistant_message(message)
        # Match tau-bench's official ToolCallingAgent exactly: one assistant
        # turn executes at most the first decoded tool call.  Executing every
        # call from a malformed multi-call response gives the training policy
        # transitions that the official evaluator never exposes.
        parsed_action = parsed[0] if parsed else None
        official_action = _official_action(
            message,
            parsed_action=parsed_action,
            action_class=self._action_class,
            respond_action_name=self._respond_action_name,
        )
        response = instance.env.step(official_action)
        emitted = [
            _emitted_message(
                message,
                parsed_action=parsed_action,
                observation=str(response.observation),
                instance_id=instance_id,
                turn_index=len(instance.step_infos or ()),
            )
        ]
        instance.reward = float(response.reward)
        instance.terminated = bool(response.done)
        if instance.terminated:
            instance.evaluation = _reward_payload(
                getattr(response.info, "reward_info", None),
                fallback_reward=instance.reward,
            )
        if instance.trajectory_sidecar is not None:
            certificate = instance.trajectory_sidecar.observe(
                assistant_message=message,
                observations=emitted,
            )
            assert instance.step_infos is not None
            instance.step_infos.append(certificate.to_turn_info())
        return {
            "state": emitted,
            "emitted_state": emitted,
            "reward": instance.reward,
            "is_terminated": instance.terminated,
            "info": {
                "tools": list(instance.env.tools_info),
                "oracle_source": "real_environment",
                "trajectory_sidecar_enabled": (
                    instance.trajectory_sidecar is not None
                ),
                "trajectory_action_proposals": _trajectory_action_proposals(
                    instance.trajectory_sidecar
                ),
            },
        }

    def evaluate(
        self,
        *,
        instance_id: str,
        messages: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        if instance.evaluation is None:
            # Match tau-bench's official ``ToolCallingAgent.solve`` contract:
            # reward is produced by the last executed ``env.step`` and an
            # unfinished episode stays at zero when the interaction budget is
            # exhausted.  Calling ``calculate_reward`` here used to turn a
            # state-complete but non-terminated rollout into a positive
            # session, even though official evaluation would return the last
            # step's zero reward.
            instance.evaluation = _reward_payload(
                None, fallback_reward=instance.reward
            )
        return {
            "score": instance.reward,
            "accuracy": instance.reward,
            "reward": instance.reward,
            "completed": instance.terminated,
            "session_completed": instance.terminated,
            "outcome": (
                "success"
                if instance.reward >= 1.0
                else "failure" if instance.terminated else "incomplete"
            ),
            "official_reward_info": instance.evaluation,
            "step_infos": list(instance.step_infos or ()),
            "failure_infos": [],
        }

    def get_info(
        self,
        *,
        instance_id: str,
        messages: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        return {
            "tools": list(instance.env.tools_info),
            "domain": instance.domain,
            "environment_task_id": instance.task_id,
            "terminated": instance.terminated,
        }

    def release_instance(self, instance_id: str) -> bool:
        return self.instances.pop(instance_id, None) is not None

    def close_all(self) -> None:
        self.instances.clear()

    def _instance(self, instance_id: str) -> _Instance:
        try:
            return self.instances[instance_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown tau-bench instance_id: {instance_id}"
            ) from exc

    def _prefix_plan(
        self, guidance: Mapping[str, Any]
    ) -> VerifiedPrefixPlan | None:
        strategy = str(guidance.get("guidance_reset_strategy") or "initial")
        if strategy == "initial":
            return None
        if strategy != "verified_prefix":
            raise ValueError(f"unsupported tau-bench reset strategy: {strategy}")
        if self.verified_prefix_index is None:
            raise ValueError(
                "verified_prefix reset requires configured trajectory evidence"
            )
        reference = str(guidance.get("guidance_reset_reference") or "")
        return self.verified_prefix_index.get(reference)

    def _replay_verified_prefix(
        self,
        *,
        env: Any,
        plan: VerifiedPrefixPlan,
        instance_id: str,
        trajectory_sidecar: RealTrajectorySession | None = None,
    ) -> list[dict[str, Any]]:
        history = _initial_state(env, plan.initial_user_message["content"])
        for turn_index, turn in enumerate(plan.turns):
            message = dict(turn.assistant)
            parsed = actions_from_assistant_message(message)
            parsed_action = parsed[0] if parsed else None
            official_action = _official_action(
                message,
                parsed_action=parsed_action,
                action_class=self._action_class,
                respond_action_name=self._respond_action_name,
            )
            response = env.step(official_action)
            actual = _emitted_message(
                message,
                parsed_action=parsed_action,
                observation=str(response.observation),
                instance_id=instance_id,
                turn_index=turn_index,
            )
            if not observation_matches(actual, turn.expected):
                raise ValueError(
                    "verified-prefix observation mismatch at turn "
                    f"{turn_index}: expected={turn.expected!r} actual={actual!r}"
                )
            if bool(response.done):
                raise ValueError(
                    "verified prefix terminated before its declared frontier"
                )
            if trajectory_sidecar is not None:
                # This certificate is used only to advance the verifier's
                # evidence-bounded state.  Prefix actions are fixed reset
                # history, so their labels must never be exposed as actor
                # credit in ``step_infos``.
                certificate = trajectory_sidecar.observe(
                    assistant_message=message,
                    observations=[actual],
                )
                if certificate.candidate_state_count <= 0:
                    raise ValueError(
                        "verified prefix is absent from trajectory-sidecar "
                        f"state evidence at turn {turn_index}"
                    )
            history.append(message)
            history.append(actual)
        return history


def build_manager(
    config: Mapping[str, Any] | None = None,
) -> TauBenchOfficialManager:
    return TauBenchOfficialManager(config)


def _configured_llm_user(
    *,
    model: str,
    provider: str | None,
    temperature: float,
    enable_thinking: bool,
    api_base: str,
    max_retries: int = 60,
    request_timeout_seconds: float = 180.0,
    retry_initial_seconds: float = 2.0,
    retry_max_seconds: float = 30.0,
) -> Any:
    """Build tau-bench's native LLM user without its discarded first reset."""

    from litellm import completion
    from tau_bench.envs.user import LLMUserSimulationEnv

    class _ConfiguredUser(LLMUserSimulationEnv):
        def __init__(self) -> None:
            # Deliberately do not call LLMUserSimulationEnv.__init__: that
            # constructor performs an unobservable generation with no task.
            self.messages: list[dict[str, Any]] = []
            self.model = model
            self.provider = provider
            self.total_cost = 0.0

        def generate_next_message(
            self, messages: list[dict[str, Any]]
        ) -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "custom_llm_provider": self.provider,
                "messages": messages,
                "temperature": temperature,
                "timeout": request_timeout_seconds,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    }
                },
            }
            if api_base:
                kwargs["api_base"] = api_base
            for attempt in range(max_retries + 1):
                try:
                    response = completion(**kwargs)
                    break
                except Exception as exc:
                    if attempt >= max_retries or not _retryable_user_error(exc):
                        raise
                    delay = min(
                        retry_max_seconds,
                        retry_initial_seconds * (2 ** min(attempt, 8)),
                    )
                    # A small jitter prevents all environment workers from
                    # retrying a recovered endpoint in the same instant.
                    time.sleep(delay * random.uniform(0.8, 1.2))
            message = response.choices[0].message
            self.messages.append(message.model_dump())
            hidden = getattr(response, "_hidden_params", {}) or {}
            self.total_cost = float(hidden.get("response_cost") or 0.0)
            return str(message.content or "")

    return _ConfiguredUser()


def _retryable_user_error(error: Exception) -> bool:
    """Return whether an external user-model failure is transient.

    This is deliberately transport-only: retries do not synthesize a user
    message or alter the official environment transition/reward contract.
    """

    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code is not None:
        return status_code in {408, 409, 425, 429} or status_code >= 500

    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
            "service unavailable",
            "rate limit",
            "status code: 429",
            "status code: 500",
            "status code: 502",
            "status code: 503",
            "status code: 504",
        )
    )


def _prefix_replay_user(live_user: Any, plan: VerifiedPrefixPlan) -> Any:
    """Replay recorded user turns, then continue with the configured LLM user."""

    scripted = [
        (
            str(turn.assistant.get("content") or ""),
            str(turn.expected.get("content") or ""),
        )
        for turn in plan.turns
        if not (turn.assistant.get("tool_calls") or [])
    ]

    class _PrefixReplayUser:
        def __init__(self) -> None:
            self.cursor = 0

        def reset(self, instruction: str | None = None) -> str:
            self.cursor = 0
            live_user.messages = [
                {
                    "role": "system",
                    "content": live_user.build_system_prompt(instruction),
                },
                {"role": "user", "content": "Hi! How can I help you today?"},
                {
                    "role": "assistant",
                    "content": str(plan.initial_user_message.get("content") or ""),
                },
            ]
            return str(plan.initial_user_message.get("content") or "")

        def step(self, content: str) -> str:
            live_user.messages.append({"role": "user", "content": content})
            if self.cursor < len(scripted):
                expected_agent, recorded_user = scripted[self.cursor]
                if content != expected_agent:
                    raise ValueError(
                        "verified-prefix assistant text mismatch: "
                        f"expected={expected_agent!r} actual={content!r}"
                    )
                self.cursor += 1
                live_user.messages.append(
                    {"role": "assistant", "content": recorded_user}
                )
                return recorded_user
            return live_user.generate_next_message(live_user.messages)

        def get_total_cost(self) -> float:
            return float(live_user.get_total_cost())

    return _PrefixReplayUser()


def _same_start_initial_user(
    live_user: Any,
    *,
    cache: dict[tuple[str, str], str],
    key: tuple[str, str],
    lock: threading.Lock,
) -> Any:
    """Freeze one official LLM-user reset message per training task."""

    class _SameStartInitialUser:
        def reset(self, instruction: str | None = None) -> str:
            with lock:
                if key not in cache:
                    cache[key] = str(live_user.reset(instruction) or "")
                message = cache[key]
            # Every environment instance must continue from the exact cached
            # visible reset, including the live user's private chat history.
            live_user.messages = [
                {
                    "role": "system",
                    "content": live_user.build_system_prompt(instruction),
                },
                {"role": "user", "content": "Hi! How can I help you today?"},
                {"role": "assistant", "content": message},
            ]
            return message

        def step(self, content: str) -> str:
            live_user.messages.append({"role": "user", "content": content})
            return live_user.generate_next_message(live_user.messages)

        def get_total_cost(self) -> float:
            return float(live_user.get_total_cost())

    return _SameStartInitialUser()


def _validate_prefix_identity(
    plan: VerifiedPrefixPlan,
    *,
    domain: str,
    task_id: str,
    wiki: str,
    initial_observation: str,
) -> None:
    if plan.domain != domain or plan.task_id != task_id:
        raise ValueError(
            "verified-prefix reset identity mismatch: "
            f"plan={plan.domain}:{plan.task_id} runtime={domain}:{task_id}"
        )
    if str(plan.system_message.get("content") or "") != wiki:
        raise ValueError("verified-prefix system policy does not match backend")
    if str(plan.initial_user_message.get("content") or "") != initial_observation:
        raise ValueError("verified-prefix initial user observation mismatch")


def _initial_state(env: Any, observation: str) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    wiki = str(getattr(env, "wiki", "") or "")
    if wiki.strip():
        state.append({"role": "system", "content": wiki})
    state.append({"role": "user", "content": str(observation)})
    return state


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _official_action(
    message: Mapping[str, Any],
    *,
    parsed_action: Mapping[str, Any] | None,
    action_class: Any,
    respond_action_name: str,
) -> Any:
    if parsed_action is not None:
        return action_class(
            name=str(parsed_action["name"]),
            kwargs=dict(parsed_action.get("arguments") or {}),
        )
    return action_class(
        name=respond_action_name,
        kwargs={"content": str(message.get("content") or "")},
    )


def _emitted_message(
    message: Mapping[str, Any],
    *,
    parsed_action: Mapping[str, Any] | None,
    observation: str,
    instance_id: str,
    turn_index: int,
) -> dict[str, Any]:
    if parsed_action is None:
        return {"role": "user", "content": str(observation)}
    return {
        "role": "tool",
        "tool_call_id": _tool_call_id(
            message,
            instance_id=instance_id,
            turn_index=turn_index,
        ),
        "name": str(parsed_action["name"]),
        "content": str(observation),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _require_env_type(env_type: str) -> None:
    if env_type != "tau_bench":
        raise ValueError(
            f"tau-bench manager cannot serve env_type={env_type!r}"
        )


def _real_guidance(
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = params if isinstance(params, Mapping) else {}
    raw = params.get("trajectory_guidance")
    guidance = dict(raw) if isinstance(raw, Mapping) else {}
    if guidance.get("execution_mode") != "real":
        raise ValueError("tau-bench official manager requires real guidance")
    reset_strategy = str(guidance.get("guidance_reset_strategy") or "")
    if reset_strategy not in {"initial", "verified_prefix"}:
        raise ValueError(
            "tau-bench adapter supports initial and verified_prefix resets"
        )
    if reset_strategy == "verified_prefix" and not str(
        guidance.get("guidance_reset_reference") or ""
    ).strip():
        raise ValueError("verified_prefix requires guidance_reset_reference")
    if guidance.get("episode_execution_boundary") is not True:
        raise ValueError("real episode boundary is not declared")
    if guidance.get("per_action_fallback_allowed") is not False:
        raise ValueError("per-action oracle fallback is forbidden")
    return guidance


def _assistant_message(
    action: dict[str, Any] | list[dict[str, Any]] | str,
) -> dict[str, Any]:
    if isinstance(action, str):
        return {"role": "assistant", "content": action}
    if isinstance(action, list):
        if not action:
            raise ValueError("empty action list")
        action = action[-1]
    if not isinstance(action, Mapping):
        raise TypeError(f"unsupported action type: {type(action).__name__}")
    return dict(action)


def _trajectory_action_proposals(
    session: RealTrajectorySession | None,
) -> list[dict[str, Any]]:
    if session is None:
        return []
    return [proposal.to_dict() for proposal in session.action_proposals()]


def _tool_call_id(
    message: Mapping[str, Any],
    *,
    instance_id: str,
    turn_index: int,
) -> str:
    """Return the first provider ID or a stable transport-local fallback.

    Qwen's chat template uses the observation role to render a tool response;
    the identifier is transport metadata.  Still, preserving a provider ID
    when present keeps the live training history structurally identical to
    tau-bench's official agent history.
    """

    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], Mapping):
        value = str(calls[0].get("id") or "").strip()
        if value:
            return value
    return f"call_{instance_id}_{turn_index}_0"


def _reward_payload(value: Any, *, fallback_reward: float) -> dict[str, Any]:
    if value is None:
        return {"reward": fallback_reward}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return {"reward": fallback_reward, "detail": str(value)}


__all__ = ["TauBenchOfficialManager", "build_manager"]
