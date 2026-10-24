"""Thin bridge from the official Tau2 Gym API to the shared RL lifecycle.

All task state, user simulation, tool execution, and evaluation stay inside
``tau2.gym.gym_agent.AgentGymEnv``.  This module only translates transport
shapes; it never reads trajectory observations while an episode is running.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import threading
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from trajectory2gym.openai_response import extract_tool_calls_from_content
from trajectory2gym.tau2_projection import tau2_agent_system_prompt
from trajectory2gym.text import strip_visible_thinking
from trajectory2gym.training.real_trajectory_sidecar import (
    RealTrajectoryCreditIndex,
    RealTrajectorySession,
)


@dataclass
class _Instance:
    env: Any
    domain: str
    task_id: str
    tools: list[dict[str, Any]]
    reward: float = 0.0
    terminated: bool = False
    info: dict[str, Any] | None = None
    trajectory_sidecar: RealTrajectorySession | None = None
    step_infos: list[dict[str, Any]] | None = None
    native_history_length: int = -1


class Tau2OfficialManager:
    """Lifecycle adapter around Tau2's own ``AgentGymEnv``."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = dict(config or {})
        vendor_root = config.pop("vendor_root", None)
        if vendor_root:
            source_root = Path(str(vendor_root)).expanduser().resolve()
            if (source_root / "src").is_dir():
                source_root = source_root / "src"
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))

        injected = config.pop("_gym_factory", None)
        self._gym_factory: Callable[..., Any] = (
            injected
            if callable(injected)
            else _load_symbol(
                str(
                    config.pop(
                        "gym_factory",
                        "tau2.gym.gym_agent:AgentGymEnv",
                    )
                )
            )
        )
        self.domains = tuple(
            str(value)
            for value in config.pop(
                "domains", ("retail", "airline", "telecom")
            )
        )
        self.task_split = str(config.pop("task_split", "train"))
        self.seed = int(config.pop("seed", 0))
        self.gym_kwargs = dict(config.pop("gym_kwargs", {}))
        self.same_start_initial_user_cache = bool(
            config.pop("same_start_initial_user_cache", False)
        )
        self._initial_user_turns: dict[tuple[str, str], tuple[Any, list[Any]]] = {}
        self._initial_user_locks: dict[tuple[str, str], threading.Lock] = {}
        self._initial_user_lock_guard = threading.Lock()
        injected_sidecar = config.pop("_trajectory_sidecar_index", None)
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
        _configure_vendor_llm_defaults(
            nl_assertion_llm=config.pop("nl_assertion_llm", None),
            nl_assertion_llm_args=config.pop("nl_assertion_llm_args", None),
            environment_interface_llm=config.pop(
                "environment_interface_llm", None
            ),
            environment_interface_llm_args=config.pop(
                "environment_interface_llm_args", None
            ),
        )
        if config:
            raise ValueError(
                "unsupported Tau2 manager config: "
                + ", ".join(sorted(config))
            )
        self.instances: dict[str, _Instance] = {}

    def get_env_profile(
        self,
        *,
        env_type: str = "tau2_bench",
        split: str = "train",
        params: dict[str, Any] | None = None,
    ) -> list[str]:
        _require_env_type(env_type)
        try:
            load_tasks = _load_symbol("tau2.runner.helpers:load_tasks")
        except (ImportError, AttributeError):
            return []
        task_ids: list[str] = []
        for domain in self.domains:
            for task in load_tasks(domain, split or self.task_split):
                task_ids.append(f"{domain}::{task.id}")
        return task_ids

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
        domain = str(guidance.get("environment_domain") or "")
        if domain not in self.domains:
            raise ValueError(
                f"Tau2 domain {domain!r} is not enabled; expected one of "
                + ", ".join(self.domains)
            )
        instance_id = instance_id or f"tau2_{uuid.uuid4().hex[:12]}"
        if instance_id in self.instances:
            raise ValueError(f"duplicate instance_id: {instance_id}")

        env = self._gym_factory(
            domain=domain,
            task_id=str(task_id),
            **self.gym_kwargs,
        )
        if self.same_start_initial_user_cache:
            key = (domain, str(task_id))
            with self._initial_user_lock_guard:
                task_lock = self._initial_user_locks.setdefault(
                    key, threading.Lock()
                )
            _install_same_start_initial_user(
                env,
                cache=self._initial_user_turns,
                key=key,
                lock=task_lock,
            )
        observation, raw_info = env.reset(seed=self.seed)
        observation = _synchronize_initial_observation(env, observation)
        native_initial = _native_agent_observation(env)
        tools = _tool_schemas(raw_info)
        public_info = _public_info(raw_info)
        native_state = (
            native_initial[0]
            if native_initial is not None
            else _state(observation, done=False)
        )
        agent_initiated = bool(native_initial is not None and not native_state)
        raw_task = raw_info.get("task") if isinstance(raw_info, Mapping) else None
        if callable(getattr(raw_task, "model_dump", None)):
            task_reset_recipe = raw_task.model_dump(mode="json")
        elif isinstance(raw_task, Mapping):
            task_reset_recipe = dict(raw_task)
        else:
            task_reset_recipe = str(raw_task or task_id)
        initial_state: list[dict[str, Any]] = []
        domain_policy = str(public_info.get("policy") or "")
        if domain_policy:
            initial_state.append(
                {
                    "role": "system",
                    "content": tau2_agent_system_prompt(domain_policy),
                }
            )
        initial_state.extend(copy.deepcopy(native_state))
        initial_context_sha256 = _canonical_digest(initial_state)
        state_hash_version = "tau2_task_reset_recipe.v1"
        state_fingerprint = _canonical_digest(
            {
                "domain": domain,
                "policy": str(public_info.get("policy") or ""),
                "seed": self.seed,
                "task_id": str(task_id),
                "task_reset_recipe": task_reset_recipe,
                "tools": tools,
            }
        )
        reset_identity_sha256 = _canonical_digest(
            {
                "initial_context_sha256": initial_context_sha256,
                "state_fingerprint": state_fingerprint,
                "state_hash_version": state_hash_version,
            }
        )
        trajectory_sidecar = (
            self.trajectory_sidecar_index.start(
                domain=domain,
                reset_id=str(task_id),
                initial_messages=initial_state,
            )
            if self.trajectory_sidecar_index is not None
            else None
        )
        sidecar_reset_prefix_turns = 0
        self.instances[instance_id] = _Instance(
            env=env,
            domain=domain,
            task_id=str(task_id),
            tools=tools,
            info=public_info,
            trajectory_sidecar=trajectory_sidecar,
            step_infos=[],
            native_history_length=(
                native_initial[1] if native_initial is not None else -1
            ),
        )
        return {
            "state": initial_state,
            "info": {
                "tools": tools,
                "domain": domain,
                "environment_task_id": str(task_id),
                "state_fingerprint": state_fingerprint,
                "state_hash_version": state_hash_version,
                "initial_visible_context_sha256": initial_context_sha256,
                "reset_identity_sha256": reset_identity_sha256,
                "same_start_initial_user_cache": (
                    self.same_start_initial_user_cache
                ),
                # The official Gym supplies this as the agent's system policy.
                # Expose it at reset so external policies receive the same
                # agent-visible contract as the benchmark's native runner.
                "policy": str(public_info.get("policy") or ""),
                "oracle_source": "real_environment",
                "agent_initiated": agent_initiated,
                "policy_visible_transport": (
                    "official_system_plus_native_api_messages.v2"
                    if native_initial is not None
                    else "official_system_plus_lossy_gym_string_fallback.v2"
                ),
                "trajectory_sidecar_enabled": (
                    self.trajectory_sidecar_index is not None
                ),
                "trajectory_sidecar_reset_prefix_turns": (
                    sidecar_reset_prefix_turns
                ),
                "trajectory_action_proposals": _trajectory_action_proposals(
                    self.instances[instance_id].trajectory_sidecar
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
        try:
            assistant_message = _assistant_message(action)
            action_text = _action_text(assistant_message)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            # An empty/malformed model response is a policy failure, not an
            # environment infrastructure failure.  Terminate with zero
            # official reward and a typed observation so a single bad sample
            # cannot turn the whole GRPO group into an HTTP-500 failure.
            instance.reward = 0.0
            instance.terminated = True
            instance.info = {
                "failure_type": "invalid_assistant_action",
                "failure_detail": str(error),
            }
            return {
                "state": [
                    {
                        "role": "user",
                        "content": "[INVALID_ASSISTANT_ACTION]",
                    }
                ],
                "emitted_state": [
                    {
                        "role": "user",
                        "content": "[INVALID_ASSISTANT_ACTION]",
                    }
                ],
                "reward": 0.0,
                "is_terminated": True,
                "info": {
                    "tools": instance.tools,
                    "oracle_source": "real_environment",
                    "failure_type": "invalid_assistant_action",
                    "trajectory_sidecar_enabled": (
                        instance.trajectory_sidecar is not None
                    ),
                },
            }
        observation, reward, terminated, truncated, raw_info = (
            instance.env.step(action_text)
        )
        instance.reward = float(reward)
        instance.terminated = bool(terminated or truncated)
        instance.info = _public_info(raw_info)
        native_emitted = (
            _native_agent_observation(
                instance.env,
                previous_length=instance.native_history_length,
                done=instance.terminated,
            )
            if instance.native_history_length >= 0
            else None
        )
        if native_emitted is not None:
            emitted, instance.native_history_length = native_emitted
        else:
            emitted = _state(observation, done=instance.terminated)
        if instance.trajectory_sidecar is not None:
            certificate = instance.trajectory_sidecar.observe(
                assistant_message=assistant_message,
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
                "tools": instance.tools,
                "oracle_source": "real_environment",
                "policy_visible_transport": (
                    "official_system_plus_native_api_messages.v2"
                    if native_emitted is not None
                    else "official_system_plus_lossy_gym_string_fallback.v2"
                ),
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
            "step_infos": list(instance.step_infos or []),
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
            "tools": instance.tools,
            "domain": instance.domain,
            "environment_task_id": instance.task_id,
            "terminated": instance.terminated,
            "official_info": instance.info or {},
            "trajectory_sidecar_enabled": (
                instance.trajectory_sidecar is not None
            ),
        }

    def release_instance(self, instance_id: str) -> bool:
        instance = self.instances.pop(instance_id, None)
        if instance is None:
            return False
        # Tau2's GymAgent waits for an external action in a daemon thread.
        # End an unfinished episode through the official stop action so the
        # orchestrator can clean up its own resources.
        if not instance.terminated:
            try:
                instance.env.step("done()")
            except Exception:
                pass
        close = getattr(instance.env, "close", None)
        if callable(close):
            close()
        return True

    def close_all(self) -> None:
        for instance_id in list(self.instances):
            self.release_instance(instance_id)

    def _instance(self, instance_id: str) -> _Instance:
        try:
            return self.instances[instance_id]
        except KeyError as exc:
            raise KeyError(f"unknown Tau2 instance_id: {instance_id}") from exc


def build_manager(config: Mapping[str, Any] | None = None) -> Tau2OfficialManager:
    return Tau2OfficialManager(config)


def _install_same_start_initial_user(
    env: Any,
    *,
    cache: dict[tuple[str, str], tuple[Any, list[Any]]],
    key: tuple[str, str],
    lock: threading.Lock,
) -> None:
    """Freeze Tau2's first user turn and synchronize its hidden history."""

    original_get_user = getattr(env, "_get_user", None)
    if not callable(original_get_user):
        raise TypeError(
            "same_start_initial_user_cache requires AgentGymEnv._get_user"
        )

    def cached_get_user(_env: Any) -> Any:
        user = original_get_user()
        original_generate = user.generate_next_message
        first_call = True

        def generate_next_message(
            _user: Any, message: Any, state: Any
        ) -> tuple[Any, Any]:
            nonlocal first_call
            if not first_call:
                return original_generate(message, state)
            first_call = False
            with lock:
                cached = cache.get(key)
                if cached is None:
                    response, updated = original_generate(message, state)
                    cache[key] = (
                        copy.deepcopy(response),
                        copy.deepcopy(updated.messages),
                    )
                    return response, updated
                response, messages = cached
                state.messages = copy.deepcopy(messages)
                return copy.deepcopy(response), state

        user.generate_next_message = types.MethodType(
            generate_next_message, user
        )
        return user

    env._get_user = types.MethodType(cached_get_user, env)


def _trajectory_action_proposals(
    session: RealTrajectorySession | None,
) -> list[dict[str, Any]]:
    if session is None:
        return []
    return [proposal.to_dict() for proposal in session.action_proposals()]


def _load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("symbol must use module.path:attribute")
    module_name, name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def _configure_vendor_llm_defaults(
    *,
    nl_assertion_llm: Any,
    nl_assertion_llm_args: Any,
    environment_interface_llm: Any,
    environment_interface_llm_args: Any,
) -> None:
    """Configure Tau2 LLM-backed official components without editing vendor code.

    ``AgentGymEnv`` exposes the user simulator in its constructor, but Tau2's
    NL-assertion evaluator and environment interface import their defaults at
    module import time.  Merely setting ``OPENAI_API_BASE`` therefore still
    sends those calls to the default GPT model.  Patch only the public config
    constants consumed by the official components so Real RL and the official
    evaluator use the same explicitly declared model route.
    """

    if nl_assertion_llm is not None:
        model = str(nl_assertion_llm)
        args = dict(nl_assertion_llm_args or {})
        config = importlib.import_module("tau2.config")
        evaluator = importlib.import_module(
            "tau2.evaluator.evaluator_nl_assertions"
        )
        config.DEFAULT_LLM_NL_ASSERTIONS = model
        config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
        evaluator.DEFAULT_LLM_NL_ASSERTIONS = model
        evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    elif nl_assertion_llm_args is not None:
        raise ValueError("nl_assertion_llm_args requires nl_assertion_llm")

    if environment_interface_llm is not None:
        model = str(environment_interface_llm)
        args = dict(environment_interface_llm_args or {})
        config = importlib.import_module("tau2.config")
        interface = importlib.import_module(
            "tau2.environment.utils.interface_agent"
        )
        config.DEFAULT_LLM_ENV_INTERFACE = model
        config.DEFAULT_LLM_ENV_INTERFACE_ARGS = args
        interface.DEFAULT_LLM_ENV_INTERFACE = model
        interface.DEFAULT_LLM_ENV_INTERFACE_ARGS = args
    elif environment_interface_llm_args is not None:
        raise ValueError(
            "environment_interface_llm_args requires environment_interface_llm"
        )


def _require_env_type(env_type: str) -> None:
    if env_type != "tau2_bench":
        raise ValueError(f"Tau2 manager cannot serve env_type={env_type!r}")


def _real_guidance(
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = params if isinstance(params, Mapping) else {}
    raw = params.get("trajectory_guidance")
    guidance = dict(raw) if isinstance(raw, Mapping) else {}
    if guidance.get("execution_mode") != "real":
        raise ValueError("Tau2 official manager requires real guidance")
    if guidance.get("guidance_reset_strategy") != "initial":
        raise ValueError(
            "Tau2 AgentGymEnv currently exposes task-level initial reset only"
        )
    if guidance.get("episode_execution_boundary") is not True:
        raise ValueError("real episode boundary is not declared")
    if guidance.get("per_action_fallback_allowed") is not False:
        raise ValueError("per-action oracle fallback is forbidden")
    return guidance


def _action_text(
    action: dict[str, Any] | list[dict[str, Any]] | str,
) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, list):
        if not action:
            raise ValueError("empty action list")
        action = action[-1]
    if not isinstance(action, Mapping):
        raise TypeError(f"unsupported action type: {type(action).__name__}")
    content = action.get("content")
    if isinstance(content, str) and content.strip():
        return content
    tool_calls = action.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) == 1:
        call = tool_calls[0]
        function = call.get("function", {}) if isinstance(call, Mapping) else {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return json.dumps(
            {"name": function.get("name"), "arguments": arguments},
            ensure_ascii=False,
        )
    raise ValueError("assistant action has neither content nor one tool call")


def _assistant_message(
    action: dict[str, Any] | list[dict[str, Any]] | str,
) -> dict[str, Any]:
    if isinstance(action, str):
        action = {"role": "assistant", "content": action}
    if isinstance(action, list):
        if not action:
            raise ValueError("empty action list")
        action = action[-1]
    if not isinstance(action, Mapping):
        raise TypeError(f"unsupported action type: {type(action).__name__}")
    message = dict(action)
    content = strip_visible_thinking(str(message.get("content") or ""))
    if not message.get("tool_calls") and content:
        parsed = extract_tool_calls_from_content(content)
        if parsed:
            # The in-process RL sampler returns serialized Hermes tool tags,
            # whereas the OpenAI collection/evaluation path returns native
            # ``tool_calls``.  Normalize both representations before handing
            # the action to the official Tau2 gym so they execute identically.
            message["tool_calls"] = parsed
            content = ""
    message["content"] = content
    return message


def _tool_schemas(info: Any) -> list[dict[str, Any]]:
    if not isinstance(info, Mapping):
        return []
    schemas: list[dict[str, Any]] = []
    for tool in info.get("tools") or []:
        schema = getattr(tool, "openai_schema", None)
        if isinstance(schema, Mapping):
            schemas.append(dict(schema))
        elif isinstance(tool, Mapping):
            schemas.append(dict(tool))
    return schemas


def _public_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        return {}
    return {
        key: value
        for key, value in info.items()
        if key not in {"task", "simulation_run", "tools"}
        and isinstance(value, (str, int, float, bool, type(None), list, dict))
    }


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _state(observation: Any, *, done: bool) -> list[dict[str, str]]:
    text = str(observation or "")
    if text:
        return [{"role": "user", "content": text}]
    if done:
        return [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]
    return [{"role": "user", "content": ""}]


def _native_agent_observation(
    env: Any, *, previous_length: int | None = None, done: bool = False
) -> tuple[list[dict[str, Any]], int] | None:
    """Read the official GymAgent's API-native visible messages.

    ``AgentGymEnv.step`` renders native ToolMessage/UserMessage objects into a
    lossy string such as ``"tool: {...}"``. Wrapping that string as a user
    message changes the policy input relative to the official evaluator. The
    in-process adapter can read the same public message objects retained by
    GymAgent. It emits only messages added after the externally supplied
    assistant action and never reads task labels or evaluator state.
    """

    agent = getattr(env, "_agent", None)
    lock = getattr(agent, "_lock", None)
    if agent is None or lock is None:
        return None
    with lock:
        raw_messages = list(getattr(agent, "observation", ()) or ())
    if any(not callable(getattr(message, "model_dump", None)) for message in raw_messages):
        return None
    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        raw = message.model_dump(mode="json", exclude_none=True)
        role = str(raw.get("role") or "")
        if role not in {"user", "tool"}:
            # The full native history contains prior assistant actions. They
            # remain in the model's transcript already and are used only to
            # locate the newly emitted tail below.
            messages.append({"role": role, "content": raw.get("content")})
            continue
        item: dict[str, Any] = {
            "role": role,
            "content": raw.get("content"),
        }
        for key in ("name", "tool_call_id", "tool_calls"):
            if key in raw:
                item[key] = raw[key]
        messages.append(item)
    total = len(messages)
    if previous_length is None:
        return messages, total
    # AgentGym may expose either cumulative agent state or only the current
    # observation tail on terminal/multi-tool paths. In both cases the native
    # messages visible after the most recent assistant action are exactly the
    # newly emitted environment messages. Do not infer this boundary from a
    # monotonically increasing list length.
    last_assistant = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ),
        default=-1,
    )
    emitted = [
        message
        for message in messages[last_assistant + 1 :]
        if message.get("role") in {"user", "tool"}
    ]
    if not emitted:
        if done:
            return [
                {"role": "user", "content": "[CONVERSATION_COMPLETED]"}
            ], total
        raise RuntimeError(
            "official environment step emitted no native user/tool message"
        )
    return emitted, total


def _synchronize_initial_observation(env: Any, observation: Any) -> Any:
    """Close Tau2's reset/observation publication race without changing it.

    ``GymAgent.generate_next_message`` clears its ``_agent_turn_finished``
    event before publishing ``_observation`` while holding ``_lock``.  The
    stock ``AgentGymEnv.reset`` checks the event without taking that lock, so
    under concurrent resets it can return the still-empty observation from
    the tiny interval between those two operations.  Acquiring the agent lock
    here establishes the missing happens-before edge and reads only the same
    official initial observation that ``reset`` intended to return.

    A genuinely completed simulation remains empty.  This helper never
    invents state, advances the episode, retries the user simulator, or alters
    tools/rewards.
    """

    if str(observation or ""):
        return observation
    done = getattr(env, "_simulation_done", None)
    if done is not None and callable(getattr(done, "is_set", None)):
        if done.is_set():
            return observation
    agent = getattr(env, "_agent", None)
    lock = getattr(agent, "_lock", None)
    formatter = getattr(env, "_format_observation", None)
    if agent is None or lock is None or not callable(formatter):
        return observation
    with lock:
        official_messages = list(getattr(agent, "observation", ()) or ())
    if not official_messages:
        return observation
    return formatter(official_messages)


__all__ = ["Tau2OfficialManager", "build_manager"]
