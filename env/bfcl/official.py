"""Thin lifecycle bridge to the official BFCL evaluator and executor.

The official ``bfcl_eval`` package remains the task loader, multi-turn state
machine, transition executor, and accuracy checker.  This module only adapts
its batch-oriented protocol to Trajectory2Gym's shared episode lifecycle.
Trajectory evidence is never consulted while an instance is running.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from trajectory2gym.tool_calls import actions_from_assistant_message
from trajectory2gym.training.real_trajectory_sidecar import (
    RealTrajectoryCreditIndex,
    RealTrajectorySession,
)


DEFAULT_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
)

OFFICIAL_QWEN_FC_PROMPT_TRANSPORT = "official_qwen_fc_raw_v1"
LEGACY_OPENAI_TOOLS_PROMPT_TRANSPORT = "legacy_openai_tools_v1"

_OPENAI_TYPES = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


@dataclass
class _Instance:
    entry: dict[str, Any]
    ground_truth: Any
    category: str
    group: str
    task_id: str
    run_key: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_aliases: dict[str, str] = field(default_factory=dict)
    turn_index: int = 0
    calls_by_turn: list[list[list[str]]] = field(default_factory=list)
    assistant_steps_by_turn: list[list[dict[str, Any]]] = field(
        default_factory=list
    )
    assistant_step_count: int = 0
    single_actions: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    reward: float = 0.0
    terminated: bool = False
    force_terminated: bool = False
    evaluation: dict[str, Any] | None = None
    turn_certificates: list[dict[str, Any]] | None = None
    public_state_fingerprint: str = ""
    trajectory_sidecar: RealTrajectorySession | None = None
    sidecar_step_infos: list[dict[str, Any]] = field(default_factory=list)


class _OfficialBFCLBackend:
    """Direct imports of the public official BFCL package."""

    def __init__(
        self,
        *,
        categories: tuple[str, ...],
        allowed_task_ids: set[str] | None,
        model_name: str,
    ):
        from bfcl_eval.constants.default_prompts import (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
            MAXIMUM_STEP_LIMIT,
        )
        from bfcl_eval.utils import (
            is_multi_turn,
            is_relevance_or_irrelevance,
            load_dataset_entry,
            load_ground_truth_entry,
        )

        self.additional_function_prompt = (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
        )
        self.max_steps_per_turn = int(MAXIMUM_STEP_LIMIT) + 1
        self.model_name = model_name
        self._is_multi_turn = is_multi_turn
        self._is_relevance = is_relevance_or_irrelevance
        self._entries: dict[str, tuple[dict[str, Any], Any, str]] = {}
        self._state_fingerprints: dict[str, str] = {}

        # A source checkout without BFCL's declared dependencies can load the
        # JSON data while silently failing every scored action later.  Import
        # the official execution/checking components now so the HTTP service
        # fails closed before a training run starts.
        importlib.import_module(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"
        )
        importlib.import_module(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker"
        )
        if any(
            not is_multi_turn(category)
            and not is_relevance_or_irrelevance(category)
            for category in categories
        ):
            importlib.import_module(
                "bfcl_eval.eval_checker.ast_eval.ast_checker"
            )
            importlib.import_module("bfcl_eval.constants.enums")

        for category in categories:
            entries = load_dataset_entry(
                category,
                include_prereq=False,
                include_language_specific_hint=False,
            )
            if is_relevance_or_irrelevance(category):
                ground_truth_rows: list[Any] = [None] * len(entries)
            else:
                ground_truth_rows = load_ground_truth_entry(category)
                if len(entries) != len(ground_truth_rows):
                    raise ValueError(
                        f"official BFCL {category} prompt/answer length mismatch: "
                        f"{len(entries)} != {len(ground_truth_rows)}"
                    )
            for entry, answer in zip(entries, ground_truth_rows):
                task_id = str(entry["id"])
                if allowed_task_ids is not None and task_id not in allowed_task_ids:
                    continue
                if task_id in self._entries:
                    raise ValueError(f"duplicate official BFCL task id: {task_id}")
                ground_truth = (
                    None
                    if answer is None
                    else copy.deepcopy(answer["ground_truth"])
                )
                self._entries[task_id] = (
                    copy.deepcopy(entry),
                    ground_truth,
                    category,
                )

        if allowed_task_ids is not None:
            missing = allowed_task_ids.difference(self._entries)
            if missing:
                raise ValueError(
                    "BFCL task allowlist contains unknown IDs: "
                    + ", ".join(sorted(missing)[:10])
                )

    @property
    def task_ids(self) -> list[str]:
        return sorted(self._entries)

    def load_task(
        self, task_id: str
    ) -> tuple[dict[str, Any], Any, str]:
        try:
            entry, ground_truth, category = self._entries[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown official BFCL task id: {task_id}") from exc
        return (
            copy.deepcopy(entry),
            copy.deepcopy(ground_truth),
            category,
        )

    def execute(
        self,
        calls: list[str],
        *,
        entry: Mapping[str, Any],
        run_key: str,
    ) -> list[str]:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        results, instances = execute_multi_turn_func_call(
            func_call_list=calls,
            initial_config=dict(entry.get("initial_config") or {}),
            involved_classes=list(entry["involved_classes"]),
            model_name=run_key,
            test_entry_id=str(entry["id"]),
            long_context=(
                "long_context" in str(entry["id"])
                or "composite" in str(entry["id"])
            ),
            is_evaL_run=False,
        )
        self._state_fingerprints[run_key] = _public_state_fingerprint(
            instances
        )
        return list(results)

    def initialize(
        self,
        *,
        entry: Mapping[str, Any],
        run_key: str,
    ) -> str:
        """Instantiate the official backend and fingerprint its public state."""

        self.execute([], entry=entry, run_key=run_key)
        return self._state_fingerprints[run_key]

    def public_state_fingerprint(self, run_key: str) -> str:
        return self._state_fingerprints[run_key]

    def evaluate_single(
        self,
        *,
        entry: Mapping[str, Any],
        ground_truth: Any,
        category: str,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._is_relevance(category):
            valid = not actions if "irrelevance" in category else bool(actions)
            return {
                "valid": valid,
                "error_type": (
                    None
                    if valid
                    else "relevance_error:unexpected_function_call"
                    if "irrelevance" in category
                    else "relevance_error:missing_function_call"
                ),
            }

        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        language = (
            Language.JAVA
            if "java" in category and "javascript" not in category
            else Language.JAVASCRIPT
            if "javascript" in category
            else Language.PYTHON
        )
        decoded = [
            {str(action["name"]): dict(action.get("arguments") or {})}
            for action in actions
        ]
        return dict(
            ast_checker(
                list(entry["function"]),
                decoded,
                ground_truth,
                language,
                category,
                self.model_name,
            )
        )

    def evaluate_multi(
        self,
        *,
        entry: Mapping[str, Any],
        ground_truth: Any,
        category: str,
        calls_by_turn: list[list[list[str]]],
        run_key: str,
    ) -> dict[str, Any]:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
        )

        return dict(
            multi_turn_checker(
                calls_by_turn,
                ground_truth,
                copy.deepcopy(dict(entry)),
                category,
                run_key,
            )
        )

    def certify_multi_turns(
        self,
        *,
        entry: Mapping[str, Any],
        ground_truth: Any,
        calls_by_turn: list[list[list[str]]],
        run_key: str,
    ) -> list[dict[str, Any]]:
        """Return official-checker certificates without early termination.

        BFCL's official session checker intentionally returns at the first
        failed user turn.  That is correct for benchmark accuracy, but it
        cannot identify which already-completed turn should receive actor
        credit.  This audit follows the same executor, state checker, and
        response checker while retaining one result per user turn.  It never
        changes the binary session grade.
        """

        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            response_checker,
            state_checker,
        )
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
            is_empty_execute_response,
        )

        initial_config = dict(entry.get("initial_config") or {})
        involved_classes = list(entry["involved_classes"])
        test_entry_id = str(entry["id"])
        long_context = bool(
            "long_context" in test_entry_id or "composite" in test_entry_id
        )
        model_key = run_key + "_turn_eval"
        ground_truth_key = run_key + "_turn_ground_truth_eval"
        all_model_results: list[str] = []
        certificates: list[dict[str, Any]] = []
        model_turns = list(calls_by_turn)
        ground_truth_turns = list(ground_truth or ())

        for turn_index, expected_calls in enumerate(ground_truth_turns):
            model_steps = (
                model_turns[turn_index]
                if turn_index < len(model_turns)
                else []
            )
            model_results: list[str] = []
            model_instances: dict[str, Any] = {}
            execution_error = ""
            for model_calls in model_steps:
                try:
                    step_results, model_instances = execute_multi_turn_func_call(
                        func_call_list=model_calls,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=model_key,
                        test_entry_id=test_entry_id,
                        long_context=long_context,
                        is_evaL_run=True,
                    )
                    model_results.extend(step_results)
                except Exception as exc:
                    execution_error = f"model_exec_error:{type(exc).__name__}"
                    break
            all_model_results.extend(model_results)

            try:
                expected_results, expected_instances = (
                    execute_multi_turn_func_call(
                        func_call_list=expected_calls,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=ground_truth_key,
                        test_entry_id=test_entry_id,
                        long_context=long_context,
                        is_evaL_run=True,
                    )
                )
            except Exception as exc:
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": None,
                        "reason": f"ground_truth_exec_error:{type(exc).__name__}",
                    }
                )
                continue

            if execution_error:
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": False,
                        "reason": execution_error,
                    }
                )
                continue
            if not expected_calls:
                attempted = bool(model_steps) and not is_empty_execute_response(
                    model_steps
                )
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": not attempted,
                        "reason": (
                            "spurious_tool_call"
                            if attempted
                            else "correct_abstention"
                        ),
                    }
                )
                continue
            if not model_steps or is_empty_execute_response(model_steps):
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": False,
                        "reason": "empty_turn_model_response",
                    }
                )
                continue
            if set(model_instances) != set(expected_instances):
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": False,
                        "reason": "instance_mismatch",
                    }
                )
                continue
            try:
                state_valid = bool(
                    state_checker(model_instances, expected_instances).get(
                        "valid", False
                    )
                )
            except Exception as exc:
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": None,
                        "reason": f"state_checker_error:{type(exc).__name__}",
                    }
                )
                continue
            if not state_valid:
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": False,
                        "reason": "state_mismatch",
                    }
                )
                continue
            try:
                response_valid = bool(
                    response_checker(
                        all_model_results,
                        expected_results,
                        turn_index,
                    ).get("valid", False)
                )
            except Exception as exc:
                certificates.append(
                    {
                        "turn_index": turn_index,
                        "valid": None,
                        "reason": f"response_checker_error:{type(exc).__name__}",
                    }
                )
                continue
            certificates.append(
                {
                    "turn_index": turn_index,
                    "valid": response_valid,
                    "reason": "pass" if response_valid else "response_mismatch",
                }
            )
        return certificates

    def is_multi_turn(self, category: str) -> bool:
        return bool(self._is_multi_turn(category))

    def cleanup(self, *, entry: Mapping[str, Any], run_key: str) -> None:
        """Release BFCL's module-global executable instances for this rollout."""

        self._state_fingerprints.pop(run_key, None)
        if not self.is_multi_turn(str(entry["id"])):
            return
        module = importlib.import_module(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"
        )
        for model_key in (
            run_key,
            run_key + "_eval",
            run_key + "_ground_truth_eval",
            run_key + "_turn_eval",
            run_key + "_turn_ground_truth_eval",
        ):
            for class_name in entry["involved_classes"]:
                name = re.sub(
                    r"[-./:]",
                    "_",
                    f"{model_key}_{entry['id']}_{class_name}_instance",
                )
                module.__dict__.pop(name, None)

    def provenance(self) -> dict[str, Any]:
        try:
            version = importlib.metadata.version("bfcl_eval")
        except importlib.metadata.PackageNotFoundError:
            version = "source-checkout"
        module = importlib.import_module("bfcl_eval")
        module_file = getattr(module, "__file__", None)
        if module_file:
            module_path = str(Path(module_file).resolve())
        else:
            namespace_paths = list(getattr(module, "__path__", []))
            module_path = (
                str(Path(namespace_paths[0]).resolve())
                if namespace_paths
                else "namespace:bfcl_eval"
            )
        return {
            "package": "bfcl_eval",
            "version": version,
            "module_path": module_path,
        }


class BFCLOfficialManager:
    """Official BFCL task lifecycle with no trajectory-action fallback."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = dict(config or {})
        bfcl_root = config.pop("bfcl_root", None)
        if bfcl_root:
            _add_bfcl_root(Path(str(bfcl_root)).expanduser().resolve())

        categories = tuple(
            str(value)
            for value in config.pop("categories", DEFAULT_CATEGORIES)
        )
        allowed_task_ids = _allowed_task_ids(
            config.pop("task_ids", None),
            config.pop("task_ids_file", None),
        )
        self.model_name = str(
            config.pop("model_name", "Qwen/Qwen3-8B-FC")
        )
        injected_backend = config.pop("_backend", None)
        prompt_transport = config.pop("prompt_transport", None)
        if prompt_transport is None:
            if injected_backend is None:
                raise ValueError(
                    "BFCL prompt_transport is required; use "
                    f"{OFFICIAL_QWEN_FC_PROMPT_TRANSPORT!r} for mainline "
                    "Qwen-FC execution"
                )
            # Preserve isolated fake-backend unit tests. Real BFCL execution
            # must opt into, and receipt, the official raw-completion prompt.
            prompt_transport = LEGACY_OPENAI_TOOLS_PROMPT_TRANSPORT
        self.prompt_transport = str(prompt_transport)
        if self.prompt_transport not in {
            OFFICIAL_QWEN_FC_PROMPT_TRANSPORT,
            LEGACY_OPENAI_TOOLS_PROMPT_TRANSPORT,
        }:
            raise ValueError(
                f"unsupported BFCL prompt_transport: {self.prompt_transport!r}"
            )
        self._backend = injected_backend or _OfficialBFCLBackend(
            categories=categories,
            allowed_task_ids=allowed_task_ids,
            model_name=self.model_name,
        )
        self.max_steps_per_turn = int(
            config.pop(
                "max_steps_per_turn",
                self._backend.max_steps_per_turn,
            )
        )
        if self.max_steps_per_turn <= 0:
            raise ValueError("max_steps_per_turn must be positive")
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
        if config:
            raise ValueError(
                "unsupported BFCL manager config: "
                + ", ".join(sorted(config))
            )
        self.instances: dict[str, _Instance] = {}

    def get_env_profile(
        self,
        *,
        env_type: str = "bfcl",
        split: str = "train",
        params: dict[str, Any] | None = None,
    ) -> list[str]:
        _require_env_type(env_type)
        return list(self._backend.task_ids)

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
        entry, ground_truth, category = self._backend.load_task(str(task_id))
        instance_id = instance_id or f"bfcl_{uuid.uuid4().hex[:12]}"
        if instance_id in self.instances:
            raise ValueError(f"duplicate instance_id: {instance_id}")
        run_key = re.sub(r"[^A-Za-z0-9_]", "_", f"t2g_{instance_id}")
        questions = list(entry.get("question") or [])
        if not questions:
            raise ValueError(f"official BFCL task {task_id} has no question")
        instance = _Instance(
            entry=entry,
            ground_truth=ground_truth,
            category=category,
            group=_category_group(category),
            task_id=str(task_id),
            run_key=run_key,
            calls_by_turn=[[] for _ in questions],
            assistant_steps_by_turn=[[] for _ in questions],
            trajectory_sidecar=None,
        )
        initialize = getattr(self._backend, "initialize", None)
        if self._backend.is_multi_turn(category) and callable(initialize):
            instance.public_state_fingerprint = str(
                initialize(entry=instance.entry, run_key=instance.run_key)
            )
        else:
            instance.public_state_fingerprint = _stable_digest(
                entry.get("initial_config") or {}
            )
        state = self._prepare_turn(instance, 0)
        if self.trajectory_sidecar_index is not None:
            instance.trajectory_sidecar = self.trajectory_sidecar_index.start(
                domain=_evidence_domain(category),
                reset_id=str(task_id),
                initial_messages=state,
            )
        initial_context_sha256 = _stable_digest(state)
        reset_identity_sha256 = _stable_digest(
            {
                "category": instance.category,
                "initial_context_sha256": initial_context_sha256,
                "state_fingerprint": instance.public_state_fingerprint,
                "state_hash_version": "bfcl_public_attributes.v1",
                "task_id": instance.task_id,
            }
        )
        self.instances[instance_id] = instance
        info = self._info(instance, guidance=guidance)
        info["initial_visible_context_sha256"] = initial_context_sha256
        info["reset_identity_sha256"] = reset_identity_sha256
        return {
            "state": state,
            "info": info,
        }

    def step(
        self,
        *,
        instance_id: str,
        action: dict[str, Any] | list[dict[str, Any]] | str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        if instance.terminated:
            raise RuntimeError(f"BFCL instance {instance_id} is terminated")
        message = _assistant_message(action)
        parsed = [
            _restore_official_action(item, instance.tool_aliases)
            for item in actions_from_assistant_message(message)
        ]
        assistant_step_index = instance.assistant_step_count
        instance.assistant_step_count += 1
        instance.assistant_steps_by_turn[instance.turn_index].append(
            {
                "assistant_step_index": assistant_step_index,
                "actions": copy.deepcopy(parsed),
                "has_tool_calls": bool(parsed),
            }
        )
        instance.assistant_text = str(message.get("content") or "")

        if not self._backend.is_multi_turn(instance.category):
            instance.single_actions = parsed
            instance.terminated = True
            self._evaluate(instance)
            emitted = [
                {
                    "role": "user",
                    "content": "[BFCL_SINGLE_TURN_COMPLETED]",
                }
            ]
            self._observe_sidecar(instance, message, emitted)
            return self._step_payload(instance, emitted)

        if parsed:
            calls = [
                _action_to_python_call(item, instance.tool_aliases)
                for item in parsed
            ]
            instance.calls_by_turn[instance.turn_index].append(calls)
            results = self._backend.execute(
                calls,
                entry=instance.entry,
                run_key=instance.run_key,
            )
            fingerprint = getattr(
                self._backend, "public_state_fingerprint", None
            )
            if callable(fingerprint):
                instance.public_state_fingerprint = str(
                    fingerprint(instance.run_key)
                )
            if self.prompt_transport == OFFICIAL_QWEN_FC_PROMPT_TRANSPORT:
                emitted = [
                    {
                        "role": "tool",
                        "name": str(parsed[index].get("name") or ""),
                        "tool_call_id": f"bfcl_{assistant_step_index}_{index}",
                        "content": str(result),
                    }
                    for index, result in enumerate(results)
                ]
            else:
                emitted = [
                    {"role": "user", "content": str(result)}
                    for result in results
                ]
            if (
                len(instance.calls_by_turn[instance.turn_index])
                >= self.max_steps_per_turn
            ):
                instance.force_terminated = True
                instance.terminated = True
                self._evaluate(instance)
                emitted.append(
                    {
                        "role": "user",
                        "content": "[BFCL_MAXIMUM_STEP_LIMIT_REACHED]",
                    }
                )
            self._observe_sidecar(instance, message, emitted)
            return self._step_payload(instance, emitted)

        # The official generation loop advances only after a response no
        # longer decodes to a function call.
        next_turn = instance.turn_index + 1
        if next_turn >= len(instance.calls_by_turn):
            instance.terminated = True
            self._evaluate(instance)
            emitted = [
                {"role": "user", "content": "[BFCL_CONVERSATION_COMPLETED]"}
            ]
            self._observe_sidecar(instance, message, emitted)
            return self._step_payload(instance, emitted)
        emitted = self._prepare_turn(instance, next_turn)
        self._observe_sidecar(instance, message, emitted)
        return self._step_payload(instance, emitted)

    def evaluate(
        self,
        *,
        instance_id: str,
        messages: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._instance(instance_id)
        if instance.evaluation is None:
            if instance.terminated:
                self._evaluate(instance)
            else:
                instance.evaluation = {
                    "valid": False,
                    "error_type": "multi_turn:incomplete",
                    "error_message": (
                        "The rollout ended before the official BFCL "
                        "conversation reached its terminal turn."
                    ),
                }
        valid = bool(instance.evaluation.get("valid", False))
        reward = 1.0 if valid else 0.0
        official_step_infos = self._official_turn_step_infos(instance)
        step_infos = (
            copy.deepcopy(instance.sidecar_step_infos)
            if instance.trajectory_sidecar is not None
            else official_step_infos
        )
        failure_infos = [
            copy.deepcopy(info)
            for info in step_infos
            if float(info.get("official_turn_credit_label", 0.0)) < 0.0
        ]
        return {
            "score": reward,
            "accuracy": reward,
            "reward": reward,
            "success": valid,
            "session_success": valid,
            "completed": instance.terminated,
            "session_completed": instance.terminated,
            "outcome": (
                "success"
                if valid
                else "failure" if instance.terminated else "incomplete"
            ),
            "task_id": instance.task_id,
            "domain": instance.group,
            "source": "official_bfcl",
            "official_category": instance.category,
            "official_result": copy.deepcopy(instance.evaluation),
            "step_infos": step_infos,
            "turn_infos": copy.deepcopy(step_infos),
            "official_turn_infos": official_step_infos,
            "failure_infos": failure_infos,
            "official_turn_certificate_count": sum(
                int(abs(float(info.get("official_turn_credit_label", 0.0))) > 0)
                for info in step_infos
            ),
            "trajectory_sidecar_enabled": (
                instance.trajectory_sidecar is not None
            ),
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
            "tools": copy.deepcopy(instance.tools),
            "domain": instance.group,
            "trajectory_evidence_domain": _evidence_domain(instance.category),
            "environment_task_id": instance.task_id,
            "official_category": instance.category,
            "official_protocol": (
                "executable_multi_turn"
                if self._backend.is_multi_turn(instance.category)
                else "single_turn_checker"
            ),
            "turn_index": instance.turn_index,
            "terminated": instance.terminated,
            "official_backend": self._backend.provenance(),
            "trajectory_sidecar_enabled": (
                instance.trajectory_sidecar is not None
            ),
            "trajectory_action_proposals": _trajectory_action_proposals(
                instance.trajectory_sidecar
            ),
        }

    def release_instance(self, instance_id: str) -> bool:
        instance = self.instances.pop(instance_id, None)
        if instance is None:
            return False
        self._backend.cleanup(entry=instance.entry, run_key=instance.run_key)
        return True

    def close_all(self) -> None:
        for instance_id in list(self.instances):
            self.release_instance(instance_id)

    def _prepare_turn(
        self, instance: _Instance, turn_index: int
    ) -> list[dict[str, Any]]:
        instance.turn_index = turn_index
        held_out = (instance.entry.get("missed_function") or {}).get(
            str(turn_index)
        )
        messages = copy.deepcopy(instance.entry["question"][turn_index])
        if held_out:
            if self.prompt_transport == OFFICIAL_QWEN_FC_PROMPT_TRANSPORT:
                raise RuntimeError(
                    "official Qwen-FC raw prompt transport cannot mutate the "
                    "initial tools preamble in-place; missing-function BFCL "
                    "categories require a dedicated prompt-state adapter"
                )
            known = {
                str(function.get("name"))
                for function in instance.entry.get("function") or []
            }
            for function in held_out:
                if str(function.get("name")) not in known:
                    instance.entry.setdefault("function", []).append(
                        copy.deepcopy(function)
                    )
            if not messages:
                messages = [
                    {
                        "role": "user",
                        "content": self._backend.additional_function_prompt,
                    }
                ]
        functions = instance.entry.get("function") or []
        if self.prompt_transport == OFFICIAL_QWEN_FC_PROMPT_TRANSPORT:
            instance.tools = []
            instance.tool_aliases = {
                str(function.get("name") or ""): str(function.get("name") or "")
                for function in functions
                if str(function.get("name") or "")
            }
            if turn_index == 0:
                messages = _official_qwen_fc_initial_messages(
                    messages,
                    functions,
                )
        else:
            instance.tools, instance.tool_aliases = _openai_tools(functions)
        return [dict(message) for message in messages]

    def _evaluate(self, instance: _Instance) -> None:
        if instance.force_terminated:
            instance.evaluation = {
                "valid": False,
                "error_type": "multi_turn:force_terminated",
                "error_message": (
                    "Official BFCL per-turn maximum step limit reached."
                ),
            }
        elif self._backend.is_multi_turn(instance.category):
            try:
                instance.evaluation = _json_safe(
                    self._backend.evaluate_multi(
                        entry=instance.entry,
                        ground_truth=instance.ground_truth,
                        category=instance.category,
                        calls_by_turn=copy.deepcopy(instance.calls_by_turn),
                        run_key=instance.run_key,
                    )
                )
            except Exception as exc:
                instance.evaluation = {
                    "valid": False,
                    "error_type": "multi_turn:official_checker_exception",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
        else:
            try:
                instance.evaluation = _json_safe(
                    self._backend.evaluate_single(
                        entry=instance.entry,
                        ground_truth=instance.ground_truth,
                        category=instance.category,
                        actions=copy.deepcopy(instance.single_actions),
                    )
                )
            except Exception as exc:
                instance.evaluation = {
                    "valid": False,
                    "error_type": "single_turn:official_checker_exception",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
        instance.reward = (
            1.0 if instance.evaluation.get("valid", False) else 0.0
        )

    def _official_turn_step_infos(
        self, instance: _Instance
    ) -> list[dict[str, Any]]:
        """Align official user-turn checks to retained assistant decisions.

        One certified decision is emitted per BFCL user turn.  Correct turns
        before the first error receive positive local credit.  The first
        model-attributable failed turn receives negative credit.  Remaining
        decisions are neutral because their state may be downstream of that
        error.  Protocol-closing responses and extra substeps are also
        neutral, preventing one user turn from being counted multiple times.
        """

        certify = getattr(self._backend, "certify_multi_turns", None)
        if not self._backend.is_multi_turn(instance.category) or not callable(
            certify
        ):
            return []
        if instance.turn_certificates is None:
            try:
                instance.turn_certificates = _json_safe(
                    certify(
                        entry=instance.entry,
                        ground_truth=instance.ground_truth,
                        calls_by_turn=copy.deepcopy(instance.calls_by_turn),
                        run_key=instance.run_key,
                    )
                )
            except Exception:
                # Certification is optional actor evidence.  The official
                # session score above remains authoritative; missing local
                # evidence must fail closed to zero credit.
                instance.turn_certificates = []

        by_turn = {
            int(item.get("turn_index", -1)): item
            for item in instance.turn_certificates
            if isinstance(item, dict)
        }
        infos_by_step: dict[int, dict[str, Any]] = {}
        first_failure_seen = False
        for turn_index, assistant_steps in enumerate(
            instance.assistant_steps_by_turn
        ):
            certificate = by_turn.get(turn_index, {})
            certified_valid = certificate.get("valid")
            reason = str(certificate.get("reason") or "uncertified")
            credit_step = None
            if certified_valid is True and not first_failure_seen:
                credit_step = next(
                    (
                        step
                        for step in assistant_steps
                        if step.get("has_tool_calls")
                    ),
                    assistant_steps[0] if assistant_steps else None,
                )
                credit_label = 1.0
                branch = "valid_official_turn"
            elif certified_valid is False and not first_failure_seen:
                # A missing-call error belongs to the no-tool closing answer;
                # other errors belong to the first attempted tool decision.
                prefer_no_tool = reason == "empty_turn_model_response"
                credit_step = next(
                    (
                        step
                        for step in assistant_steps
                        if bool(step.get("has_tool_calls")) != prefer_no_tool
                    ),
                    assistant_steps[0] if assistant_steps else None,
                )
                credit_label = -1.0
                branch = "invalid_official_first_error"
                first_failure_seen = True
            else:
                credit_label = 0.0
                branch = (
                    "neutral_official_downstream"
                    if first_failure_seen
                    else "neutral_official_uncertified"
                )

            for step in assistant_steps:
                step_index = int(step["assistant_step_index"])
                selected = credit_step is step and credit_label != 0.0
                label = credit_label if selected else 0.0
                step_branch = branch if selected else "neutral_official_protocol"
                infos_by_step[step_index] = {
                    "success": bool(label > 0.0),
                    "accepted": bool(label > 0.0),
                    "label": "accepted" if label > 0.0 else "rejected" if label < 0.0 else "abstained",
                    "branch_type": step_branch,
                    "verifier_status": (
                        "valid" if label > 0.0 else "invalid" if label < 0.0 else "ambiguous"
                    ),
                    "non_blocking_abstention": bool(label == 0.0),
                    "verified_support": bool(label > 0.0),
                    "progress_advanced": bool(label > 0.0),
                    "actual_actions": copy.deepcopy(step.get("actions") or []),
                    "reason": reason,
                    "official_user_turn_index": turn_index,
                    "official_turn_credit_label": label,
                    "official_turn_certificate": "official_checker_turn_v1",
                    "policy_context_id": (
                        f"bfcl:{instance.task_id}:user_turn:{turn_index}"
                    ),
                }
        return [
            infos_by_step.get(
                index,
                {
                    "success": False,
                    "accepted": False,
                    "label": "abstained",
                    "branch_type": "neutral_official_uncertified",
                    "verifier_status": "ambiguous",
                    "non_blocking_abstention": True,
                    "official_turn_credit_label": 0.0,
                    "official_turn_certificate": "",
                    "policy_context_id": f"bfcl:{instance.task_id}:assistant:{index}",
                },
            )
            for index in range(instance.assistant_step_count)
        ]

    def _step_payload(
        self,
        instance: _Instance,
        emitted: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "state": emitted,
            "emitted_state": emitted,
            "reward": instance.reward,
            "is_terminated": instance.terminated,
            "info": self._info(instance),
        }

    def _observe_sidecar(
        self,
        instance: _Instance,
        assistant_message: Mapping[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        if instance.trajectory_sidecar is None:
            return
        certificate = instance.trajectory_sidecar.observe(
            assistant_message=assistant_message,
            observations=observations,
        )
        instance.sidecar_step_infos.append(certificate.to_turn_info())

    def _info(
        self,
        instance: _Instance,
        *,
        guidance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = {
            "tools": copy.deepcopy(instance.tools),
            "policy_prompt_transport": self.prompt_transport,
            "official_function_schema_sha256": _stable_digest(
                instance.entry.get("function") or []
            ),
            "domain": instance.group,
            "trajectory_evidence_domain": _evidence_domain(instance.category),
            "environment_task_id": instance.task_id,
            "official_category": instance.category,
            "turn_index": instance.turn_index,
            "state_fingerprint": instance.public_state_fingerprint,
            "state_hash_version": "bfcl_public_attributes.v1",
            "oracle_source": "real_environment",
            "trajectory_guided_credit_enabled": True,
            "trajectory_credit_source": (
                "context_certified_trajectory_sidecar_v2"
                if instance.trajectory_sidecar is not None
                else "official_bfcl_turn_checker_v1"
            ),
            "trajectory_sidecar_enabled": (
                instance.trajectory_sidecar is not None
            ),
            "trajectory_action_proposals": _trajectory_action_proposals(
                instance.trajectory_sidecar
            ),
        }
        if guidance is not None:
            info["trajectory_guidance_level"] = guidance.get("guidance_level")
            info["official_backend"] = self._backend.provenance()
        return info

    def _instance(self, instance_id: str) -> _Instance:
        try:
            return self.instances[instance_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown official BFCL instance_id: {instance_id}"
            ) from exc


def build_manager(
    config: Mapping[str, Any] | None = None,
) -> BFCLOfficialManager:
    return BFCLOfficialManager(config)


def _add_bfcl_root(path: Path) -> None:
    candidates = (
        path,
        path / "berkeley-function-call-leaderboard",
    )
    for candidate in candidates:
        if (candidate / "bfcl_eval" / "utils.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(
        "bfcl_root must contain bfcl_eval/utils.py directly or under "
        "berkeley-function-call-leaderboard/"
    )


def _allowed_task_ids(
    raw_ids: Any,
    raw_path: Any,
) -> set[str] | None:
    if raw_ids is not None and raw_path is not None:
        raise ValueError("set only one of task_ids or task_ids_file")
    if raw_path is not None:
        payload = json.loads(
            Path(str(raw_path)).expanduser().read_text(encoding="utf-8")
        )
        if isinstance(payload, Mapping):
            values = [
                task_id
                for task_ids in payload.values()
                for task_id in task_ids
            ]
        else:
            values = payload
        if not isinstance(values, list):
            raise ValueError("BFCL task_ids_file must contain a list or mapping")
        return {str(value) for value in values}
    if raw_ids is None:
        return None
    if not isinstance(raw_ids, (list, tuple, set)):
        raise ValueError("BFCL task_ids must be a sequence")
    return {str(value) for value in raw_ids}


def _openai_tools(
    functions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Match BFCL's Gorilla-to-OpenAPI transport conversion."""

    tools: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for raw in functions:
        function = copy.deepcopy(raw)
        original_name = str(function["name"])
        exposed_name = re.sub(r"\.", "_", original_name)
        aliases[exposed_name] = original_name
        aliases[original_name] = original_name
        function["name"] = exposed_name
        parameters = function.setdefault("parameters", {})
        parameters["type"] = "object"
        parameters["properties"] = _convert_properties(
            parameters.get("properties") or {}
        )
        tools.append({"type": "function", "function": function})
    return tools, aliases


def _official_qwen_fc_initial_messages(
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render BFCL's Qwen-FC raw tools preamble through the native template.

    The official evaluator hand-renders raw Gorilla function schemas and then
    calls the completions API.  Passing OpenAI-wrapped tools to Qwen's native
    chat template changes function names, schema types, JSON structure, and
    tokenization.  Supplying the exact official preamble as a system message
    and no native ``tools`` argument produces the same Qwen prompt byte for
    byte while remaining compatible with the training rollout stack.
    """

    copied = [copy.deepcopy(dict(message)) for message in messages]
    if not functions:
        return copied

    leading_system = ""
    if copied and str(copied[0].get("role") or "") == "system":
        leading_system = str(copied.pop(0).get("content") or "") + "\n\n"
    content = (
        leading_system
        + "# Tools\n\n"
        + "You may call one or more functions to assist with the user query.\n\n"
        + "You are provided with function signatures within <tools></tools> "
        + "XML tags:\n<tools>"
    )
    for function in functions:
        # Match bfcl_eval's QwenFCHandler._format_prompt exactly: default
        # json.dumps spacing, insertion order, and ASCII escaping.
        content += "\n" + json.dumps(function)
    content += (
        "\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )
    return [{"role": "system", "content": content}, *copied]


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Preserve checker structure while making its diagnostics transportable."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_safe(item) for item in value),
            key=lambda item: _stable_digest(item),
        )
    return str(value)


def _public_value(value: Any, active: set[int] | None = None) -> Any:
    """Convert an executor object into a stable, public-state-only value."""

    if active is None:
        active = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    marker = id(value)
    if marker in active:
        return {"cycle": type(value).__name__}
    if isinstance(value, Mapping):
        active.add(marker)
        result = {
            str(key): _public_value(item, active)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
        active.remove(marker)
        return result
    if isinstance(value, (list, tuple)):
        active.add(marker)
        result = [_public_value(item, active) for item in value]
        active.remove(marker)
        return result
    if isinstance(value, (set, frozenset)):
        active.add(marker)
        result = [_public_value(item, active) for item in value]
        active.remove(marker)
        return sorted(result, key=lambda item: _stable_digest(item))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        active.add(marker)
        result = {
            str(key): _public_value(item, active)
            for key, item in sorted(attributes.items())
            if not str(key).startswith("_")
            and str(key) != "parent"
            and not callable(item)
        }
        active.remove(marker)
        return {"class": type(value).__name__, "attributes": result}
    return {"type": type(value).__name__, "value": str(value)}


def _public_state_fingerprint(instances: Mapping[str, Any]) -> str:
    public_state = {
        str(name): _public_value(instance)
        for name, instance in sorted(instances.items())
    }
    return _stable_digest(public_state)


def _convert_properties(
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, raw in properties.items():
        value = copy.deepcopy(dict(raw))
        original_type = str(value.get("type") or "string")
        value["type"] = _OPENAI_TYPES.get(original_type, "string")
        if original_type == "float":
            value["format"] = "float"
            value["description"] = (
                str(value.get("description") or "")
                + " This is a float type value."
            ).strip()
        if "properties" in value:
            value["properties"] = _convert_properties(value["properties"])
        if isinstance(value.get("items"), Mapping):
            value["items"] = _convert_item_schema(value["items"])
        converted[str(key)] = value
    return converted


def _convert_item_schema(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(raw))
    value["type"] = _OPENAI_TYPES.get(
        str(value.get("type") or "string"),
        "string",
    )
    if "properties" in value:
        value["properties"] = _convert_properties(value["properties"])
    if isinstance(value.get("items"), Mapping):
        value["items"] = _convert_item_schema(value["items"])
    return value


def _restore_official_action(
    action: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    exposed_name = str(action.get("name") or "")
    return {
        "name": aliases.get(exposed_name, exposed_name),
        "arguments": dict(action.get("arguments") or {}),
    }


def _action_to_python_call(
    action: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> str:
    requested_name = str(action.get("name") or "")
    allowed_names = set(aliases.values())
    if requested_name not in allowed_names:
        return (
            "__unsupported_bfcl_tool__("
            f"requested_tool={requested_name!r})"
        )
    arguments = dict(action.get("arguments") or {})
    parts: list[str] = []
    for key, value in arguments.items():
        if not str(key).isidentifier():
            return (
                "__unsupported_bfcl_arguments__("
                f"requested_tool={requested_name!r})"
            )
        parts.append(f"{key}={value!r}")
    return f"{requested_name}({', '.join(parts)})"


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


def _category_group(category: str) -> str:
    if "multi_turn" in category:
        return "multi_turn"
    if "live" in category:
        return "live"
    return "non_live"


def _evidence_domain(category: str) -> str:
    """Match the canonical trajectory domain used by BFCL data compilers."""

    return "multi_turn" if "multi_turn" in category else category


def _trajectory_action_proposals(
    session: RealTrajectorySession | None,
) -> list[dict[str, Any]]:
    if session is None:
        return []
    return [proposal.to_dict() for proposal in session.action_proposals()]


def _require_env_type(env_type: str) -> None:
    if env_type != "bfcl":
        raise ValueError(f"BFCL manager cannot serve env_type={env_type!r}")


def _real_guidance(
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = params if isinstance(params, Mapping) else {}
    raw = params.get("trajectory_guidance")
    guidance = dict(raw) if isinstance(raw, Mapping) else {}
    if guidance.get("execution_mode") != "real":
        raise ValueError("official BFCL manager requires real guidance")
    if guidance.get("guidance_reset_strategy") != "initial":
        raise ValueError(
            "official BFCL bridge exposes task-level initial reset only"
        )
    if guidance.get("episode_execution_boundary") is not True:
        raise ValueError("real episode boundary is not declared")
    if guidance.get("per_action_fallback_allowed") is not False:
        raise ValueError("per-action oracle fallback is forbidden")
    return guidance


__all__ = ["BFCLOfficialManager", "DEFAULT_CATEGORIES", "build_manager"]
