from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .loader import load_trajectory_episodes
from .observation_db import ObservationDatabase
from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep
from .verifiability import VerifiabilityAnalyzer, VerifiabilityDecision
from .tool_semantics import DEFAULT_TOOL_SEMANTICS, ToolSemantics


@dataclass(frozen=True)
class OracleVerificationRow:
    episode_id: str
    step_index: int
    action: dict[str, Any]
    partial_label: str
    partial_action_class: str
    partial_accepted: bool
    oracle_label: str
    oracle_answerable: bool
    observation_match: bool | None
    outcome: str
    partial_observation_messages: tuple[dict[str, Any], ...] = ()
    oracle_observation_messages: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    oracle_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "action": copy.deepcopy(self.action),
            "partial_label": self.partial_label,
            "partial_action_class": self.partial_action_class,
            "partial_accepted": self.partial_accepted,
            "oracle_label": self.oracle_label,
            "oracle_answerable": self.oracle_answerable,
            "observation_match": self.observation_match,
            "outcome": self.outcome,
            "partial_observation_messages": copy.deepcopy(list(self.partial_observation_messages)),
            "oracle_observation_messages": copy.deepcopy(list(self.oracle_observation_messages)),
            "reason": self.reason,
            "oracle_reason": self.oracle_reason,
        }


@dataclass(frozen=True)
class OracleVerificationReport:
    dataset: str
    source_path: str
    oracle_mode: str
    compile_episode_count: int
    oracle_episode_count: int
    candidate_count: int
    accepted_count: int
    false_accept_count: int
    oracle_answerable_abstain_count: int
    reject_count: int
    reject_correct_count: int
    invalid_probe_count: int
    hallucinated_observation_count: int
    outcome_counts: dict[str, int]
    partial_label_counts: dict[str, int]
    oracle_label_counts: dict[str, int]
    rows: tuple[OracleVerificationRow, ...] = field(default_factory=tuple)

    @property
    def false_accept_rate(self) -> float:
        return _safe_div(self.false_accept_count, self.accepted_count)

    @property
    def oracle_answerable_abstain_rate(self) -> float:
        return _safe_div(self.oracle_answerable_abstain_count, self.candidate_count)

    @property
    def reject_correctness(self) -> float:
        return _safe_div(self.reject_correct_count, self.reject_count)

    @property
    def hallucinated_observation_rate(self) -> float:
        return _safe_div(self.hallucinated_observation_count, self.candidate_count)

    def to_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "source_path": self.source_path,
            "oracle_mode": self.oracle_mode,
            "compile_episode_count": self.compile_episode_count,
            "oracle_episode_count": self.oracle_episode_count,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "false_accept_count": self.false_accept_count,
            "false_accept_rate": self.false_accept_rate,
            "oracle_answerable_abstain_count": self.oracle_answerable_abstain_count,
            "oracle_answerable_abstain_rate": self.oracle_answerable_abstain_rate,
            "reject_count": self.reject_count,
            "reject_correct_count": self.reject_correct_count,
            "reject_correctness": self.reject_correctness,
            "invalid_probe_count": self.invalid_probe_count,
            "hallucinated_observation_count": self.hallucinated_observation_count,
            "hallucinated_observation_rate": self.hallucinated_observation_rate,
            "outcome_counts": dict(self.outcome_counts),
            "partial_label_counts": dict(self.partial_label_counts),
            "oracle_label_counts": dict(self.oracle_label_counts),
        }
        if include_rows:
            payload["rows"] = [row.to_dict() for row in self.rows]
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"# Trajectory2Gym Oracle Verification: {self.dataset}",
            "",
            f"Source: `{self.source_path}`",
            "",
            f"Oracle mode: `{self.oracle_mode}`",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| compile episodes | {self.compile_episode_count} |",
            f"| oracle episodes | {self.oracle_episode_count} |",
            f"| action candidates | {self.candidate_count} |",
            f"| accepted candidates | {self.accepted_count} |",
            f"| false accepts | {self.false_accept_count} |",
            f"| false-accept rate | {self.false_accept_rate:.6f} |",
            f"| oracle-answerable abstains | {self.oracle_answerable_abstain_count} |",
            f"| oracle-answerable abstain rate | {self.oracle_answerable_abstain_rate:.6f} |",
            f"| rejects | {self.reject_count} |",
            f"| reject correct | {self.reject_correct_count} |",
            f"| reject correctness | {self.reject_correctness:.6f} |",
            f"| hallucinated observations | {self.hallucinated_observation_count} |",
            f"| hallucinated observation rate | {self.hallucinated_observation_rate:.6f} |",
            "",
            f"Outcome counts: `{json.dumps(self.outcome_counts, sort_keys=True)}`",
            "",
            f"Partial labels: `{json.dumps(self.partial_label_counts, sort_keys=True)}`",
            "",
            f"Oracle labels: `{json.dumps(self.oracle_label_counts, sort_keys=True)}`",
            "",
            "## Interpretation",
            "",
            "- `hallucinated_observation_count` is a by-construction audit: invalid or ambiguous decisions must not emit observations.",
            "- `false_accept_rate` is empirical: accepted observations are compared against the held-out oracle observation.",
            "- `oracle_answerable_abstain_rate` measures conservative abstention on actions the held-out oracle can answer.",
            "- `reject_correctness` is measured on schema-invalid probes generated from held-out gold actions.",
            "",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ExecutableOracleResult:
    label: str
    answerable: bool
    observation_messages: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutableOracleBackend(Protocol):
    def evaluate_action(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        action: dict[str, Any] | list[dict[str, Any]],
    ) -> ExecutableOracleResult:
        ...


def audit_heldout_oracle(
    episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
    *,
    dataset: str,
    source_path: str = "",
    train_ratio: float = 0.8,
    seed: int = 0,
    max_oracle_episodes: int | None = None,
    include_invalid_probes: bool = True,
    tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
) -> OracleVerificationReport:
    compile_episodes, oracle_episodes = split_episodes(
        list(episodes),
        train_ratio=train_ratio,
        seed=seed,
        max_oracle_episodes=max_oracle_episodes,
    )
    partial_analyzer = VerifiabilityAnalyzer(
        observation_db=ObservationDatabase.from_episodes(compile_episodes),
        allow_step_replay=False,
        tool_semantics=tool_semantics,
    )
    oracle_analyzer = VerifiabilityAnalyzer.from_episodes(
        oracle_episodes, tool_semantics=tool_semantics
    )

    rows: list[OracleVerificationRow] = []
    for episode in oracle_episodes:
        for step in episode.steps:
            if not step.expected_actions:
                continue
            action = _action_payload(step)
            rows.append(
                _verify_action(
                    episode=episode,
                    step=step,
                    action=action,
                    partial_decision=partial_analyzer.classify(episode, step, action),
                    oracle_decision=oracle_analyzer.classify(episode, step, action),
                )
            )
            if include_invalid_probes and len(step.expected_actions) == 1:
                invalid = schema_invalid_probe(step.expected_actions[0])
                if invalid is not None:
                    invalid_payload = invalid.to_dict()
                    rows.append(
                        _verify_action(
                            episode=episode,
                            step=step,
                            action=invalid_payload,
                            partial_decision=partial_analyzer.classify(episode, step, invalid_payload),
                            oracle_decision=oracle_analyzer.classify(episode, step, invalid_payload),
                        )
                    )

    return _report(
        dataset=dataset,
        source_path=source_path,
        oracle_mode="heldout_trajectory_split",
        compile_episode_count=len(compile_episodes),
        oracle_episode_count=len(oracle_episodes),
        rows=rows,
    )


def audit_executable_oracle(
    episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
    *,
    backend: ExecutableOracleBackend,
    dataset: str,
    source_path: str = "",
    oracle_mode: str = "executable",
    train_ratio: float = 0.8,
    seed: int = 0,
    max_oracle_episodes: int | None = None,
    include_invalid_probes: bool = True,
    tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
) -> OracleVerificationReport:
    compile_episodes, oracle_episodes = split_episodes(
        list(episodes),
        train_ratio=train_ratio,
        seed=seed,
        max_oracle_episodes=max_oracle_episodes,
    )
    partial_analyzer = VerifiabilityAnalyzer(
        observation_db=ObservationDatabase.from_episodes(compile_episodes),
        allow_step_replay=False,
        tool_semantics=tool_semantics,
    )
    schema_oracle = VerifiabilityAnalyzer.from_episodes(
        oracle_episodes, tool_semantics=tool_semantics
    )

    rows: list[OracleVerificationRow] = []
    for episode in oracle_episodes:
        for step in episode.steps:
            if not step.expected_actions:
                continue
            action = _action_payload(step)
            oracle_result = backend.evaluate_action(episode, step, action)
            rows.append(
                _verify_action(
                    episode=episode,
                    step=step,
                    action=action,
                    partial_decision=partial_analyzer.classify(episode, step, action),
                    oracle_decision=_decision_from_executable_result(episode, step, oracle_result),
                )
            )
            if include_invalid_probes and len(step.expected_actions) == 1:
                invalid = schema_invalid_probe(step.expected_actions[0])
                if invalid is not None:
                    invalid_payload = invalid.to_dict()
                    rows.append(
                        _verify_action(
                            episode=episode,
                            step=step,
                            action=invalid_payload,
                            partial_decision=partial_analyzer.classify(episode, step, invalid_payload),
                            oracle_decision=schema_oracle.classify(episode, step, invalid_payload),
                        )
                    )

    return _report(
        dataset=dataset,
        source_path=source_path,
        oracle_mode=oracle_mode,
        compile_episode_count=len(compile_episodes),
        oracle_episode_count=len(oracle_episodes),
        rows=rows,
    )


def split_episodes(
    episodes: list[TrajectoryEpisode],
    *,
    train_ratio: float,
    seed: int,
    max_oracle_episodes: int | None = None,
) -> tuple[list[TrajectoryEpisode], list[TrajectoryEpisode]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(episodes) < 2:
        raise ValueError("held-out oracle verification requires at least two episodes")
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
    compile_episodes = shuffled[:train_count]
    oracle_episodes = shuffled[train_count:]
    if max_oracle_episodes is not None:
        oracle_episodes = oracle_episodes[: max(0, max_oracle_episodes)]
    if not oracle_episodes:
        raise ValueError("held-out split produced no oracle episodes")
    return compile_episodes, oracle_episodes


def schema_invalid_probe(action: ToolAction) -> ToolAction | None:
    if not action.arguments:
        return ToolAction(name=action.name, arguments={"__invalid_extra_field__": "invalid"})
    arguments = copy.deepcopy(action.arguments)
    first_key = next(iter(arguments))
    value = arguments[first_key]
    if isinstance(value, str):
        arguments[first_key] = {"invalid_type": value}
    elif isinstance(value, bool):
        arguments[first_key] = "not_a_boolean"
    elif isinstance(value, int):
        arguments[first_key] = "not_an_integer"
    elif isinstance(value, float):
        arguments[first_key] = "not_a_number"
    elif isinstance(value, list):
        arguments[first_key] = {"invalid_type": "list"}
    elif isinstance(value, dict):
        arguments[first_key] = "not_an_object"
    else:
        arguments[first_key] = {"invalid_type": str(value)}
    return ToolAction(name=action.name, arguments=arguments)


def _decision_from_executable_result(
    episode: TrajectoryEpisode,
    step: TrajectoryStep,
    result: ExecutableOracleResult,
) -> VerifiabilityDecision:
    metadata = {"episode_id": episode.id, "step_index": step.index, "step_kind": step.kind, **result.metadata}
    if result.answerable:
        return VerifiabilityDecision(
            label=result.label,
            action_class="valid",
            accepted=True,
            observation_messages=result.observation_messages,
            reason=result.reason,
            metadata=metadata,
        )
    return VerifiabilityDecision(
        label=result.label,
        action_class="ambiguous",
        accepted=False,
        reason=result.reason,
        metadata=metadata,
    )


def _verify_action(
    *,
    episode: TrajectoryEpisode,
    step: TrajectoryStep,
    action: dict[str, Any] | list[dict[str, Any]],
    partial_decision: VerifiabilityDecision,
    oracle_decision: VerifiabilityDecision,
) -> OracleVerificationRow:
    oracle_answerable = oracle_decision.accepted
    observation_match: bool | None = None
    if partial_decision.accepted and oracle_answerable:
        observation_match = _observation_signature(partial_decision.observation_messages) == _observation_signature(
            oracle_decision.observation_messages
        )

    if partial_decision.accepted:
        if oracle_decision.label.endswith("_prefix_error"):
            outcome = "oracle_prefix_error"
        elif not oracle_answerable:
            outcome = "false_accept_oracle_unanswerable"
        elif observation_match:
            outcome = "true_accept"
        else:
            outcome = "false_accept_observation_mismatch"
    elif partial_decision.is_invalid:
        outcome = "correct_reject" if oracle_decision.is_invalid else "incorrect_reject"
    elif partial_decision.is_ambiguous:
        outcome = "oracle_answerable_abstain" if oracle_answerable else "justified_abstain"
    else:
        outcome = "unknown"

    return OracleVerificationRow(
        episode_id=episode.id,
        step_index=step.index,
        action=copy.deepcopy(action),
        partial_label=partial_decision.label,
        partial_action_class=partial_decision.action_class,
        partial_accepted=partial_decision.accepted,
        partial_observation_messages=partial_decision.observation_messages,
        oracle_label=oracle_decision.label,
        oracle_answerable=oracle_answerable,
        oracle_observation_messages=oracle_decision.observation_messages,
        observation_match=observation_match,
        outcome=outcome,
        reason=partial_decision.reason,
        oracle_reason=oracle_decision.reason,
    )


def _report(
    *,
    dataset: str,
    source_path: str,
    oracle_mode: str,
    compile_episode_count: int,
    oracle_episode_count: int,
    rows: list[OracleVerificationRow],
) -> OracleVerificationReport:
    outcomes = Counter(row.outcome for row in rows)
    partial_labels = Counter(row.partial_label for row in rows)
    oracle_labels = Counter(row.oracle_label for row in rows)
    accepted = sum(1 for row in rows if row.partial_accepted)
    false_accept = outcomes["false_accept_oracle_unanswerable"] + outcomes["false_accept_observation_mismatch"]
    rejects = sum(1 for row in rows if row.partial_action_class == "invalid")
    reject_correct = outcomes["correct_reject"]
    invalid_probes = sum(1 for row in rows if row.oracle_label.startswith("invalid_"))
    hallucinated = sum(
        1
        for row in rows
        if row.partial_action_class in {"invalid", "ambiguous"} and row.partial_observation_messages
    )
    return OracleVerificationReport(
        dataset=dataset,
        source_path=source_path,
        oracle_mode=oracle_mode,
        compile_episode_count=compile_episode_count,
        oracle_episode_count=oracle_episode_count,
        candidate_count=len(rows),
        accepted_count=accepted,
        false_accept_count=false_accept,
        oracle_answerable_abstain_count=outcomes["oracle_answerable_abstain"],
        reject_count=rejects,
        reject_correct_count=reject_correct,
        invalid_probe_count=invalid_probes,
        hallucinated_observation_count=hallucinated,
        outcome_counts=dict(sorted(outcomes.items())),
        partial_label_counts=dict(sorted(partial_labels.items())),
        oracle_label_counts=dict(sorted(oracle_labels.items())),
        rows=tuple(rows),
    )


def _action_payload(step: TrajectoryStep) -> dict[str, Any] | list[dict[str, Any]]:
    if len(step.expected_actions) == 1:
        return step.expected_actions[0].to_dict()
    return [action.to_dict() for action in step.expected_actions]


def _observation_signature(observations: tuple[dict[str, Any], ...]) -> str:
    return json.dumps(
        [_normalize_observation_message(message) for message in observations],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_observation_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(message)
    if "content" in normalized:
        normalized["content"] = _normalize_observation_content(normalized.get("content"))
    return _normalize_value(normalized)


def _normalize_observation_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    stripped = content.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return " ".join(stripped.split())
    if isinstance(parsed, dict) and "response" in parsed:
        return parsed.get("response")
    return parsed


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value) if value[key] is not None}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        if re.fullmatch(r"-?\d+\.\d+", value.strip()):
            return float(value.strip())
        return " ".join(value.strip().split())
    return value


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify partial-environment decisions against an oracle.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-oracle-episodes", type=int, default=None)
    parser.add_argument("--no-invalid-probes", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--rows-jsonl", type=Path)
    args = parser.parse_args()

    episodes = load_trajectory_episodes(
        args.data_path,
        source=args.source,
        max_entries=args.max_entries,
        max_tasks=args.max_tasks,
    )
    report = audit_heldout_oracle(
        episodes,
        dataset=args.dataset,
        source_path=args.data_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_oracle_episodes=args.max_oracle_episodes,
        include_invalid_probes=not args.no_invalid_probes,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report.to_dict(include_rows=False), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report.to_markdown(), encoding="utf-8")
    if args.rows_jsonl:
        args.rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_jsonl.open("w", encoding="utf-8") as handle:
            for row in report.rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    if not args.output_json and not args.output_md and not args.rows_jsonl:
        print(report.to_markdown())


if __name__ == "__main__":
    main()
