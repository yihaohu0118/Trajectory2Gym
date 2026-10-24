from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case_study import ActionCaseRow, analyze_episodes
from .loader import load_trajectory_episodes


@dataclass(frozen=True)
class UnsafeSimulatorAblation:
    dataset: str
    source_path: str
    action_count: int
    verified_accepts: int
    ambiguous_abstains: int
    unsafe_accepts: int
    fabricated_observations: int
    hallucination_rate_among_unsafe_accepts: float
    label_counts: dict[str, int]
    fabricated_label_counts: dict[str, int]
    fabricated_tool_type_counts: dict[str, int]
    fabricated_examples: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_path": self.source_path,
            "action_count": self.action_count,
            "verified_accepts": self.verified_accepts,
            "ambiguous_abstains": self.ambiguous_abstains,
            "unsafe_accepts": self.unsafe_accepts,
            "fabricated_observations": self.fabricated_observations,
            "hallucination_rate_among_unsafe_accepts": self.hallucination_rate_among_unsafe_accepts,
            "label_counts": self.label_counts,
            "fabricated_label_counts": self.fabricated_label_counts,
            "fabricated_tool_type_counts": self.fabricated_tool_type_counts,
            "fabricated_examples": list(self.fabricated_examples),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Trajectory2Gym Unsafe Simulator Ablation: {self.dataset}",
            "",
            f"Source: `{self.source_path}`",
            "",
            "This is a negative-control audit. It does not run an LLM simulator.",
            "It asks what would happen if every ambiguous abstain were accepted",
            "with a fabricated observation.",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| action candidates | {self.action_count} |",
            f"| verified partial-environment accepts | {self.verified_accepts} |",
            f"| ambiguous abstains | {self.ambiguous_abstains} |",
            f"| hypothetical unsafe accepts | {self.unsafe_accepts} |",
            f"| fabricated observations | {self.fabricated_observations} |",
            f"| hallucination rate among unsafe accepts | {self.hallucination_rate_among_unsafe_accepts:.4f} |",
            "",
            f"Runtime labels: `{json.dumps(self.label_counts, sort_keys=True)}`",
            "",
            f"Fabricated labels: `{json.dumps(self.fabricated_label_counts, sort_keys=True)}`",
            "",
            f"Fabricated tool types: `{json.dumps(self.fabricated_tool_type_counts, sort_keys=True)}`",
            "",
        ]
        if self.fabricated_examples:
            lines.extend(
                [
                    "## Fabrication Examples",
                    "",
                    "| trajectory | step | tool | type | label | evidence |",
                    "| --- | ---: | --- | --- | --- | --- |",
                ]
            )
            for item in self.fabricated_examples:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(item["trajectory_id"]),
                            str(item["step_index"]),
                            str(item["tool_name"]),
                            str(item["tool_type"]),
                            str(item["runtime_label"]),
                            str(item["evidence"]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        return "\n".join(lines)


def audit_unsafe_simulator(report: Any, *, example_limit: int = 12) -> UnsafeSimulatorAblation:
    labels = Counter(row.runtime_label for row in report.action_rows)
    verified_accepts = sum(count for label, count in labels.items() if label.startswith("valid_"))
    ambiguous_rows = [row for row in report.action_rows if row.runtime_label.startswith("ambiguous_")]
    fabricated_observations = len(ambiguous_rows)
    unsafe_accepts = verified_accepts + fabricated_observations
    hallucination_rate = fabricated_observations / unsafe_accepts if unsafe_accepts else 0.0
    fabricated_label_counts = Counter(row.runtime_label for row in ambiguous_rows)
    fabricated_tool_type_counts = Counter(row.tool_type for row in ambiguous_rows)
    examples = tuple(_example(row) for row in ambiguous_rows[:example_limit])
    return UnsafeSimulatorAblation(
        dataset=report.dataset,
        source_path=report.source_path,
        action_count=len(report.action_rows),
        verified_accepts=verified_accepts,
        ambiguous_abstains=fabricated_observations,
        unsafe_accepts=unsafe_accepts,
        fabricated_observations=fabricated_observations,
        hallucination_rate_among_unsafe_accepts=hallucination_rate,
        label_counts=dict(sorted(labels.items())),
        fabricated_label_counts=dict(sorted(fabricated_label_counts.items())),
        fabricated_tool_type_counts=dict(sorted(fabricated_tool_type_counts.items())),
        fabricated_examples=examples,
    )


def _example(row: ActionCaseRow) -> dict[str, Any]:
    return {
        "trajectory_id": row.trajectory_id,
        "step_index": row.step_index,
        "tool_name": row.tool_name,
        "tool_type": row.tool_type,
        "runtime_label": row.runtime_label,
        "evidence": row.evidence,
        "arguments": row.arguments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the unsafe hallucinated-simulator negative control.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--max-entries", type=int, default=100)
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    episodes = load_trajectory_episodes(args.data_path, source=args.source, max_entries=args.max_entries)
    report = analyze_episodes(episodes, dataset=args.dataset, source_path=args.source_label or args.data_path)
    audit = audit_unsafe_simulator(report, example_limit=args.example_limit)
    if args.output_json:
        args.output_json.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(audit.to_markdown(), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(audit.to_markdown())


if __name__ == "__main__":
    main()
