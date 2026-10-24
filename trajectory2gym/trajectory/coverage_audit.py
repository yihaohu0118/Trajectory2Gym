from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case_study import analyze_episodes
from .loader import load_trajectory_episodes


@dataclass(frozen=True)
class CoverageAudit:
    dataset: str
    source_path: str
    trajectory_count: int
    action_count: int
    strict_replay_accepts: int
    partial_environment_accepts: int
    invalid_rejects: int
    ambiguous_abstains: int
    hallucinated_observation_count: int
    label_counts: dict[str, int]
    accept_equivalent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_path": self.source_path,
            "trajectory_count": self.trajectory_count,
            "action_count": self.action_count,
            "strict_replay_accepts": self.strict_replay_accepts,
            "partial_environment_accepts": self.partial_environment_accepts,
            "invalid_rejects": self.invalid_rejects,
            "ambiguous_abstains": self.ambiguous_abstains,
            "hallucinated_observation_count": self.hallucinated_observation_count,
            "label_counts": self.label_counts,
            "accept_equivalent": self.accept_equivalent,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"# Trajectory2Gym Coverage Audit: {self.dataset}",
                "",
                f"Source: `{self.source_path}`",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| trajectories | {self.trajectory_count} |",
                f"| action candidates | {self.action_count} |",
                f"| strict replay accepts | {self.strict_replay_accepts} |",
                f"| partial environment accepts | {self.partial_environment_accepts} |",
                f"| invalid rejects | {self.invalid_rejects} |",
                f"| ambiguous abstains | {self.ambiguous_abstains} |",
                f"| hallucinated observations | {self.hallucinated_observation_count} |",
                f"| accept equivalent arguments | {str(self.accept_equivalent).lower()} |",
                "",
                f"Runtime labels: `{json.dumps(self.label_counts, sort_keys=True)}`",
                "",
            ]
        )


def audit_case_study(report: Any, *, accept_equivalent: bool = True) -> CoverageAudit:
    labels = Counter(row.runtime_label for row in report.action_rows)
    strict_accepts = labels["valid_replay_exact"]
    partial_accepts = sum(count for label, count in labels.items() if label.startswith("valid_"))
    invalid_rejects = sum(count for label, count in labels.items() if label.startswith("invalid_"))
    ambiguous_abstains = sum(count for label, count in labels.items() if label.startswith("ambiguous_"))
    hallucinated = 0
    for row in report.action_rows:
        if row.classification in {"invalid", "ambiguous"} and row.runtime_feedback == "observation":
            hallucinated += 1
    return CoverageAudit(
        dataset=report.dataset,
        source_path=report.source_path,
        trajectory_count=len(report.trajectory_rows),
        action_count=len(report.action_rows),
        strict_replay_accepts=strict_accepts,
        partial_environment_accepts=partial_accepts,
        invalid_rejects=invalid_rejects,
        ambiguous_abstains=ambiguous_abstains,
        hallucinated_observation_count=hallucinated,
        label_counts=dict(sorted(labels.items())),
        accept_equivalent=accept_equivalent,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict replay and partial-environment coverage.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--max-entries", type=int, default=100)
    parser.add_argument("--source-label", default=None)
    parser.add_argument(
        "--no-accept-equivalent",
        action="store_true",
        help="disable heuristic equivalent-argument replay and report conservative exact/database coverage",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    episodes = load_trajectory_episodes(args.data_path, source=args.source, max_entries=args.max_entries)
    accept_equivalent = not args.no_accept_equivalent
    report = analyze_episodes(
        episodes,
        dataset=args.dataset,
        source_path=args.source_label or args.data_path,
        accept_equivalent=accept_equivalent,
    )
    audit = audit_case_study(report, accept_equivalent=accept_equivalent)
    if args.output_json:
        args.output_json.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(audit.to_markdown(), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(audit.to_markdown())


if __name__ == "__main__":
    main()
