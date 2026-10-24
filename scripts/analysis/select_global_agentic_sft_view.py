#!/usr/bin/env python3
"""Select one benchmark-neutral SFT view from certified semantic trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory2gym.training.global_sft_selection import (
    FORMAT,
    prepare_candidates,
    select_global_candidates,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _token_map(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass" or int(payload.get("truncated_rows", -1)) != 0:
        raise ValueError("token audit is not a passed zero-truncation audit")
    result: dict[str, int] = {}
    for item in payload.get("row_lengths") or []:
        trace = str(item.get("trace_hash") or "")
        tokens = int(item.get("supervised_tokens") or 0)
        if not trace or tokens <= 0 or trace in result:
            raise ValueError("token audit has invalid or duplicate trace identity")
        result[trace] = tokens
    return result


def model_boundary_score(mean_target_nll: float) -> float:
    """Learning-potential proxy shared by every model scale.

    exp(-NLL) is the geometric mean probability of a certified target token.
    The symmetric p(1-p) factor is small for already-saturated and currently
    unreachable targets, and maximal at the imitation boundary.
    """
    if not math.isfinite(mean_target_nll) or mean_target_nll < 0:
        raise ValueError("mean target NLL must be finite and non-negative")
    probability = math.exp(-mean_target_nll)
    return 4.0 * probability * (1.0 - probability)


def capability_aligned_score(percentile: float) -> float:
    """Retain reliable behavior and emphasize the learner's boundary."""
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 1.0:
        raise ValueError("global learner NLL percentile must be in [0,1]")
    return max(1.0 - percentile, 4.0 * percentile * (1.0 - percentile))


def _policy_gap_map(
    path: Path,
    *,
    mode: str = "multiscale_mean_v1",
    model_scale: str | None = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    if mode not in {
        "none_v1",
        "multiscale_mean_v1",
        "multiscale_capability_aligned_v2",
        "model_boundary_v1",
        "capability_aligned_v1",
    }:
        raise ValueError("unsupported policy-gap mode")
    if mode in {"model_boundary_v1", "capability_aligned_v1"} and model_scale not in {"4b", "8b", "14b", "32b"}:
        raise ValueError("model-boundary selection requires a model scale")
    result: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    for row in _read_rows(path):
        if mode == "none_v1":
            raise ValueError("none_v1 does not accept a policy-gap file")
        if row.get("format") != "trajectory2gym_multiscale_certified_action_gap_v1":
            raise ValueError("unsupported multi-scale policy-gap row")
        trace = str(row.get("parent_trace_hash") or "")
        if mode == "multiscale_mean_v1":
            value = float(row.get("multi_scale_policy_gap"))
            detail = {"mode": mode, "value": value}
        elif mode == "multiscale_capability_aligned_v2":
            scales = row.get("scales") or {}
            required_scales = ("4b", "8b", "14b")
            if any(not isinstance(scales.get(scale), dict) for scale in required_scales):
                raise ValueError(
                    "missing frozen 4B/8B/14B percentile for multiscale readout"
                )
            percentiles = {
                scale: float(scales[scale]["global_percentile_rank"])
                for scale in required_scales
            }
            aligned = {
                scale: capability_aligned_score(percentile)
                for scale, percentile in percentiles.items()
            }
            value = sum(aligned.values()) / len(aligned)
            detail = {
                "mode": mode,
                "frozen_scales": list(required_scales),
                "global_percentile_rank": percentiles,
                "per_scale_capability_aligned_score": aligned,
                "multiscale_capability_aligned_score": value,
            }
        elif mode == "model_boundary_v1":
            scale_row = (row.get("scales") or {}).get(model_scale)
            if not isinstance(scale_row, dict):
                raise ValueError(f"missing {model_scale} Base NLL")
            nll = float(scale_row.get("mean_certified_target_nll"))
            value = model_boundary_score(nll)
            detail = {
                "mode": mode,
                "model_scale": model_scale,
                "mean_certified_target_nll": nll,
                "geometric_mean_target_probability": math.exp(-nll),
                "boundary_score": value,
            }
        else:
            scale_row = (row.get("scales") or {}).get(model_scale)
            if not isinstance(scale_row, dict):
                raise ValueError(f"missing {model_scale} Base percentile")
            percentile = float(scale_row.get("global_percentile_rank"))
            value = capability_aligned_score(percentile)
            detail = {
                "mode": mode,
                "model_scale": model_scale,
                "global_percentile_rank": percentile,
                "retention_score": 1.0 - percentile,
                "boundary_score": 4.0 * percentile * (1.0 - percentile),
                "capability_aligned_score": value,
            }
        if not trace or trace in result or not 0.0 <= value <= 1.0:
            raise ValueError("invalid or duplicate multi-scale policy gap")
        result[trace] = value
        metadata[trace] = detail
    return result, metadata


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=12_000_000)
    parser.add_argument("--max-per-task", type=int, default=8)
    parser.add_argument("--max-per-task-teacher", type=int, default=4)
    parser.add_argument("--multiscale-policy-gap", type=Path)
    parser.add_argument("--require-multiscale-policy-gap", action="store_true")
    parser.add_argument(
        "--policy-gap-mode",
        choices=(
            "none_v1",
            "multiscale_mean_v1",
            "multiscale_capability_aligned_v2",
            "model_boundary_v1",
            "capability_aligned_v1",
        ),
        default="none_v1",
    )
    parser.add_argument("--model-scale", choices=("4b", "8b", "14b", "32b"))
    parser.add_argument("--coverage-closure", action="store_true")
    parser.add_argument("--minimum-per-task", type=int, default=0)
    parser.add_argument("--stop-after-closure", action="store_true")
    parser.add_argument("--teacher-cap-only-with-explicit-provenance", action="store_true")
    parser.add_argument("--maximum-selected-trajectories", type=int)
    parser.add_argument(
        "--loop-redundancy-mode",
        choices=("coarse_signature_v1", "certified_exact_adjacent_v2"),
        default="coarse_signature_v1",
    )
    parser.add_argument(
        "--path-family-mode",
        choices=("declared_or_task_v1", "capability_path_v2"),
        default="declared_or_task_v1",
    )
    parser.add_argument("--max-per-capability-path", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.input_dir / "manifest.json"
    train_path = args.input_dir / "train.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_configs = [
        item.get("selection_config") or {}
        for item in manifest.get("sources") or []
    ]
    if not (
        manifest.get("status") == "pass"
        and manifest.get("training_objective") == "assistant_token_causal_cross_entropy"
        and (manifest.get("certified_policy_target") or {}).get("success_only") is True
        and source_configs
        and all(config.get("selection_mode") == "candidate_pool" for config in source_configs)
    ):
        raise ValueError(
            "input is not a preselection-free environment-certified candidate union"
        )
    rows = _read_rows(train_path)
    gap_sha256 = None
    if args.policy_gap_mode == "none_v1" and args.multiscale_policy_gap is not None:
        raise ValueError("none_v1 forbids a policy-gap input")
    if args.multiscale_policy_gap is not None:
        gap, gap_metadata = _policy_gap_map(
            args.multiscale_policy_gap,
            mode=args.policy_gap_mode,
            model_scale=args.model_scale,
        )
        row_traces = {str(row.get("trace_hash") or "") for row in rows}
        if set(gap) != row_traces:
            raise ValueError("multi-scale policy-gap coverage differs from candidates")
        rows = [
            {
                **row,
                "multi_scale_policy_gap": gap[str(row["trace_hash"])],
                "model_conditioned_selection": gap_metadata[str(row["trace_hash"])],
            }
            for row in rows
        ]
        gap_sha256 = _sha(args.multiscale_policy_gap)
    elif args.require_multiscale_policy_gap:
        raise ValueError("required multi-scale policy-gap file is missing")
    tokens = _token_map(args.token_audit)
    candidates, preparation = prepare_candidates(
        rows,
        supervised_tokens=tokens,
        require_multiscale_policy_gap=args.require_multiscale_policy_gap,
        loop_redundancy_mode=args.loop_redundancy_mode,
        path_family_mode=args.path_family_mode,
    )
    selected, audit, ledger = select_global_candidates(
        candidates,
        token_budget=args.token_budget,
        max_per_task=args.max_per_task,
        max_per_task_teacher=args.max_per_task_teacher,
        policy_gap_enabled=args.require_multiscale_policy_gap,
        coverage_closure=args.coverage_closure,
        minimum_per_task=args.minimum_per_task,
        fill_remaining=not args.stop_after_closure,
        teacher_cap_requires_explicit_provenance=args.teacher_cap_only_with_explicit_provenance,
        max_selected_trajectories=args.maximum_selected_trajectories,
        max_per_capability_path=args.max_per_capability_path,
    )
    audit["policy_gap_mode"] = args.policy_gap_mode
    audit["model_scale"] = args.model_scale
    temporary = args.output_dir.with_name(f".{args.output_dir.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        _write_jsonl(temporary / "train.jsonl", selected)
        _write_jsonl(temporary / "validation.jsonl", [])
        _write_jsonl(temporary / "selection_ledger.jsonl", ledger)
        output_manifest = {
            "format": "trajectory2gym_global_selected_semantic_sft_manifest_v1",
            "status": "pass",
            "method_role": "unified_active_interaction_synthesis",
            "config": {
                "seed": 0,
                "validation_fraction": 0.0,
                "max_traces_per_task": args.max_per_task,
                "success_only": True,
                "assume_success_when_missing": False,
                "enable_thinking": False,
                "supervision_mode": "verified_agentic",
                "selection_mode": "global_learning_value",
                "loop_redundancy_mode": args.loop_redundancy_mode,
                "path_family_mode": args.path_family_mode,
                "max_per_capability_path": args.max_per_capability_path,
                "policy_gap_mode": args.policy_gap_mode,
                "model_scale": args.model_scale,
                "coverage_closure": args.coverage_closure,
                "minimum_per_task": args.minimum_per_task,
            },
            "selection_format": FORMAT,
            "training_objective": "assistant_token_causal_cross_entropy",
            "new_optimizer_claim": False,
            "certified_policy_target": {
                "success_only": True,
                "supervision_mode": "verified_agentic",
                "enable_thinking": False,
                "complete_trajectory_context": True,
                "user_tool_external_messages_are_context_only": True,
                "rejected_error_action_is_context_only": True,
                "environment_observation_or_official_action_certificate_required": True,
            },
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": _sha(manifest_path),
            "source_train_sha256": _sha(train_path),
            "source_token_audit": str(args.token_audit.resolve()),
            "source_token_audit_sha256": _sha(args.token_audit),
            "multiscale_policy_gap": (
                str(args.multiscale_policy_gap.resolve())
                if args.multiscale_policy_gap is not None
                else None
            ),
            "multiscale_policy_gap_sha256": gap_sha256,
            "model_conditioned_selection": args.policy_gap_mode == "model_boundary_v1",
            "primary_training_authorized": True,
            "preparation_audit": preparation,
            "selection_audit": audit,
            "train_rows": len(selected),
            "validation_rows": 0,
            "train_rows_by_corpus": dict(
                sorted(Counter(str(row.get("source_corpus") or "") for row in selected).items())
            ),
            "train_rows_by_domain": dict(
                sorted(Counter(str(row.get("domain") or "") for row in selected).items())
            ),
            "validation_rows_by_corpus": {},
            "validation_rows_by_domain": {},
            "missing_schema_domains": [],
            "task_overlap": [],
            "family_overlap": [],
            "train_sha256": _sha(temporary / "train.jsonl"),
            "selection_ledger_sha256": _sha(temporary / "selection_ledger.jsonl"),
            "enable_thinking": False,
            "benchmark_specific_selection": False,
            "benchmark_specific_loss_weights": False,
            "all_selected_rows_used_for_training": True,
            "candidate_pool_preselection_free": True,
            "non_mainline_method_deviation": manifest.get(
                "non_mainline_method_deviation"
            )
            is True,
            "included_corpora": manifest.get("included_corpora"),
            "excluded_corpora": manifest.get("excluded_corpora"),
            "official_evaluation_information_used": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "SELECTION_READY").touch()
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(output_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
