#!/usr/bin/env python3
"""Publish an immutable train-ready view after transport and token gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity-contract", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.input_dir / "manifest.json"
    train_path = args.input_dir / "train.jsonl"
    validation_path = args.input_dir / "validation.jsonl"
    audit_path = args.input_dir / "token_audit.json"
    readiness_path = args.input_dir / "selection_readiness.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("format") == "trajectory2gym_global_selected_rendered_sft_manifest_v1"
        and manifest.get("status") == "pass"
        and manifest.get("benchmark_specific_selection") is False
        and manifest.get("benchmark_specific_loss_weights") is False
        and (manifest.get("bfcl_transport_projection") or {}).get("status") == "pass"
    ):
        raise ValueError("input is not a passed globally selected rendered view")
    selection = manifest.get("selection_audit") or {}
    policy_gap_mode = selection.get("policy_gap_mode")
    capacity_contract = None
    if args.capacity_contract is not None:
        capacity_contract = json.loads(args.capacity_contract.read_text())
        scale = str((manifest.get("config") or {}).get("model_scale") or "")
        if not (
            capacity_contract.get("format")
            in {
                "trajectory2gym_capacity_conditioned_three_source_primary_v1",
                "trajectory2gym_capacity_conditioned_three_source_primary_v2",
                "trajectory2gym_capacity_conditioned_three_source_primary_v3",
            }
            and capacity_contract.get("status") == "frozen"
            and capacity_contract.get("active_training_sources")
            == ["bfcl", "tau1", "tau2"]
            and set(manifest.get("train_rows_by_corpus") or {})
            == {"bfcl", "tau1", "tau2"}
            and (capacity_contract.get("tau3") or {}).get("training_paused") is True
        ):
            raise ValueError("capacity-conditioned three-source contract mismatch")
        budget_law = capacity_contract["capacity_budget_law"]
        if budget_law.get("unit") == "selected semantic trajectories":
            expected_budget = int(budget_law["candidate_token_ceiling"])
            if int(selection.get("max_selected_trajectories") or 0) != int(
                budget_law["budgets"][scale]
            ):
                raise ValueError("semantic-parent capacity budget mismatch")
        else:
            expected_budget = int(budget_law["budgets"][scale])
    else:
        expected_budget = {
            "none_v1": 16_000_000,
            "required_multiscale": 12_000_000,
            "multiscale_capability_aligned_v2": 8_000_000,
            "model_boundary_v1": 8_000_000,
            "capability_aligned_v1": 8_000_000,
        }.get(policy_gap_mode)
    if not (
        selection.get("format") == "trajectory2gym_global_sft_selection_v1"
        and expected_budget is not None
        and int(selection.get("token_budget") or 0) == expected_budget
        and (
            policy_gap_mode not in {"model_boundary_v1", "capability_aligned_v1"}
            or (
                (manifest.get("config") or {}).get("policy_gap_mode")
                == policy_gap_mode
                and (manifest.get("config") or {}).get("model_scale")
                in {"4b", "8b", "14b", "30b-a3b"}
            )
        )
    ):
        raise ValueError("global selection receipt is incomplete")
    if not (
        audit.get("format") == "trajectory2gym_sft_token_audit_v0"
        and audit.get("status") == "pass"
        and int(audit.get("truncated_rows", -1)) == 0
        and int(audit.get("rows", -1)) == int(manifest["train_rows"])
        and audit.get("input_sha256") == _sha(train_path)
    ):
        raise ValueError("rendered token audit is missing, truncated, or unbound")
    if not (
        readiness.get("format")
        == "trajectory2gym_unified_sft_selection_readiness_v1"
        and readiness.get("status") == "pass"
        and readiness.get("training_permitted") is True
        and all((readiness.get("checks") or {}).values())
    ):
        raise ValueError("global selection readiness gate did not pass")
    rendered_tokens = int(audit["supervised_tokens"]["total"])
    semantic_tokens = int(selection["selected_supervised_tokens"])
    projection = manifest.get("bfcl_transport_projection") or {}
    if projection.get("effective_supervised_decisions_preserved") is not True:
        raise ValueError("transport did not preserve every supervised decision")
    if validation_path.stat().st_size != 0:
        raise ValueError("all globally selected rows must be used for training")
    temporary = args.output_dir.with_name(f".{args.output_dir.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        _link(train_path, temporary / "train.jsonl")
        _link(validation_path, temporary / "validation.jsonl")
        _link(audit_path, temporary / "token_audit.json")
        _link(readiness_path, temporary / "selection_readiness.json")
        output_manifest = {
            **manifest,
            "parent_manifest": str(manifest_path.resolve()),
            "parent_manifest_sha256": _sha(manifest_path),
            "train_sha256": _sha(temporary / "train.jsonl"),
            "validation_sha256": _sha(temporary / "validation.jsonl"),
            "token_audit": str((args.output_dir / "token_audit.json").resolve()),
            "token_audit_sha256": _sha(temporary / "token_audit.json"),
            "selection_readiness": str(
                (args.output_dir / "selection_readiness.json").resolve()
            ),
            "selection_readiness_sha256": _sha(
                temporary / "selection_readiness.json"
            ),
            "semantic_to_transport_token_accounting": {
                "format": "trajectory2gym_semantic_to_transport_token_accounting_v1",
                "status": "pass",
                "semantic_supervised_tokens": semantic_tokens,
                "rendered_supervised_tokens": rendered_tokens,
                "difference": rendered_tokens - semantic_tokens,
                "exact_token_count_conservation_required": False,
                "effective_supervised_decisions_preserved": True,
                "optimizer_mass_conservation_pending": True,
            },
            "all_selected_rows_used_for_training": True,
            "training_ready": True,
        }
        if capacity_contract is not None:
            output_manifest.update(
                method_contract_id=capacity_contract["format"],
                capacity_conditioned_primary_contract=str(
                    args.capacity_contract.resolve()
                ),
                capacity_conditioned_primary_contract_sha256=_sha(
                    args.capacity_contract
                ),
                active_training_sources=["bfcl", "tau1", "tau2"],
                excluded_training_sources=["tau3"],
            )
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "DATA_READY").touch()
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            output_manifest["semantic_to_transport_token_accounting"],
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
