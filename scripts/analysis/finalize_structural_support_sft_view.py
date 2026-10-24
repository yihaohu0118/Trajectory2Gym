#!/usr/bin/env python3
"""Publish an immutable train-ready structural-support SFT transport view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def finalize(semantic: Path, rendered: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    sm = json.loads((semantic / "manifest.json").read_text())
    rm = json.loads((rendered / "manifest.json").read_text())
    token = json.loads((rendered / "token_audit.json").read_text())
    nt = json.loads((rendered / "nonthinking_audit.json").read_text())
    cert = json.loads((rendered / "certified_target_audit.json").read_text())
    ready = json.loads((rendered / "selection_readiness.json").read_text())
    selection = sm.get("selection_audit") or {}
    if not (
        sm.get("format") == "trajectory2gym_structural_support_semantic_manifest_v1"
        and sm.get("status") == "pass"
        and sm.get("training_ready") is False
        and sm.get("base_nll_used") is False
        and sm.get("official_evaluation_information_used") is False
        and sm.get("benchmark_specific_selection") is False
        and sm.get("benchmark_specific_loss_weights") is False
        and selection.get("nll_used") is False
        and selection.get("paths_selected") == selection.get("paths_candidate")
        and selection.get("tasks_selected") == selection.get("tasks_candidate")
        and selection.get("capability_atoms_selected") == selection.get("capability_atoms_candidate")
    ):
        raise ValueError("semantic structural-support gate failed")
    if not (
        rm.get("format") == "trajectory2gym_global_selected_rendered_sft_manifest_v1"
        and rm.get("status") == "pass"
        and rm.get("selection_audit") == selection
        and rm.get("train_sha256") == _sha(rendered / "train.jsonl")
        and (rm.get("bfcl_transport_projection") or {}).get("status") == "pass"
    ):
        raise ValueError("official transport gate failed")
    if not (
        token.get("format") == "trajectory2gym_sft_token_audit_v0"
        and token.get("status") == "pass"
        and int(token.get("truncated_rows") or 0) == 0
        and int(token.get("rows") or -1) == int(rm.get("train_rows") or -2)
        and token.get("input_sha256") == _sha(rendered / "train.jsonl")
        and nt.get("status") == "pass"
        and int(nt.get("violation_count") or 0) == 0
        and cert.get("status") == "pass"
        and int(cert.get("violation_count") or 0) == 0
        and ready.get("status") == "pass"
        and ready.get("training_permitted") is True
        and all((ready.get("checks") or {}).values())
        and (rendered / "validation.jsonl").stat().st_size == 0
    ):
        raise ValueError("token, target, or readiness gate failed")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "token_audit.json",
            "nonthinking_audit.json",
            "certified_target_audit.json",
            "selection_readiness.json",
        ):
            _link(rendered / name, temporary / name)
        active_sources = sorted(
            set(sm.get("active_training_sources") or rm.get("active_training_sources") or [])
        )
        if not active_sources:
            raise ValueError("semantic/rendered view does not declare active sources")
        manifest = {
            **rm,
            "method_contract_id": "structural_support_scaling_v1",
            "method_role": "certified_retention_task_capability_and_ordered_path_support",
            "semantic_selection": str(semantic.resolve()),
            "semantic_selection_sha256": sm["train_sha256"],
            "parent_manifest": str((rendered / "manifest.json").resolve()),
            "parent_manifest_sha256": _sha(rendered / "manifest.json"),
            "train_sha256": _sha(temporary / "train.jsonl"),
            "validation_sha256": _sha(temporary / "validation.jsonl"),
            "token_audit": str((output / "token_audit.json").resolve()),
            "selection_readiness": str((output / "selection_readiness.json").resolve()),
            "active_training_sources": active_sources,
            "excluded_training_sources": sorted(
                {"bfcl", "tau1", "tau2", "tau3"} - set(active_sources)
            ),
            "missing_schema_domains": [],
            "task_overlap": [],
            "primary_training_authorized": True,
            "non_mainline_method_deviation": False,
            "training_objective": "assistant_token_causal_cross_entropy",
            "row_loss_weight_field": "",
            "training_ready": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "DATA_READY").touch()
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "pass",
        "scale": selection["scale"],
        "semantic_parents": selection["semantic_parents"],
        "rendered_rows": rm["train_rows"],
        "certified_target_tokens": token["supervised_tokens"]["total"],
        "train_sha256": manifest["train_sha256"],
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.semantic_dir, args.rendered_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
