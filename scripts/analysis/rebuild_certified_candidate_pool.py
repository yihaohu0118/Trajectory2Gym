#!/usr/bin/env python3
"""Recompile every evidence-eligible success from an existing source manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory2gym.training.sft import (
    SFTBuildConfig,
    build_sft_dataset,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    config = source.get("config") or {}
    inputs = [
        (
            item.get("domain_override"),
            Path(item["path"]),
        )
        for item in source.get("source_inputs") or []
    ]
    schemas = [Path(item["path"]) for item in source.get("schema_inputs") or []]
    if not inputs or not schemas:
        raise ValueError("source manifest lacks reproducible source/schema inputs")
    missing = [str(path) for _, path in inputs if not path.exists()]
    missing.extend(str(path) for path in schemas if not path.exists())
    if missing:
        raise FileNotFoundError(f"candidate source inputs are missing: {missing[:8]}")
    candidate_config = SFTBuildConfig(
        seed=0,
        validation_fraction=0.0,
        max_traces_per_task=1,
        success_only=True,
        assume_success_when_missing=False,
        enable_thinking=False,
        supervision_mode="verified_agentic",
        selection_mode="candidate_pool",
        include_successful_policy_anchors=False,
        policy_frontier_only=False,
        policy_frontier_loss_only=False,
        require_supported_references=bool(config.get("require_supported_references")),
    )
    train, validation, manifest = build_sft_dataset(
        inputs,
        schema_sources=schemas,
        config=candidate_config,
    )
    assert not validation
    assert manifest["selection_audit"]["training_permitted"] is False
    manifest["candidate_pool_parent_manifest"] = str(args.source_manifest.resolve())
    manifest["candidate_pool_parent_selection_mode"] = config.get("selection_mode")
    manifest["candidate_pool_reconstructed_before_learning_value_selection"] = True
    temporary = args.output_dir.with_name(f".{args.output_dir.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        write_jsonl(temporary / "train.jsonl", train)
        write_jsonl(temporary / "validation.jsonl", validation)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "CANDIDATE_POOL_READY").touch()
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"rows": len(train), "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
