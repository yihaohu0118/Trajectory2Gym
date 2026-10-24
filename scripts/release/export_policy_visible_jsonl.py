"""Export a frozen certified candidate file without operational metadata."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.release_utils import (
    MANIFEST, LITERAL_REVIEW, RULES, create_manifest, is_reviewed_literal, parse_literal_review,
)

KEEP = {
    "format", "trace_hash", "task_id", "source_task_id", "source_corpus",
    "source_domain", "domain", "split", "success", "enable_thinking", "messages",
    "tools", "supervised_assistant_turn_indices", "verified_assistant_turn_indices",
    "assistant_turns", "supervised_assistant_turns", "context_assistant_turns",
    "trajectory_role", "trajectory_family_id",
}
MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id", "function_call", "reasoning_content"}
ROLES = {"system", "user", "assistant", "tool"}


def project(row: dict, ordinal: int, reviewed: dict | None = None) -> dict:
    if row.get("success") is not True or row.get("enable_thinking") is not False:
        raise ValueError(f"Row {ordinal}: expected successful non-thinking source")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row {ordinal}: messages missing")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise ValueError(f"Row {ordinal}: unsupported message role")
        if set(message) - MESSAGE_KEYS:
            raise ValueError(f"Row {ordinal}: non-standard message metadata needs review")
        if message.get("reasoning_content") not in (None, ""):
            raise ValueError(f"Row {ordinal}: non-empty reasoning content needs review")
        if not isinstance(message.get("content"), (str, type(None))):
            raise ValueError(f"Row {ordinal}: non-text content needs review")
    turns = sum(m["role"] == "assistant" for m in messages)
    indices = row.get("supervised_assistant_turn_indices")
    if (not isinstance(indices, list) or not indices or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < turns for i in indices)):
        raise ValueError(f"Row {ordinal}: invalid supervision indices")
    if not isinstance(row.get("trace_hash"), str) or not row["trace_hash"]:
        raise ValueError(f"Row {ordinal}: source trace identity missing")
    result = {k: row[k] for k in sorted(KEEP & set(row))}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for name, pattern in RULES.items():
        if any(not is_reviewed_literal(name, m.group(), reviewed or {})
               for m in pattern.finditer(encoded)):
            raise ValueError(f"Row {ordinal}: publication rule {name} requires review")
    return result


def export(source: Path, output: Path, expected_sha256: str, expected_rows: int,
           shard_rows: int = 5000, literal_review: dict | None = None) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new export directory")
    if source.is_symlink() or not source.is_file():
        raise ValueError("Source must be a regular JSONL file")
    if shard_rows < 1 or expected_rows < 1 or len(expected_sha256) != 64:
        raise ValueError("Explicit valid source hash and row count are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = source.stat()
    reviewed = parse_literal_review(literal_review, expected_sha256) if literal_review else {}
    digest = hashlib.sha256()
    count, sources, identities = 0, Counter(), set()
    with tempfile.TemporaryDirectory(prefix=output.name + ".staging-", dir=output.parent) as temporary:
        staging = Path(temporary)
        payload = staging / "payload"
        (payload / "data").mkdir(parents=True)
        stream = None
        try:
            with source.open("rb") as reader:
                for line in reader:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = project(json.loads(line), count, reviewed)
                    if row["trace_hash"] in identities:
                        raise ValueError("Duplicate source trace identity")
                    identities.add(row["trace_hash"])
                    sources[str(row.get("source_corpus", "unknown"))] += 1
                    if count % shard_rows == 0:
                        if stream is not None:
                            stream.close()
                        path = payload / "data" / f"train-{count // shard_rows:05d}.jsonl.gz"
                        stream = gzip.GzipFile(filename=str(path), mode="wb", compresslevel=3, mtime=0)
                    stream.write((json.dumps(row, ensure_ascii=False,
                                             separators=(",", ":")) + "\n").encode("utf-8"))
                    count += 1
                    if count % shard_rows == 0:
                        print(json.dumps({"phase": "export", "rows": count}), flush=True)
        finally:
            if stream is not None:
                stream.close()
        final = source.stat()
        if (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns):
            raise ValueError("Source changed during export")
        if digest.hexdigest() != expected_sha256 or count != expected_rows:
            raise ValueError("Frozen source hash or row count mismatch")
        provenance = {
            "format": "trajectory2gym_policy_visible_export_v1",
            "source_input_sha256": digest.hexdigest(), "rows": count,
            "rows_by_source": dict(sorted(sources.items())),
            "kept_top_level_fields": sorted(KEEP),
            "messages_tools_and_supervision_indices": "preserved without modification",
            "removed": "non-allowlisted top-level operational/provenance metadata",
            "trace_hash_semantics": "identifier inherited from source, not a hash of the exported JSON row",
            "public_data_review_required": True,
        }
        (payload / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        if literal_review:
            (payload / LITERAL_REVIEW).write_text(json.dumps(literal_review, indent=2, sort_keys=True) + "\n")
        card = (
            "---\nlanguage:\n- en\ntask_categories:\n- text-generation\n"
            "tags:\n- tool-use\n- trajectories\n- private-staging\n"
            "configs:\n- config_name: semantic_candidates\n  data_files:\n"
            "  - split: train\n    path: data/*.jsonl.gz\n---\n\n"
            "# Trajectory2Gym: policy-visible candidate interactions\n\n"
            f"This private staging release contains {count:,} complete candidate interactions. "
            "These are semantic parents, not the rendered SFT examples or a final scale-specific view.\n\n"
            "Messages, tool schemas, and supervised assistant-turn indices are inherited unchanged "
            "from the frozen source. Top-level operational metadata and private source paths are omitted. "
            "The export does not distribute all native reset states or execution certificates and is "
            "not a standalone native-environment replay package.\n\n"
            "## Provenance and training scope\n\n"
            f"Original input SHA-256: {digest.hexdigest()}\n\n"
            "See provenance.json for source counts and the export schema, and RELEASE_MANIFEST.json "
            "for hashes of the exported files. Exported bytes differ from the original training input.\n\n"
            "The corpus is training-side construction data. Tau1 Airline has the disclosed seen-task "
            "scope; Tau2 construction uses its official train partition. This split label does not "
            "assert template-level separation or unseen-task generalization for Tau1 Airline.\n\n"
            "## Content and redistribution review\n\n"
            "Native environment entities and user/tool text are preserved because they define "
            "observation-to-action dependencies. Pattern scanning is not a complete personal-data audit. "
            "Keep this repository private until content and applicable upstream data licenses are reviewed. "
            "No blanket data license is assigned by the code license.\n"
        )
        if literal_review:
            card += ("\nBenchmark-native address literals and generic path placeholders were retained "
                     "through a source-bound, exact-value-hash review. See NATIVE_LITERAL_REVIEW.json. "
                     "Credential and internal-host rules cannot be exempted. This is not an independent human privacy audit.\n")
        (payload / "README.md").write_text(card, encoding="utf-8")
        manifest = create_manifest(payload, "dataset")
        (payload / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(payload, output)
    return {"rows": count, "shards": (count + shard_rows - 1) // shard_rows,
            "source_input_sha256": expected_sha256, "uploaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--shard-rows", type=int, default=5000)
    parser.add_argument("--literal-review", type=Path,
                        help="Optional exact-value review bound to the frozen input hash; never exempts credentials")
    args = parser.parse_args()
    review = json.loads(args.literal_review.read_text()) if args.literal_review else None
    print(json.dumps(export(args.source, args.output, args.expected_sha256,
                            args.expected_rows, args.shard_rows, review)), flush=True)


if __name__ == "__main__":
    main()
