import gzip
import hashlib
import json
import pytest

from scripts.release.export_policy_visible_jsonl import export, project
from scripts.release.release_utils import validate_manifest


def input_file(tmp_path, content="done"):
    row = {"trace_hash": "demo", "source_corpus": "synthetic",
           "source": "operational-metadata", "success": True, "enable_thinking": False,
           "messages": [{"role": "user", "content": "hello"},
                        {"role": "assistant", "content": content}],
           "tools": [], "supervised_assistant_turn_indices": [0]}
    data = (json.dumps(row) + "\n").encode()
    source = tmp_path / "source.jsonl"
    source.write_bytes(data)
    return source, hashlib.sha256(data).hexdigest(), row


def test_export_preserves_payload_and_removes_metadata(tmp_path):
    source, sha, row = input_file(tmp_path)
    output = tmp_path / "export"
    export(source, output, sha, 1)
    with gzip.open(output / "data/train-00000.jsonl.gz", "rt") as f:
        result = json.loads(f.readline())
    assert "source" not in result
    assert result["messages"] == row["messages"]
    assert [list(m) for m in result["messages"]] == [list(m) for m in row["messages"]]
    assert result["supervised_assistant_turn_indices"] == [0]
    assert validate_manifest(output)["kind"] == "dataset"


def test_wrong_hash_never_publishes_export(tmp_path):
    source, _, _ = input_file(tmp_path)
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        export(source, output, "0" * 64, 1)
    assert not output.exists()


def test_secret_in_message_blocks_export(tmp_path):
    source, sha, _ = input_file(tmp_path, "hf_" + "a" * 30)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="credential_token"):
        export(source, output, sha, 1)
    assert not output.exists()


def test_existing_export_is_never_overwritten(tmp_path):
    source, sha, _ = input_file(tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export(source, output, sha, 1)


def test_reviewed_native_literal_is_exact_and_source_bound(tmp_path):
    literal = ".".join(["192", "168", "1", "1"])
    source, sha, row = input_file(tmp_path, "Synthetic router: " + literal)
    review = {"format": "trajectory2gym_native_literal_review_v1",
              "source_input_sha256": sha, "rationale": "Synthetic test fixture",
              "allowed_sha256_by_rule": {"private_network": [hashlib.sha256(literal.encode()).hexdigest()]}}
    output = tmp_path / "export"
    export(source, output, sha, 1, literal_review=review)
    manifest = validate_manifest(output)
    assert manifest["automated_scan"]["reviewed_native_literal_occurrences"] == 1
    review["source_input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        export(source, tmp_path / "other", sha, 1, literal_review=review)


def test_review_cannot_exempt_credentials(tmp_path):
    source, sha, _ = input_file(tmp_path)
    review = {"format": "trajectory2gym_native_literal_review_v1",
              "source_input_sha256": sha, "rationale": "Not permitted",
              "allowed_sha256_by_rule": {"credential_token": ["0" * 64]}}
    with pytest.raises(ValueError):
        export(source, tmp_path / "export", sha, 1, literal_review=review)


def test_nullable_reasoning_field_is_preserved(tmp_path):
    _, _, row = input_file(tmp_path)
    row["messages"][1]["reasoning_content"] = None
    assert project(row, 0)["messages"] == row["messages"]


def test_nonempty_reasoning_field_is_rejected(tmp_path):
    _, _, row = input_file(tmp_path)
    row["messages"][1]["reasoning_content"] = "private fixture text"
    with pytest.raises(ValueError, match="reasoning content"):
        project(row, 0)
