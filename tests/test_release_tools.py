import json
import tarfile
from types import SimpleNamespace

import pytest

from scripts.release.release_utils import MANIFEST, create_manifest, scan, validate_manifest
from scripts.release.build_code_archive import build
from scripts.release.upload_private_dataset import publish


def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "README.md").write_text("# Synthetic test data\n")
    (root / "train.jsonl").write_text('{"messages": [], "synthetic_test_only": true}\n')
    (root / MANIFEST).write_text(json.dumps(create_manifest(root, "dataset")))
    return root


def test_manifest_checks_bytes(tmp_path):
    root = dataset(tmp_path)
    assert validate_manifest(root)["kind"] == "dataset"
    (root / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        validate_manifest(root)


def test_extra_file_rejected(tmp_path):
    root = dataset(tmp_path)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_manifest(root)


@pytest.mark.parametrize("name", [".env", ".env.local", "credentials.json", "id_rsa", "private.pem"])
def test_sensitive_filenames_rejected_without_content_read(tmp_path, name):
    (tmp_path / name).write_text("placeholder")
    with pytest.raises(ValueError):
        create_manifest(tmp_path, "code")


def test_secret_findings_are_value_redacted(tmp_path):
    secret = "hf_" + "x" * 30
    (tmp_path / "example.py").write_text("value = " + repr(secret))
    report = scan(tmp_path)
    assert report["status"] == "needs_review"
    assert secret not in json.dumps(report)
    assert report["findings"][0]["rule"] == "credential_token"


def test_symlink_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    original = tmp_path / "original.py"
    original.write_text("pass")
    (root / "copy.py").symlink_to(original)
    with pytest.raises(ValueError):
        scan(root)


def test_symlink_root_rejected(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        scan(alias)


def test_archive_is_code_only_and_owner_neutral(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "README.md").write_text("# Demo")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "cached.pyc").write_bytes(b"cache")
    archive = tmp_path / "code.tar.gz"
    build(root, archive)
    with tarfile.open(archive) as handle:
        assert handle.getnames() == ["Trajectory2Gym/README.md", "Trajectory2Gym/" + MANIFEST]
        assert all(m.uid == m.gid == 0 and not m.uname and not m.gname for m in handle)
    with pytest.raises(FileExistsError):
        build(root, archive)


class FakeApi:
    def __init__(self, private=True):
        self.private = private
        self.uploaded = False
        self.names = []

    def whoami(self):
        return {"name": "test-account"}

    def create_repo(self, **kwargs):
        assert kwargs["private"] is True and kwargs["repo_type"] == "dataset"

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kwargs):
        self.uploaded = True
        self.names = kwargs["allow_patterns"]
        return SimpleNamespace(oid="test-commit")

    def list_repo_files(self, **kwargs):
        return self.names


def test_upload_is_private_and_exactly_manifested(tmp_path):
    api = FakeApi()
    receipt = publish(dataset(tmp_path), "Trajectory2gym", api)
    assert receipt["repo_id"] == "test-account/Trajectory2gym"
    assert receipt["private"] is True and api.uploaded
    assert set(api.names) == {"README.md", "train.jsonl", MANIFEST}


def test_public_repository_rejected(tmp_path):
    api = FakeApi(private=False)
    with pytest.raises(ValueError):
        publish(dataset(tmp_path), "Trajectory2gym", api)
    assert not api.uploaded


def test_another_namespace_cannot_be_injected(tmp_path):
    api = FakeApi()
    with pytest.raises(ValueError):
        publish(dataset(tmp_path), "someone-else/dataset", api)
    assert not api.uploaded


def test_path_traversal_in_manifest_is_rejected(tmp_path):
    root = dataset(tmp_path)
    manifest = json.loads((root / MANIFEST).read_text())
    manifest["files"][0]["path"] = "../outside.json"
    (root / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_manifest(root)
