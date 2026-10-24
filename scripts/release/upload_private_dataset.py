"""Upload only a sealed dataset export to the authenticated user's private repo."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.release_utils import MANIFEST, sha256, validate_manifest


def publish(folder: Path, repo_name: str, api):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", repo_name):
        raise ValueError("Provide a repository name, not another account's namespace")
    manifest = validate_manifest(folder)
    if manifest["kind"] != "dataset":
        raise ValueError("Only dataset manifests are accepted")
    who = api.whoami()
    owner = who["name"]
    repo_id = owner + "/" + repo_name
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if info.private is not True:
        raise ValueError("Refusing to upload to an existing public repository")
    names = [r["path"] for r in manifest["files"]] + [MANIFEST]
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(folder),
        allow_patterns=names, commit_message="Upload reviewed private trajectory release")
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if info.private is not True:
        raise RuntimeError("Repository privacy changed during upload")
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    if not set(names) <= remote_files:
        raise RuntimeError("Uploaded repository is missing manifest files")
    return {"repo_id": repo_id, "private": True, "files": len(names),
            "commit": getattr(commit, "oid", None),
            "manifest_sha256": sha256(folder / MANIFEST),
            "verification": "local SHA-256 manifest, remote file presence, repository privacy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-name", default="Trajectory2gym")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--token-stdin", action="store_true",
                        help="Read a token from standard input; never a command-line argument")
    args = parser.parse_args()
    try:
        manifest = validate_manifest(args.folder)
        if manifest["kind"] != "dataset":
            raise ValueError("Expected a dataset manifest")
        if not args.execute:
            print(json.dumps({"status": "dry_run", "network_requests": 0,
                              "files": len(manifest["files"]), "private_only": True}))
            return 0
        from huggingface_hub import HfApi
        credential = (sys.stdin.readline().strip() if args.token_stdin
                      else os.environ.get("HF_TOKEN") or getpass.getpass("HF token (hidden): "))
        if not credential:
            raise ValueError("No Hugging Face credential supplied")
        result = publish(args.folder, args.repo_name,
                         HfApi(endpoint="https://huggingface.co", token=credential))
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        # Do not print request bodies, headers, or arbitrary server exception text.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
