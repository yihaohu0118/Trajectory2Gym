"""Seal an explicitly prepared JSONL export; does not select or rewrite data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.release_utils import MANIFEST, create_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    target = args.folder / MANIFEST
    if target.exists():
        raise FileExistsError("Use a fresh export directory; manifest already exists")
    manifest = create_manifest(args.folder, "dataset")
    if not any(r["path"].endswith((".jsonl", ".jsonl.gz")) for r in manifest["files"]):
        raise ValueError("No JSONL trajectory file in prepared export")
    if "README.md" not in {r["path"] for r in manifest["files"]}:
        raise ValueError("A dataset card README.md is required")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"files": len(manifest["files"]), "manifest": MANIFEST,
                      "uploaded": False}))


if __name__ == "__main__":
    main()
