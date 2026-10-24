"""Build a checked code-only archive, excluding history and runtime artifacts."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.release_utils import MANIFEST, create_manifest, sha256


def build(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError("Archive output must be outside the source package")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing archive")
    manifest = create_manifest(root, "code")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for item in manifest["files"]:
            path = root / item["path"]
            info = archive.gettarinfo(str(path), arcname="Trajectory2Gym/" + item["path"])
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("Trajectory2Gym/" + MANIFEST)
        info.size, info.mode = len(body), 0o644
        archive.addfile(info, io.BytesIO(body))
    return {"files": len(manifest["files"]), "archive_sha256": sha256(output),
            "archive_bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
