"""Run local, value-redacting publication checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.release_utils import scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--kind", choices=["code", "dataset"], default="code")
    args = parser.parse_args()
    try:
        result = scan(args.root, args.kind)
    except (ValueError, OSError, UnicodeError) as exc:
        result = {"status": "needs_review", "error": str(exc)}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
