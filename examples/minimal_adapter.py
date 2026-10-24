"""The only adapter logic a new trajectory dataset normally needs."""

import sys
from pathlib import Path

# Keep the example executable from a fresh checkout before the package is
# installed; installed users simply resolve the normal package import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory2gym.trajectory import DeclarativeTrajectoryAdapter, ToolSemantics


def build_adapter() -> DeclarativeTrajectoryAdapter:
    return DeclarativeTrajectoryAdapter(
        source="my_dataset",
        tool_semantics=ToolSemantics(
            observation_db_safe_exact_names=("lookup_weather", "search_docs"),
        ),
    )


if __name__ == "__main__":
    print(build_adapter().source)
