from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkInstantiation:
    benchmark: str
    adapter: str
    verifier_plugins: tuple[str, ...]
    grounding: str
    shared_compiler_used: bool
    training_runtime: str = "partial"
    trajectory_role: str = "environment evidence + policy sampling"
    status: str = "supported"

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "adapter": self.adapter,
            "verifier_plugins": list(self.verifier_plugins),
            "grounding": self.grounding,
            "shared_compiler_used": self.shared_compiler_used,
            "training_runtime": self.training_runtime,
            "trajectory_role": self.trajectory_role,
            "status": self.status,
        }


BENCHMARK_REGISTRY: dict[str, BenchmarkInstantiation] = {
    "tau_bench": BenchmarkInstantiation(
        benchmark="tau_bench",
        adapter="DeclarativeTrajectoryAdapter",
        verifier_plugins=("CompilerVerifier",),
        grounding="trajectory_replay + observation_database",
        shared_compiler_used=True,
        training_runtime="real",
        trajectory_role="data mining + task/state guidance",
    ),
    "tau2_bench": BenchmarkInstantiation(
        benchmark="tau2_bench",
        adapter="DeclarativeTrajectoryAdapter",
        verifier_plugins=("CompilerVerifier",),
        grounding="trajectory_replay + observation_database",
        shared_compiler_used=True,
        training_runtime="real",
        trajectory_role="data mining + task/state guidance",
    ),
    "bfcl": BenchmarkInstantiation(
        benchmark="bfcl",
        adapter="DeclarativeTrajectoryAdapter",
        verifier_plugins=("CompilerVerifier",),
        grounding="trajectory_replay + observation_database",
        shared_compiler_used=True,
        training_runtime="real",
        trajectory_role="data mining + task/state guidance",
    ),
    "wildtoolbench": BenchmarkInstantiation(
        benchmark="wildtoolbench",
        adapter="DeclarativeTrajectoryAdapter",
        verifier_plugins=("CompilerVerifier",),
        grounding="trajectory_replay + observation_database",
        shared_compiler_used=True,
        training_runtime="partial",
        trajectory_role="environment evidence + state guidance",
    ),
    "apigen_mt": BenchmarkInstantiation(
        benchmark="apigen_mt",
        adapter="DeclarativeTrajectoryAdapter",
        verifier_plugins=("CompilerVerifier",),
        grounding="trajectory_replay + observation_database",
        shared_compiler_used=True,
        training_runtime="analysis_only",
        status="analysis_only",
    ),
    "hardgen": BenchmarkInstantiation(
        benchmark="hardgen",
        adapter="DeclarativeTrajectoryAdapter",
        verifier_plugins=("CompilerVerifier",),
        grounding="trajectory_replay + observation_database",
        shared_compiler_used=True,
        training_runtime="analysis_only",
        status="analysis_only",
    ),
}


def get_benchmark_instantiation(name: str) -> BenchmarkInstantiation:
    return BENCHMARK_REGISTRY[name]


def benchmark_registry_rows() -> list[dict[str, object]]:
    return [entry.to_dict() for entry in BENCHMARK_REGISTRY.values()]
