"""Benchmark integrations for Trajectory2Gym.

Benchmark loaders, adapters, and environment facades live outside the core
partial-environment compiler so benchmark-specific glue stays isolated from the
paper-facing method implementation.
"""

from env.base import Benchmark, EvaluationResult, ExecutionResult, TestCase

__all__ = ["Benchmark", "EvaluationResult", "ExecutionResult", "TestCase"]
