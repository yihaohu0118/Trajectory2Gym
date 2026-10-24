"""Pure-SFT training utilities included in the paper supplement."""

from .sft import SFTBuildConfig, build_sft_dataset

__all__ = ["SFTBuildConfig", "build_sft_dataset"]
