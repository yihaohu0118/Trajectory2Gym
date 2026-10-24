"""Export full model weights at the explicitly pinned inference precision."""
from __future__ import annotations

import json
from pathlib import Path


def bf16_export_state(state_dict):
    """Make an export dictionary without modifying parameters or optimizer state."""
    import torch

    result = {}
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor state entry: {name}")
        if value.device.type == "meta":
            raise ValueError(f"unmaterialized state entry: {name}")
        result[name] = value.detach().to(
            device="cpu", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype
        ).contiguous()
    return result


class CheckpointStorageMixin:
    checkpoint_storage_dtype = "native"

    def _save(self, output_dir=None, state_dict=None):
        if self.checkpoint_storage_dtype == "native":
            return super()._save(output_dir, state_dict=state_dict)
        if self.checkpoint_storage_dtype != "bf16":
            raise ValueError("unknown checkpoint storage dtype")
        if not self.args.save_only_model or not self.args.bf16:
            raise ValueError("BF16 storage requires BF16 training and model-only export")
        if state_dict is None:
            if getattr(self, "is_fsdp_enabled", False):
                raise ValueError("FSDP export requires its gathered full state dict")
            state_dict = self.model.state_dict()
        exported = bf16_export_state(state_dict)
        result = super()._save(output_dir, state_dict=exported)
        destination = Path(output_dir or self.args.output_dir)
        config_path = destination / "config.json"
        config = json.loads(config_path.read_text())
        config["dtype"] = "bfloat16"
        if "torch_dtype" in config:
            config["torch_dtype"] = "bfloat16"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        receipt = {
            "format": "trajectory2gym_checkpoint_storage_v1",
            "status": "pass",
            "storage_dtype": "bfloat16",
            "official_evaluation_dtype": "bfloat16",
            "conversion": "FP32 full-state tensors -> BF16 before serialization",
            "floating_tensors": sum(value.is_floating_point() for value in exported.values()),
            "stored_tensor_bytes": sum(value.numel() * value.element_size() for value in exported.values()),
            "optimizer_updates_changed": False,
            "training_parameters_modified": False,
            "full_precision_training_resume_supported": False,
            "equivalence_scope": "same tensor values as native full-state checkpoint loaded explicitly at BF16",
        }
        (destination / "checkpoint_storage.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return result
