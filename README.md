# Trajectory2Gym

[![Trajectory2Gym](https://yihaohu0118.github.io/Trajectory2Gym-page/assets/poster.png)](https://yihaohu0118.github.io/Trajectory2Gym-page/)

**Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents**

[Project Homepage](https://yihaohu0118.github.io/Trajectory2Gym-page/) ·
[Paper](https://yihaohu0118.github.io/Trajectory2Gym-page/assets/Trajectory2Gym.pdf) ·
[Dataset](https://huggingface.co/datasets/mis2/Trajectory2gym) ·
[Checkpoints](#published-checkpoints)

This is the code release for the paper's supervised tool-use method. It
contains the interaction representation, decision-grounded target construction,
structural-support selection, parent-normalized SFT implementation, thin native
environment adapters, and self-contained regression tests.

## Method-to-code map

| Paper component | Main code |
| --- | --- |
| Compositional interaction representation and synthesis | `trajectory2gym/trajectory/`, `scripts/analysis/build_targeted_active_synthesis_overlay.py` |
| Policy-visible evidence and supervised target admission | `trajectory2gym/training/policy_view.py`, `trajectory2gym/training/sft.py`, `scripts/analysis/rebuild_certified_candidate_pool.py` |
| Structural-support allocation | `trajectory2gym/training/global_sft_selection.py`, `scripts/analysis/select_structural_support_scaling_views.py` |
| Rendering and parent-normalized learning | `scripts/analysis/finalize_unified_global_sft_view.py`, `scripts/analysis/apply_transport_parent_mass_correction.py`, `scripts/train/train_trajectory2gym_sft_view.py` |
| Native benchmark interfaces | `env/taubench/`, `env/tau2_bench/`, `env/bfcl/` |

The snapshot excludes historical RL utilities. Its training-package
initializer exposes only the SFT interface retained here; the reported
experiments use ordinary assistant-token cross-entropy, not an RL objective.

## Install and test

Use Python 3.10 or newer. Clone the repository, then install the package:

```bash
git clone https://github.com/yihaohu0118/Trajectory2Gym.git
cd Trajectory2Gym
python3 -m pip install -e ".[dev]"
```

The full regression suite runs on Linux and requires PyTorch and symbolic-link
support. For CPU tests, install the CPU build before running the suite:

```bash
python3 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pytest -q tests
```

For GPU training, select the compatible build from the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).
GitHub Actions runs the complete CPU suite, example, and release-manifest checks
on Python 3.11; its lightweight test dependencies are in `requirements-ci.txt`.

The included tests use small synthetic inputs and do not require benchmark
credentials or model checkpoints. Publication-tool tests additionally cover
manifest integrity, sensitive-file rejection, and private-only upload guards.

The selected command-line entry points provide their arguments through
`--help`. For example:

```bash
python3 scripts/analysis/rebuild_certified_candidate_pool.py --help
python3 scripts/analysis/select_structural_support_scaling_views.py --help
python3 scripts/analysis/finalize_unified_global_sft_view.py --help
python3 scripts/train/train_trajectory2gym_sft_view.py --help
```

## Scope and reproducibility

The public candidate interactions are available at
[mis2/Trajectory2gym](https://huggingface.co/datasets/mis2/Trajectory2gym):
34,772 semantic interactions in seven compressed JSONL shards (about 125 MB).
These are the common candidate pool, not the rendered scale-specific SFT
examples or the expanded 30B-A3B corpus. The dataset includes file checksums,
source provenance, target masks, and upstream license notices.

## Published checkpoints

All five paper checkpoints are available with native tokenizers, chat templates,
model cards, upstream licenses, and per-file release manifests:

| Model | Stored precision | Weight files |
|---|---|---:|
| [1.7B](https://huggingface.co/mis2/Trajectory2Gym-1.7B) | float32 | 8.1 GB |
| [4B](https://huggingface.co/mis2/Trajectory2Gym-4B) | float32 | 17.6 GB |
| [8B](https://huggingface.co/mis2/Trajectory2Gym-8B) | float32 | 32.8 GB |
| [14B](https://huggingface.co/mis2/Trajectory2Gym-14B) | bfloat16 | 29.5 GB |
| [30B-A3B](https://huggingface.co/mis2/Trajectory2Gym-30B-A3B) | bfloat16 | 61.1 GB |

Weights retain their archived precision and exact safetensor bytes; no
quantization or retraining was performed for publication. Checksums and public
access were verified. The reported scores are inherited experiment results,
not a new benchmark rerun during release. Original-precision loading can require
more memory than a lower-precision deployment.

## Code-only archive boundaries

This archive intentionally omits raw interaction logs, training/evaluation
task instances, checkpoint weights, private API requests, experiment receipts,
host-specific launchers, and vendored third-party runtimes. Official benchmark
packages and model weights must be obtained under their own licenses. Running
the full paper experiments additionally requires the released benchmark
environments and the selected training data; this code snapshot alone does not
recreate the reported numeric results.

See [WORKFLOW](docs/WORKFLOW.md) for construction and explicit training options,
and [DATA_RELEASE](docs/DATA_RELEASE.md) for reviewed data exports and private
Hugging Face staging. The generic trainer includes historical diagnostic
options; its defaults are not a complete paper-reproduction configuration.

The CPU-only example at examples/structural_support_demo.py demonstrates
nested support allocation using synthetic fixtures, without model downloads.
The paper's depth and learning-rate schedule is provided separately in
configs/paper_support_depths.json. The 30B-A3B study uses an expanded corpus.

The release utilities scan code and data locally, produce SHA-256 manifests,
and build a source archive without Git history or development caches.
Uploads require an explicit execution flag and always use a private dataset
repository under the authenticated user's namespace.

## License

This code retains the existing Apache-2.0 license in LICENSE.
External models, benchmark packages, tasks, and derived datasets retain their
respective terms; the code license does not grant redistribution rights to
those assets. See [third-party notes](THIRD_PARTY.md).
