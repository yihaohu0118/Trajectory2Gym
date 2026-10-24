# Construction and training workflow

Keep complete semantic interactions, rendered examples, and token-level
supervision separate. Rendering can create multiple examples from one parent;
it must not silently multiply that parent's optimization mass.

## 1. Construct and certify

Use native training-side resets, tools and observations. Collect the realized
interaction, project the assistant-visible history, and certify eligible
decisions. The candidate builder accepts a source manifest containing
source_inputs (each with a path and optional domain_override) and schema_inputs
(each with a path). Paths refer to local, reviewed inputs.

~~~bash
python scripts/analysis/rebuild_certified_candidate_pool.py \
  --source-manifest /path/to/source-manifest.json \
  --output-dir outputs/candidates
~~~

A candidate pool is not automatically train-ready. Preserve native execution
evidence and the visibility/target checks required by the pipeline.
The synthesis overlay operates on supplied native records; it does not create
missing backend execution evidence from a task description.

## 2. Allocate structural support

The selector consumes a passed candidate manifest, a non-truncated token audit,
and a frozen retention-core reference. Source names below are provenance labels,
not quotas or scoring features.

~~~bash
python scripts/analysis/select_structural_support_scaling_views.py \
  --candidate-dir /path/to/audited-candidates \
  --reference-dir /path/to/frozen-reference \
  --include-source tau1 --include-source tau2 --include-source bfcl \
  --scale-depth 1.7b:1 --scale-depth 4b:2 \
  --scale-depth 8b:3 --scale-depth 14b:4 \
  --output-dir outputs/semantic-views
~~~

The paper's 30B-A3B configuration has depth 8 and a separate expanded corpus.
Run it against matching candidate/reference assets. Do not infer a complete
experimental contract from helper defaults.

## 3. Render and conserve parent mass

The rendering and finalization scripts check semantic, transport, token,
non-thinking and target audits before publishing a train-ready view.
apply_transport_parent_mass_correction.py derives the row coefficient

\[
\omega_{ij}=\frac{M}{P}\frac{n_{ij}}{\sum_{j'}n_{ij'}}.
\]

Here P counts semantic parents, M counts rendered examples, and n counts
audited supervised target tokens. The field is transport_parent_loss_weight.
Do not change the tokenizer/template and retain coefficients derived from old
token counts.

## 4. Train with explicit objective options

For the parent-normalized assistant-token objective, explicitly request:

~~~text
--assistant-only-loss
--sparse-supervised-logits
--row-loss-weight-field transport_parent_loss_weight
--loss-normalization global_batch_parent_mean
--num-train-epochs 1
--dtype bf16
--weight-decay 0.01
--lr-scheduler-type cosine
--warmup-steps 10
--max-grad-norm 1.0
~~~

Provide train-jsonl, val-jsonl, model, output-dir, and an audited max-length.
The frozen train-ready views have empty validation files; do not substitute
evaluation tasks. Match global batch size eight through world size, per-device
batch and gradient accumulation. Paper learning rates are 1.1e-5 (1.7B),
7e-6 (4B/8B), and 3.7e-6 (14B/30B-A3B).
Full-parameter SFT does not use LoRA or domain-balancing sampling.
Non-thinking depends on rendered data and the native template.

Use --help and --dry-run first. A compatible PyTorch installation and native
benchmark dependencies are required for GPU training and full evaluation.
This guide alone is not an original distributed-runtime configuration.
