# Dataset preparation and private Hub upload

Code and data are separate artifacts. The upload utility never uploads an
arbitrary research directory, publishes a public repository, or accepts a token
as a command-line argument.

## Prepare a new export directory

~~~text
reviewed-export/
  README.md
  data/
    train-00000.jsonl.gz
  provenance.json
~~~

JSONL and gzipped JSONL preserve nested messages, tool calls, schemas and target
masks. State whether records are semantic parents or rendered examples; do not
count rendered children as new interactions.

Before staging:

1. Bind the source to its frozen training manifest and hash. Do not export an
   evaluation directory or label a historical diagnostic as the main corpus.
2. Keep policy-visible messages and existing target masks. Private collector
   state and teacher-only hints do not belong in the learner-visible history.
3. Remove operational metadata such as credentials, private endpoints, host
   paths and internal account identifiers. Do not change native fictional
   references independently of their observations and tool arguments.
4. Record transformations, input hashes, exported hashes, row counts and source
   terms. A transformed export is not byte-identical to an original experiment
   file, even if messages and targets are unchanged.
5. Review content and redistribution rights. Pattern scanning does not establish
   that every possible piece of personal information has been detected.

The code license does not automatically license derived datasets. Record data
terms separately and do not invent a license field. Disclose Tau1 Airline's
seen-task scope; Tau2 task partitions do not imply template-level separation.

## Seal and inspect locally

For a frozen, already policy-visible candidate JSONL, the optional exporter
projects an explicit top-level field allowlist, preserves messages/tools/masks,
and writes deterministic compressed shards. It requires the pre-existing input
hash and row count, refuses existing outputs, and rejects suspicious content:

~~~bash
python scripts/release/export_policy_visible_jsonl.py \
  --source /path/to/certified/train.jsonl \
  --expected-sha256 EXPECTED_INPUT_SHA256 --expected-rows EXPECTED_ROW_COUNT \
  --output /path/to/new-reviewed-export
~~~

It seals its own output manifest. It does not infer visibility from raw logs or
re-certify native execution. For other manually prepared exports, seal them with
the following commands instead:

~~~bash
python scripts/release/prepare_dataset_manifest.py --folder /path/to/reviewed-export
python scripts/release/check_release.py --root /path/to/reviewed-export --kind dataset
python scripts/release/upload_private_dataset.py --folder /path/to/reviewed-export
~~~

The first command creates RELEASE_MANIFEST.json with exact relative filenames,
sizes and SHA-256 hashes. It does not rewrite trajectories or overwrite existing
manifests. The last command is a dry run with no network requests. Do not add
extra files after sealing the folder.

Benchmark tasks can legitimately contain simulated network addresses or generic
path placeholders. Do not remove such values independently from dependent tool
calls. If inspection establishes that a finding is a native illustrative value,
an optional --literal-review file may list exact literal hashes bound to the
frozen source hash. Only address/path rules can be reviewed this way; credentials
and internal-domain findings cannot be exempted. The review is included in the
dataset and the manifest reports how many occurrences were retained. An assistant
review must not be described as an independent human privacy audit.

## Upload directly from the data host

Install huggingface-hub in the upload environment. This standalone utility can
use the host's Hub SDK separately from the pinned model-training environment.

~~~bash
python scripts/release/upload_private_dataset.py \
  --folder /path/to/reviewed-export --repo-name Trajectory2gym --execute
~~~

The command uses HF_TOKEN from the process environment, or prompts with input
hidden. Controlled runners can use --token-stdin. Never paste credentials into
source files, command-line options, READMEs or release manifests. This utility
does not save credentials.

The namespace comes from the authenticated account. The repository is created
private; existing public repositories are rejected. Only manifest files are
uploaded. The receipt distinguishes local checksum validation and remote file
presence/privacy checks from a remote byte readback.

There is no public-publishing option. Review private data and its licenses before
making a separate visibility change.

## Official API references

- https://huggingface.co/docs/huggingface_hub/en/guides/upload
- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
