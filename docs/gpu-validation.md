# Reproducible simulator validation

`tools/validate_simulators.py` prepares standard Harbor 1.4 tasks and, only with
`--run`, executes the deterministic Oracle agent. It does not select an LLM,
download assets, create simulator images, or import a simulator on the host.
Oracle programs are explicitly project-authored, not official benchmark experts.

## Prepare and inspect

After building the SDK wheel and preparing an immutable-image deployment map:

```bash
.venv/bin/python tools/validate_simulators.py \
  --deployments deployments/local/map.json \
  --sdk-wheel build/wheels/roboenv-0.2.0-py3-none-any.whl \
  --backends robocasa --modes reference --seeds 0 \
  --out .cache/validation-plan
```

No Docker or GPU command runs during preparation. Generated task directories are
checked by Harbor's actual loader, contain the ordinary environment and verifier,
and use the existing task generator and taskset runner. Inspect `receipt.json`
and `taskset/`. Preparing without a backend selection covers the six small
reference examples across RoboCasa, RoboTwin, and RoboDojo at seed 0.

Execution requires another new output directory; existing evidence is never
overwritten. Select the permitted GPU using `ROBOENV_GPU` as for any Roboenv task:

```bash
.venv/bin/python tools/validate_simulators.py \
  --deployments deployments/local/map.json \
  --sdk-wheel build/wheels/roboenv-0.2.0-py3-none-any.whl \
  --backends libero --cases libero/spatial_0 \
  --modes probe --seeds 0 1 2 --run \
  --out .cache/validation-run
```

Trials run sequentially with no automatic retry. A nonzero result must be
investigated, not silently retried until a convenient success appears. Historical
example programs are tested as supplied; portability to other seeds is not assumed.

## Read-only device probes

Use `--modes probe` for boot, registry and device-read checks without motion. By
default this exercises each selected backend's default robot binding plans:
LIBERO Panda, RoboCasa Panda/Omron, RoboTwin Piper and ALOHA-AgileX, RoboDojo ARX X5
and YAM. `--robots` restricts that list. YAM explicitly selects `env_cfg_type=yam`;
changing only the robot descriptor is not sufficient.

The seventh plan, BEHAVIOR R1Pro, requires an **additional explicit task source**:

```bash
.venv/bin/python tools/validate_simulators.py \
  --deployments deployments/local/map.json \
  --sdk-wheel build/wheels/roboenv-0.2.0-py3-none-any.whl \
  --backends behavior --modes probe \
  --source generated/selected-behavior/tasks \
  --cases EXPLICIT_TASK_ID --out .cache/behavior-probe
```

The selected normalized task must declare `backend_kwargs.scene_model`; no scene
or task instance is guessed. An additional source cannot replace a built-in
example ID. BEHAVIOR remains a symbolic-primitives condition, not continuous
motor control. The command above still only prepares; add `--run` to execute.

## What each receipt field proves

| Field | Required evidence |
|---|---|
| `boot_verified` | Independent verifier observed the expected episode and a ready worker |
| `device_read_probe_passed` | Probe completed, every expected device has a successful read in the independently validated journal, no mutating probe call |
| `reference_program_completed` | Oracle completion record matches the saved program hash and seed, with no recorded agent failure |
| `final_verifier_success` | Valid, matching journal plus the independent full-success verdict and Harbor reward 1 |
| `reference_replay_passed` | Reference mode, program completed, boot verified, and final verifier success |
| `selected_evidence_complete` | Every selected check passed and inspected image/build provenance is present; not approval of the entire project |

A probe can pass with reward 0 because it checks connectivity, not task completion.
That receipt always keeps `reference_replay_passed=false`. A reference with reward
0 **never** passes, even if Oracle exited normally. Program completion alone is
not a benchmark verdict. Each trial starts through Harbor, but the current live
verifier is not a separate second-process replay certificate; the receipt states
`fresh_process_replay_certificate=false`.

Receipts record SDK/program/task/binding hashes, inspected immutable image IDs,
the deployment-declared runtime wheel hash, source revisions and upstream lock
hash. Build provenance is labeled as a declaration from the deployment manifest,
not reconstructed from the running filesystem. Missing build metadata remains
visible and prevents `selected_evidence_complete` from becoming true.

Only allowlisted fields enter `receipt.json`; raw Harbor logs, errors, local
deployment locations and simulator observations remain private. Do not upload
the complete output tree. Preserve its journal locally when investigating a
failure; the public receipt records its hash.

## Trusted manual GPU CI

The `gpu-validation.yml` workflow is manual-only and restricted to the default
branch. It never runs self-hosted GPU jobs on public pull requests. Operators must
configure the `gpu-validation` environment with required reviewers and a
default-branch deployment policy, and provide an isolated, trusted runner labeled
`roboenv-gpu` with Docker, NVIDIA support and prebuilt depots/images.

Protected environment variables name the operator's local deployment map
(`ROBOENV_GPU_DEPLOYMENT_MAP`), optional explicit additional task source
(`ROBOENV_GPU_TASK_SOURCE`) and device (`ROBOENV_GPU_DEVICE`). These locations are
not repository content. No API key is needed. The workflow uploads only the
sanitized receipt after a successful privacy check and preserves a failed check's
red status. Creating the YAML does not configure GitHub protection rules or prove
that the external runner has executed it.
