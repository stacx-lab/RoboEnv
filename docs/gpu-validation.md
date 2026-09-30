# Reproducible simulator validation

`tools/validate_simulators.py` prepares standard Harbor 1.4 tasks and, only with
`--run`, executes the deterministic Oracle agent. It does not select an LLM,
download assets, create simulator images, or import a simulator on the host.
Oracle programs are explicitly project-authored, not official benchmark experts.

## Prepare and inspect

After building the SDK wheel and preparing an immutable-image deployment map:

```sh
.venv/bin/python tools/validate_simulators.py \
  --deployments deployments/local/map.json \
  --sdk-wheel build/wheels/roboenv-0.9.1-py3-none-any.whl \
  --backends robocasa --modes probe --seeds 0 \
  --out .cache/validation-plan
```

No Docker or GPU command runs during preparation. Generated task directories are
checked by Harbor's actual loader, contain the ordinary environment and verifier,
and use the existing task generator and taskset runner. Inspect `receipt.json`
and `taskset/`. Select backends and modes explicitly. `--modes reference` requires
a supplied `solution.py` beside a task in an explicit `--source` catalog. The
checked-in catalog contains no reference solutions. Use `--modes probe` for
device connectivity checks, or supply a controller using the current public
device API for reference validation.

Execution requires another new output directory; existing evidence is never
overwritten. Select the permitted GPU using `ROBOENV_GPU` as for any Roboenv task:

```bash
.venv/bin/python tools/validate_simulators.py \
  --deployments deployments/local/map.json \
  --sdk-wheel build/wheels/roboenv-0.9.1-py3-none-any.whl \
  --backends libero --cases libero/bowl-onto-plate \
  --modes probe --seeds 0 1 2 --run \
  --out .cache/validation-run
```

Trials run sequentially with no automatic retry. A nonzero result must be
investigated, not silently retried until a convenient success appears. Reference
programs are tested as supplied; portability to other seeds is not assumed.

## Read-only device probes

Use `--modes probe` for boot, registry and device-read checks without motion. By
default this exercises supported rigs with matching selected tasks: scene-specific
LIBERO Panda, RoboCasa Panda/Omron, RoboTwin Piper and ALOHA-AgileX, and RoboDojo
ARX-X5. `--robots` restricts that list using catalog-relative paths such as
`robodojo/robodojo-arx-x5`.

BEHAVIOR R1Pro requires an **additional explicit task source**:

```bash
.venv/bin/python tools/validate_simulators.py \
  --deployments deployments/local/map.json \
  --sdk-wheel build/wheels/roboenv-0.9.1-py3-none-any.whl \
  --backends behavior --modes probe \
  --source generated/selected-behavior/tasks \
  --cases EXPLICIT_TASK_ID --out .cache/behavior-probe
```

The selected normalized task must declare `backend_kwargs.scene_model`; no scene
or task instance is guessed. An additional source cannot replace a built-in
example ID. BEHAVIOR uses continuous typed motor devices with physical grasping.
All native calls run on their creation thread. Physics advances only for active
commands; native GPU balance, rendering and replay require separate validation.

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

## Native device checks

Native tools use prepared simulator interpreters and assets. Select the tool for
the backend and use its `--help` for task, rig, clock and output arguments:

| Tool | Scope |
|---|---|
| `tools/accept_libero.py` | LIBERO device, transport, recording and fresh-process replay gates without model calls |
| `tools/check_native_robotwin.py`, `tools/check_native_robodojo.py`, `tools/check_native_behavior.py` | Backend-specific native device conformance |
| `tools/check_native_clock.py` | Worker clock progress, motion, terminal freeze and independent replay |
| `tools/check_native_transport.py` | Concurrent episodes, deduplication, lost replies, public/owner separation and endpoint reuse |
| `tools/replay_native_devices.py` | Fresh-process replay of matching RoboTwin, RoboDojo or BEHAVIOR owner records |

Transport checks select an RGB/RGB-D camera from the live inventory; `--camera`
overrides it. `--seeds FIRST SECOND` selects native layouts for RoboTwin and
RoboDojo. BEHAVIOR also needs distinct `--instances FIRST SECOND`; seeds alone
do not select different presampled activity instances. The check rejects
effectively identical loaded states.

For a single GPU on an Isaac host, use `CUDA_VISIBLE_DEVICES` and leave
`OMNIGIBSON_GPU_ID` unset so Kit and PhysX can select the masked device. Verify the
actual renderer/physics device in native logs. A source or import check does not
establish GPU readiness.

Record the effective task, rig, clock, initialization, source/artifact identities
and commands with each result. Preserve failed attempts. CPU/schema checks,
native motion, recording, task success and command replay are separate evidence.
For real Kinex sessions and frozen controllers, use
[Kinex Harbor integration](../integrations/kinex/README.md); for manual recording,
use [local Kinex use](kinex-local.md).

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

Kinex smoke and rig checks use the built `packages/core` runtime with Node >=22.19.
Select a checkout with `--kinex` (or `KINEX_ROOT`) and a Node executable with
`--node` (or `NODE`) in the LIBERO and SIMPLE acceptance runners. Rebuild Kinex core
after updating its source before running these checks. Python driver tests use
that checkout's `packages/core/assets/python` library.
