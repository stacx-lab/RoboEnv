# Reproducible simulator validation

`tools/validate_simulators.py` validates prepared simulators on one host with the
same three commands a harness uses: `roboenv launch` starts the world as its owner,
`roboenv replay` runs one program against the public endpoint as an agent would, and
`roboenv verify` certifies the run through the owner socket before the world is
stopped. It never selects an LLM, downloads assets, builds images or imports a
simulator into its own process. Its programs are explicitly project-authored, not
official benchmark experts.

Two modes produce separate evidence categories:

| Mode | Program | Passes when |
|---|---|---|
| `probe` | Generated read-only device connectivity program: reads every declared readable device, never writes | Boot verified, program completed, certificate valid, every expected device read in the retained journal, no mutating call |
| `reference` | The task's checked-in `solution.py` | Boot verified, program completed, and a valid certificate with full success (reward 1) |

## Prepare and inspect

Preparation selects cases and writes their inputs without starting anything:

```sh
.venv/bin/python tools/validate_simulators.py \
  --backends libero --modes probe --seeds 0 \
  --out .cache/validation-plan
```

Inspect `receipt.json` (phase `prepared`) and `cases/*/{task.yaml,program.py,
program-source.json}`. Select backends and modes explicitly. `--modes reference`
requires a `solution.py` beside the task; a retired reference (`solution-source.json`
with `retired: true`) is skipped by default and refused when selected with `--cases`.
The two checked-in RoboCasa references are retired (they were written for the tool
API that typed devices replaced), so supply `--source DIRECTORY` with a current
reference program. An additional source cannot replace a built-in example ID.

## Run

Execution needs a new output directory; existing evidence is never overwritten.
Worlds start from the project's `roboenv prepare` registrations, or from one explicit
interpreter for a single selected backend. Select the GPU as for any local run:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
.venv/bin/python tools/validate_simulators.py \
  --backends libero --cases libero/bowl-onto-plate \
  --modes probe --seeds 0 1 2 --run \
  --interpreter .venv-sim/libero/bin/python \
  --out .cache/validation-run
```

Cases run sequentially with no automatic retry; each one launches its own world,
replays, verifies and stops that world before the next starts. A nonzero exit must
be investigated, not retried until a convenient success appears. Reference programs
are tested as supplied; portability to other seeds is not assumed.

For packaged startup, `--runtime-root ROOT --catalog CATALOG --interpreter PYTHON`
and `--python-path DIR` pass through to `roboenv launch` unchanged. `--python-path`
is also how a checkout whose simulator interpreters were prepared against another
checkout selects the SDK and runtime the launched world imports; without it the
interpreter's own installation is used.

## Read-only device probes

Use `--modes probe` for boot, registry and device-read checks without motion. By
default this exercises supported rigs with matching selected tasks: scene-specific
LIBERO Panda, RoboCasa Panda/Omron, RoboTwin Piper and ALOHA-AgileX, and RoboDojo
ARX-X5. `--robots` restricts that list using catalog-relative paths such as
`robodojo/robodojo-arx-x5`. A backend without a maintained rig list, such as the
test backend, probes each selected task's own robot.

BEHAVIOR R1Pro requires an **additional explicit task source**:

```bash
.venv/bin/python tools/validate_simulators.py \
  --backends behavior --modes probe \
  --source generated/selected-behavior/tasks \
  --cases EXPLICIT_TASK_ID --out .cache/behavior-probe
```

The selected normalized task must declare `backend_kwargs.scene_model`; no scene
or task instance is guessed. BEHAVIOR uses continuous typed motor devices with
physical grasping. All native calls run on their creation thread. Physics advances
only for active commands; native GPU balance, rendering and replay require separate
validation.

## What each receipt field proves

| Field | Required evidence |
|---|---|
| `boot_verified` | `roboenv launch` reported readiness and the certificate names the expected task and seed with a ready worker |
| `program_completed` | `roboenv replay` exited 0 and reported `program_completed` |
| `verifier_valid` | `roboenv verify` produced a valid certificate, and the retained `actions.jsonl` matches its recorded hash |
| `device_read_probe_passed` | Probe mode: every expected device has a successful read-only read in the retained journal and nothing mutated the world |
| `final_verifier_success` | Valid certificate, matching identity, `trial_success` and reward 1 |
| `reference_replay_passed` | Reference mode: program completed, boot verified and final verifier success |
| `world_stopped` | The launched world was stopped after certification; a leaked world fails the case |
| `all_selected_checks_passed` | Every case passed its mode's check and stopped its world; not approval of the entire project |

`journal` records the retained journal's hash, size and request count; `budget`
the certified budget; `timing_s` the launch, replay and verify wall times;
`replay_exit_code` and `verify_exit_code` the two agent/verifier processes.
`failure_category` names the first failed gate: `launch-failed`, `stage-timeout`,
`probe-program-failed`, `reference-program-failed`, `device-read-failed`,
`probe-mutated-environment`, `incomplete-device-read-evidence`,
`invalid-verifier-evidence`, `reference-not-successful` or `malformed-evidence`.

A probe can pass with reward 0 because it checks connectivity, not task completion;
that receipt always keeps `reference_replay_passed=false`. A reference with reward
0 **never** passes, even if its program exited normally. Program completion alone
is not a benchmark verdict. The live verifier certifies the run it observed; it is
not a separate second-process replay certificate, so the receipt states
`fresh_process_replay_certificate=false`.

Only allowlisted fields enter `receipt.json`: identities, hashes, verdict fields,
exit codes and timings. Raw launch, replay and verify output goes to
`validation.private.log` (mode 0600) beside the run records and retained journals.
Do not upload the output tree; the public receipt records the journal hash.

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
For manual Kinex recording, use [local Kinex use](kinex-local.md); scheduled agent
trials belong to the benchmark harness built on this SDK.

## Trusted manual GPU CI

The `gpu-validation.yml` workflow is manual-only and restricted to the default
branch. It never runs self-hosted GPU jobs on public pull requests. Operators must
configure the `gpu-validation` environment with required reviewers and a
default-branch deployment policy, and provide an isolated, trusted runner labeled
`roboenv-gpu` with NVIDIA support and simulators prepared with `roboenv prepare`.

The workflow launches worlds from its checkout, so it needs that checkout's pinned
upstream sources and the runner's prepared interpreters: it runs
`roboenv sources sync --backends BACKEND` for the selected backend before
validation, and it checks out with `clean: false` so that the interpreters under
`.venv-sim/` and their registrations under `.cache/` survive between runs. An
operator who keeps the interpreter outside the workspace names it with
`ROBOENV_GPU_INTERPRETER` instead.

Protected environment variables name an optional additional task source
(`ROBOENV_GPU_TASK_SOURCE`), an optional explicit simulator interpreter for the
selected backend (`ROBOENV_GPU_INTERPRETER`) and the device (`ROBOENV_GPU_DEVICE`,
carried by the job as `VALIDATION_GPU` and exported as `CUDA_VISIBLE_DEVICES` and
`MUJOCO_EGL_DEVICE_ID`; a `ROBOENV_*` name would never reach the world, because
launch strips every `ROBOENV_*` variable). These locations are not repository
content. No API key is needed. The workflow uploads only the sanitized receipt
after a successful privacy check and preserves a failed check's red status.
Creating the YAML does not configure GitHub protection rules or prove that the
external runner has executed it.

Kinex smoke and rig checks use the built `packages/core` runtime with Node >=22.19.
Select a checkout with `--kinex` (or `KINEX_ROOT`) and a Node executable with
`--node` (or `NODE`) in the LIBERO and SIMPLE acceptance runners. Rebuild Kinex core
after updating its source before running these checks. Python driver tests use
that checkout's `packages/core/assets/python` library.
