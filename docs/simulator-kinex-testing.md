# Testing simulator integration with Kinex

Use [test-simulator-kinex](../.agents/skills/test-simulator-kinex/SKILL.md) to run
real jobs on a prepared simulator. Use
[integrate-simulator-kinex](../.agents/skills/integrate-simulator-kinex/SKILL.md)
when native code, source preparation or catalogs need implementation. Integration
includes testing; standalone testing has its own entry. See [skills.md](skills.md).

## Job modes and claims

| Mode | Model calls | What a pass establishes |
| --- | --- | --- |
| Probe | None | Real sensors, measured control through Kinex, finalization, recording and replay |
| Develop | Authorized exact provider/model/effort | A real agent observed and developed a controller; task outcomes remain separate |
| Validate | None for frozen control | The immutable controller ran in a fresh workspace/world and met selected trial checks |
| Integration acceptance | Authorized live agent sessions | Installation, native, transport, visual agent interaction, original outcomes and replay |
| Task-performance acceptance | Development only | Frozen task trials meeting the requested native success threshold |
| Harbor job | As specified | Results inside the actual exported agent/verifier and image boundary |

An unspecified test starts with a bounded no-model probe. It does not automatically
spend model budget or start a complete development campaign. Completing an
integration requires the full integration matrix. Task success rate is reported
separately and does not gate integration readiness. Discovery coverage is not
solved-task coverage.

## Owners and executable entrypoints

RoboEnv owns task resolution, the native world, physical clock, original predicate,
command evidence and native replay. Kinex owns tool discovery, device activation,
Python execution, recording and model-driven development. The runner composes
these owners without exposing private simulator observations to the agent.

Inspect the selected Kinex checkout's `examples/simulator-acceptance/README.md`,
`campaign.py`, `infrastructure.py`, `aggregate.py` and parsers. These are
working-tree capabilities, not guaranteed by a historical base commit. The current
generic campaign/replay path targets RoboTwin, RoboDojo and BEHAVIOR. LIBERO and
RoboCasa also have separate native/protocol entrypoints; use the supported format.

Run the campaign with RoboEnv's development interpreter. `simulator_python`
selects the native environment; `kinex python` builds the agent environment.
Do not install simulator packages into the agent environment.

## A concrete probe specification

This is a template for the current parser, not an installed environment or a
passed job. Replace paths with inspected executables and use real device names.
Paths resolve from the runner's working directory; absolute paths avoid ambiguity.

```json
{
  "roboenv": "/path/to/RoboEnv",
  "simulator_python": "/path/to/native-env/bin/python",
  "bun": "/path/to/bun",
  "task": "robotwin/stack-blocks-three",
  "seed": 0,
  "timeouts": {"python": 300, "boot": 1200, "execution": 300, "media": 600, "replay": 1200},
  "recording": {"head_camera": {"rate": 2}, "left_arm": {"rate": 5}},
  "args": {"arm": "left_arm", "camera": "head_camera"}
}
```

Configure GPU and renderer settings for the backend after checking availability.
License acceptance is a separately authorized setup operation. Large dependencies
and assets can use external storage; preserve old environments and evidence.

Check both owners' actual output locations before a long job. The current local
SDK creates native run evidence under the selected RoboEnv project's
`.cache/roboenv/runs`; a campaign `--out` on another filesystem does not move those
captures. Prefer arranging sufficient storage before launch. If storage must be
relocated, first close affected worlds and launchers, retain old evidence, verify
copied content and original path access, and record the new destination. Changing
resolved paths during an active episode can break sensor artifact ownership checks.

Set shell variables `ROBOENV_REPO`, `KINEX_REPO`, `RUN_SPEC` and `JOB_OUT` to the
inspected roots, private JSON file and a **new** output directory, then:

```sh
cd "$KINEX_REPO"
"$ROBOENV_REPO/.venv/bin/python" examples/simulator-acceptance/campaign.py --help
"$ROBOENV_REPO/.venv/bin/python" examples/simulator-acceptance/campaign.py \
  probe "$RUN_SPEC" --out "$JOB_OUT"
```

Pinned `bun` and `uv` must also be on child process `PATH`. Inspect `attempt.json`,
`execution.json`, `owner/verdict.json`, `media.json`, `replay.json` and failure logs.
A simple movement probe may have native task success false while its narrower
infrastructure checks pass. It is not task acceptance.
Require actual passing media and replay files and match replay to the owner
record checksum. A zero exit code alone is insufficient: some Isaac shutdown
paths exit Python before a pending return or exception reaches the caller.

## Full integration acceptance

The detailed rules live in the testing skill's
[acceptance contract](../.agents/skills/test-simulator-kinex/references/acceptance.md).
Select the task/rig/clock matrix and tolerances before controller development.

| Gate | Required evidence |
| --- | --- |
| A | Final recipe in a new prefix, dependencies/imports/assets, portable profiles |
| B | Measured control, geometry, contact, rendered calibration, clocks and lifecycle across rigs/states |
| C | Concurrent transport/identity fences, real no-model Kinex, media and independent replay |
| D | Actual visual model session, real control, owner outcome, recording/replay and failure diagnosis |
| Optional task performance | Immutable fresh-process trials meeting a requested success threshold |

The current `infrastructure.py` validates checksummed static, native, transport,
Kinex, recording and replay evidence before `campaign.py develop`. It does not
independently verify that preparation used a new prefix. For Gate A, also retain
the installation registration and log proving fresh environment creation,
dependency consistency and isolated import origins; match their lock and package
configuration to the current recipe and pair them with verified native assets.
Reusing an interpreter or passing this aggregator alone does not certify Gate A.
Development adds a `model` object (`provider`, `id`,
`effort`), private `auth_directory`, `prompt` and `infrastructure` to the run spec.
Verify actual outgoing provider identity and successful image requests. Credentials
and owner state never belong in copied controller content.

For integration acceptance, inspect representative session messages, raw tool
results, device calls and video frames alongside the source-matched receipts.
Require measured motion, intact sensors, correctly owned clocks/verdicts, cleanup
and independent replay. A controller reporting a missed grasp or failed perception
can establish a working loop. A broken driver, missing observation or replay
mismatch cannot be dismissed as model reasoning. Report unresolved intermittent
issues and supporting regression evidence explicitly.

Keep `integration_passed`, `controller_completed` and `task_success` separate in
the owner audit. The current `infrastructure.py` and `campaign.py develop` outputs
support that review; `aggregate.py` is a strict task-performance aggregator.
It has no task-success-optional integration CLI mode. Preserve its original results
and do not pass invented fields to change its meaning.

## Optional task-performance acceptance

Select from development results, use `campaign.py freeze`, then
`campaign.py validate --artifact ...` in fresh directories. Default finals require
three successes per task when task-performance acceptance is requested: two equivalent initial states and a third distinct state
excluded from development. BEHAVIOR must pin `activity_instance_id` in
`backend_kwargs`; a changed seed alone is insufficient. Its current native path
loads the base scene and then restores the selected training instance's task
state and robot pose before starting the episode clock. Verify the state-file
hash and effective initialization in owner evidence. RoboDojo YAM requires the
matching `env_cfg_type`, not just a changed robot label. Its released ARX-X5
layouts require the explicit `layout_config: arx_x5` practice adaptation.

For the current freezer, the selected tool name must match both its manifest and
directory: `name: solve_task` belongs at `.kinex/tool/solve_task/tool.yaml`.
Normal Kinex discovery can accept a different directory name, so a successful
development call alone does not prove that the artifact can be frozen. Normalize
and verify the draft before final selection.

`aggregate.py` consumes a separate final spec with infrastructure/artifact paths,
task IDs, complete development/exposure history and trials. Each trial supplies
`directory` and `role` (`repeatability` or `held_out`). `role` is not a per-job
field. The frozen tool returns `completed: true` without error; the native predicate
independently establishes task success. Both are required alongside actual writes,
unchanged inputs, valid media and command replay.

## Failures and resumption

Retain failed trials and logs. Matching replay of failure remains failed task
acceptance. Using final feedback for more development exposes those cases; record
them as tuning data and reserve a new held-out case. Do not edit frozen code during
validation or silently replace controller failures with retries.

If Kinex interrupts repeated identical calls for loop confirmation, retain that
execution failure. For operations such as visual scanning, use a finite plan with
explicit camera/body viewpoints, visited state and a clear stopping condition.
Do not disable safeguards or repeat the denied sequence. Resume an appropriate
bounded workflow under existing authorization; missing permission remains a
separate prerequisite.

Record source/dirty hashes, dependencies, assets, effective initialization, stage
commands and result paths. On resume, recheck evidence against current inputs.
Rerun affected gates; reuse unchanged evidence only with a matching identity.
Recheck external prerequisites and retain later license/resource authorizations
instead of reintroducing old blockers.

An exited campaign parent does not imply that its Kinex runner or native world
has stopped. Match active processes to the saved workspace/session before recovery.
Let bounded work finish, collect owner evidence through the existing finalizer,
and close the owned world through its shutdown API. Record recovered evidence
separately from the interruption; missing completion evidence remains missing.

For Harbor, follow [Harbor tasks](harbor-tasks.md), validate with the real loader
and run within the requested image and agent/verifier boundary. A local job cannot
certify deployment isolation, and this skill grants no publication authority.
