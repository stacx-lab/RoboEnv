# RoboCasa

RoboEnv runs one RoboCasa task per service. All devices share that world's clock,
state, and verdict. The agent receives task instructions, device facts, and sensor
readings; perception, planning, and policy execution belong to its harness.

## Run a task

After [preparing RoboCasa](setup.md), launch a catalog task:

```sh
roboenv prepare robocasa
roboenv launch robocasa/pick-place-microwave --port 8123
```

Use separate ports for concurrent worlds, `--port 0` for automatic assignment, or
`--detach` for a background run. `roboenv stop ENDPOINT` stops a locally owned run.
The default host is `127.0.0.1`; containers can bind `0.0.0.0`.

```python
import roboenv

with roboenv.connect("http://127.0.0.1:8123") as env:
    print(env.task.instruction)
    arm = env.device("arm")
    arm.write(arm.read()["positions"])
    arm.stop()
    image = env.device("camera_left").read()
    print(image["file"])
```

A Python owner can use `with roboenv.launch(task=TASK, port=8123) as run:` and
connect to `run.endpoint`. Closing an attached client releases client resources,
not the world. Device results contain data directly; execution receipts stay in
the service journal. An uncertain motion result carries a request ID and is never
automatically retried.

For two independent worlds, run these commands in separate terminals:

```sh
roboenv launch robocasa/pick-place-microwave --port 8123
roboenv launch robocasa/arrange-vegetables --port 8124
```

Copy the same robot directory into two harness workspaces and set each process's
`ROBOENV_ENDPOINT` to its world. Changing endpoints does not require editing
profiles. See [Complex local testing](complex-local.md).

## Implementation boundary

[RoboCasaAdapter](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py) connects
lifecycle, instruction, native success checking, budget, and owner evidence to the
shared worker. Its [backend](../runtime/src/roboenv_runtime/adapters/robocasa/backend.py)
creates one `robosuite` environment and configures its controllers. The
[native binding](../runtime/src/roboenv_runtime/bindings/backends/robocasa/devices.py)
attaches selected devices to that environment through `robocasa.device/1`.
The shared `WorldSession` schedules realtime ticks and serializes them with device
operations and finalization. Drivers neither own physics nor start another world.

The backend still contains private scene-inspection helpers, and adapter `observe`
returns privileged object poses for owner use. These are not public device
capabilities or examples for a sensor-only integration. Local IK/FK belongs in the
exported driver; task-solving programs belong in the harness. Use the
[integration guide](simulator-integration.md) when adapting another simulator.

## Devices and control

The standard [Panda–Omron catalog](../catalog/robots/robocasa-panda-omron/workspace.yaml)
has eight devices. Driven devices also implement `load`, `unload`, and `status`;
these do not reset or close the world.

| Device | Type | Service read / write |
|---|---|---|
| `arm` | `manipulator` | Seven joint positions/velocities and flange pose; seven absolute joint targets in radians |
| `torso` | `manipulator` | Lift state and flange pose; one absolute target in metres |
| `body` | Mobile `body` | Planar joint state and world pose; normalized forward, left, yaw-rate commands |
| `gripper` | `end_effector` | Finger joint state; one signed command in `[-1,1]`, positive closes |
| `camera_left`, `camera_right`, `camera_wrist` | `rgbd` | Aligned RGB and metric depth files; no write |
| `world` | `world` | Profile only; reference frame and open workspace bounds |

Actuators implement `write` and `stop`. Arm/torso targets follow
`intrinsics.joints` order and replace the active goal. Repeated targets continue
motion without restarting interpolation. Their native joint-position controllers
use torque control. Body and gripper commands persist until replaced or stopped;
writing one device preserves all other goals.

The backend selects absolute `JOINT_POSITION` arm control and absolute torso
inputs while retaining native base and gripper semantics. During each interval it
uses the normal `robosuite` step, preserving physics, observables, and task hooks.
It replaces the composite goal update for that call so arm/lift goals persist and
base/gripper inputs are refreshed, then restores the hook even on failure.

A successful write returns `accepted: true`, never arrival. `stop` holds the
arm/torso/gripper or requests zero base velocity. It neither teleports joints nor
pauses realtime physics; an in-flight synchronous interval finishes before stop.
Reads include faults and simulation acquisition `sample_time_s` and do not step.

Choose the clock at launch; these mutually exclusive flags override the task YAML
for this run without changing the catalog:

```sh
roboenv launch robocasa/pick-place-microwave --realtime
roboenv launch robocasa/pick-place-microwave --sync
```

Without either flag, launch uses the task's `backend_kwargs.realtime`. Checked-in
and discovered RoboCasa tasks explicitly set `realtime: true`. Python owners can
pass `realtime=True` or `realtime=False` to `roboenv.launch`; `None` uses the task
setting. Clock selection currently supports RoboCasa only; other simulators reject
an explicit override.

To change the task default before launch:

```yaml
backend_kwargs:
  task_name: PickPlaceCounterToMicrowave
  realtime: false  # true is the default
```

| Mode | Write and time behavior |
|---|---|
| Realtime | Return after accepting the goal; background physics continues while the agent thinks. |
| Synchronous | Update the goal, advance the whole world by one native control interval, then return and pause. |

The service owns the control period, normally 0.05 simulated seconds. Profiles do
not expose it to controller tools. Reads, recording, status, local IK/FK, and stop
never request advancement. Complex's `servo_to` and `track` use the same loop in
both modes; their wall-clock rate cap does not set simulation time.

Both modes count `action_frames` with `max: null`; the original horizon and legacy
`max_steps` do not terminate this configuration. Original task success rules remain.
Submission atomically freezes the terminal state. Realtime pacing is best effort:
slow simulation or camera requests can cause lag, but physics steps are not skipped.
Replaying commands without their simulation-frame timing does not reproduce a run.

## Catalog and generation

Standard kitchen tasks reuse the same hardware configuration. Scene layout and
initial placement do not change its static profiles. Different robots, controllers,
resolutions, or camera arrangements require a matching catalog configuration.

```text
catalog/robots/robocasa-panda-omron/
  workspace.yaml
  devices/<name>/profile.yaml
  devices/<name>/driver.py       # except world
catalog/bindings/robocasa-panda-omron.yaml
catalog/tasks/<simulator>/<task-name>/task.yaml
```

The robot directory contains only `workspace.yaml` and `devices/`, ready to copy
into `.complex/robots/`. Binding plans and task/session configuration stay outside
that copy. Launch reads the catalog without rewriting it.

Profiles derive from the pinned compiled MJCF, robot initialization, native
controllers, and RoboCasa camera definitions. A real Panda URDF is not a substitute
for the simulated model. Model-derived sections use `source: introspected`; home
joints are initialization defaults, not a task's current pose. Exact source hashes
are recorded in [the provenance receipt](robocasa-config.provenance.json).

Regenerate explicitly with the prepared RoboCasa interpreter:

```sh
/path/to/robocasa/python tools/generate_robocasa_config.py \
  --provenance docs/robocasa-config.provenance.json
```

The [generator](../tools/generate_robocasa_config.py) compiles the robot in an empty
arena, installs robot-mounted cameras, and initializes controllers. It loads no
kitchen task, renders nothing, and takes no physics steps. `--out DIRECTORY` writes
a separate robot directory. Repeated generation is deterministic; rigid mounts
are also checked after changing arm/lift configuration.

A custom task's `device_bindings` replaces the complete default selection. Logical
names, profiles, and mounts must match the published device directories.
The standard map binds logical `body` to native entity `base`; logical and native
names need not match. Renaming requires changing the directory, profile name,
binding key, and dependent mount references together. The binding also supports a
custom composite `robot` actuator with a native action vector; it cannot be selected
alongside its component actuators and is not part of the standard eight-device rig.

`config.camera` selects an existing sensor, which must also be enabled through
`backend_kwargs.camera_names`. A profile does not create a camera.

## Portable drivers and local kinematics

Catalog drivers use standard-library HTTP/Unix transport and **do not import
RoboEnv**. Set `ROBOENV_ENDPOINT` for the current service; the catalog carries no
session connection file. New one-shot drivers discover the selected episode;
runners can pin session/generation explicitly. Established SDK clients retain
their original episode identity.

Arm/torso drivers additionally implement `fk(joints)` and `ik(pose, seed?)` using
`complex.kinematics` and `intrinsics.kinematics.chain`. Complex supplies its Python
library on the driver's PYTHONPATH; parsing profiles requires PyYAML. The simulator
service exposes no FK/IK RPC and has no solver dependency on Complex.

The chain stores fixed link origins, joint axes, anchors, and reference positions
in profile joint order. FK returns the flange in the device base frame: metres and
unit quaternion xyzw. Explicitly seeded IK and FK work offline. Without a seed, IK
reads current joints once and projects only that local seed into limits. The damped
least-squares solver may fail to converge; it does not validate a motion path.

```sh
python3 driver.py fk '{"joints": [0, 0, 0, -2, 0, 2, 0.7]}'
python3 driver.py write '{"target": [0, 0, 0, -2, 0, 2, 0.7]}'
```

### Joint limits and recovery

Panda's nominal limits are in radians; the lift spans 0–0.34 m, with model home 0.2 m.

| Joint | Minimum | Maximum |
|---|---:|---:|
| 1 | -2.8973 | 2.8973 |
| 2 | -1.7628 | 1.7628 |
| 3 | -2.8973 | 2.8973 |
| 4 | -3.0718 | -0.0698 |
| 5 | -2.8973 | 2.8973 |
| 6 | -0.0175 | 3.7525 |
| 7 | -2.8973 | 2.8973 |

MuJoCo's soft constraints permit small measured overshoots. Profiles retain native
constraint settings under `joint_limit_semantics`. Reads preserve actual joints
and measured flange transforms; explicit write/FK targets and IK seeds remain
range-checked. Stop holds the measured configuration, not home. Neither accepted
commands nor successful IK prove arrival or recovery from contact.

Panda motor control ranges are torque limits, not velocity limits. Arm/torso servo
control provides no hard measured velocity cap. Base actuator ranges bound velocity
targets, and gripper `speed` is a normalized increment. Workspace velocity caps are
explicitly `null` (no additional cap), distinct from unresolved `???`. Open world
AABB sides and calibration readiness do not imply collision-free or reachable space.

## Frames and cameras

The robot frame component is rooted at `body`; `world` is separate. The body profile
omits `extrinsics.mount`. Its measured world pose does not implicitly create that
edge, and an absent mount is not identity.

The torso remains a one-joint manipulator because Complex represents its live
base-to-flange transform. The arm and left/right cameras mount on `torso:flange`;
the gripper and wrist camera mount on `arm:flange`. The gripper mount is its rigid
flange-to-TCP offset. Merging the lift into the body would require named moving
body frames in Complex.

| Camera | Native sensor | Vertical FOV | Rigid parent |
|---|---|---:|---|
| Left | `robot0_agentview_left` | 60° | Torso flange |
| Right | `robot0_agentview_right` | 60° | Torso flange |
| Wrist | `robot0_eye_in_hand` | 75° | Arm flange |

All three output 256×256 images with principal point `(128,128)` and no camera
randomization. Mounts use OpenCV optical axes: +X right, +Y down, +Z forward,
converted from MuJoCo's camera convention. Robot-mounted views are reusable across
kitchens; a layout-specific viewer camera is not a static robot sensor.

RGB and depth render together without stepping and are flipped upright. `file` is
RGB PNG; `depth_file` is a 16-bit binary PGM (P5, big-endian, maxval 65535). Multiply
raw depth by `depth_scale = 0.001` for optical-axis metres. Zero marks invalid,
background, or unrepresentable depth; values are not clipped into fake surfaces.
Conversion uses the live model's near/far planes, not fixed sensor-range calibration.
This is quantized synthetic depth, with no assumed hardware noise model.

HTTP downloads are episode-scoped and client-local. SDK files last until client
cleanup; one-shot driver files remain readable after process exit. RGB-only custom
profiles remain supported. Other benchmark observation policies need separate review.

## Isolation, Harbor, and validation

Task context is empty. Object poses, fixtures, layout/style, scene seeds, and verdicts
stay private to the owner/verifier. The HTTP service exposes device operations and
safe discovery/health/request status, with no scene API, task macros, or policy runner.
Operation-status queries cannot retrieve cached owner results.

Harbor uses the same task resolver and an isolated service network per trial, with
`ROBOENV_ENDPOINT=http://sim:8123`. Readiness includes device binding and trial-local
profile/driver publication under `/run/roboenv/robot`. The agent runs as `agent`;
the verifier accesses the protected owner socket and evidence as `root`. Each restart
creates a fresh episode identity. Harbor cleans up even when the agent fails.

`bash tools/check.sh` covers CPU, privacy, format, and packaging checks. With a
prepared RoboCasa interpreter, tests exercise real Panda–Omron controllers in a
CPU Lift scene, both clocks, repeated goals, stop, invalid inputs, local FK/IK without
state mutation, and rejected solver RPCs. They also check deterministic generation,
Complex schema conformance, copied drivers without RoboEnv imports, HTTP episode
isolation, camera routing, file lifetime, and unchanged catalogs after launch.
Set `ROBOENV_ROBOCASA_TEST_PYTHON` to select the interpreter explicitly.

Camera transport tests use synthetic pixels. Kitchen boot, rendered calibration,
task success, and fresh-process replay require separate evidence. See
[Complex local testing](complex-local.md), [Harbor tasks](harbor-tasks.md), and
[simulator integration](simulator-integration.md) for those workflows.
