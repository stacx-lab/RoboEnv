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
profiles. See [Kinex local testing](kinex-local.md).

## Implementation boundary

[RoboCasaAdapter](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py) connects
lifecycle, instruction, native success checking, budget, and owner evidence to the
shared worker. Its [backend](../runtime/src/roboenv_runtime/adapters/robocasa/backend.py)
creates one `robosuite` environment and configures its controllers. The
[binding factories](../runtime/src/roboenv_runtime/bindings/backends/robocasa/__init__.py)
attach selected joint, velocity, gripper, camera and world endpoints to that environment.
The shared `WorldSession` schedules realtime ticks and serializes them with device
operations and finalization. Drivers neither own physics nor start another world.

The backend still contains private scene-inspection helpers, and adapter `observe`
returns privileged object poses for owner use. These are not public device
capabilities or examples for a sensor-only integration. Local IK/FK belongs in the
exported driver; task-solving programs belong in the harness. Use the
[integration guide](simulator-integration.md) when adapting another simulator.

## Devices and control

The standard [Panda–Omron catalog](../catalog/robots/robocasa/robocasa-panda-omron/workspace.yaml)
has eight devices. Driven devices also implement `load`, `unload`, and `status`;
these do not reset or close the world.

| Device | Type | Service read / write |
|---|---|---|
| `arm` | `joint` | Seven joint positions/velocities and flange pose; seven absolute joint targets in radians |
| `torso` | `joint` | Lift state and top pose; one absolute target in metres |
| `base` | `velocity` | World pose and measured chassis-origin twist; normalized body-frame `vx`, `vy`, `wz` commands |
| `gripper` | `gripper` | Finger joint state; one signed command in `[-1,1]`, positive closes |
| `camera_left`, `camera_right`, `camera_wrist` | `rgbd` | Aligned RGB and metric depth files; no write |
| `world` | `world` | Profile only; reference frame and open workspace bounds |

Actuators implement `write` and `stop`. Arm/torso targets follow
`intrinsics.model.joints` order and replace the active goal. Repeated targets continue
motion without restarting interpolation. Their native joint-position controllers
use torque control. Base and gripper commands persist until replaced or stopped;
writing one device preserves all other goals.

The backend selects absolute `JOINT_POSITION` arm control and absolute torso
inputs while retaining native base and gripper semantics. During each interval it
uses the normal `robosuite` step, preserving physics, observables, and task hooks.
It replaces the composite goal update for that call so arm/lift goals persist and
base/gripper inputs are refreshed, then restores the hook even on failure.

A successful write returns `accepted: true`, never arrival. `stop` holds the
arm/torso/gripper or requests zero base velocity. It neither teleports joints nor
pauses realtime physics; an in-flight synchronous interval finishes before stop.
Actuator reads include faults and simulation acquisition `sample_time_s`. Camera
acquisition timestamps use Unix seconds, as required by Kinex. Reads do not step.

Choose the clock at launch; these mutually exclusive flags override the task YAML
for this run without changing the catalog:

```sh
roboenv launch robocasa/pick-place-microwave --realtime
roboenv launch robocasa/pick-place-microwave --sync
```

Without either flag, launch uses the task's `backend_kwargs.realtime`. Checked-in
and discovered RoboCasa tasks explicitly set `realtime: true`. Python owners can
pass `realtime=True` or `realtime=False` to `roboenv.launch`; `None` uses the task
setting. Clock override support is backend-specific; unsupported overrides fail
explicitly.

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

The service owns the control period, normally 0.05 simulated seconds. Joint and
velocity devices declare `intrinsics.control.execution`: realtime
with `command_timeout_s: null` (retain commands until replaced/stopped), or stepped
with `interval_s: 0.05` in a live synchronous export. The static catalog describes
the default realtime mode. Reads, recording, status, local IK/FK, and stop
never request advancement. Kinex's `servo_to` and `track` use the same loop in
both modes; their wall-clock rate cap does not set simulation time.

Both modes count `action_frames`. The default `max_steps: null` is unbounded;
a positive `max_steps` sets a hard native control-step limit. Set
`stop_on_success: true` to freeze on native success, including success on the last
allowed step. After termination, reads and status remain available but writes fail
with `episode_finished`. Original task success rules remain.

`roboenv status --json` uses `ROBOENV_ENDPOINT` when no run is supplied and reports
simulation state and budget without success or failure metrics, including after
termination. Device responses and terminal errors also omit the judgment.
Launch with `--allow-stop` to let the assigned agent call `roboenv stop --json`;
this seals and returns the final verdict, releases the worker, and retains an
immutable receipt. The local owner can always stop and close its run.
`--runtime-root` and `--run-dir` support packaged deployments without mounting a
source checkout. Private evidence includes `terminal.json` and the action journal.
Submission atomically freezes the terminal state. Realtime pacing is best effort:
slow simulation or camera requests can cause lag, but physics steps are not skipped.
Replaying commands without their simulation-frame timing does not reproduce a run.

## Catalog and generation

Standard kitchen tasks reuse the same hardware configuration. Scene layout and
initial placement do not change its static profiles. Different robots, controllers,
resolutions, or camera arrangements require a matching catalog configuration.

```text
catalog/robots/robocasa/robocasa-panda-omron/
  workspace.yaml
  devices/<name>/profile.yaml
  devices/<name>/driver.py       # except world
  devices/_roboenv.py             # one shared portable transport per collection
src/roboenv/devices/bindings/defaults/robocasa/robocasa-panda-omron.yaml
catalog/tasks/<simulator>/<task-name>/task.yaml
```

The robot directory contains only `workspace.yaml` and `devices/`, ready to copy
as the workspace’s `.kinex/` directory. Binding plans and task/session configuration stay outside
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
The standard map binds logical `base` to native entity `base`; custom logical names
are supported. Renaming updates the profile, connection target, stop scope, and
dependent mounts. Native controller composition remains internal to the world;
public devices expose their individual typed operations.

`config.camera` selects an existing sensor, which must also be enabled through
`backend_kwargs.camera_names`. A profile does not create a camera.

Profiles use Kinex's strict per-type schemas. Access
uses typed `access.connection` and explicit driver actions. Manipulators store home
positions by joint name, optional speed capability (`null` here), feedback channels,
and stop scope. Gripper command and measured finger dimensions are independent;
unsupported sensors and aperture conversion are not invented. World has no access
or extrinsics. Required unknown facts use `???`; optional unavailable values use
`null` according to each schema.

## Portable drivers and local kinematics

Catalog drivers share `devices/_roboenv.py` for HTTP/Unix transport and **do not
import RoboEnv**. Copy the device collection, including this one shared file.
Each `devices/<name>/driver.py` locates it relative to its own file, independently
of the working directory. Set `ROBOENV_ENDPOINT` for the current service; the catalog carries no
session connection file. Drivers default to persistent processes and discover the
selected episode once; runners can pin session/generation explicitly. Drivers and
SDK clients retain their original episode identity. Reload the driver after changing
the profile or episode connection. Configuration, HTTP connections, and local
kinematics models are reused within the process.

Arm/torso drivers additionally implement `fk(joints)` and `ik(pose, seed?)` using
`kinex.joints` and `intrinsics.model`, a serial chain with an
explicit numerical IK configuration. Base and gripper adapters also use Kinex’s
typed command/feedback validators. Kinex supplies its Python
library on the driver's PYTHONPATH; parsing profiles requires PyYAML. The simulator
service exposes no FK/IK RPC and has no solver dependency on Kinex.

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

MuJoCo's soft constraints permit small measured overshoots. Reads preserve actual joints
and measured flange transforms; explicit write/FK targets and IK seeds remain
range-checked. Stop holds the measured configuration, not home. Neither accepted
commands nor successful IK prove arrival or recovery from contact.

Panda motor control ranges are torque limits, not velocity limits. Arm/torso servo
control provides no hard measured velocity cap. Base actuator ranges bound velocity
targets, and gripper `speed` is a normalized increment. Workspace velocity caps are
explicitly `null` (no additional cap), distinct from unresolved `???`. Open world
AABB sides and calibration readiness do not imply collision-free or reachable space.

## Frames and cameras

The mobile `base` omits a static `extrinsics.mount` and declares
`intrinsics.feedback.pose_frame: world`. Its live pose supplies the world-to-base
edge. Measured base velocities are at the chassis origin in body coordinates, in
m/s and rad/s; they are not normalized commands or raw generalized joint velocity.

The torso is a one-joint endpoint with a declared `top` frame. The arm and
left/right cameras mount on `torso:top`; the gripper and wrist camera mount on
`arm:flange`. Both frames are declared as null in `intrinsics.frames`; `read.frame_poses`
supplies their measured transforms relative to the
canonical device frame. Names such as top/flange are declared, never implicit.

| Camera | Native sensor | Vertical FOV | Rigid parent |
|---|---|---:|---|
| Left | `robot0_agentview_left` | 60° | Torso top |
| Right | `robot0_agentview_right` | 60° | Torso top |
| Wrist | `robot0_eye_in_hand` | 75° | Arm flange |

All three output 256×256 images with principal point `(128,128)` and no camera
randomization. Mounts use OpenCV optical axes: +X right, +Y down, +Z forward,
converted from MuJoCo's camera convention. Robot-mounted views are reusable across
kitchens; a layout-specific viewer camera is not a static robot sensor.

RGB and depth render together without stepping and are flipped upright. `file` is
RGB PNG; `depth_file` is a 16-bit binary PGM (P5, big-endian, maxval 65535). Multiply
raw depth by `intrinsics.depth.scale_m_per_unit` (0.001) for optical-axis metres. Zero marks invalid,
background, or unrepresentable depth; values are not clipped into fake surfaces.
Conversion uses the live model's near/far planes, not fixed sensor-range calibration.
This is quantized synthetic depth, with no assumed hardware noise model. Color and
depth reads carry matching acquisition Unix timestamps in `sample_time_s` and
`depth_sample_time_s`, plus explicit dimensions. They timestamp the paired render
while the world state is locked; transport receipt time is separate. Scale and encoding live in the profile. Color facts
use `intrinsics.color` with pinhole projection, distortion, and derived FOV; depth
uses `intrinsics.depth` with color registration and optical-Z quantity. The camera
table lists nominal native FOV; exported FOV follows Kinex’s pixel-edge convention.

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
Kinex schema conformance, copied drivers without RoboEnv imports, HTTP episode
isolation, camera routing, file lifetime, and unchanged catalogs after launch.
Set `ROBOENV_ROBOCASA_TEST_PYTHON` to select the interpreter explicitly.

Camera transport tests use synthetic pixels. Kitchen boot, rendered calibration,
task success, and fresh-process replay require separate evidence. See
[Kinex local testing](kinex-local.md), [Harbor tasks](harbor-tasks.md), and
[simulator integration](simulator-integration.md) for those workflows.


## Control contracts and assembly

Owner bindings are explicit: `robocasa.joint/1` (configured native arm/lift part),
`robocasa.velocity/1`, `robocasa.gripper/1`, `robocasa.camera/1`, and `robocasa.world/1`.
They share one world and expose individual read/write/stop operations.
Kinex devices() submits mixed reads and writes concurrently. Each write in a stepped
world advances its own interval; realtime writes replace that device's goal. Native
world access remains serialized. The operator ensures enabled devices do not overlap;
there is no group coordinator or automatic controller switching.
