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
    # Submit a hold reference; write returns immediately.
    arm.write([arm.read()["positions"]], duration_s=0.5)
    env.device("world").wait(0.5)
    print(arm.read()["positions"])  # Measured feedback, not an arrival guarantee.
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
The shared `WorldSession` advances pending commands and serializes them with device
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
| `arm` | `joint` | Seven joint positions/velocities; seven absolute joint targets in radians. Compute flange pose with explicit FK. |
| `torso` | `joint` | Lift joint state; one absolute target in metres. Compute the top frame with explicit FK. |
| `base` | `velocity` | Measured chassis-origin twist; normalized body-frame `vx`, `vy`, `wz` commands with advertised bounds [-1, 1] |
| `gripper` | `gripper` | Measured aperture in metres; direction in `[-1, 1]`, positive closes |
| `camera_left`, `camera_right`, `camera_wrist` | `rgbd` | Aligned RGB and metric depth files; no write |
| `world` | `world` | World clock, reference frame and optional workspace bounds |

Actuators implement `write` and `stop`. Arm/torso targets follow
`intrinsics.model.joints` order and replace the active goal. Joint write streams complete action vectors without planning or arrival waits. Their native joint-position controllers
use torque control. Base and gripper commands persist until replaced or stopped;
writing one device preserves all other goals.

The backend selects absolute `JOINT_POSITION` arm control and absolute torso
inputs while retaining native base and gripper semantics. During each interval it
uses the normal `robosuite` step, preserving physics, observables, and task hooks.
It replaces the composite goal update for that call so arm/lift goals persist and
base/gripper inputs are refreshed, then restores the hook even on failure.

Writes return `execution_id` and a measured `state` snapshot.
Wait for execution with `world.wait(duration_s)`.
Stop holds measured arm/torso posture, retains the accumulated gripper target and
preload, or zeroes base velocity without stepping. Explicit open releases the grasp. Reads
include faults and acquisition timestamps. Realtime continues while idle;
stepped advances during world waits.

```sh
roboenv launch robocasa/pick-place-microwave
```

See [the common execution contract](simulator-integration.md#execution-and-cancellation)
for schemas, timing, cancellation and native controller support.

## Catalog and generation

Standard kitchen tasks reuse the same hardware configuration. Scene layout and
initial placement do not change its static profiles. Different robots, controllers,
resolutions, or camera arrangements require a matching catalog configuration.

```text
catalog/robots/robocasa/robocasa-panda-omron/
  workspace.yaml
  devices/<name>/profile.yaml
  devices/<name>/driver.py
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
joints are initialization defaults, not a task's current pose.

Regenerate explicitly with the prepared RoboCasa interpreter:

```sh
/path/to/robocasa/python tools/generate_robocasa_config.py
```

The [generator](../tools/generate_robocasa_config.py) compiles the robot in an empty
arena, installs robot-mounted cameras, and initializes controllers. It loads no
kitchen task, renders nothing, and takes no physics steps. `--out DIRECTORY` writes
a separate robot directory. Repeated generation is deterministic; rigid mounts
are also checked after changing arm/lift configuration.

A custom task selects a robot arrangement whose owner-side `bindings.yaml` replaces
the complete default selection. Logical names, profiles, and mounts must match the
published device directories. Keep this binding plan out of public workspace exports.
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
python3 driver.py write '{"actions": [[0, 0, 0, -2, 0, 2, 0.7]], "duration_s": 0.05}'
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
and measured flange transforms. Writes delegate limits to the native simulator;
FK targets and IK seeds retain geometric solver checks. Stop holds the measured configuration, not home. Neither accepted
commands nor successful IK prove arrival or recovery from contact.

Panda motor control ranges are torque limits, not velocity limits. Arm/torso servo
control provides no hard measured velocity cap. Base actuator ranges bound velocity
targets, and gripper `speed` is a normalized increment. Workspace velocity caps are
explicitly `null` (no additional cap), distinct from unresolved `???`. Open world
AABB sides and calibration readiness do not imply collision-free or reachable space.

## Frames and cameras

The mobile `base` is an unlocalized root: its profile omits `extrinsics.mount`
and `feedback.pose_frame`, and reads/write states omit global `pose` and empty
`frame_poses`. Robot-local transforms still work; a world-relative query requires
an explicit estimate. Measured base velocities are at the chassis origin in body
coordinates, in m/s and rad/s, independently of command units.

Writes use named axes, for example
`{mode: "drive", velocity: {vx: 0.05, vy: 0, wz: 0}, duration_s: 0.5}`.
The profile publishes the native controller's normalized input bounds `[-1, 1]`
for all three axes. `limit_handling: native` preserves native behavior: inputs
are converted into actuator coordinates and robosuite clips the resulting actuator
outputs; MuJoCo also enforces actuator limits. Kinex adds no input clipping or
rejection. These input bounds do not promise measured speed limits. The pinned
controller computes a scaled input but uses its original input in goal conversion,
so this is not equivalent to clipping each body-frame input before conversion.

The torso is a one-joint endpoint with a declared `top` frame. The arm and
left/right cameras mount on `torso:top`; the gripper and wrist camera mount on
`arm:flange`. Each device's `intrinsics.kinematics.frame` declares its endpoint;
no duplicate entry in `intrinsics.frames` is needed. Joint reads contain joint measurements; local FK computes
these poses relative to the device base. FK agrees with MuJoCo body transforms;
the CPU conformance test checks 101 configurations for each device.

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
depth reads carry one simulation-clock `timestamp`; dimensions remain in the profile.
It timestamps the paired render while the world state is locked.
Scale and encoding live in the profile. Color facts
use `intrinsics.color` with pinhole projection, distortion, and derived FOV; depth
uses `intrinsics.depth` with color registration and optical-Z quantity. The camera
table lists nominal native FOV; exported FOV follows Kinex’s pixel-edge convention.

HTTP downloads are episode-scoped and client-local. SDK files last until client
cleanup; one-shot driver files remain readable after process exit. RGB-only custom
profiles remain supported. Other benchmark observation policies need separate review.

## Isolation and validation

Task context is empty. Object poses, fixtures, layout/style, scene seeds, and verdicts
stay private to the owner/verifier. The HTTP service exposes device operations and
safe discovery/health/request status, with no scene API, task macros, or policy runner.
Operation-status queries cannot retrieve cached owner results.

External orchestrators launch the same service and deliver its public endpoint to
agents. They own container users, network isolation, resource allocation and cleanup.
Owner sockets and evidence must remain inaccessible to agent containers. Each local
launch creates a fresh episode identity; clients pin that identity on attachment.

`bash tools/check.sh` covers CPU, privacy, format, and packaging checks. With a
prepared RoboCasa interpreter, tests exercise real Panda–Omron controllers in a
CPU Lift scene, both clocks, repeated goals, stop, invalid inputs, local FK/IK without
state mutation, and rejected solver RPCs. They also check deterministic generation,
Kinex schema conformance, copied drivers without RoboEnv imports, HTTP episode
isolation, camera routing, file lifetime, and unchanged catalogs after launch.
Set `ROBOENV_ROBOCASA_TEST_PYTHON` to select the interpreter explicitly. When the
synced RoboCasa source and its kitchen assets are also present, a CPU test builds one
kitchen at one seed three times in each of two processes and requires the same layout,
style, instruction, object models and poses every time.

Camera transport tests use synthetic pixels. Kitchen boot, rendered calibration,
task success, and fresh-process replay require separate evidence. See
[Kinex local testing](kinex-local.md), [agent integration](device-documentation.md), and
[simulator integration](simulator-integration.md) for those workflows.


## Control contracts and assembly

Owner bindings are explicit: `robocasa.joint/1` (configured native arm/lift part),
`robocasa.velocity/1`, `robocasa.gripper/1`, `robocasa.camera/1`, and `robocasa.world/1`.
They share one world and expose individual read/write/stop operations.
Kinex devices() submits mixed reads and writes concurrently. Each write in a stepped
world advances pending commands through shared native intervals. Native
world access remains serialized. The operator ensures enabled devices do not overlap;
there is no group coordinator or automatic controller switching.
