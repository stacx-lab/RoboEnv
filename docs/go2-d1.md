# Go2 + D1-550 in Isaac Lab

`isaaclab/go2-d1-control` is a flat-ground control practice environment. The
Go2 base uses a feedback trot and analytic leg IK. The D1 uses independent arm
joint control with local numerical IK. There is no whole-body policy, learned
checkpoint, external base force, or root-pose shortcut during control.

The accepted wrist D405 and head D435 mounts are preserved. The initial arm is
folded (`Joint2=-0.65`, `Joint3=1.2`, `Joint5=-0.55` radians) for locomotion.
The control practice world contains a floor and the robot. Two additional task
worlds use the same controllers: `go2-d1-reach` and `go2-d1-pick-place` (below).

## Prepare and launch

Use the isolated Isaac Lab installation described in [isaaclab.md](isaaclab.md).
Build the robot and cameras using [the geometry tools](../tools/robots/README.md),
then prepare a separate dynamics bundle:

```sh
# Use the visualization interpreter with trimesh and pycollada installed.
python tools/prepare_go2_d1.py --source generated/go2-d1-cameras \
  --out .cache/isaaclab/go2-d1

source .venv/bin/activate
roboenv launch isaaclab/go2-d1-control --port 8123
```

The Go2–D1 simulator worker defaults `OMNI_KIT_ACCEPT_EULA=YES` for this launch.
An explicitly set value takes precedence.

Preparation converts mesh files to OBJ, records file checksums, and refuses to
overwrite an existing bundle. To refine the URDF, prepare a new bundle, update the
affected device profiles under `catalog/robots/isaaclab/isaaclab-go2-d1/devices/`
to match its geometry, and point the task's `backend_kwargs.asset_manifest` to the
new `assets.json`. Catalog/runtime geometry disagreement fails at startup.
Do not edit a bundle during an episode; recordings pin its identity.

Robot profiles and the three task definitions are checked in under
`catalog/robots/isaaclab/isaaclab-go2-d1/` and `catalog/tasks/isaaclab/`.
Launch uses these definitions directly; no catalog generation step is required.
Internal binding defaults are separate from the public catalog. The installed
SDK includes the Go2–D1 default at
`roboenv/devices/bindings/defaults/isaaclab/isaaclab-go2-d1.yaml`, so copied catalogs
resolve without a neighboring source-tree binding directory.

Portable drivers share one `devices/_roboenv.py`. Copy the whole device collection,
including that file; an individual device directory is not a complete bundle.
SDK/CLI agents can also use the [device documentation exporter](device-documentation.md)
with live descriptors, independently of the Kinex file-driver projection.

## Inputs and outputs

| Device | Input | Measured output |
| --- | --- | --- |
| `base` | Body-frame `vx`, `vy`, `wz`; ground-relative `height`, `roll`, `pitch` | World pose, height, six body-frame velocities, faults, simulation timestamp |
| `arm` | Six absolute D1 joint angles in URDF order `Joint1` through `Joint6` | Positions, velocities, flange pose relative to arm mount |
| `gripper` | Normalized opening: 0 closed, 1 open | Two finger joint positions and velocities |
| `camera_head` | Read | D435-mounted RGB and registered optical-axis depth |
| `camera_wrist` | Read | D405-mounted RGB and registered optical-axis depth |
| `world` | `read`, `wait(duration_s)` | Shared simulation time; static world reference |

Base commands accept finite numeric values for all six velocity/posture components;
the public profile declares no range bounds. The demonstration scripts exercise
`vx ±1.0 m/s`, `vy ±0.5 m/s`, `wz ±π/6 rad/s` (±30°/s),
`height 0.26–0.34 m`, and `roll/pitch ±0.12 rad` (about 6.9 degrees).
Tracking depends on the combined command and arm configuration. The gait rate
limits its internal command and uses velocity/posture feedback. Its stride and
feedback corrections are bounded, and private leg targets are limited to reachable positions
and joint bounds. Foot lift is 6.5 cm, the gait period is 0.32 s, and each leg spends 60% of
its cycle in stance. Acceleration limits are 0.5 m/s² longitudinally,
0.4 m/s² laterally, and 1 rad/s² in yaw. The neutral support center is derived
from the URDF masses and initial folded-arm pose. Velocity-error foot placement
and filtered attitude-rate damping improve tracking in reverse.

The stance-time/foot-placement design was informed by
[Di Carlo et al., Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control](https://dspace.mit.edu/entities/publication/bc8c7e1e-5830-443f-a879-787947111fcf)
and the [authors' foot-placement implementation](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/user/MIT_Controller/Controllers/convexMPC/ConvexMPCLocomotion.cpp).
This environment uses gait/IK and feedback corrections; it does not implement
the paper's MPC force optimizer or a whole-body arm/leg controller.

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
base = env.device("base")
command = {
    "mode": "locomotion",
    "velocity": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
    "posture": {"height": 0.30, "roll": 0.0, "pitch": 0.0},
}
world = env.device("world")
base.write(**command, duration_s=3.0)
world.wait(3.0)  # allow the initial stance to settle
command["velocity"]["vx"] = 0.05
base.write(**command, duration_s=5.0)
world.wait(5.0)
base.stop()
print(base.read()["data"])
print(env.device("camera_wrist").read()["data"])
```

Writes submit validated buffers and return immediately with `execution_id`,
submission `elapsed_s` and measured `state`. Arm writes use
`env.device("arm").write(actions=[[q1, q2, q3, q4, q5, q6]], duration_s=1.0)`;
gripper writes use `env.device("gripper").write(actions=[0.8], duration_s=0.2)`.
Use `world.wait(...)` to wait in simulation time; submitting a write does not
establish measured arrival.

Arm FK/IK runs locally in the exported Kinex joint driver, using its complete
serial-chain profile. Its pose is the flange pose in the arm mount frame, in
metres and an XYZW quaternion. The device's local `ik` action returns joint
angles, which are then sent through `write`; it does not step physics. The HTTP
joint interface accepts joint targets, not an RPC IK request. The native test
also exercises the standalone URDF solver in
`roboenv_runtime.adapters.isaaclab.go2_d1.adapter.RobotModel`.

## Clock and stop behavior

Physics runs at 200 Hz with a 50 Hz control interval. The task catalogs default
to realtime; use `--execution-mode stepped` to advance only during `world.wait`.
Reads, writes, geometry queries and stops never request physics steps.
Concurrent world waits overlap and all devices share each native tick. Waits may
round up to the next control interval.

New writes replace pending references without stopping first. Invalid buffers
leave existing motion intact. Arm and gripper retain their final reference;
base velocity expires after its requested `duration_s` in world time, maintaining
posture. The shared world scheduler is the only clock and command-expiry authority;
there is no backend realtime mode or wall-clock watchdog. Base stop requests zero
velocity, not instantaneous physical braking; arm/gripper stop hold measured
positions. Each stop affects
only that device, and late cancellation cannot stop a newer replacement.

All Go2–D1 tasks have no step limit in realtime mode (`budget.max: null`).
In stepped mode, control, reach and pick-and-place default to 6,000 control
intervals (120 simulation seconds). The catalog's `max_steps` applies only to
stepped execution; the shared world's selected mode also applies to CLI overrides.
A low body height or excessive tilt terminates the episode; final reads remain
available. Control practice has no success predicate. Reach and pick-and-place
freeze on success, task failure or budget exhaustion. Reset does not hide a
settling phase.

Both cameras output 480×270 RGB and registered depth. They use nominal RGB
horizontal FOVs (D405 87°, D435 69°) and a 0.015–20 m clipping range. These are
ideal simulated pinhole sensors, without RealSense stereo noise, distortion,
per-unit calibration, or the D435's separate wider depth FOV. Moving-camera poses
follow the current arm FK. The portable RGB-D driver decodes the depth transport
into metres.

## Dynamics assumptions and validation

The source D1 URDF has unusable zero effort/velocity limits. Preparation and the
native arm actuators use the torque mapping in
[`limits.py`](../runtime/src/roboenv_runtime/adapters/isaaclab/go2_d1/limits.py):
3.3 Nm for `d1_Joint1` and `d1_Joint2`, and 1.7 Nm for `d1_Joint3` through
`d1_Joint6`, sourced from the D1 specification referenced there. That table does
not distinguish continuous, peak or stall torque. Arm joint positions retain the
URDF bounds; arm velocity is explicitly uncapped in simulation. Finger joints use
the practice approximations 40 N / 0.1 m/s. Camera inertias are approximated by
uniform boxes. The Go2 source torque limits are retained.
Robot self-collisions are disabled in this first practice
implementation; collision-free arm planning and hardware deployment are outside
its validated scope.

Run CPU/source checks with `bash tools/check.sh`. With the registered Isaac
interpreter, run native checks and fresh-process replay:

```sh
ISAAC_PY=/path/to/isolated/isaaclab/bin/python
"$ISAAC_PY" tests/go2_d1_native.py --out .cache/go2-native-check
python tools/check_go2_d1_transport.py --execution-mode stepped --out .cache/go2-transport-check
"$ISAAC_PY" tools/replay_go2_d1.py .cache/go2-transport-check/record.json \
  --out .cache/go2-transport-check/replay.json
python tools/check_go2_d1_transport.py --execution-mode realtime --out .cache/go2-realtime-check
```

The native check measures forward/sideways/yaw tracking, height/roll/pitch,
arm IK, and both cameras. Transport checks exercise real launch, validation,
clock ownership, stop, and recording. Replay compares a fresh simulation with
recorded commands, assets, source identities, and final state; it never restores
final state. Logs and generated assets stay outside Git. A reused installation,
native run, CPU check, Kinex execution, task success, and model-driven acceptance
are distinct evidence; the included checks do not certify the latter two.

New recordings identify the Go2 implementation, the Isaac Lab dependency lock,
and the Isaac Lab source entries and their dependencies/patch checksums from
`upstreams.lock.json`. Unrelated backend updates and JSON formatting changes do
not change this source identity. Changes to an identified input require a fresh
recording and replay on the updated checkout.

Older recordings hashed the entire upstream lock. They cannot be replayed by this
updated implementation: strict source-identity comparison rejects them. Preserve
those recordings and their original checkout for historical validation; collect
new evidence for this version instead of editing recorded hashes or relabeling
old results. For each native or transport check above, choose a new output
directory and replay its new `record.json` in a fresh Isaac process.

## Velocity demonstration videos

Record three 22-second MP4 clips (forward/backward, sideways, turning) with a fixed
spectator camera and commanded/measured velocity overlays:

```sh
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tools/record_go2_d1_demos.py \
  --out generated/go2-d1-velocity-demos
```

The recorder requires FFmpeg, Pillow and the DejaVu Sans font. It adds only a
spectator camera; all motion comes from the existing native controller and physics.
Playback is 25 fps at normal simulation speed. The measured curve has a declared
0.25-second low-pass filter; adjacent telemetry JSON files retain raw body-frame
measurements, root poses and commands. Owner records retain source and asset
identities. Output directories must be new. Use `--cases sideways` to select a
single demo or `--smoke` for a short framing check.


For the higher-speed, 24-second demos, add `--high-speed`. These request ±1 m/s
longitudinally, ±0.5 m/s laterally, and ±30°/s yaw, with a following spectator
camera. The timeline is stand 0–2 s, positive command 2–10 s, stop 10–13 s,
negative command 13–21 s, and stop 21–24 s. The extra stop interval accommodates
the acceleration-limited controller. Yaw overlays use degrees/s; recorded
commands and raw telemetry remain in SI units.

The dedicated speed check uses independent forward, backward, left, right, CCW
and CW trials. It measures the 11–15 s hold window and the 17.5–18 s stop window,
checks mean tracking within 10%, reports raw RMS error separately, and checks
posture and falls. These are flat-ground, folded-arm trials, not validation of
simultaneous maximum-axis commands or arbitrary arm/payload configurations.

```sh
"$ISAAC_PY" tests/go2_d1_speed_native.py --out .cache/go2-speed-check
"$ISAAC_PY" tools/record_go2_d1_demos.py --high-speed --out generated/go2-speed-videos
```

## Posture demonstration videos

Use `--posture` to record separate height, roll and pitch clips. Each clip starts
neutral, holds the positive endpoint, returns to neutral, holds the negative
endpoint, then returns to neutral. Height endpoints are 0.34 and 0.26 m; roll and
pitch endpoints are ±0.12 rad (±6.88°). The angle overlays use degrees; commands
and raw telemetry retain radians. Height is the base origin above the flat floor.

```sh
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tools/record_go2_d1_demos.py \
  --posture --out generated/go2-posture-videos
# Optionally test the same posture commands while walking:
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tools/record_go2_d1_demos.py \
  --posture --walk-speed 0.5 --out generated/go2-walking-posture-videos
```

Use `--cases height`, `roll`, or `pitch` to select clips. Each is 24 seconds at
25 fps. These exercise one posture axis at a time with the independent arm target
held constant; they do not establish simultaneous maximum posture and velocity
tracking. Raw telemetry and source-matched owner records accompany each video.


## Mobile manipulation tasks

```sh
roboenv launch isaaclab/go2-d1-reach --port 8123
roboenv launch isaaclab/go2-d1-pick-place --port 8123
```

These are deterministic, custom practice tasks, not published benchmark replicas.
They retain the six existing devices and the accepted robot/camera mounts. The
agent selects base velocity/posture and independent arm IK/joint and gripper
targets. There is no task-solving policy inside the environment. Public inputs are
the task instruction, robot proprioception, and the two RGB-D cameras. Object
poses, contact forces, task progress and verdicts remain owner-only. Reset seeds
are journaled; the first task versions use fixed layouts without randomization.

**Mobile reach:** a cyan marker at world `[1.4, 0, 0.65]` m specifies the gripper
TCP position; colored axes specify orientation xyzw `[0, 0.70710678, 0, 0.70710678]`.
The task TCP is a grasp-center reference 0.105 m along flange +Z (not the flange
origin). This distant pose is outside the arm's fixed-base workspace. Success
requires at least 0.5 m base displacement and a continuous one-second hold within
25 mm / 0.10 rad, with base translational speed below 0.08 m/s. Merely commanding
the pose, passing through it, or moving only the arm does not count.

**Pick, carry, place:** an orange 40 × 40 × 60 mm, 80 g dynamic block starts on the
floor at `[1.25, 0, 0.03]` m. The green destination is a 0.30 × 0.30 m floor square
centered at `[2.5, 0, 0]`. Grasping uses the existing finger collision meshes,
friction and actuators; no weld, attachment, gravity disable or payload teleport
is used. After bilateral finger contact and a sustained lift above 0.12 m, the
base must travel at least 0.6 m while carrying. Ground clearance below 8 mm outside a qualified
placement, or loss of the grasp for over 0.2 s, fails the episode. Placement must
put the entire block inside the destination. Success requires an open gripper,
no finger contact, a nearly stationary block on the floor, and a stopped base for
one second. A terminal drop cannot be repaired into success in that episode.

The owner feasibility probe approaches the ground with 0.26 m base height and
+0.12 rad pitch, then uses arm IK to lower, grasp, lift and place. It is a
privileged test fixture using known geometry, not an agent/perception baseline.
The existing approximate dynamics and disabled robot self-collisions still apply;
these tests do not certify collision-free motion or hardware operation.

```sh
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tests/go2_d1_tasks_native.py \
  --task go2-d1-reach --cameras --out .cache/go2-reach-probe
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tests/go2_d1_tasks_native.py \
  --task go2-d1-pick-place --cameras --out .cache/go2-pick-place-probe
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" tools/replay_go2_d1.py \
  .cache/go2-pick-place-probe/record.json --out .cache/go2-pick-place-replay.json
# Add --video generated/pick-place.mp4 for a spectator recording of that replay.
```

Records include task definition/source identity, ordered device commands, complete
payload pose/velocity, grasp contact forces and scoring state. Independent replay
checks robot state, payload pose and velocity, and the task stage/verdict. A
replayed failure remains a task failure. CPU tests cover invalid shortcuts,
premature release, object bounds, hold reset and terminal failure behavior.
