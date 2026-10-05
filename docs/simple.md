# SIMPLE G1 devices

SIMPLE uses the current control-contract device API. Native WBC owns legs and
waist through one mobile `base`; arms and fingers retain independent joint targets.
RoboEnv owns the world, clocks, native verdict and private replay evidence.

## Prepare and launch

```sh
roboenv prepare simple
roboenv tasks list --simulator simple
roboenv launch simple/g1-xmove-pick
roboenv launch simple/g1-transfer-between-tables
```

Execution is synchronous with `sim_mode: mujoco_isaac`: MuJoCo physics with Isaac
Sim 4.5 RGB rendering. Preparation uses an isolated Python 3.10 environment and
checksum-verified assets (about 4.3 GB of archives). Isaac requires compatible
NVIDIA RTX rendering hardware and Vulkan. SIMPLE launches with Isaac rendering
default `OMNI_KIT_ACCEPT_EULA=YES`, accepting NVIDIA's Omniverse EULA automatically.
An explicitly set environment value is preserved. MuJoCo-only launches do not
set this variable. Launch never downloads assets.

For diagnostic MuJoCo rendering, copy a representative task YAML, set
`backend_kwargs.sim_mode: mujoco`, and launch that file. Rendering changes the
images available to an agent, so results from the two renderers are distinct.

| Example | Native registration | Synchronous horizon |
| --- | --- | --- |
| `simple/g1-xmove-pick` | `G1WholebodyXMovePickTeleop-v0` | 800 intervals / 16 s |
| `simple/g1-transfer-between-tables` | `G1WholebodyLocomotionPickBetweenTablesTeleop-v0` | 1200 intervals / 24 s |
| `simple/g1-bend-pick` | `G1WholebodyBendPickTeleop-v0` | 800 intervals / 16 s |

Discovery reads the pinned decoupled G1 teleop registrations without importing
the simulator. The checked-in assets and examples cover these three tasks;
other discovered tasks may need additional assets. Domain randomization defaults
to native level 0, with levels 1 and 2 selectable in task configuration.

### Assets, calibration and shutdown

`prepare simple` extracts `vMaterials_2.zip` at the depot's data root. Outdated
extraction receipts are rebuilt from checksum-verified archives; existing matching
files and old asset paths are preserved. Both native data resolvers honor
`SIMPLE_DATA_DIR`, so no links into the source checkout are needed. Verification
only reads assets; run preparation to update an old layout.

The head cameras use 640×360 RGB, a 110° horizontal field of view, and a 5 cm stereo
baseline. In public pixel-center coordinates, `fx = fy ≈ 224.066412` and
`(cx, cy) = (319.5, 179.5)`. Isaac's pixel-edge coordinates are converted once by
subtracting half a pixel. Public optical axes are +X right, +Y down, +Z forward;
quaternions are xyzw, while MuJoCo and Isaac pose APIs use wxyz.

The USD camera-link fixed joint agrees with the catalog's torso-relative mount.
The shared G1 USD uses older waist offsets; the managed patch aligns its roll and
pitch joint origins with Sonic's pinned MuJoCo model. Isaac mirrors full-precision
MuJoCo state, refreshes articulation transforms, and captures without integrating
another physics interval. Startup compares the actual optical pose with the measured
MuJoCo torso, rather than the articulation's authored USD transform. Mount checks
retain the strict `1e-4` matrix tolerance; there is no tolerance override.

Isaac fast shutdown is disabled so cleanup returns to the worker or replay tool.
Reset errors remain visible in the worker log, and replay failures retain a
nonzero exit status. Standalone native and replay commands flush output and exit
explicitly after cleanup, as the worker does, to avoid native destructors running
after Kit plugins have unloaded. Replay output must use a fresh path.

On multi-GPU hosts, select a GPU with `CUDA_VISIBLE_DEVICES` before launching and
allow for cold shader compilation within the startup deadline (900 seconds by
default). `roboenv.launch` strips inherited `ROBOENV_*` variables before setting
its own worker configuration.

Kinex's `run --record` requires devices to be loaded; exported profiles initially
have `loaded: false`. RoboEnv's `launch --record` instead records independent
spectator views. Neither changes the owner-record transport limit: long realtime
sessions can exceed the 8 MiB response bound because control intervals accumulate
in the native replay record. Keep diagnostic sessions bounded.

## Arm gravity compensation

Arm gravity compensation is enabled by default. To select it explicitly at startup:

```sh
roboenv launch simple/g1-xmove-pick --arm-gravity-compensation
roboenv launch simple/g1-xmove-pick --no-arm-gravity-compensation
```

Python owners can pass `arm_gravity_compensation=False` to `roboenv.launch`.
Task YAML can set `backend_kwargs.arm_gravity_compensation: false`. An explicit
launch argument overrides the task setting; an omitted setting defaults to true.
The option accepts booleans, applies only to SIMPLE, and stays fixed for the run.
The resolved service configuration and owner record retain the effective value.

At each 5 ms physics substep, the arm controller adds gravity-only feedforward
to PD torque for the fourteen arm joints, then applies the existing torque limits.
The calculation includes hand mass and current robot orientation, using scratch
state without changing the live physics state. Legs, waist, finger actuation,
PD gains, and joint-position command semantics retain their existing behavior.
Compensation remains active when stop latches measured arm positions.

Disabling the option restores the uncompensated PD path. Neither setting promises
exact tracking under contact, torque saturation, or dynamic disturbances.
Gravity compensation does not make a fixed arm-local target follow an object
when the torso moves; frame-aware Cartesian control remains a separate concern.

In the seed-0 XMove standing diagnostic, settled torso-relative flange error
dropped from about 31 mm / 10.6 degrees to 0.4–0.5 mm / 0.6 degrees. Maximum
movement during two seconds after arm stop dropped from about 33 mm to below
0.1 mm. These measurements cover the probe's configuration, not all poses or
loaded grasps. The fixed-torso and zero-gravity controls isolate gravity from
balance motion; the native comparison uses a fresh process for each setting
because upstream registries cache robot configuration within a process.

## Device contracts

| Endpoint | Type | Command |
| --- | --- | --- |
| `base` | `velocity` | Body velocity and torso posture references |
| `left_arm`, `right_arm` | `joint` | Seven absolute joint positions per arm |
| `left_hand`, `right_hand` | `dexterous_hand` | Seven absolute joint positions per hand |
| `camera_head_left`, `camera_head_right` | `rgb` | Sensor reads |
| `world` | `world` | Reference profile only |

```python
import roboenv

with roboenv.connect() as env:
    env.device("base").write(
        mode="locomotion",
        velocity={"vx": 0.15, "vy": 0.0, "wz": 0.1},
        posture={"height": 0.72, "roll": 0.0, "pitch": 0.05, "yaw": 0.0},
        duration_s=1.0,
    )
```

A base write is a bounded segment: the command applies for `duration_s` world
seconds. Velocity is expressed in pelvis-frame m/s and yaw rad/s. Height is a
ground-relative pelvis-height reference. Torso roll/pitch/yaw are radians
relative to the WBC heading frame; torso yaw is not turning velocity. All
declared components are required and must be finite. These command values have
no added range bounds. Physical joint constraints and native torque clipping
remain active. The managed metric hook bypasses upstream teleop turn-flag semantics.

WBC alone writes legs and waist. There is no waist device. Base feedback declares
`intrinsics.frames.torso: {parent_frame: null, translation_m: null, rotation_xyzw: null}` and supplies `read.frame_poses.torso`, the measured
torso pose relative to pelvis. `height_m` measures pelvis Z above the flat scene
reference plane at world Z=0; it is not terrain clearance. Base reads omit global
pose, and world-relative transforms require explicit estimates. Arms and head cameras mount on `base:torso`; hands
mount on their arm's `flange`. Local arm FK/IK uses exported serial geometry and
Kinex's standard joint driver. Hands provide portable finger geometry and named
frames as described below; they do not advertise finger IK or grasp confirmation.
Measured state is never projected into joint limits.

## Hand geometry

Both `dexterous_hand` profiles include optional `intrinsics.geometry`; ordinary
`joint` devices are unchanged. The canonical hand frame remains its arm flange.
`palm` is the fixed native palm-link reference. `thumb_tip`, `index_tip`, and
`middle_tip` are dynamic nominal fingertip frames computed explicitly with local
hand geometry/FK from measured joint positions.
Their positions include offsets from the distal joints to the mesh ends (about
5.2 cm), and their axes follow the distal links. The generator derives each point
from the center of the distal bounding-box face of the native collision mesh:
+X for index/middle, -Y for the left thumb, +Y for the right thumb.

`grasp_center` is the midpoint between the thumb tip and the average of index and
middle tips: weights 0.5, 0.25, 0.25. Its axes stay aligned to the hand frame.
`grasp_reference` fixes this center at the declared home configuration, providing
a stable reference for arm positioning. Neither reference is a measured object
center, an optimal grasp target, or a contact sensor. Mesh bounds and hashes are
recorded in generation provenance; no object or task-private geometry is exported.

Within a Kinex Python tool, prediction is local and requests no simulation steps:

```python
from kinex import device, frames, profile
from kinex.hand import forward
from kinex.joints import JointDevice

hand = profile("left_hand")
state = device("left_hand")["data"]
tips_now = forward(hand, state["positions"])
candidate = list(state["positions"])
candidate[0] += 0.05
JointDevice(hand).command({"actions": [candidate], "duration_s": 0.1})  # Validate a proposed command.
tips_if_commanded = forward(hand, candidate)     # Prediction only; no write.
tip_in_arm = frames.pose("left_hand:index_tip", "left_arm", readings={"left_hand": state})
```

The helper accepts finite measured positions without clamping, so it also works
with native soft-limit overshoot. Hand `kinematics: null` means no single-endpoint
FK/IK API; finger-chain prediction uses `kinex.hand.forward` instead.

To position the fixed reference using arm IK, express the desired reference pose
in the arm's canonical frame, then remove the flange-to-reference transform:

```python
from kinex.geometry import compose_rt, invert_rt, pose_to_rt, xyzq_to_rt
from kinex.motion import flat

# desired_reference is [x,y,z,qx,qy,qz,qw] in left_arm coordinates.
offset = frames.pose("left_hand:grasp_reference", "left_arm:flange")
flange_goal = flat(compose_rt(xyzq_to_rt(desired_reference), invert_rt(pose_to_rt(offset))))
arm = JointDevice(profile("left_arm"))
seed = arm.project(device("left_arm")["data"]["positions"])
solution = arm.ik(flange_goal, seed)  # Still no motion or collision guarantee.
```

Use the current Kinex checkout and regenerate existing workspaces to acquire the
geometry-aware hand schema and profiles. The moving `grasp_center` is unsuitable
as a fixed arm offset when changing finger configurations.

## Time, stop and concurrent calls

Joint and hand writes buffer full absolute vectors over `duration_s`. Goal trajectories are supplied by the harness.
The lower-body WBC continues to own legs and waist. The executor rounds action
boundaries onto its 20 ms native interval (four 5 ms MuJoCo substeps).

Realtime advances continuously; stepped advances during world waits.
Independent devices share each tick;
new commands replace prior motion on the same device. Stop cancels that device's active and
queued commands, zeroing base travel or holding measured joints. It retains peer
targets and posture. Stepped retains the native horizon; realtime removes the
step cap. Native non-timeout termination conditions remain active in both modes.
Reads, status, stop, recording and local FK/IK never advance physics.

## Kinex and replay

```sh
python tools/prepare_kinex_workspace.py simple/g1-xmove-pick --out NEW_WORKSPACE
```

Copied drivers use the standard portable transport, Kinex joint/velocity helpers,
and ordinary recording. Use Kinex tools such as `servo` for individual arms
and device calls for base and hand commands. The native Kinex integration check
installs `tests/fixtures/device-composition/` only into its temporary test workspace
to exercise these device contracts together.

Private records contain initialization, individual accepted writes/stops and timeout
stops at physics boundaries, retained targets, MuJoCo integration state, controller
history/interpolation state, budgets and cached native verdicts. Fresh-process replay
reconstructs initialization and command timing without restoring final state.
Reads and scoring never re-run stateful native success predicates.

This is practice protocol `simple-g1-metric-wbc/4`, with record schema
`roboenv/simple-record/3`. Replay requires the recorded gravity-compensation setting;
older record/protocol versions are rejected. Preserve historical recordings and
generate new recordings with the current implementation. It is not
an official SIMPLE baseline or a task-performance result.

## Validation

The current revision passes native control, Kinex stereo recording and fresh-process
replay checks with MuJoCo rendering in both clocks for all three examples. Isaac
rendering remains unvalidated; selecting it as the default does not certify that path.
Hand geometry is checked against native MuJoCo transforms across 102 configurations,
including both hands, individual joint perturbations, limits, and measured overshoot.
The Kinex checks exercise live fingertip frames, local prediction, and fixed grasp
reference conversion through arm IK. These checks do not establish grasp success.

```sh
python tools/accept_simple.py --python SIMULATOR_PYTHON --out NEW_RESULTS
# Diagnostic matrix with MuJoCo images:
python tools/accept_simple.py --python SIMULATOR_PYTHON --out NEW_RESULTS --sim-mode mujoco
# Uncompensated native checks and replay:
python tools/accept_simple.py --python SIMULATOR_PYTHON --out NEW_RESULTS --sim-mode mujoco --native-only --no-arm-gravity-compensation
# Gravity force, fixed-torso, tracking and post-stop diagnostics (both settings):
SIMULATOR_PYTHON tests/simple_gravity_probe.py --out NEW_RESULTS
# Geometry only, without rendering or starting a task:
SIMULATOR_PYTHON tests/simple_hand_probe.py --out NEW_REPORT.json --kinex ../kinex
```

The runner covers all three tasks in both clocks, fresh-process replay, altered-record
rejection, Kinex local IK and stereo recordings, and public-service velocity expiry.
It defaults to Isaac rendering. `--native-only` omits the Kinex/service checks;
`--tasks xmove` selects a smaller matrix. No model calls or task-solving attempts run.

Report offline conformance, native boot/control, rendering, Kinex recording and replay
separately. Previous PR validation used obsolete contracts and does not certify this
implementation. Isaac acceptance requires a fresh successful run on an appropriately
configured host; MuJoCo checks alone do not establish Isaac support or visual parity.
Container isolation certification and task-performance evaluation remain separate.

Source and asset pins, managed patch checksums and dependency locks are versioned.
`tools/generate_simple_config.py` regenerates the catalog and private model facts
from pinned robot geometry and camera definitions. Pass
`--bindings-out src/roboenv/devices/bindings/defaults` separately when maintaining
internal binding defaults. Startup checks native
geometry against those facts without rewriting static profiles.
