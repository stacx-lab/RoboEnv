# Static RoboCasa device configuration

The standard RoboEnv RoboCasa rig is PandaOmron: a seven-joint Panda arm, Omron
planar mobile base, one lift joint, Panda parallel gripper, and three
robot-mounted RGB-D cameras (left, right, and wrist). All standard tasks use the same device definitions.
Kitchen tasks change the external scene and initial placement, not that rig.
Upstream also supports other robots and camera variants; those are different
hardware configurations, not variants of this catalog entry.

## Research findings

The primary sources are RoboCasa's [Kitchen implementation](https://github.com/robocasa/robocasa/blob/main/robocasa/environments/kitchen/kitchen.py)
and [camera definitions](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/camera_utils.py),
and robosuite's [PandaOmron composition](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/robots/compositional.py)
and [default controller configuration](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/config/robots/default_pandaomron.json).
Generation uses the locally pinned source versions; exact input SHA-256 values
are recorded in [the source receipt](robocasa-config.provenance.json). Online
default branches can differ. The copyable robot directory contains only
`workspace.yaml` and `devices/`; RoboEnv bindings live separately in
`catalog/bindings/robocasa-panda-omron.yaml`.

Kitchen defaults to 20 Hz control, 256-pixel camera images, and no camera
randomization. Its camera setup adds robot-mounted cameras independently of the
layout-specific viewer camera. RoboEnv uses `robot0_agentview_left` and `robot0_agentview_right`, mounted on
`mobilebase0_support`, plus `robot0_eye_in_hand` on the arm's hand. The external
views move with the lift; the wrist view moves with the arm. Their rigid mounts
are reusable across tasks, unlike a free kitchen viewer camera.

| Device | Pre-generated facts |
|---|---|
| Arm | 7 joint limits and model home joints; 6 normalized OSC commands; translation scaling ±0.05 m, rotation scaling ±0.5 rad per interval |
| Body | 3 planar joints; forward/left/yaw commands; controller scaling and 20 Hz interval |
| Torso | 1 prismatic joint, 0–0.34 m; model home 0.2 m; normalized joint delta scaled to ±0.05 m |
| Gripper | 2 finger joint limits; one signed command; flange-to-TCP transform |
| Left/right cameras | 256×256 aligned RGB-D, vertical FOV 60°, cx=cy=128 px; optical frames relative to torso flange |
| Wrist camera | 256×256 aligned RGB-D, vertical FOV 75°, cx=cy=128 px; optical frame relative to arm flange |
| World | Root coordinate frame; explicitly open global workspace AABB; no predeclared kitchen surfaces |

Control scaling is the native controller's target scaling, not a guarantee of
realized displacement. FK/IK and mount quaternions use xyzw. The camera optical
transform converts MuJoCo's -Z view direction to +Z forward and +Y down.

RoboCasa/robosuite supports RGB-D via `camera_depths`; the upstream default is
RGB-only. RoboEnv enables depth for its standard rig, requests the RGB and depth
buffers in the same render, converts depth with robosuite's
[`get_real_depth_map`](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/camera_utils.py),
and flips both images upright. Conversion uses the live model's near/far planes,
which depend on scene extent; these are not fixed sensor range calibration.
The profile declares `depth.aligned: true` and `depth.scale: 0.001` for the
16-bit millimetre PGM output. Background and unrepresentable depths become 0,
never a fabricated surface at maximum distance. This is quantized synthetic
optical-axis depth, not an assumed real RGB-D sensor noise model. RGB-only
custom profiles remain supported. Other benchmark configurations may restrict
modalities; RGB-D support does not establish their official observation rules.

The compiled MJCF model is authoritative for this simulated robot; importing a
real Panda URDF would not describe the same actuators or their enforced limits.
Profiles mark model-derived sections `source: introspected`. Home joints come
from robot initialization defaults, never the current task pose. Arm/torso FK
uses `mj_kinematics`; IK uses `mj_jacBody` with damped least squares and joint
range constraints. Both run on scratch data and express the flange in the
device base frame. IK may fail for unreachable targets or an unsuitable seed.

### Arm joint limits and recovery

The compiled Panda model enables limits on all seven revolute joints. These
are nominal position ranges in radians, in the same order as `arm.read().positions`
and FK/IK joint arrays. They are exported as `intrinsics.joints[].limits`, with
per-joint `source: introspected`; `intrinsics.joint_limit_semantics` records units,
constraint solver settings and the controller's behavior.

| Joint | Minimum (rad) | Maximum (rad) | Approximate degrees |
|---|---:|---:|---|
| 1 | -2.8973 | 2.8973 | -166 to 166 |
| 2 | -1.7628 | 1.7628 | -101 to 101 |
| 3 | -2.8973 | 2.8973 | -166 to 166 |
| 4 | -3.0718 | -0.0698 | -176 to -4 |
| 5 | -2.8973 | 2.8973 | -166 to 166 |
| 6 | -0.0175 | 3.7525 | -1 to 215 |
| 7 | -2.8973 | 2.8973 | -166 to 166 |

MuJoCo enforces these through
[soft joint-limit constraints](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#body-joint),
so measured joints may slightly cross a nominal bound. The compiled rig uses
zero activation margins, `solref=[0.02,1]` and `solimp=[0.9,0.95,0.001,0.5,2]`.
These values describe native physics, not a conservative operating margin.
Reading state must preserve such an overshoot: `read` computes the measured
flange pose on scratch data without rejecting or clamping the measured joints.
Explicit FK inputs and explicit IK seeds remain range-checked. Default IK seeds
are projected into range on scratch data only, so a slight live overshoot does
not prevent the solver from starting. IK still may fail to converge.

The native OSC controller commands six Cartesian deltas through torques and
attracts nullspace posture toward the initial joints. It has no explicit
joint-range avoidance or guaranteed recovery strategy. At a limit, near a
singularity, or in contact, repeatedly sending a Cartesian delta may not produce
the intended motion. `stop` resets the servo goal; it does not move the robot
home. `home_joints` is not a valid target for this six-dimensional `write` API,
and a successful read-only IK call does not execute motion or certify a path.

CPU regression tests place every arm/lift joint 0.001 rad/m beyond both limits
and verify that read and implicit-seed IK work without modifying live state;
out-of-range explicit targets still fail. This reproduces a former read failure,
not a user's particular kitchen blockage. Diagnosing that blockage requires the
measured joints, commands and contact context; these checks do not certify recovery.

### Velocity limits

The pinned Panda MJCF uses torque motors: its `ctrlrange` values are torque
limits, not rad/s. The native OSC and torso position controller do not impose
a hard measured velocity cap. Dividing their delta scales by 0.05 seconds does
not produce one. Omron's velocity actuators have bounded velocity targets;
their compiled ranges are exported under `velocity_limits.actuator_target_ranges`
with units. Tracking dynamics can still produce different measured velocities.
This distinction follows MuJoCo's separate
[control and force limits](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#actuator-general-ctrlrange).
The raw compiled actuator ranges are included for inspection, with flags stating
whether each is limited. Gripper `speed` is a normalized command increment,
not a physical finger velocity.

The workspace safety velocity fields are explicitly `null`: this catalog imposes
no additional velocity cap. This is a policy choice, distinct from `???` (not yet
specified). Native controller limits and dynamics still apply; RoboEnv does not
invent a numeric cap or change benchmark control dynamics. The sibling Complex
schema, calibration DAG, and summary all preserve this distinction.

The torso's changing height belongs in `torso.read().flange_pose`. The camera's
rigid mount is relative to that flange. The mobile chassis is named `body` with
type `body` and `intrinsics.mobility: mobile`, matching Complex's mobile-base
contract. Its `extrinsics` contains only `source: default`: there is no `mount`,
unknown pose, or calibration request. The robot's frame component is rooted at
`body`; `world` is a separate reference with its own bounds. Devices can relate
through their common robot ancestor without a path to `world`.

`body.read().pose` still reports the simulator's current robot-to-world pose.
Its availability does not implicitly connect these profile frames, nor does an
absent edge imply identity. A custom binding can explicitly declare a dynamic
world mount when that relation is wanted. The standard generator and runtime
preserve the independent mobile root.

Scene geometry, object poses, fixture placement, and layout/style metadata are
not agent observations. There is no scene device and no ground-truth scene in
task context. The agent must perceive and estimate these from its camera images.
Robot joint state, robot pose, calibration, and kinematics remain device facts.
Task discovery provides the task ID and instruction, without the scene seed.
Owner-only verification may use simulator state; the HTTP API cannot retrieve
owner observations, including through operation-status lookups.

## Generation and verification

The profiles and drivers are already checked in. Regeneration is an explicit
maintenance operation, never part of `roboenv launch`:

```sh
.venv-sim/robocasa-resolved/bin/python tools/generate_robocasa_config.py \
  --provenance docs/robocasa-config.provenance.json
```

Use the registered RoboCasa interpreter if your installation is at a different
prefix. The script compiles only the Panda–Omron model in an empty arena, adds
the upstream robot camera definitions, and initializes the native controllers.
It does not initialize a task, acquire kitchen assets, render, advance physics,
or call a model. Intrinsics and rigid mounts are checked again after changing
arm/lift configuration. Numeric round-off and metadata are normalized so repeated
generation produces identical files. The checked-in source hashes record the
actual model/controller inputs.

Tests compare regenerated files byte-for-byte, validate profiles and interfaces
with the current sibling Complex schema and calibration DAG when available,
exercise all three camera routes over HTTP without spending action frames,
reject disabled cameras, exercise unchanged copied drivers against
two CPU services, and ensure launch leaves the source catalog unchanged. These
checks do not certify actual kitchen GPU rendering, TUI recording, task success,
or fresh-process task replay. Follow [the local test plan](complex-local.md) for
those separate integration checks.

The sibling Complex update removes simulator introspection and `kind: sim`
dispatch. This catalog supplies ordinary URL drivers and precomputed facts.
Its calibration DAG now checks driver existence and required verbs independently
of `/ping` liveness. All device calibration nodes pass. `world.bounds` declares
no additional global AABB restriction using `min` and `max` of `[null, null, null]`,
which Complex represents as open sides. This is a catalog default, not a claim
that the kitchen is collision-free or every point is reachable. Native joint
limits and collision physics remain in effect. Actual scene geometry still
comes from perception. Explicit null safety caps satisfy the safety policy node,
so this catalog satisfies `task-ready`; that status does not certify task success.
Compatibility
tests read the sibling's working sources, including local edits, rather than a
stale cached schema; Complex is not a RoboEnv runtime dependency.

## Why the torso stays separate

The torso is an independently actuated lift. Complex currently supports live
base-to-flange edges for `manipulator` devices, while each `body` owns one frame.
Keeping the torso as a one-joint manipulator preserves the moving mounts of the
arm and external cameras. A combined body/lift driver could accept four command
components, but correct frame composition would also require extending Complex
to support named moving body frames. This revision keeps its existing contract.
