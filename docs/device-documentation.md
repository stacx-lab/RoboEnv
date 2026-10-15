# Device access

Any coding agent can connect through the SDK or CLI. RoboEnv exposes a task and
its devices; the harness owns reasoning, programs, permissions, and scheduling.
Kinex is optional, and the SDK/runtime do not depend on it.

## Public API

```python
import roboenv

with roboenv.connect("http://127.0.0.1:8123") as env:
    print(env.task.instruction)
    print(env.devices())
    print(env.device("left_arm").read())
```

| API | Meaning |
| --- | --- |
| `env.task` | Immutable snapshot of public task ID, instruction, and permitted context |
| `env.refresh_task()` | Refresh that snapshot for native instruction transitions |
| `env.devices()` | Declared device profiles and supported operations |
| `env.device(name)` | Handle for a declared device; unknown names fail |
| `device.profile` | Configuration and capabilities for that device |
| Device operation | Device data directly, or a structured failure |

Closing an attached client releases its resources, including downloaded images,
without stopping the world. Clients pin episode identity: reusing a port cannot
redirect an established client's commands into a replacement episode. See
[launch and ownership](setup.md#start-a-task).

New integrations default to sensor-only observations. Robot state and declared
measurements are public; object poses, scene seeds, scoring, journals, and owner
snapshots stay private. Task context must follow the backend's explicit observation
policy. A world device exposes its clock and static reference-frame profile, not
an arbitrary simulator dispatcher or general scene-state API. See the
[public service boundary](simulator-integration.md#6-preserve-the-public-service-boundary).

## Device operations and frames

Read the live descriptor for supported operations. Sensor reads never advance
physics. Actuator writes require `duration_s` and return immediately with an
execution identity and measured state. Joint/hand `actions` contain absolute joint
vectors; gripper/suction `actions` are flat numeric sequences, including `[value]`
for a single command. For example, hold RoboTwin's left arm at its measured position:

```python
arm = env.device("left_arm")
target = arm.read()["positions"]
arm.write(actions=[target], duration_s=0.5)
env.device("world").wait(0.5)
print(arm.read())
```

Velocity devices use named axes and modes declared in their profiles.

Consult the profile for units, bounds, supported modes, and stop scope. Use
`env.device("world").read()` for `time_s` and `.wait(duration_s)` to wait in
simulation time. In stepped mode, pending waits drive shared native ticks; reads
and writes do not request ticks. Independent devices can write concurrently.
Planning and measured arrival checks belong to the harness. See
[execution and cancellation](simulator-integration.md#execution-and-cancellation)
for buffers, clocks, budgets, replacement, and failure handling.

Joint, hand, and scalar-gripper reads omit `frame_poses`. Velocity reads and
write-result states omit global `pose`, and their profiles omit
`feedback.pose_frame` and fixed world mounts. Available body measurements such as
height and local frames remain public. World-relative transforms require explicit
estimates; bases are unlocalized roots.

Static frames use `{parent_frame, translation_m, rotation_xyzw}`. Null parent means
the device base; null transform components mean unknown. Joint `kinematics.frame`
and hand geometry define computed frames. `intrinsics.frames` contains additional
references, including fixed descendants of computed frames. Moving transforms
come from FK at measured joint positions, not a fabricated static transform.

Exported Kinex joint drivers provide local `fk(joints, frame=..., reference_frame=...)`
and `ik(pose, seed=..., frame=..., reference_frame=...)`. Defaults are the modeled
endpoint and device base; results include both frame names. Fixed endpoint
descendants and fixed base references are supported. FK accepts measured joint-limit
overshoot without clamping. Omitting an IK seed needs only a joint-state read;
solving does not mutate the simulator. Joint-only devices need no solver, and
advertised FK/IK must work. Kinex's Python helpers and PyYAML support these drivers;
the agent environment needs no simulator engine or RoboEnv SDK.

Hand geometry may describe fingers, optional kinematic chains, and a weighted
center. Moving links name model joints; fixed links contain an origin. Chains
start at the device base with explicit tip offsets. Palm/grasp references are
ordinary fixed frames. Description-only geometry does not advertise FK; a center
requires fingers and predicts geometry, not contact or grasp success. See
[scalar gripper geometry](grippers.md) for the separate gripper contract.

Observations use a finite numeric `timestamp` in seconds in the device's clock;
RoboEnv uses simulation time. Use acquisition time, or receipt time if unavailable.
Combined observations use the latest component time, while cached readings retain
the original time. A camera must work before the first motion, with files surviving
one-shot driver exit. Image dimensions must match the profile. RGB-D requires
registered metric depth and calibration. IDs are opaque; generated identifiers use
16 random hexadecimal characters with any existing prefix.

## Export a device reference

`roboenv.devices.documentation` exports a language reference for coding agents
that connect through the RoboEnv SDK or CLI. It does not depend on an agent framework.
The caller owns task goals, instructions, and evaluation policy.

```python
from roboenv.devices.documentation import export_documentation

export_documentation(
    robot,                       # public robot workspace manifest
    env.devices(),               # descriptors from the running environment
    output_directory,
    {"endpoint": endpoint, "session": session, "generation": 0},
)
```

Use a fresh output directory per environment session. The output is `harness.md`,
`devices/<name>.yaml`, `devices.json`, `connection.json`, and `workspace.yaml`.
No drivers or agent auto-discovery instruction files are written. Device names
must be unique logical names. Public exports omit binding selectors, access/driver
configuration, and implementation metadata. Supply only a public robot manifest.

The Markdown explains connections, supported action arguments and result schemas,
zero-based joint vector order and units, gripper direction commands, named base
velocity fields, feedback frames, mount transforms, and RGB/depth interpretation.
The YAML preserves exact public geometry and schemas for code to consume. Use
live descriptors so action support, dimensions, limits, and control settings match
the running environment; static driver action lists are not SDK capabilities.

`robot_description(robot, devices)` supplies a short model/simulator/inventory
summary for embedding in an orchestrator's instruction. `render_documentation`
returns the full Markdown without writing files. Neither function starts or
changes a simulator, and neither defines benchmark goals or rewards.

Kinex-style file drivers remain a separate projection through
`roboenv.devices.export.export_devices`; orchestration adapters choose when to
create those resources. That projection places a single `_roboenv.py` in the
shared `devices/` root, alongside the individual device directories.

## Camera conventions

Camera projection uses integer pixel centers: the top-left pixel is `(0, 0)`.
For a centered image the principal point is `((width - 1) / 2, (height - 1) / 2)`.
Registered depth is optical-axis Z in the profile's declared units, sampled at
those same pixel centers. RoboCasa uses single-sample offscreen rendering for
paired RGB-D; RoboTwin retains native supersampled RGB and captures pixel-center
depth with a matching raster camera. These corrections change rendered sensor
data, so older image recordings retain their original calibration.

Camera `timestamp` uses the simulator clock, matching joint feedback.
RGBD returns one timestamp for the pair; transport preserves it. Separate
reads in realtime mode can still describe different simulation instants. Matching
clock units alone does not make an image and a later joint read an atomic snapshot.

RoboTwin gripper `tcp` frames follow the native robot's TCP position and orientation,
including the end-effector joint anchor and embodiment calibration. Finger surface
calibration still determines aperture and closing geometry; its midpoint does not
define the RoboTwin TCP. This corrects earlier profiles that used that midpoint.

Re-export devices when upgrading a runtime or changing camera or TCP calibration so
the copied profiles reflect the current intrinsics and extrinsics.

## Camera read output

SDK camera reads accept `camera.read(output="file")` (the default),
`output="array"`, or `output="base64"`. File mode returns `file` and optional
`depth_file`. Array mode returns owned NumPy `color` (uint8 RGB, H×W×3) and
optional `depth` (uint16, H×W); install `roboenv[camera]` for decoding.
Base64 mode returns PNG/PGM16 strings under `color`/`depth`. Every mode includes
one `timestamp`; profile dimensions and depth units are unchanged.

The server protocol and artifact production are unchanged. Clients convert the
received images; array/base64 HTTP reads avoid local temporary files. File mode
supports artifacts larger than the existing in-memory response limit. Exported
Kinex drivers accept the same option; arrays cross their JSON boundary as base64
and are decoded in the calling Kinex Python runtime.

## Changing task instructions

For tasks with changing instructions, call `env.refresh_task()` between stages.
It updates `env.task` through the existing public description endpoint.
`roboenv describe` and copied Kinex workspaces' `python task.py` read the same
current instruction without exposing native task state.

The description's `task.native_instruction` is the simulator's own text (empty
when unavailable); `task.instruction` falls back to the launch catalog when that
text is absent. `task.context` exposes only declared public language facts, such
as sampled object names, colors, target burner, or assigned arm. Reading these
facts does not resample the scene or language. It does not expose object poses,
seeds, or scoring state.

A benchmark harness may own a separate catalog goal. It can bind that goal's
declared fields from `task.context` after initialization, before starting an
agent. RoboEnv does not rewrite that goal or combine it with native wording.

## Refreshing copied Kinex workspaces

From the agent workspace, use `roboenv setup ROBOT_ID --force` and
`ROBOENV_ENDPOINT` as described in [local Kinex setup](kinex-local.md).
This replaces `.kinex/workspace.yaml` and the entire `.kinex/devices/` collection,
including profiles, typed drivers, and shared helpers. Save custom connection and
calibration settings before replacement and reapply them afterward.
Portable persistent drivers use correlated JSONL requests and may reply out of
order. Reload Kinex devices after refreshing their files, and restart owned
RoboEnv services to load runtime changes. Borrowed worlds stay under their owner's
lifecycle control. Update custom consumers to numeric `timestamp`; old-field
aliases are not supported.
