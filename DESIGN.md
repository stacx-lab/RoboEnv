# Design of RoboEnv

RoboEnv exposes **a task and a set of devices** backed by one simulation service.
The task supplies an instruction. Devices supply basic sensor readings and actuator
commands. The agent perceives the scene, plans, and writes control programs.

All eight simulator backends use this task and typed-device contract. Native
control, rendering and success predicates remain simulator-specific.
See [RoboCasa devices](docs/robocasa.md) for implemented semantics and validation limits.

## Ownership and boundaries

| Component | Responsibility |
|---|---|
| RoboEnv | World initialization, native device control, time, budgets, authoritative verdicts, evidence |
| Coding-agent harness or external orchestrator | Agent execution, permissions, planning, model calls, trial scheduling, resource allocation, and cleanup |

An arm exposes read/write/stop and kinematics, not `pick_object` or `open_door`.
A camera returns images, not detected objects or simulator object poses. Higher-level
operations belong in harness tools. Native motor/servo controllers remain in RoboEnv;
there is no second agent loop or parallel driver stack inside the simulator.

A single world owns lifecycle, atomic native state access, and budgets. Bindings attach
to its existing entities. Loading a device never creates another world or resets a task.

A validated new motion replaces existing motion on its physical resources; observations bypass motion admission.
All native calls execute on the simulation creation thread. Active commands yield
between native control intervals, allowing camera reads, stops and peer commands.
The world advances once for all pending devices. Tasks default to realtime (time advances continuously); explicit
`execution_mode: stepped` advances physics only during world.wait.
Launch uses that mode unless explicitly overridden. No physics steps are skipped.
Realtime has no episode step cap: configured limits and native timeout conditions
are disabled, while counters continue and budget maxima are null. Stepped keeps
configured/native limits. Success and non-timeout failure conditions remain native.
Stop cancels active and queued commands in its scope. Reset, submission and shutdown
settle pending work before changing lifecycle state. Native budgets and failures
end commands with their measured terminal state and partial progress.

## Static robot configuration and private task configuration

A robot configuration describes a stable embodiment and sensor arrangement. It is
pre-generated and reusable across tasks using that arrangement. Changing a task,
seed, object placement, or service port must not rewrite the robot catalog.

```text
catalog/
  robots/<robot-id>/
    workspace.yaml
    devices/<name>/
      profile.yaml
      driver.py
  tasks/<simulator>/<task-name>/task.yaml
```

The copyable robot directory contains only `workspace.yaml` and `devices/`.
Simulator binding defaults are internal package resources under
`src/roboenv/devices/bindings/defaults/<backend>/<robot>.yaml`, selected by backend
and robot directory name. They are packaged with the SDK so launch and device preparation can
resolve them without importing simulator engines. Owner-bundle `bindings.yaml`
plans override built-in defaults; tasks select the arrangement through `robot`. Source receipts and integration
notes belong in `docs/`; generators belong in `tools/`; test-only profiles belong in
test fixtures.
Do not add task/session files, deployment paths, or simulator metadata to the robot
workspace. A particular device may need local implementation resources, but those
are not a reason to copy the simulator or its scene assets into the agent workspace.

| Data | Location and visibility |
|---|---|
| Joint limits, command units/scaling, sensor intrinsics, rigid mounts | Static device profiles; agent-visible |
| Basic device bridge | Static `driver.py`; agent-visible |
| Entity selectors and native controller configuration | Binding/runtime configuration; owner-side |
| Task instruction | Public `env.task.instruction` |
| Seed, scene layout, objects, fixture geometry, success predicate | Private task/world state |
| Endpoint and optional episode pin | Deployment environment, not generated catalog files |
| Journal, privileged snapshots, verdict | Owner/verifier evidence |

Profiles come from authoritative model/controller/sensor definitions. An explicit
maintenance generator may compile an isolated robot model on CPU to resolve stable
facts; it must not bake a sampled kitchen or task state into those facts. Verify
reproducible output from the pinned sources. Unknown facts
remain unknown; an identity transform is not a substitute for missing calibration.

Different embodiments or sensor arrangements need different configurations. Standard
RoboCasa tasks share one Panda–Omron configuration. A custom device subset or
binding map belongs to a separate robot arrangement; its owner binding plan
replaces the whole default list. A copied Kinex workspace must match the selected configuration.
A new task with the same hardware should simply reuse the existing robot ID.

Task definitions derive the backend from the namespaced public ID. Native task
selectors occur only in `backend_kwargs`. Shared policy (`seed`, `budget_mode`,
`execution_mode`, `stop_on_success`) stays at the top level and is emitted explicitly
by the canonical serializer used for discovery, generators and private launch files.
`budget_mode` gates practice controls; it does not select the clock or horizon.

## Launch, attachment, and concurrency

```sh
roboenv prepare robocasa
roboenv launch robocasa/pick-place-microwave --port 8123
```

`prepare` installs a selected backend in an isolated interpreter and acquires its
assets. `launch` uses a prepared installation; it does not install dependencies or
write configurations into the catalog. `launch` is the only startup command.

Startup resolves the private task and binding plan, creates the world, binds devices,
validates their actual interfaces and measurements, and then reports readiness.
Live device discovery may describe the running implementation; it does not regenerate
the checked-in robot configuration. A hardware/configuration mismatch must be surfaced,
not hidden by silently changing advertised units or static calibration.

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
instruction = env.task.instruction
profiles = env.devices()
state = env.device("arm").read()
image = env.device("camera_left").read()
```

Python owners can use `with roboenv.launch(task=..., port=8123) as run:` and connect
to `run.endpoint`. Closing an attached client releases its resources, not the world.
The owner closes the service or uses `roboenv stop ENDPOINT` for a locally owned run.

All devices in a trial share one endpoint. Parallel worlds use distinct ports in the
same network namespace; `--port 0` assigns an available port. An occupied explicit
port fails without disturbing an existing service. A bind address such as `0.0.0.0`
is not the client address: a container client might connect to `http://sim:8123`.
Isolated container networks can reuse that internal port. GPU allocation remains
a scheduling concern; different ports do not create additional GPU capacity.

SDK clients discover and retain episode identity. Port reuse cannot redirect an
established client's commands into a replacement episode. A new one-shot driver
invocation attaches to the currently selected service; a runner can provide an
explicit session/generation pin when required. These values never enter static profiles.

## Small public API and observation policy

| API | Meaning |
|---|---|
| `env.task` | Immutable snapshot of public task ID, instruction, and permitted context |
| `env.refresh_task()` | Refresh the snapshot for native instruction transitions |
| `env.devices()` | Declared device profiles and supported operations |
| `env.device(name)` | Handle for a declared device; unknown names fail |
| `device.profile` | Configuration and capabilities for that device |
| Device operation | Useful device data directly, or a structured failure |

The default policy for new integrations is **sensor-only**. Task context is not a
back door to object positions, fixture placement, scene geometry, layout/style IDs,
or a scene seed. The agent must perceive and estimate scene state from sensors.
Robot joint state and declared robot-state measurements are allowed; they do not
justify exposing the entire simulator state. RoboCasa currently has empty task context.

There is no scene device, general object-state API, arbitrary simulator dispatcher,
or agent-visible scoring shortcut. `world` exposes its clock and static reference-frame profile.
Any benchmark needing privileged observations requires a separately documented,
explicit policy; never inherit ground-truth access from an older backend by accident.

Enforce this boundary in transport and data projection, not only SDK convenience
methods. HTTP rejects owner operations. Public discovery and health omit private
metadata. Operation-status queries cannot unwrap owner/verifier results, including
cached results. Journals and privileged snapshots are owner-only; agents receive
only sensor artifacts through the public artifact path.

A local harness running as the simulator's Unix user can still read that user's
files. Enforced observation isolation requires a separate agent identity/container
without simulator assets, private task definitions, owner sockets, or evidence access.
An external orchestrator must enforce that separation. A read-only
mount alone does not make privileged data safe to expose.

## Device interfaces and frames

| Type | Required basic driver operations |
|---|---|
| Sensors, including `rgb` and `rgbd` | `read`, `load`, `unload`, `status` |
| `velocity`, `gripper`, `suction` | Above, plus `write`, `stop` |
| `joint`, `dexterous_hand` | Above, plus `write`, `stop`; optional `ik`, `fk` |
| `world` | `read`, `wait`, lifecycle |
| Passive/specialized types | Their actual Kinex type contract |

Use Kinex's schema and required-action definitions. Joint-only devices need no solver. Advertised FK/IK must actually work.
Unsupported actions fail explicitly. Do not pad devices with unrelated methods.

Joint `kinematics.frame` and hand geometry declare their computed frames relative
to the device base. `intrinsics.frames` contains only additional references, which
may attach to computed frames. Exported drivers resolve these through Kinex FK;
reads remain measurement-only.
Hand geometry may contain a physical description, optional finger chains and an
optional weighted center. Moving finger links name model joints directly; fixed
links contain only an origin. Chains start at the device base, with explicit tip
offsets. Palm and grasp references are ordinary fixed frames, not geometry
selectors. Description-only geometry does not advertise FK. A center requires
fingers and remains a geometric prediction, not contact or grasp feedback.

- `read` samples a sensor or actuator without advancing physics. Actuator state
  includes positions, velocities, efforts, faults and timestamp for joint devices;
  joint, hand and gripper reads omit `frame_poses`. Velocity reads and write states
  omit global `pose`; their profiles omit `feedback.pose_frame` and fixed mounts.
  Body measurements such as height and nonempty local `frame_poses` remain available.
  Bases are unlocalized roots; world-relative queries require explicit estimates.
  Other devices may supply named live
  poses relative to each frame's declared parent, in metres and quaternion xyzw.
- Joint `write` submits absolute vectors; gripper and suction submit flat numeric
  sequences. Velocity devices submit velocity/posture channels. All require a duration.
- `stop` clears the affected buffers and applies the declared hold/halt behavior
  without advancing physics.
- `fk(joints, frame=..., reference_frame=...)` and `ik(pose, seed=..., frame=...,
  reference_frame=...)` convert joint and selected-frame pose representations
  locally in the joint driver using static profile geometry and Kinex
  utilities. Defaults are the modeled endpoint and null (device base); results
  include both frame names. Fixed endpoint descendants and fixed base references
  are supported. FK accepts measured joint-limit overshoot without clamping.
  RoboCasa does not expose simulator IK/FK RPCs. An omitted seed requires
  only a joint-state read; solving never alters live state. Exported drivers use
  standalone transport and do not require the RoboEnv SDK in the agent environment.
- Lifecycle operations affect the device binding, not episode ownership.

Device observations use one finite numeric `timestamp` in seconds, including
measured state in write results. Use acquisition time, or receipt time when it is
unavailable, in the device's clock. Combined observations use the latest component
time; cached observations retain it. RoboEnv timestamps use simulation time.
Camera reads return file paths and timestamp; fixed dimensions stay in profiles
and are validated against image contents.
Generated UUIDs are replaced by 16 random hexadecimal characters; IDs remain opaque.

A camera read must work before the first motion and return a client-accessible image
file plus metadata. The file must survive a one-shot driver process so Kinex can
record it. Explicit SDK client cleanup may remove its downloaded artifacts. RGBD
requires registered metric depth and calibration, not an unrelated second image.

Device boundaries must support correct frame composition. Static profiles contain
parent-relative frame definitions `{parent_frame, translation_m, rotation_xyzw}`.
Null parent means the device base; null transform components mean unknown.
Moving joint transforms use FK at measured positions.
Device names identify canonical frames; named moving and fixed frames are explicit.
RoboCasa's lift is a joint endpoint with frame `top`, and its arm declares `flange`.
Mounted descendants refer to these names without type-specific frame traversal.

The standard RoboCasa devices are:

| Name | Role |
|---|---|
| `base` | Omron planar mobile base |
| `torso` | Vertical lift; live top frame carries the arm and external cameras |
| `arm` | Seven-joint Panda with absolute joint-position control |
| `gripper` | Panda parallel gripper |
| `camera_left`, `camera_right` | Two third-person RGB-D views, mounted on the lift |
| `camera_wrist` | RGB-D view mounted on the arm flange |
| `world` | Root frame, without scene ground truth |

## One world, one execution authority

`write({actions, duration_s})` validates and replaces a device buffer.
Gripper and suction accept flat numeric sequences; joint and dexterous-hand
devices accept joint-vector sequences. End-effectors have no separate target write.
It returns immediately with `execution_id` and measured `state`.
The native executor consumes references at its fixed cadence; input frequency is
`len(actions) / duration_s`. References may be resampled; the final reference is
retained. Velocity commands expire to zero velocity while preserving posture.
Replacement does not call stop. Explicit stop clears its declared scope.
For every scalar gripper, stop retains the last applied grip target and actuator
preload, never the measured aperture or an unconsumed buffered goal. Direction
mode freezes its accumulated target; position mode retains its applied reference.
Release requires an explicit open command. Reset seeds a new hold from measured
posture; stopping before the first command must not initiate closure.

`world.read()` returns `time_s` and, for native `action_frames` budgets, an optional
`budget: {used_s, max_s}` in seconds. Both values use the native control interval;
`max_s` is null when uncapped, including realtime mode. Consumption includes any
counted initialization frames, so `used_s` can differ from `time_s`. This is
native episode accounting, not wall-clock time or an orchestrator deadline. Other budget
units (`sim_steps`, `api_calls`, and unknown units) omit the field because their
duration is not established. `world.status()` remains `{loaded: true}`.

`world.wait(duration_s)` waits in world time and
returns `time_s` and actual `waited_s`. Concurrent waits overlap; they do not
multiply physics steps. Tick rounding may extend elapsed time by less than one
control interval. Reads never advance physics.

Realtime defaults to continuous advancement. Stepped advances only while a
world.wait is pending, even when buffers remain. Both modes share the device
API and native control thread. Realtime has no step/time horizon; stepped retains
episode limits. Both preserve non-timeout termination conditions. Goal planning
and arrival checks belong to the harness.

Background motion retains resource ownership after submission. Faults stop the
affected motion and remain visible through reads. `devices.cancel` cancels only
the named request's wait or owned motion; late cleanup cannot stop a replacement.
Portable actuator drivers cancel their own request when accepted feedback is
malformed, including when the returned execution identity is missing. Uncertain
transport outcomes are cancelled by the outgoing request identity without retrying
the write. Cleanup failures report `stop_failed` and block further driver writes
until recovery.
Reset, freeze and shutdown settle pending waits and motion before releasing resources.
A new episode invalidates old identities. Drivers never terminate borrowed worlds.

## Coding-agent integration

Any coding agent can use the Python SDK or CLI against the same device service.
`roboenv.devices.documentation` exports public operation schemas, geometry and
connection details without executable drivers or privileged task state. The agent
harness owns reasoning, programs, sessions and task scheduling.

Kinex is an optional consumer. Use `tools/prepare_kinex_workspace.py` to copy a
resolved catalog into a fresh workspace and set `ROBOENV_ENDPOINT`. Static drivers
use standard-library transport; local IK/FK requires Kinex's Python library and
PyYAML. They never start the simulator or import simulator engines. The core SDK
and runtime do not require Kinex.

RoboEnv owns native success rules, budgets and episode evidence. Owner tooling can
use `roboenv.verification` to validate score/budget consistency and supervisor
journals. These are single-episode checks, not fresh-process replay certificates.
External orchestrators arrange container isolation and keep owner sockets,
private task definitions and journals inaccessible to agents.

## Adding a simulator and claiming support

Follow [simulator integration](docs/simulator-integration.md). Add the native adapter and binding,
pinned source/installation recipe, reusable robot configuration, lightweight task
discovery, packaging/deployment support, and observation-boundary tests. A new backend
must not add public environment methods or copy an old primitive API wholesale.

Verification separates source/config validity, CPU device conformance, actual GPU
boot, device checks, task success, and fresh-process replay. Test static regeneration,
unchanged catalogs across launch, Kinex schema/driver compatibility, three-camera
routing where applicable, concurrency, and privileged-data rejection. Preserve old
evidence and record new results honestly; no GPU/model run is implicit in adding docs
or passing CPU tests.

See [RoboCasa and its catalog](docs/robocasa.md),
[local Kinex recording](docs/kinex-local.md), and
[agent-neutral device documentation](docs/device-documentation.md) for concrete workflows.


## Concurrent device control

Devices expose individual read/write/stop operations. Kinex's devices() helper can
submit mixed calls concurrently and returns per-call results in request order.
Independent devices write concurrently; writes to the same bound entity remain ordered.
Reads and recording continue during motion. The world lock protects individual native
operations. Concurrent commands share native intervals and finish with their own
acquisition-time snapshots while retaining peer references.
Control/observation types remain independent of robot morphology.
