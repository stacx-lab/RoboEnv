# Integrating a simulator

Add a simulator by connecting its native devices to one RoboEnv world and shipping
a reusable device catalog. Follow [DESIGN.md](../DESIGN.md); [RoboCasa](robocasa.md)
is the reference implementation.

Coding-agent workflows are defined in
[integrate-simulator-kinex](../.agents/skills/integrate-simulator-kinex/SKILL.md)
for implementation and
[test-simulator-kinex](../.agents/skills/test-simulator-kinex/SKILL.md) for acceptance.
See [agent integration](device-documentation.md) for SDK/CLI consumers and
[local Kinex use](kinex-local.md) for optional device compatibility.

## Choose the scope

| Change | Files and behavior to update |
|---|---|
| Same rig, new task/seed/layout | Private task configuration and instruction; reuse the catalog |
| Concurrent tasks | Separate endpoints and harness workspaces or isolated networks |
| Rename a device | Directory, profile name, binding key, and mount references |
| Add or replace a camera | Enabled native sensor, binding selection, intrinsics, and rigid mount |
| Change hardware or controller | Matching rig catalog, native binding, assets, and validation |
| Add a simulator | Source/installation recipe, adapter, bindings, task discovery, deployment, and tests |

A new task does not need a new adapter when the existing backend and rig already
support it. Changing device boundaries can require frame-interface support, not
just new directories: moving sensors need a declared live parent frame. See
[RoboCasa's catalog and frames](robocasa.md#catalog-and-generation) for the concrete rig.

## 1. Define the boundary

Identify the upstream tasks, robot models, controllers, sensors, frames, units,
budgets, and success rules. Reuse one catalog entry across tasks with the same rig.
Hardware or sensor changes need a matching configuration; scene layout, object poses,
and seeds belong to private task state.

| Component | Responsibility |
|---|---|
| Adapter | Initialize/close the simulator, native control, task instruction, budget, and score |
| WorldSession / worker | One world, serialized effects, lifecycle, clock, and evidence |
| Native binding | Basic operations on existing entities in that world |
| DeviceService / supervisor | Validation, dispatch, episode identity, deduplication, and transport boundaries |
| Driver | Portable device access and local utilities such as IK/FK |
| Agent harness | Perception, planning, controller loops, and model calls |
| External orchestrator | Trial scheduling, deployment, agent execution, and cleanup |

New integrations default to sensor-only observations: instructions, device facts,
and declared sensor readings. Keep scene truth and verdicts private. A benchmark
requiring privileged observations needs an explicit, separately tested policy.
Preserve native success rules and non-timeout terminations. Apply the shared clock
policy: realtime has no episode step cap; stepped retains native/configured horizons.

## 2. Pin sources and prepare an interpreter

Register selected upstreams in `upstreams.lock.json` and `third_party/`. Record exact
revisions, dependencies, asset identities, and checksum-verified patches. Follow the
upstream update rules in [AGENTS.md](../AGENTS.md); do not rely on unrecorded edits.

```sh
roboenv sources check-updates --backends BACKEND
roboenv sources sync --backends BACKEND
roboenv sources status --backends BACKEND
```

Add the installation recipe to `runtime/environments.json` and the preparation and
diagnostic support in `src/roboenv/provisioning/`, so these commands work:

```sh
roboenv prepare BACKEND
roboenv doctor BACKEND
```

Keep simulator dependencies in an isolated interpreter. SDK imports, task discovery,
and catalog reads must not boot a GPU or download models. Record renderer and asset
requirements alongside the Python lock; see [setup](setup.md).

## 3. Implement the adapter and bindings

Implement [EnvAdapter](../runtime/src/roboenv_runtime/core/adapter.py): identity,
reset/close, task instruction, budget, score, and required owner observation/media
support. Keep observations explicit and owner-only. Register the adapter
with `@register_adapter` and add its lazy module path to `BUILTIN_ADAPTERS`; import
heavy packages only when starting the backend.

The runtime no longer exports `Capability` or exposes adapter `capabilities`
metadata. `AdapterBase.describe()` returns only `name` and `embodiment`; update
consumers that imported those constants or read that field. Optional practice
snapshots use `checkpoint()` and `restore(token)` methods (the
`SupportsCheckpoint` protocol); the worker detects these methods directly.
Simulation-status capability fields are separate and unchanged.

Use [RoboCasa's adapter](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py)
and [native bindings](../runtime/src/roboenv_runtime/bindings/backends/robocasa/)
as examples. Register a fixed binding ID in the
[runtime registry](../runtime/src/roboenv_runtime/bindings/__init__.py). Configuration
must not select arbitrary Python imports.

Keep simulator-specific lifecycle and stepping in a backend wrapper, with a thin
adapter connecting it to RoboEnv. RoboCasa separates these concerns as follows:

| Reference implementation | Role |
|---|---|
| [adapter.py](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py) | Adapter identity, reset/close, instruction, native verdict, budget, owner observation, and native control hooks |
| [backend.py](../runtime/src/roboenv_runtime/adapters/robocasa/backend.py) | One native environment, controller setup, persistent goals, native intervals, and rendering |
| [bindings/backends/robocasa/](../runtime/src/roboenv_runtime/bindings/backends/robocasa/) | Per-type bindings (joint, velocity, gripper, camera, world): entity selection, command validation, measurements, and basic actions |
| [world.py](../runtime/src/roboenv_runtime/runtime/world.py) | Shared execution lock, command scheduling, and terminal freeze |
| Typed client assets and [_roboenv.py](../src/roboenv/assets/_roboenv.py) | Per-type validation/local utilities and shared portable transport |

The backend's internal helpers are not an agent API. In particular, RoboCasa's
owner observation includes object poses; never publish it as a device read or
public task context. Bind to the initialized episode. Native reset can replace
robot/controller/model objects, so rebuild bindings and invalidate old handles
rather than retaining references across episodes.

The [binding model](../src/roboenv/devices/model.py) defines `DeviceDefinition`,
`Action`, `BoundDevice`, and `ExecutionContext`. A factory takes `(spec, root, context)`:

- `spec` selects a logical name, entity, profile, and validated configuration.
- `root` bounds profile access to the robot definition directory.
- `context.native` supplies the existing world; verify backend and embodiment.
- `context.devices` maps entities to logical names, including mount parents.

Validate dimensions, finite values, ranges, and preconditions before effects. Route
mutations through `execution.perform(..., mutating=True)`. Declare accurate input
and output schemas and reject unsupported operations. A binding neither creates a
second world nor calls its own locked worker through a socket. Closing it releases
only binding resources.

Joint writes submit `actions` with `duration_s` and return immediately.
The executor samples buffers on native ticks. Planning and arrival checks belong
to the harness. Reuse the [binding executor](../runtime/src/roboenv_runtime/bindings/motion.py)
for validation, replacement and cancellation. Scalar and velocity writes use the
same duration contract. World `read` returns `time_s` and an optional
`budget: {used_s, max_s}` in seconds for native `action_frames` budgets. The shared
binding multiplies native consumption and cap by the control interval; an uncapped
budget has `max_s: null`. Counted initialization frames remain included, so budget
elapsed time can differ from the world clock. This does not represent wall-clock
time or orchestrator deadlines. Other units, including `sim_steps` and `api_calls`, omit
the field because their duration is not established. `status` stays `{loaded: true}`;
`wait(duration_s)` returns only `time_s` and the wait's `waited_s`.
Realtime advances continuously; stepped advances during waits.

Configure native controllers to implement the advertised action space. RoboCasa
selects absolute `JOINT_POSITION` control for the arm and absolute lift goals;
copying an upstream Cartesian policy action vector would not implement this contract.
Retain the native physics, observables, and task hooks when adapting goal updates.
Compose all active parts before one native interval; separately stepping each part
would change time and task dynamics. RoboCasa exposes only typed component devices; its native composite controller is
an internal implementation detail. Do not expose a whole-robot action-vector shortcut.

Expose `control_interval_s` and `advance_control` on the adapter. Each pending
command yields between intervals; the world advances once for all active devices.
All native operations run on the simulation creation thread. Realtime advances
while idle; stepped mode advances only during world waits. Count native intervals
in both modes; realtime reports an unbounded limit, while stepped enforces its horizon.

Put FK/IK in the joint driver. Export static geometry where needed; a serial
chain can use Kinex's `kinex.kinematics`. The simulator supplies measured state
and geometry, not a solver RPC. Local computation must not mutate live state.

## 4. Build a static, portable catalog

```text
catalog/robots/SIMULATOR/ROBOT/
  workspace.yaml
  devices/NAME/profile.yaml
  devices/NAME/driver.py       # except world
  devices/_roboenv.py          # shared portable transport for the collection
src/roboenv/devices/bindings/defaults/SIMULATOR/ROBOT.yaml
catalog/tasks/SIMULATOR/TASK/task.yaml
```

The robot directory contains only `workspace.yaml` and `devices/`, ready to copy to
a harness. Keep binding defaults in the internal package, generators in `tools/`,
and explanations and source receipts in `docs/`. Launch consumes the catalog without modifying it.
Task `robot` fields use `SIMULATOR/ROBOT` relative catalog paths; robot identity
IDs remain flat. Binding resources are selected by backend and robot directory name.
Generators that emit bindings accept `--bindings-out DIR`, separate from `--out`
(the public catalog). Omit it to generate only the public catalog and any explicitly
requested facts or receipts. To maintain built-in defaults, pass
`--bindings-out src/roboenv/devices/bindings/defaults`.

| Profile field | Content |
|---|---|
| `name`, `type` | Logical directory name and Kinex device type |
| `intrinsics` | Type-specific kinematic model, control, feedback, stop scope, or sensor calibration |
| `extrinsics.mount` | Explicit parent and parent frame, `translation_m`, and unit `rotation_xyzw` |
| `access` | Typed `connection`, driver entry/interpreter, and declared actions |
| `meta` | Required creation/update metadata |

Use the actual Kinex schemas and required-action definitions to validate exports.
Kinex’s type templates live in `packages/core/assets/builtin/device/templates/<type>/`;
schemas and required actions live in `packages/core/src/device/profile/` and
`packages/core/src/device/interface.ts`. Validate
against these actual definitions in development, without adding Kinex to the
simulator worker’s dependencies.

- Manipulators own `model` (including joint order/limits), `kinematics.ik`,
  named `home`, position `control`, `feedback`, and `stop` behavior/affected devices.
- Mobile `base` profiles declare command axes, frame, units, bounds, supported modes,
  and measured feedback frames and timestamps. Fixed bases are driverless static mounts.
- Grippers declare control, aperture bounds and feedback support, and stop scope.
  `extrinsics.mount` locates the gripper base; `intrinsics.frames.tcp` locates its TCP.
  `intrinsics.geometry` holds approach/closing axes and an optional finger description.
  Native finger measurements and aperture conversion stay private; reads return
  aperture, faults and timestamp. See [gripper geometry](grippers.md).
- Cameras use `intrinsics.color` for projection/distortion/FOV and `intrinsics.depth`
  for encoding, dimensions, registration, scale, invalid values, and depth quantity.
- World exposes read/wait through its driver, with optional bounds/surfaces and no extrinsics.

Required missing facts use `???`; optional unavailable facts use schema-defined
`null`. Explicit null safety caps impose no additional limit. Never infer physical
velocity caps from torque ranges or policy frequency. Remove obsolete fields rather
than adding compatibility translations.

Simulator joint and velocity devices declare `control.limit_handling: native`. Validate types,
finite values, dimensions and supported modes before effects, then let native
controllers and physics handle limits. Model joint ranges describe geometry;
they are not an extra write rejection rule. Continuous command bounds may be
`null` when no independent bound exists in that command frame. Document native
clipping and coordinate transforms in workspace safety notes; do not copy URDF
velocity/effort ratings into profiles unless the running simulator enforces them.
Scalar grippers retain Kinex's normal rejection: normalized position commands
must be in `[0, 1]`, and direction commands in `[-1, 1]`. The native controller
still owns finger motion, saturation and physical constraints after admission.
See the [source audit](simulator-limits.md).

Extract stable facts from pinned robot/controller/camera sources, using a standalone
CPU model if necessary. Home joints come from initialization defaults. Do not export
random task placements or scene geometry as calibration. Support generation into a
temporary directory and test deterministic output. The
[RoboCasa generator](../tools/generate_robocasa_config.py) demonstrates this process.

Mount poses use metres and quaternion xyzw. Cameras use OpenCV optical axes
(+X right, +Y down, +Z forward); convert native conventions explicitly. A joint device
provides a live base-to-flange edge. Mount moving sensors on that flange instead of
baking the current joint position into a rigid transform. Mobile bases have no static
world mount or global pose feedback. Local body measurements can supply moving
attachment frames; world-relative transforms require explicit estimates. An
undeclared edge is not identity.

Provide working drivers in the catalog, with JSON arguments and one JSON result on
stdout. Diagnostics go to stderr. Success returns `ok: true` plus data; failure
returns `ok: false` and an error. See the
[joint adapter](../src/roboenv/assets/joint-client.py) and shared
[transport](../src/roboenv/assets/_roboenv.py). Export one `devices/_roboenv.py`
shared by all `devices/NAME/driver.py` files. Drivers resolve that parent directory
from `__file__`; locating the transport needs neither the RoboEnv package nor
a transport-specific Python path. Typed drivers still use Kinex's Python helpers.
`write_driver` installs the shared module and removes obsolete per-device copies.
For static workspace copies, include `shared_driver_resources(collection)` once
and `driver_resources(device)` for each selected device, including local geometry.
Move the collection as a unit, or include the shared file when copying a subset.

| Device | Required driver actions |
|---|---|
| Sensor | `read`, `load`, `unload`, `status` |
| Actuator | Sensor actions plus `write`, `stop` |
| `joint`, `dexterous_hand` | Actuator actions; optional declared local `ik`, `fk` |
| `world` | Profile only; no driver |

Drivers run in the agent environment, which may have no RoboEnv or simulator
packages. Transport uses the standard library; typed validation and local
kinematics use Kinex’s Python helpers and PyYAML. Drivers default to persistent
JSON-lines dialogs (`driver.py --serve`), sharing cached profile data and an HTTP
connection across calls. Test the copied files in that environment. Camera `file` and
`depth_file` paths must remain readable after a one-shot driver exits.

Use `ROBOENV_ENDPOINT` to select a deployment without rewriting profiles. Catalogs
contain no episode/session files; a runner may supply pinned connection identity.
Changing a device name requires updating its directory, profile, binding key, and
mount references together.

## 5. Connect tasks and deploy the service

### Task format

Every simulator uses the same flat `task.yaml`, parsed by `roboenv.tasks.model.Task`.
Local startup serializes that resolved definition unchanged into
private worker configuration. There is no second runtime task schema.

```yaml
id: libero/bowl-onto-plate
backend: libero
backend_task: libero.spatial_0
robot: libero/libero-panda-tabletop
instruction: Pick the akita black bowl between the plate and the ramekin and place it on the plate
backend_kwargs:
  suite_name: libero_spatial
  task_id: 0
  init_state_index: 0
budget_mode: practice
```

The five required fields are `id`, `backend`, `backend_task`, `robot`, and
`instruction`. Optional defaults are `backend_kwargs: {}`, `seed: 0`,
`budget_mode: official`, `device_bindings: null`, and `execution_mode: realtime`.
Backend-specific native inputs belong in `backend_kwargs`; robot profiles and
binding plans describe actual hardware rather than forcing every robot to have
the same device names. Unknown fields and obsolete task formats are rejected.

Clock precedence is launch override, explicit task setting, then realtime.
Realtime disables episode step caps and native horizon timeouts while continuing
to count intervals with `budget.max: null`. Stepped retains existing native and
configured horizons. Native non-timeout failures, success policies, explicit stop,
request deadlines and external trial timeouts remain active. World initialization
applies the selected mode before reset; configured horizons are retained separately
from effective limits. Budget mode does not select the clock.

The worker owns episode policy. Adapters expose native `score()` and `budget()`;
adapters supporting `stop_on_success` consume and validate that boolean before
constructing the native environment. It defaults to false and must not be passed
to a native constructor that does not support it. LIBERO, RoboTwin, BEHAVIOR,
RoboDojo, RoboCasa, SIMPLE and FakeShell support this option in `backend_kwargs`.
The worker checks enabled success stopping and stepped budget exhaustion after
each world control interval, then seals the terminal verdict automatically.
Adapters do not declare `bounded_episodes` or install additional tick callbacks
for this policy. Status derives the capability from success stopping and the
effective stepped budget.

Natural termination ends pending buffers and waits with `episode_finished` and
known effects. Buffer cleanup runs without an extra tick and precedes the final
observation and score; cleanup failure prevents certification. A buffer or wait
that completes exactly at the terminal frame may still return normally. Final
reads and repeated submit/stop use the sealed verdict. Existing native success
predicates, non-timeout termination and owner verification budget rules remain intact.

### Device bindings

A binding plan selects existing native entities; logical and native names may differ:

```yaml
schema_version: roboenv/bindings/1
backend: robocasa
embodiment: panda_omron
devices:
  base:
    binding: robocasa.velocity/1
    entity: base
    config: {part: base}
    profile: devices/base/profile.yaml
```

Use `src/roboenv/devices/bindings/defaults/SIMULATOR/ROBOT.yaml` for defaults. Task `device_bindings` is a complete
replacement, not a patch. Private owner bundles may include `bindings.yaml`; these
plans take precedence over packaged defaults, while explicit task bindings take
precedence over both. Public robot workspaces exclude binding plans.
Publish exactly the selected profiles and drivers.
Sensor configuration must also enable those sensors in the backend.

Catalog task IDs use `SIMULATOR/TASK`, matching their directory below
`catalog/tasks/`. `roboenv tasks list --simulator SIMULATOR` lists that backend's
IDs; pass the same ID to launch or export. Keep native `backend_task` identifiers
separate from these public names. Discovery must emit the same directory structure.
Tasks default to realtime and can explicitly select stepped mode. All backends use the shared stepped/realtime
executor. Stepped retains native horizons; realtime
disables step/time caps while preserving native non-timeout termination conditions.

Add lightweight discovery and default robot mapping in `src/roboenv/tasks/catalog.py`.
Read task IDs, arguments, instructions, and provenance from pinned registries or
configuration; record exclusions. Tasks expose typed devices. Start with
one small example before generating task collections, which stay outside Git.

```sh
roboenv launch TASK_ID --port 8123
```

SDK and CLI launch use the same resolver and service. Ready means the world
and selected devices are initialized. Connection failure must not create another
world. Parallel trials use separate endpoints or isolated networks.

Add backend mappings, runtime wheels, source/asset slices, and image support to
`tools/build_simulator_image.py` and deployment recipes. Use real image digests.
External orchestrators own readiness, endpoint delivery, owner access, and cleanup.
See [image preparation](setup.md#standalone-simulator-images).
Rebuild runtime wheels/images after runtime changes; updating task definitions alone does
not update the deployed service implementation. Preserve previous deployment
identities and validation evidence.

## 6. Preserve the public service boundary

Agents using the SDK connect with `roboenv.connect(ENDPOINT)` and access immutable
`env.task`, device profiles, and basic device operations. Returned values contain
device data, not internal journal/budget receipts. The wire protocol is `roboenv/1`
with `roboenv/devices/1` semantics: session, generation, request ID, deadline, operation,
and object arguments. HTTP GET `/` discovers identity without initializing a world.

| Public HTTP operation | Purpose |
|---|---|
| `environment.describe` | Permitted task and device information |
| `devices.list`, `devices.describe` | Device discovery |
| `devices.call` | A declared basic device operation |
| `operations.get` | Status/result of an allowed public request |
| `health` | Sanitized readiness and liveness |

Reuse [DeviceService](../src/roboenv/devices/service.py), the
[worker](../runtime/src/roboenv_runtime/runtime/worker.py), and the
[HTTP boundary](../runtime/src/roboenv_runtime/runtime/http_service.py); do not add a
parallel server stack. HTTP and owner Unix access share the world and execution lane.
Established clients
pin episode identity; an old client cannot command a replacement episode on the
same port. Deduplicate mutating requests. Lost responses must retain their request
ID as an uncertain outcome; query status instead of resubmitting under a new ID.

Implement safe public projections explicitly; generic transport does not sanitize
a new adapter automatically. Owner observe, reset, score, finalization, journals,
and snapshots remain private, including cached results through `operations.get`.
Do not expose scene truth through task context, profiles, health, errors, or artifacts.
Only sensor artifacts may be downloaded, under episode-scoped paths.

Deployment must protect owner sockets, assets, private task definitions, and evidence
from the agent. A read-only mount still reveals its contents. Local runs sharing one
Unix user test functionality, not filesystem isolation.

## 7. Verify the integration

Run `bash tools/check.sh` after adding positive and negative tests for:

- Static and live schema conformance, required actions/calibration readiness, complete
  deterministic regeneration, copied drivers with support files, and unchanged catalogs.
- Native units, rejected inputs before effects, persistent goals, stop, behavior.
- Local FK/IK without simulator imports or state mutation, including joint-limit overshoot.
- Sensor routing, acquisition timestamps, read without stepping, aligned depth and its
  declared encoding/scale, and files surviving driver exit.
- Retry uncertainty, duplicate requests, parallel episodes, and replacement-episode rejection.
- Public/owner separation, cached-result restrictions, and cleanup after partial startup.

Report CPU/config checks, native boot, device checks, task success and fresh-process
replay separately. Integration acceptance establishes the working native/agent path;
task-performance acceptance sets separate success-rate and held-out requirements.
Use sessions, device events, video and replay to diagnose failures. Preserve prior
evidence and keep GPU/model runs within the authorized scope. See
[simulator validation](gpu-validation.md) for check tools and
[local Kinex use](kinex-local.md) for workspace setup and recording.

### Native clocks

RoboTwin, RoboDojo and BEHAVIOR share native target validation, independent writes,
owner recording and adapter lifecycle. RoboDojo and BEHAVIOR use the same world executor as other backends.
