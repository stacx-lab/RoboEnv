# Integrating a simulator

Add a simulator by connecting its native devices to one RoboEnv world and shipping
a reusable device catalog. This guide defines the architecture and integration
contracts, using [RoboTwin](../runtime/src/roboenv_runtime/adapters/robotwin/) as the example.

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
[RoboTwin’s Aloha AgileX catalog](../catalog/robots/robotwin/robotwin-aloha-agilex/)
for a concrete rig.

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

An arm exposes device operations, not task primitives such as `pick_object`.
A camera supplies images, not detected objects. Native motor controllers stay in
RoboEnv; perception, planning, and task programs belong to the harness. Loading a
device attaches to the existing world and never resets a task or starts another world.

The SDK separates public connection/lifecycle (`client.py`, `environment.py`),
device contracts (`devices/`), task discovery/resolution (`tasks/`), isolated
installation (`provisioning/`), CLI dispatch (`cli/`), and optional client control
helpers (`control/`). Runtime adapters, workers, and scoring live in `runtime/`;
robot variants reuse typed native bindings with embodiment-specific profiles.

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

Practice lifecycle supports reset. Rewind operations are rejected before execution
and by journal verification. `AdapterBase.describe()` exposes `name` and
`embodiment`; simulator status is a separate contract.

RoboTwin separates the implementation into these layers:

| Reference | Role |
| --- | --- |
| [adapter.py](../runtime/src/roboenv_runtime/adapters/robotwin/adapter.py) | Task instruction, native verdict, and adapter identity |
| [backend.py](../runtime/src/roboenv_runtime/adapters/robotwin/backend.py) | Native task lifecycle and simulation |
| [devices.py](../runtime/src/roboenv_runtime/adapters/robotwin/devices.py) | Typed control, measurements, cameras, and native timing |
| [native_devices.py](../runtime/src/roboenv_runtime/bindings/native_devices.py) | Shared bindings for native device backends |
| [world.py](../runtime/src/roboenv_runtime/runtime/world.py) | Execution lock, command scheduling, and terminal freeze |
| [_roboenv.py](../src/roboenv/assets/_roboenv.py) | Portable device transport |

Register fixed binding IDs in the [runtime registry](../runtime/src/roboenv_runtime/bindings/__init__.py);
configuration must not select arbitrary Python imports. Bindings attach to the
initialized episode. Reset may replace native objects, so rebuild bindings and
invalidate old handles. Owner observations remain private.

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

### Execution and cancellation

Reuse the [binding executor](../runtime/src/roboenv_runtime/bindings/motion.py)
for validation, replacement, and cancellation.

`write({actions, duration_s})` validates and replaces a device buffer.
Gripper and suction accept flat numeric sequences; joint and dexterous-hand
devices accept joint-vector sequences. End-effectors have no separate target write.
It returns immediately with `execution_id` and measured `state`.
The native executor consumes references at its fixed cadence; input frequency is
`len(actions) / duration_s`. References may be resampled; the final reference is
retained. Velocity commands expire to zero velocity while preserving posture.
Replacement does not call stop. Explicit stop clears its declared scope.
Scalar gripper stop retains the applied target and preload; see
[gripper stop behavior](grippers.md) for direction, position, and reset semantics.

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

Independent devices accept concurrent writes; writes to the same bound entity
remain ordered. The world lock protects individual native operations and snapshots.
Reads and recording remain available during motion, and each command returns its
own acquisition-time snapshot while preserving peer references. A validated
replacement claims the same physical resources; observations bypass motion admission.

### Native controllers and local kinematics

Configure native controllers to implement the advertised action space. RoboTwin
uses absolute joint-position targets at 25 Hz over native 250 Hz physics, with
separate arm and gripper devices. This is an explicit practice protocol.
Preserve native physics, observations, and task hooks. Apply all active device
targets in one shared interval; separately stepping each arm changes task timing.
Whole-robot action vectors remain internal to the simulator.

Expose `control_interval_s` and `advance_control` on the adapter. Each pending
command yields between intervals; the world advances once for all active devices.
All native operations run on the simulation creation thread. Realtime advances
while idle; stepped mode advances only during world waits. Count native intervals
in both modes; realtime reports an unbounded limit, while stepped enforces its horizon.

The portable joint driver's `ik` accepts `pose` and optional `seed`, `frame`,
`reference_frame`. Explicit seeds avoid a state read; absent seeds use projected
measured joints. Numerical IK retries only near the original seed, with search limits
configured by profile `ik.budget`. Legacy `ik.search` is rejected. Keep exported
catalog drivers synchronized with `joint-client.py`.

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

| Data | Location and visibility |
| --- | --- |
| Joint geometry, units, sensor intrinsics, mounts | Static public profiles |
| Native entity selectors and controller configuration | Private binding/runtime configuration |
| Task instruction and permitted context | Public task snapshot |
| Seed, sampled scene state, success predicate | Private task/world state |
| Endpoint and optional episode pin | Deployment environment; never static catalog files |
| Journals, privileged snapshots, verdict | Owner/verifier evidence |

Binding defaults ship with the SDK and resolve without importing simulator engines.
Keep source receipts in `docs/`, generators in `tools/`, and test-only profiles in
fixtures. Do not copy simulator assets, private task/session files, or deployment
paths into the public robot workspace. Unknown calibration stays unknown; an
identity transform is not a substitute.

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
[native rig generator](../tools/generate_native_rigs.py) builds the RoboTwin catalogs.

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
id: robotwin/handover-block-aloha-agilex
robot: robotwin/robotwin-aloha-agilex
instruction: "use the left arm to grasp the red block on the table, hand it over to the right arm, and set it down in the center of the blue pad. Release it with the right gripper open"
seed: 0
budget_mode: practice
execution_mode: realtime
stop_on_success: true
backend_kwargs:
  task_name: handover_block
  embodiment: aloha_agilex
  task_config: demo_clean
  allow_seed_substitution: false
```

The required fields are `id`, `robot`, and `instruction`. The simulator prefix in
`id` selects the backend; native task selectors occur only in `backend_kwargs`.
Optional defaults are `seed: 0`, `budget_mode: official`,
`execution_mode: realtime`, `stop_on_success: false`, and `backend_kwargs: {}`.
Exports always expose these defaults in the order shown above. Discovery, catalog
generators and private launch snapshots use `Task.as_dict()` as their single
serializer. Launch writes the resolved definition to `server/tasks/task.yaml`.

`budget_mode: practice` permits the practice client's reset operation when
supported by the simulator. `official` disables practice reset;
the label alone does not certify benchmark eligibility. Existing catalog
protocol choices remain explicit. Neither value selects a horizon or clock.

Backend-specific native inputs belong in `backend_kwargs`; robot profiles and
binding plans describe the selected hardware. Unknown task fields are rejected.

Clock precedence is launch override, explicit task setting, then realtime.
Realtime disables episode step caps and native horizon timeouts while continuing
to count intervals with `budget.max: null`. Stepped retains existing native and
configured horizons. Native non-timeout failures, success policies, explicit stop,
request deadlines and external trial timeouts remain active. World initialization
applies the selected mode before reset; configured horizons are retained separately
from effective limits. Budget mode does not select the clock.

The worker owns episode policy. Adapters expose native `score()` and `budget()`;
all eight simulator adapters and FakeShell support top-level `stop_on_success`.
Launch overrides take precedence over the task value, which defaults to false.
Adapters consume and validate the policy without forwarding it to unsupported
native constructors.
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
backend: robotwin
embodiment: aloha_agilex
devices:
  left_arm:
    binding: robotwin.device/1
    entity: left_arm
    profile: devices/left_arm/profile.yaml
```

Use `src/roboenv/devices/bindings/defaults/SIMULATOR/ROBOT.yaml` for defaults.
Custom device arrangements use a separate robot reference with an owner-side
`bindings.yaml`. That plan replaces the complete default selection. Private owner
bundle plans take precedence over packaged defaults. Public robot workspaces exclude
binding plans; tasks carry only the selected `robot` reference.
Publish exactly the selected profiles and drivers.
Sensor configuration must also enable those sensors in the backend.

Catalog task IDs use `simulator/lowercase-kebab-case`, matching their directory below
`catalog/tasks/`. `roboenv tasks list --simulator SIMULATOR` lists that backend's
IDs; pass the same ID to launch or export. Native selectors in `backend_kwargs`
retain their simulator spelling. SIMPLE names omit the redundant `g1-` prefix;
Isaac Lab retains its meaningful `go2-d1-` task family.

Discovery matches native selectors plus robot to reuse catalog IDs and instructions.
Other names are normalized from the native name to lowercase kebab-case. LIBERO
uses suite/index fallback names such as `spatial-0`; RoboTwin's alternate ALOHA
arrangement adds `-aloha-agilex`. Ambiguous catalog identities and output name
collisions are errors. Discovery emits the same directory structure and preserves
source provenance. After task renames, export to a fresh output directory: bundle
conflict protection preserves edited files and refuses stale managed entries.
Tasks default to realtime and can explicitly select stepped mode. All backends use the shared stepped/realtime
executor. Stepped retains native horizons; realtime
disables step/time caps while preserving native non-timeout termination conditions.

Add lightweight discovery and default robot mapping in `src/roboenv/tasks/catalog.py`.
Read task IDs, arguments, instructions, and provenance from pinned registries or
configuration; record exclusions. Tasks expose typed devices. Start with
one small example before generating task collections, which stay outside Git.

```sh
roboenv launch robotwin/handover-block-aloha-agilex --port 8123
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

Agents using the SDK connect with `roboenv.connect(ENDPOINT)` and access an immutable
`env.task` snapshot, device profiles, and basic device operations. Use
`env.refresh_task()` for native instruction transitions; see the
[public API](device-documentation.md#public-api). Returned values contain
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
