# Integrating a simulator

Add a simulator by connecting its native devices to one RoboEnv world and shipping
a reusable device catalog. Follow [DESIGN.md](../DESIGN.md); [RoboCasa](robocasa.md)
is the reference implementation. Existing primitive bridges are not the interface
for new integrations.

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
| Harbor | Trial scheduling, deployment, agent/verifier execution, and cleanup |

New integrations default to sensor-only observations: instructions, device facts,
and declared sensor readings. Keep scene truth and verdicts private. A benchmark
requiring privileged observations needs an explicit, separately tested policy.
Preserve original success rules and budget semantics; document deliberate deviations,
such as RoboCasa's unbounded frame budget.

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
support. Override defaults that depend on old primitive APIs. Register the adapter
with `@register_adapter` and add its lazy module path to `BUILTIN_ADAPTERS`; import
heavy packages only when starting the backend.

Use [RoboCasa's adapter](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py)
and [native binding](../runtime/src/roboenv_runtime/bindings/backends/robocasa/devices.py)
as examples. Register a fixed binding ID in the
[runtime registry](../runtime/src/roboenv_runtime/bindings/__init__.py). Configuration
must not select arbitrary Python imports.

Keep simulator-specific lifecycle and stepping in a backend wrapper, with a thin
adapter connecting it to RoboEnv. RoboCasa separates these concerns as follows:

| Reference implementation | Role |
|---|---|
| [adapter.py](../runtime/src/roboenv_runtime/adapters/robocasa/adapter.py) | Adapter identity, reset/close, instruction, native verdict, budget, owner observation, and realtime hooks |
| [backend.py](../runtime/src/roboenv_runtime/adapters/robocasa/backend.py) | One native environment, controller setup, persistent goals, native intervals, and rendering |
| [devices.py](../runtime/src/roboenv_runtime/bindings/backends/robocasa/devices.py) | Entity selection, command validation, measurements, and basic actions |
| [world.py](../runtime/src/roboenv_runtime/runtime/world.py) | Shared execution lock, realtime scheduling, and terminal freeze |
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

Manipulator writes accept absolute joint targets in profile order, radians/metres.
A newer goal replaces the old one; repeated goals do not restart interpolation.
Realtime writes return on acceptance; synchronous writes advance one native interval
before returning. Both return `accepted: true`, never arrival. Keep other devices'
goals active and document actual stop/hold behavior. Read/status/recording do not
request steps. Manipulator profiles contain no execution clock; a mobile base
declares realtime command expiry or its stepped interval in `control.execution`.

Configure native controllers to implement the advertised action space. RoboCasa
selects absolute `JOINT_POSITION` control for the arm and absolute lift goals;
copying an upstream Cartesian policy action vector would not implement this contract.
Retain the native physics, observables, and task hooks when adapting goal updates.
Compose all active parts before one native interval; separately stepping each part
would change time and task dynamics. RoboCasa exposes only typed component devices; its native composite controller is
an internal implementation detail. Do not expose a whole-robot action-vector shortcut.

For a realtime backend, expose `realtime_interval` and `realtime_tick` to the shared
world scheduler. Do not start a physics thread per binding. Serialize ticks, device
effects, reads, and finalization through the same world authority. In synchronous
mode, route a write through the same native interval implementation. Count physical
intervals, including idle realtime intervals, according to the declared budget.

Put FK/IK in the manipulator driver. Export static geometry where needed; a serial
chain can use Kinex's `kinex.kinematics`. The simulator supplies measured state
and geometry, not a solver RPC. Local computation must not mutate live state.

## 4. Build a static, portable catalog

```text
catalog/robots/ROBOT/
  workspace.yaml
  devices/NAME/profile.yaml
  devices/NAME/driver.py       # except world and fixed base
  devices/NAME/_roboenv.py     # transport support for RoboCasa typed drivers
catalog/bindings/ROBOT.yaml
catalog/tasks/SIMULATOR/TASK/task.yaml
```

The robot directory contains only `workspace.yaml` and `devices/`, ready to copy to
a harness. Keep binding plans outside it, generators in `tools/`, and explanations
and source receipts in `docs/`. Launch consumes the catalog without modifying it.

| Profile field | Content |
|---|---|
| `name`, `type` | Logical directory name and Kinex device type |
| `intrinsics` | Type-specific kinematic model, control, feedback, stop scope, or sensor calibration |
| `extrinsics.mount` | Explicit parent and parent frame, `translation_m`, and unit `rotation_xyzw` |
| `access` | Typed `connection`, driver entry/interpreter, and declared actions |
| `meta` | Required creation/update metadata |

Use the actual Kinex schemas and required-action definitions to validate exports.
Kinex’s type templates live in `builtin/device/templates/<type>/`; schemas and
required actions live in `src/calibration/` and `src/device/interface.ts`. Validate
against these actual definitions in development, without adding Kinex to the
simulator worker’s dependencies. RoboCasa targets commit `5beca17`.

- Manipulators own `kinematics.model` (including joint order/limits), `kinematics.ik`,
  named `home`, `control.speed`, `feedback`, and `stop` behavior/affected devices.
- Mobile `base` profiles declare command axes, frame, units, bounds, execution mode,
  and measured feedback frames/clock. Fixed bases are driverless static mounts.
- End effectors declare tool/control type, independent measured channels, stop scope,
  and any known aperture conversion. Signed gripper commands remain native commands.
- Cameras use `intrinsics.color` for projection/distortion/FOV and `intrinsics.depth`
  for encoding, dimensions, registration, scale, invalid values, and depth quantity.
- World is profile-only with optional bounds/surfaces; it has no access or extrinsics.

Required missing facts use `???`; optional unavailable facts use schema-defined
`null`. Explicit null safety caps impose no additional limit. Never infer physical
velocity caps from torque ranges or policy frequency. Remove obsolete fields rather
than adding compatibility translations.

Extract stable facts from pinned robot/controller/camera sources, using a standalone
CPU model if necessary. Home joints come from initialization defaults. Do not export
random task placements or scene geometry as calibration. Support generation into a
temporary directory, record source hashes, and test deterministic output. The
[RoboCasa generator](../tools/generate_robocasa_config.py) demonstrates this process.

Mount poses use metres and quaternion xyzw. Cameras use OpenCV optical axes
(+X right, +Y down, +Z forward); convert native conventions explicitly. A manipulator
provides a live base-to-flange edge. Mount moving sensors on that flange instead of
baking the current joint position into a rigid transform. Mobile bases have no static
world mount. `feedback.pose_frame` declares the reference for their live pose edge;
an undeclared edge is not identity.

Provide working drivers in the catalog, with JSON arguments and one JSON result on
stdout. Diagnostics go to stderr. Success returns `ok: true` plus data; failure
returns `ok: false` and an error. See the
[manipulator adapter](../src/roboenv/assets/manipulator-client.py) and shared
[transport](../src/roboenv/assets/_roboenv.py). Copy the transport support file beside
each typed driver so individual device directories remain portable.

| Device | Required driver actions |
|---|---|
| Sensor | `read`, `load`, `unload`, `status` |
| Actuator | Sensor actions plus `write`, `stop` |
| Manipulator | Actuator actions plus local `ik`, `fk` |
| World / fixed base | No driver |

Drivers run in the agent environment, which may have no RoboEnv or simulator
packages. Transport uses the standard library; typed validation and local
kinematics use Kinex’s Python helpers and PyYAML. Test the copied files in that environment. Camera `file` and
`depth_file` paths must remain readable after a one-shot driver exits.

Use `ROBOENV_ENDPOINT` to select a deployment without rewriting profiles. Catalogs
contain no episode/session files; a runner may supply pinned connection identity.
Changing a device name requires updating its directory, profile, binding key, and
mount references together.

## 5. Connect tasks and deploy the service

A binding plan selects existing native entities; logical and native names may differ:

```yaml
schema_version: roboenv/bindings/1
backend: robocasa
embodiment: panda_omron
devices:
  base:
    binding: robocasa.device/1
    entity: base
    profile: devices/base/profile.yaml
```

Use `catalog/bindings/ROBOT.yaml` for defaults. Task `device_bindings` is a complete
replacement, not a patch. Publish exactly the selected profiles and drivers.
Sensor configuration must also enable those sensors in the backend.

Catalog task IDs use `SIMULATOR/TASK`, matching their directory below
`catalog/tasks/`. `roboenv tasks list --simulator SIMULATOR` lists that backend's
IDs; pass the same ID to launch or export. Keep native `backend_task` identifiers
separate from these public names. Discovery must emit the same directory structure.
RoboCasa task definitions explicitly include `backend_kwargs.realtime: true`;
launch accepts `--realtime` or `--sync` as a per-run override. Other backends must
implement the clock contract before accepting these overrides.

Add lightweight discovery and default robot mapping in `src/roboenv/tasks/catalog.py`.
Read task IDs, arguments, instructions, and provenance from pinned registries or
configuration; record exclusions. Tasks use `action_abstraction: devices`. Start with
one small example before generating task collections, which stay outside Git.

```sh
roboenv launch TASK_ID --port 8123
```

Local launch and Harbor use the same resolver and service. Ready means the world
and selected devices are initialized. Connection failure must not create another
world. Parallel trials use separate endpoints or isolated networks.

Add backend mappings, runtime wheels, source/asset slices, and image support to
`tools/build_simulator_image.py` and deployment recipes. Use real image digests.
Harbor must cover readiness, endpoint delivery, verifier access, and failure cleanup;
validate exports with its installed loader. See [Harbor tasks](harbor-tasks.md).
Rebuild runtime wheels/images after runtime changes; re-exporting tasks alone does
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
- Native units, rejected inputs before effects, persistent goals, stop, and clock behavior.
- Local FK/IK without simulator imports or state mutation, including joint-limit overshoot.
- Sensor routing, acquisition timestamps, read without stepping, aligned depth and its
  declared encoding/scale, and files surviving driver exit.
- Retry uncertainty, duplicate requests, parallel episodes, and replacement-episode rejection.
- Public/owner separation, cached-result restrictions, and cleanup after partial startup.

Report CPU/config checks, GPU boot, device checks, task success, and fresh-process
replay separately. A recording or successful Harbor load proves none of the latter.
GPU/model runs require explicit scope; preserve prior evidence. Use
[Kinex local testing](kinex-local.md) for copied workspaces and recording.
