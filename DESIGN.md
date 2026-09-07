# Design of RoboEnv

RoboEnv exposes **a task and a set of devices** backed by one simulation service.
The task supplies an instruction. Devices supply basic sensor readings and actuator
commands. The agent perceives the scene, plans, and writes control programs.

RoboCasa implements this design. Other existing backends are not all migrated;
their primitive APIs and observation policies are not templates for new integrations.
See [RoboCasa devices](docs/robocasa.md) for implemented semantics and validation limits.

## Ownership and boundaries

| Component | Responsibility |
|---|---|
| RoboEnv | World initialization, native device control, time, budgets, authoritative verdicts, evidence |
| Complex or another harness | Agent execution, permissions, perception, planning, controller programs, model calls |
| Harbor | Trial creation, resource/port allocation, harness startup, verifier execution, cleanup |

An arm exposes read/write/stop and kinematics, not `pick_object` or `open_door`.
A camera returns images, not detected objects or simulator object poses. Higher-level
operations belong in harness tools. Native motor/servo controllers remain in RoboEnv;
there is no second agent loop or parallel driver stack inside the simulator.

A single world owns lifecycle, execution serialization, and budgets. Bindings attach
to its existing entities. Loading a device never creates another world or resets a task.

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
      driver.py          # absent for world
  bindings/<robot-id>.yaml
  tasks/<simulator>/<task-name>/task.yaml
```

The copyable robot directory contains only `workspace.yaml` and `devices/`.
Simulator bindings belong outside it. Source receipts and integration notes belong
in `docs/`; generators belong in `tools/`; test-only profiles belong in test fixtures.
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
facts; it must not bake a sampled kitchen or task state into those facts. Record
source revisions/hashes separately and verify reproducible output. Unknown facts
remain unknown; an identity transform is not a substitute for missing calibration.

Different embodiments or sensor arrangements need different configurations. Standard
RoboCasa tasks share one Panda–Omron configuration. The generic task format also
supports an explicit device subset or custom binding map; that map replaces the
whole default list. A copied Complex workspace must match the selected configuration.
A new task with the same hardware should simply reuse the existing robot ID.

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
Isolated Harbor trial networks can reuse that internal port. GPU allocation remains
a scheduling concern; different ports do not create additional GPU capacity.

SDK clients discover and retain episode identity. Port reuse cannot redirect an
established client's commands into a replacement episode. A new one-shot driver
invocation attaches to the currently selected service; a runner can provide an
explicit session/generation pin when required. These values never enter static profiles.

## Small public API and observation policy

| API | Meaning |
|---|---|
| `env.task` | Immutable public task ID, instruction, and explicitly permitted context |
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
or agent-visible scoring shortcut. `world` is a reference-frame profile, not a sensor.
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
Harbor's RoboCasa deployment separates the agent and verifier users. A read-only
mount alone does not make privileged data safe to expose.

## Device interfaces and frames

| Type | Required basic driver operations |
|---|---|
| Sensors, including `rgb` and `rgbd` | `read`, `load`, `unload`, `status` |
| Mobile `body`, `end_effector` | Above, plus `write`, `stop` |
| `manipulator` | Above, plus `ik`, `fk` |
| `world` | No driver; profile only |
| Passive/specialized types | Their actual Complex type contract |

Use Complex's schema and required-action definitions. A manipulator without working
kinematics is not conformant merely because its driver lists `ik` and `fk`.
Unsupported actions fail explicitly. Do not pad devices with unrelated methods.

- `read` samples a sensor or actuator without advancing physics. Actuator state
  includes positions, velocities, and faults. Manipulator `flange_pose` is expressed
  in its own base frame, in metres and quaternion xyzw.
- `write` accepts a bounded target in the declared device action space. Profiles
  define component order, units, limits, control mode, scaling, and command duration.
  Manipulators accept absolute joint positions in chain order (rad/m); newer targets
  replace older ones, and repeated targets do not restart interpolation.
- `stop` implements the declared servo stop/hold behavior. In RoboCasa's synchronous
  stepped service, an in-flight interval finishes before stop; stop resets the goal
  without another step. This is not a physical emergency-stop guarantee.
- `fk(joints)` and `ik(pose, seed=...)` convert joint and flange-pose representations
  locally in the manipulator driver using static profile geometry and Complex
  utilities. RoboCasa does not expose simulator IK/FK RPCs. An omitted seed requires
  only a joint-state read; solving never alters live state. Exported drivers use
  standalone transport and do not require the RoboEnv SDK in the agent environment.
- Lifecycle operations affect the device binding, not episode ownership.

A camera read must work before the first motion and return a client-accessible image
file plus metadata. The file must survive a one-shot driver process so Complex can
record it. Explicit SDK client cleanup may remove its downloaded artifacts. RGBD
requires registered metric depth and calibration, not an unrelated second image.

Device boundaries must support correct frame composition. Static profiles contain
rigid transforms; moving transforms come from declared robot-state measurements.
Complex currently treats a manipulator as base plus live flange, and a body as one
frame. Therefore RoboCasa keeps its lift as `torso`, a one-joint manipulator, rather
than merging it into `body` without support for the additional moving frame.

The standard RoboCasa devices are:

| Name | Role |
|---|---|
| `body` | Omron planar mobile base |
| `torso` | Vertical lift; live flange carries the arm and external cameras |
| `arm` | Seven-joint Panda with absolute joint-position control |
| `gripper` | Panda parallel gripper |
| `camera_left`, `camera_right` | Two third-person RGB views, mounted on the lift |
| `camera_wrist` | RGB view mounted on the arm flange |
| `world` | Root frame, without scene ground truth |

## One world, one execution authority

Bindings validate inputs before effects. RoboCasa writes return `accepted: true`,
never an arrival claim. In realtime mode, background physics pursues persistent
goals; in synchronous mode, each write runs one control period then pauses. Other
parts retain their active goals. Read/status/kinematics do not request extra steps.
RoboCasa imposes no horizon in either mode; action frames count physical intervals.
Other backends retain their declared benchmark budgets. The service owns clocks;
controller tools do not branch on realtime versus synchronous execution.

Where a backend needs a joint action vector, compose it before one native step.
Do not turn one command into separately stepped actuator writes. If simultaneous
control needs a composite device, define its basic vector explicitly and reject
overlapping command ownership. That is a device contract, not a task-solving macro.

All transports share the same serialization, identity, and durable motion outcomes.
A timeout does not establish that no motion occurred. Do not blindly resend an
uncertain command with a new request ID; query the allowed original operation status.
Device close never shuts down the world. A new episode has a new identity; backend
reset/restore support, where present, must invalidate old bindings and handles.

## Complex and Harbor integration

Copy `catalog/robots/<robot-id>/` to `.complex/robots/<robot-id>/`, select the robot
in Complex's project configuration, and set `ROBOENV_ENDPOINT` before starting the
harness. Copied drivers use standard-library transport; local IK/FK requires
Complex's Python library and PyYAML. They do not import RoboEnv or simulator packages.

Static drivers forward basic operations and adapt result/error shapes. They do not
start the simulator, contain policies, or generate profiles. Complex discovers the
files, supplies tools and permissions, and records sensor outputs. RoboEnv supports
other harnesses through the same SDK without depending on Complex at runtime.

Harbor exports the private task/runtime definition and verifier separately from
agent-visible configuration. It starts a service, waits for device readiness,
provides the endpoint to the harness, runs the verifier, and cleans up the world.
The current RoboCasa Harbor exporter also stages live device descriptors and thin
drivers under `/run/roboenv/robot`; this is trial-local output, never a catalog write.
Complex workspace setup remains the harness integration's responsibility.

RoboEnv's verifier uses original success rules, budgets, and owner evidence.
Agent claims and recording outcome labels do not establish benchmark success.
Task generation must pass the real installed Harbor loader; loading a task does
not establish that the simulator boots or the agent can solve it.

## Adding a simulator and claiming support

Follow [simulator integration](docs/simulator-integration.md). Add the native adapter and binding,
pinned source/installation recipe, reusable robot configuration, lightweight task
discovery, packaging/deployment support, and observation-boundary tests. A new backend
must not add public environment methods or copy an old primitive API wholesale.

Verification separates source/config validity, CPU device conformance, actual GPU
boot, device checks, task success, and fresh-process replay. Test static regeneration,
unchanged catalogs across launch, Complex schema/driver compatibility, three-camera
routing where applicable, concurrency, and privileged-data rejection. Preserve old
evidence and record new results honestly; no GPU/model run is implicit in adding docs
or passing CPU tests.

See [RoboCasa and its catalog](docs/robocasa.md),
[local Complex recording](docs/complex-local.md), and
[Harbor task generation](docs/harbor-tasks.md) for concrete workflows.
