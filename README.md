# RoboEnv

Robotics environments with discoverable devices, shared simulation state, and
benchmark verification. Use them from Python, the CLI, Kinex, or Harbor.
RoboEnv owns the environment and verdict; the agent harness owns reasoning and
controller programs. See [DESIGN.md](DESIGN.md) for the architecture.

## Prepare a simulator

From this checkout on Linux with uv installed:

```sh
bash tools/bootstrap.sh
source .venv/bin/activate
roboenv prepare robocasa
```

Bootstrap installs the CLI into `.venv`. Then `prepare` synchronizes selected
pinned sources, installs an isolated simulator environment, and reports checks.
Replace `robocasa` with your simulator. RoboCasa preparation also downloads its
kitchen assets (about 10 GB). No wheel-building, `--install`, or
environment-path arguments are needed. Other simulators require separate asset acquisition;
see [setup](docs/setup.md) for prerequisites and diagnostics.

## Start and connect

Tasks default to `execution_mode: realtime`; `stepped` is an explicit opt-in.
`stepped` freezes idle simulation; `realtime` advances it against a monotonic clock.
`roboenv tasks list` reports each task's mode. Launch uses the resolved task setting;
`--execution-mode` (or Python `execution_mode=`) explicitly overrides it.

Realtime has no episode step cap, including configured `max_steps` and native
time limits. Frames are still counted and budgets report `max: null`. Stepped
mode retains configured/native horizons. Success policies, non-timeout native
termination, faults, explicit stop and request deadlines remain active in both.

At a stepped horizon, physics freezes and unfinished buffers or waits end with
`episode_finished`. Final reads and stop/submit remain available; budget exhaustion
does not fail the world. The worker automatically seals the native verdict and records
`step_limit` when unsuccessful. Harbor's existing reward budget rules still apply.

RoboCasa, LIBERO, RoboTwin, BEHAVIOR, RoboDojo, SIMPLE and the test backend accept
`backend_kwargs: {stop_on_success: true}` to end at the first successful control
interval in either clock mode. The default is `false`: success alone does not end
execution. Natural episode completion cleans unfinished buffers without another
physics tick before sealing the final snapshot. Explicit cancellation remains an
interruption; cleanup failures cannot produce a successful final verdict.

```sh
roboenv tasks list --simulator robocasa
roboenv launch robocasa/pick-place-microwave --port 8123
```

In another terminal:

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
print(env.task.instruction)
print(env.devices())
print(env.device("arm").read())
```

The CLI exposes the same devices: `roboenv device arm read '{}' --endpoint URL`.
Omit `--port` to assign a free port automatically. Concurrent runs use different ports;
`roboenv.connect()` also discovers a single ready local run. Select a local
service by port for lifecycle commands:

```sh
roboenv status --port 8123 --json
roboenv stop --port 8123 --json
```

`--port` targets `127.0.0.1`. Use `--endpoint http://HOST:PORT` for a remote or
container service. Either explicit selector overrides `ROBOENV_ENDPOINT` and
selects only that service; selectors cannot be combined.
HTTP stop requires launch with `--allow-stop`. Local owners can instead stop a
run by ID or `run.json` path. `launch` owns the environment until Ctrl-C;
`--detach` leaves it running until stopped.
Python owners use `with roboenv.launch(task="TASK_ID", port=8123) as run:`
and connect to `run.endpoint`. Use
`--host 0.0.0.0` when serving a task across an isolated container network.

Launch streams simulator stdout/stderr to the terminal's stderr and retains the
run logs. `--json` keeps stdout machine-readable; `--detach` streams until ready.
Python callers can enable streaming with `log_stream=sys.stderr`.

Official LIBERO uses the same typed device path with complete scene-specific Panda
catalogs in `catalog/robots/libero/libero-panda-{tabletop,floor,kitchen,living-room,study}/`.
Each task selects its robot by a relative catalog path, such as
`robot: libero/libero-panda-kitchen`; each scene directory contains its full device
profiles and native mounting geometry.
Run `roboenv prepare libero` and `roboenv launch libero/bowl-onto-plate`.
Its absolute joint control is explicitly a practice protocol, not the published OSC
benchmark action space. `tools/accept_libero.py --out OUTPUT` runs native device,
transport, Kinex recording and fresh-process replay gates without model calls;
task scheduling uses RoboEnv's [Kinex Harbor integration](integrations/kinex/README.md).

The checked-in LIBERO tasks are `bowl-onto-plate` (grasp/place),
`close-top-drawer` (close drawer), `bowl-into-bottom-drawer` (place the bowl,
then close), and `mug-into-microwave` (place the mug, then close).
The latter two are multi-stage LIBERO-Long stress cases, selected for
combined object and articulated-fixture manipulation, not an official per-task
difficulty ranking. Discovery covers all 130 native tasks and selects each scene's
robot catalog from the task's BDDL problem type. Each catalog includes portable
copies of the shared device drivers.
The additional `drawer-and-bowl` example maps to `libero_90` task 8: open the
top drawer and put the bowl inside. It uses the kitchen Panda catalog.


RoboCasa’s catalog contains arm, torso, base, gripper, three RGB-D cameras,
and world. The device API is `env.task`, `env.devices()` and `env.device(name)`.

Joint devices and dexterous hands accept buffered `write(actions, duration_s=...)`
without planning. Goal planning and measured arrival checks belong to the agent harness.
Gripper and suction writes use flat `actions: [value, ...]`; joint and hand writes
use sequences of joint vectors. A single end-effector command uses `[value]`.
Use concurrent writes for multiple devices; the simulator owns their shared clock. See
[the device contract](DESIGN.md#one-world-one-execution-authority).

Simulation defaults to realtime. Writes return immediately.
Use `env.device("world").read()` for `time_s` and `.wait(duration_s)` to wait on
simulation time. In stepped mode, pending world waits drive shared native ticks.
Reads and writes never request ticks. Joint buffers retain their final reference;
velocity buffers zero travel on expiry while preserving posture.
Exported drivers retain local IK/FK, persistent transport and scoped lifecycle.
Transport failures never resubmit motion. Task-solving controllers belong in Kinex.

All scalar grippers in RoboCasa, LIBERO, RoboTwin, RoboDojo and BEHAVIOR expose
TCP descriptions, approach/closing axes, moving contact frames and physical
aperture feedback. Mesh contact patches keep rectangular dimensions unset.
See [gripper geometry](docs/grippers.md) for frame conventions and source validation.

SIMPLE G1 uses a `velocity` base for legs and waist, separate `joint` arms,
`dexterous_hand` fingers, and stereo RGB cameras. Run `roboenv prepare simple`
and `roboenv launch simple/g1-xmove-pick`. It uses Isaac rendering over MuJoCo physics with command-driven stepping. See [SIMPLE setup, contracts and validation](docs/simple.md) for requirements
and the current validation limits. The metric control interface is a practice
protocol; native success predicates are preserved.

The experimental [Go2–D1 Isaac Lab environment](docs/go2-d1.md) provides base
velocity/height/tilt control, independent arm IK, a gripper, and two mounted RGB-D cameras.
See [Isaac Lab setup](docs/isaaclab.md) for installation and diagnostics. The former
Isaac Lab G1 practice task has been removed; SIMPLE's G1 integration remains available.

Device reads and recording remain available during bounded writes. Independent
devices accept concurrent writes through `env.device(name)`; writes to the same
physical device are ordered. Stop interrupts its declared scope and clears older
queued writes. Individual simulator steps and consistent snapshots remain atomic.

Portable persistent drivers now use correlated JSONL requests and may reply out of
order. Refresh existing workspace drivers from the updated catalog/export, then
reload Kinex devices. Restart simulator services you own to pick up runtime changes;
borrowed services remain under their owner's lifecycle control.

## Export a Harbor task

With a [prepared native deployment](docs/harbor-tasks.md):

```sh
roboenv export robocasa/pick-place-microwave --out generated/example
```

Export resolves the same task configuration as local startup, builds the SDK
wheel, and validates the Harbor format. The environment starts automatically
before the agent. Reference solutions are not included by default. Agent
installation and model configuration belong to the Harbor integration.

## Develop

```sh
bash tools/check.sh
```

Tests use shared fresh-worker fixtures for device, transport, and lifecycle
checks. CLI replay and verifier coverage exercise the real worker rather than a
second fake server. Recipe and installation checks live in `tests/test_setup.py`; test helpers live
outside test modules so tests do not import one another. The default check still runs the complete CPU suite.

Bounded-write regressions cover all six simulator backends using CPU state and
the real control/dispatch code. With LIBERO prepared and a built Kinex checkout,
run the separate native smoke (Node >=22.19):

```sh
.venv/bin/python tests/libero_control_wait.py --kinex /path/to/kinex \
  --out /tmp/libero-control-wait
```

It starts and disposes fresh services on free ports, calls Kinex `servo` and `world.wait`, and records motion outcomes without model calls
or a task-success claim. Existing services and workspaces are preserved.

The SDK is organized by responsibility:

```text
src/roboenv/
├── client.py, environment.py   Public connection and local lifecycle
├── devices/                   Device contracts, dispatch, and binding definitions
├── tasks/                     Task models, discovery, resolution, and artifacts
├── provisioning/              Isolated installation, sources, and assets
├── harbor/                    Deployment and task export
├── cli/                       Command parsing and dispatch
└── control/                   Optional client-side control helpers
```

`runtime/` contains simulator adapters, workers, and scoring. Runtime device
handlers use one native binding registry; robot variants reuse the same typed
implementations with embodiment-specific profiles.
`catalog/` exposes reusable robot definitions and small task examples. Native
binding defaults are internal resources in `src/roboenv/devices/bindings/defaults/`. `third_party/`
contains pinned submodules and recorded patches. Generated output and local
environments stay outside Git.

## Documentation

| Guide | Purpose |
|---|---|
| [Setup](docs/setup.md) | Installation, assets, local launch and device access |
| [Simulator integration](docs/simulator-integration.md) | Adapter, binding and portable catalog authoring |
| [RoboCasa](docs/robocasa.md) / [Isaac Lab](docs/isaaclab.md) | RoboCasa devices and Isaac Lab installation |
| [Simulator limit audit](docs/simulator-limits.md) | Native enforcement versus removed integration restrictions |
| [Local Kinex use](docs/kinex-local.md) | Workspace setup, recording and video playback |
| [Harbor tasks](docs/harbor-tasks.md) | Deployment, task generation and execution |
| [Kinex Harbor integration](integrations/kinex/README.md) | Agent packaging, controller development and frozen trials |
| [Simulator validation](docs/gpu-validation.md) | Native checks, receipts and replay boundaries |
| [Release engineering](docs/releasing.md) | Packaging, installation checks and publication gates |
| [Third-party notices](docs/third-party-notices.md) | Locked source notices and checksums |

This is pre-release software. CPU and format checks do not certify GPU replay
or benchmark success. Packages and simulator images are not automatically
published; see [release rules](docs/releasing.md).
