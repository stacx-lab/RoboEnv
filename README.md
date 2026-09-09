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
`roboenv.connect()` also discovers a single ready local run. Use `--endpoint URL`
when several environments are running. `launch` owns the environment until Ctrl-C;
`--detach` leaves it running for `roboenv stop http://127.0.0.1:8123`.
Python owners use `with roboenv.launch(task="TASK_ID", port=8123) as run:`
and connect to `run.endpoint`. Use
`--host 0.0.0.0` when serving a task across an isolated container network.

Launch streams simulator stdout/stderr to the terminal's stderr and retains the
run logs. `--json` keeps stdout machine-readable; `--detach` streams until ready.
Python callers can enable streaming with `log_stream=sys.stderr`.

Official LIBERO uses the same typed device path with five task-independent Panda
rig catalogs. Run `roboenv prepare libero` and `roboenv launch libero/spatial_0`.
Its absolute joint control is explicitly a practice protocol, not the published OSC
benchmark action space. `tools/accept_libero.py --out OUTPUT` runs native device,
transport, Kinex recording and fresh-process replay gates without model calls;
task-solving acceptance uses Kinex's `examples/simulator-acceptance/libero.json`.

RoboCasa’s typed catalog contains arm, torso, base, gripper, three RGB-D cameras,
and world, using Kinex’s current device schemas. It implements the device-first API: `env.task`, `env.devices()`, and
`env.device(name)`. Drivers expose only their type's basic operations. The world advances
continuously while the agent thinks. Writes update persistent controller goals;
reads do not request extra steps. Exported arm/torso drivers solve IK/FK locally
using Kinex utilities and static profile geometry; they do not require RoboEnv
in the agent environment. Pass `--sync` to `roboenv launch`, or set `backend_kwargs.realtime: false`
in the task YAML for synchronous mode, where each write advances one control period, retaining all active goals.
Arm and torso targets are absolute joint positions in radians/metres; gripper
commands keep their native signed format. Acceptance never implies arrival.
Neither mode imposes a simulation horizon.
Task-solving controllers belong in the agent harness. See [RoboCasa](docs/robocasa.md)
for action spaces, task-specific device selection, and validation coverage.

Isaac Lab is experimental and is not officially supported by RoboEnv. Its
`isaaclab/go2-posture` prototype provides native Go2 joint control with RGB-D,
IMU, and foot contacts, using synchronous stepping. See
[Isaac Lab setup and devices](docs/isaaclab.md); [G1 follow-up](plan.md) is deferred.

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
second fake server. Historical recipe checks are consolidated in
`tests/test_recipe_evidence.py`; test helpers live outside test modules so tests
do not import one another. The default check still runs the complete CPU suite.

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
handlers are grouped by simulator under `bindings/backends/`; robot variants
reuse shared implementations through an explicit binding index.
`catalog/` holds robot facts and small task examples. `third_party/`
contains pinned submodules and recorded patches. Generated output and local
environments stay outside Git.

See [simulator integration](docs/simulator-integration.md) for adapter, service,
and device catalog authoring, and [RoboCasa](docs/robocasa.md) for the reference rig.
[Installation evidence](docs/installation-reference.md) and
[GPU validation](docs/gpu-validation.md) describe tested coverage.

This is pre-release software. CPU and format checks do not certify GPU replay
or benchmark success. Packages and simulator images are not automatically
published; see [release rules](docs/releasing.md).

For a populated Kinex workspace, TUI recording, and MP4 playback, follow
[the local test plan](docs/kinex-local.md).
