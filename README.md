# RoboEnv

Robotics environments with discoverable devices, shared simulation state, and
benchmark verification. Use them from Python, the CLI, Complex, or Harbor.
RoboEnv owns the environment and verdict; the agent harness owns reasoning and
controller programs. See [DESIGN.md](DESIGN.md) for the architecture.

## Prepare a simulator

From this checkout on Linux with uv installed:

```sh
bash tools/bootstrap.sh
source .venv/bin/activate
roboenv prepare robotwin
```

Bootstrap installs the CLI into `.venv`. Then `prepare` synchronizes selected
pinned sources, installs an isolated simulator environment, and reports checks.
Replace `robotwin` with your simulator. No wheel-building, `--install`, or
environment-path arguments are needed. Acquire required assets separately;
see [setup](docs/setup.md) for prerequisites and diagnostics.

## Start and connect

```sh
roboenv tasks list --simulator robotwin
roboenv start TASK_ID --port 8123
```

In another terminal:

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
print(env.devices())
print(env.device("arm").read())
```

The CLI exposes the same devices: `roboenv device arm read '{}' --endpoint URL`.
Omit `--port` to assign a free port automatically. Concurrent runs use different ports;
`roboenv.connect()` also discovers a single ready local run. Use `--endpoint URL`
when several environments are running. `start` owns the environment until Ctrl-C;
`--detach` leaves it running for `roboenv stop http://127.0.0.1:8123`.
Python owners use `with roboenv.launch(task="TASK_ID") as env:`.

Inspect advertised capabilities before commanding. Current backends include
Cartesian and symbolic interfaces; full calibration and general IK/FK are not
available on every device.

## Export a Harbor task

With a [prepared native deployment](docs/harbor-tasks.md):

```sh
roboenv export libero-mug-into-microwave --out generated/example
```

Export resolves the same task configuration as local startup, builds the SDK
wheel, and validates the Harbor format. The environment starts automatically
before the agent. Reference solutions are not included by default. Agent
installation and model configuration belong to the Harbor integration.

## Develop

```sh
bash tools/check.sh
```

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

See [simulator integration](docs/adding-simulators.md),
[device integration](docs/adding-devices.md), and the
[device service contract](docs/device-service.md) for developer interfaces.
[Installation evidence](docs/installation-reference.md) and
[GPU validation](docs/gpu-validation.md) describe tested coverage.

This is pre-release software. CPU and format checks do not certify GPU replay
or benchmark success. Packages and simulator images are not automatically
published; see [release rules](docs/releasing.md).
