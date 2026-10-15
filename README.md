# RoboEnv

RoboEnv runs robotics simulators behind a shared device API. It owns simulation,
execution, budgets, and native task verdicts; your agent supplies the reasoning
and programs.

## Quick start

Use Linux, Git, uv, and a compatible NVIDIA GPU for simulator runs:

```sh
git clone https://github.com/stacx-lab/RoboEnv.git
cd RoboEnv
bash tools/bootstrap.sh
source .venv/bin/activate
roboenv prepare robotwin
roboenv launch robotwin/handover-block-aloha-agilex --port 8123
```

Preparation downloads the selected simulator's dependencies and assets into an
isolated environment. Keep launch running; Ctrl-C closes it. Tasks run in realtime
by default. See [setup](docs/setup.md) for installation options and GPU requirements.

In another terminal, activate `.venv` and connect:

```python
import roboenv

with roboenv.connect("http://127.0.0.1:8123") as env:
    print(env.task.instruction)
    print(env.devices())
    print(env.device("left_arm").read())
```

The CLI accesses the same devices:

```sh
roboenv tasks list --simulator robotwin
roboenv robots list --simulator robotwin
roboenv devices --endpoint http://127.0.0.1:8123
roboenv device left_arm read '{}' --endpoint http://127.0.0.1:8123
```

See [device access](docs/device-documentation.md) for control and camera APIs.
Add `--record` to launch for [spectator videos](docs/spectator.md).

## Simulators

| Simulator | Guide |
| --- | --- |
| RoboTwin | [Aloha AgileX setup](docs/setup.md) |
| RoboCasa | [Panda–Omron kitchen tasks](docs/robocasa.md) |
| LIBERO | [Panda tasks and scene catalogs](docs/setup.md#libero-tasks) |
| RoboDojo | [ARX X5 setup](docs/setup.md#prepare-a-simulator) |
| BEHAVIOR | [Household task setup](docs/setup.md#prepare-a-simulator) |
| Isaac Lab | [Setup](docs/isaaclab.md), [Go2–D1](docs/go2-d1.md), [YAM calibration](docs/yam-calibration.md) |
| SIMPLE | [G1 tasks](docs/simple.md) |
| ABC | [Bimanual YAM tasks](docs/abc.md) |

Native success and replay coverage vary by backend; see [validation](docs/gpu-validation.md).

## Coding agents

Give your agent the public endpoint and task instruction. It can use the Python
SDK, CLI, or an exported [device reference](docs/device-documentation.md).

For [Kinex](https://github.com/stacx-lab/kinex), keep the task running and open a
second terminal. `launch` uses a task ID; `setup` uses its paired robot ID. The
handover task above selects `robotwin/robotwin-aloha-agilex` in its task YAML's
`robot:` field. Export that robot's devices and connect Kinex:

```sh
source /path/to/RoboEnv/.venv/bin/activate
mkdir -p kinex-sim-workspace
cd kinex-sim-workspace
roboenv setup robotwin/robotwin-aloha-agilex
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
roboenv describe --endpoint "$ROBOENV_ENDPOINT"
kinex
```

Replace `/path/to/RoboEnv` with your checkout. Setup exports the robot into
`.kinex/` and protects existing configuration. In Kinex, run `/load all` and
`/ping all`, then supply the task instruction from `describe`.
See [task/robot pairing](docs/setup.md#choose-a-paired-task-and-robot) to select
another pair and [local Kinex use](docs/kinex-local.md) for connection and recording.

## Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md), then run:

```sh
bash tools/check.sh
```

This checks CPU behavior, privacy, and wheels. GPU validation is separate.
The [documentation index](docs/README.md) links architecture and release contracts.
See [third-party notices](docs/third-party-notices.md) for upstream acknowledgements
and applicable terms.
