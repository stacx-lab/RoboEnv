# RoboEnv

[Getting Started](#getting-started) · [Simulators](#supported-simulators) ·
[Coding Agents](#using-roboenv-with-coding-agents) ·
[Development](#contributing--development) · [Documentation](docs/README.md)

RoboEnv provides robotics simulation environments for Kinex and other coding
agents, with shared device interfaces and native task verification.

## Getting Started

Use a source checkout on Linux with Git and uv installed:

```sh
git clone https://github.com/stacx-lab/RoboEnv.git
cd RoboEnv
bash tools/bootstrap.sh
source .venv/bin/activate
roboenv prepare robocasa
roboenv launch robocasa/pick-place-microwave --port 8123
```

Bootstrap creates the Python development environment. Preparation sets up the
selected simulator separately and downloads its assets; RoboCasa needs about
10 GB of kitchen assets. See [setup](docs/setup.md) for GPU prerequisites,
other simulators, and asset requirements. Keep launch running while you connect;
Ctrl-C stops the environment.

In another terminal, activate the same environment and connect from Python:

```python
import roboenv

with roboenv.connect("http://127.0.0.1:8123") as env:
    print(env.task.instruction)
    print(env.devices())
    print(env.device("arm").read())
```

The CLI exposes the same service:

```sh
roboenv tasks list --simulator robocasa
roboenv devices --endpoint http://127.0.0.1:8123
roboenv device arm read '{}' --endpoint http://127.0.0.1:8123
roboenv status --port 8123 --json
roboenv --help
```

Tasks default to realtime; use `--execution-mode stepped` at launch for explicit
simulation-time waits. See [device access](docs/device-documentation.md) for
control semantics and [spectator recording](docs/spectator.md) for `--record`.

## Supported Simulators

| Backend | Integration and documentation |
| --- | --- |
| RoboCasa | [Panda–Omron kitchen tasks](docs/robocasa.md) |
| LIBERO | [Panda tasks and scene catalogs](docs/setup.md#libero-tasks) |
| RoboTwin | [Bimanual task setup](docs/setup.md#prepare-a-simulator) |
| RoboDojo | [ARX X5 task setup](docs/setup.md#prepare-a-simulator) |
| BEHAVIOR | [Household tasks and asset setup](docs/setup.md#prepare-a-simulator) |
| Isaac Lab | [Setup](docs/isaaclab.md), [Go2–D1](docs/go2-d1.md), and [YAM calibration](docs/yam-calibration.md) |
| SIMPLE | [G1 tasks, control, and validation](docs/simple.md) |
| ABC | [Bimanual YAM tasks and scoring](docs/abc.md) |

Integration coverage, native task success, and replay validation are separate.
See the backend guides and [validation guide](docs/gpu-validation.md) for evidence
and current limits. Some device control interfaces are practice protocols.

## Using RoboEnv with Coding Agents

Launch an environment, then give the agent its public endpoint and instruction.
Any coding agent can use the Python SDK or CLI. Export a
[device reference](docs/device-documentation.md) for operation schemas, units,
frames, and camera conventions.

[Kinex](https://github.com/stacx-lab/kinex) can also use the supplied device
catalogs; follow [local Kinex setup](docs/kinex-local.md) to prepare a workspace.
RoboEnv owns simulation, device execution, budgets, and native verdicts.
The harness owns reasoning, model configuration, permissions, and controller programs.

## Contributing & Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md), then use the
[documentation index](docs/README.md) for architecture and integration contracts.
From the bootstrapped source checkout:

```sh
bash tools/check.sh
```

This runs the CPU suite, privacy audit, and wheel checks. Native simulator
validation is separate. See [release engineering](docs/releasing.md) for version
synchronization and packaging checks; routine development does not publish artifacts.

## Acknowledgements

Thanks to the maintainers of [RoboCasa](https://github.com/robocasa/robocasa),
[robosuite](https://github.com/ARISE-Initiative/robosuite),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin),
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo),
[BEHAVIOR](https://github.com/StanfordVL/BEHAVIOR-1K),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab),
[SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE), and
[ABC](https://github.com/amazon-far/abc) for the simulators, tasks, and robot assets
that make these integrations possible. See [third-party notices](docs/third-party-notices.md)
for source provenance and applicable terms.
