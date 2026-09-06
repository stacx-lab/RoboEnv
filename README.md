# Roboenv

Agent-independent robotics environments: simulator workers, device bindings,
public SDK/CLI, task generation, budgets, and benchmark verification.

Roboenv owns the world and its verdict. Agents own their reasoning and programs.
Training and agent-specific integration belong outside this repository.

## Start here

    bash tools/bootstrap.sh
    bash tools/check.sh
    .venv/bin/roboenv --help

The SDK needs Python 3.10+ and PyYAML. Development and Harbor tooling use Python
3.12 and the pinned development lock. Each simulator needs its own incompatible
dependency environment; the CPU checks do not install or start one.

## Repository layout

    src/roboenv/       lightweight SDK, device contract, task exporter
    runtime/          simulator adapters, bindings, worker, scoring
    catalog/          robot facts and eight small task examples
    third_party/      pinned upstream submodules and managed patches
    tools/            setup, checks, image builder, unified task generator
    tests/            self-contained contract and export tests
    docs/             integration guides and validation boundaries
    generated/        local task output; ignored by Git

The published repository provides a generator, not a checked-in expansion of
every benchmark task. Historical imported task collections remain local,
ignored migration data; neither installation nor generation requires them.

## Generate Harbor tasks

    .venv/bin/roboenv sources sync --backends libero
    .venv/bin/python tools/build_simulator_image.py libero \
      --dependencies PREPARED_DEPENDENCIES \
      --runtime-wheel build/wheels/roboenv_runtime-0.2.0-py3-none-any.whl \
      --out .cache/images/libero --tag roboenv-libero:local
    .venv/bin/python tools/generate_tasks.py \
      --backends libero \
      --deployments deployments/local/map.json \
      --wheel build/wheels/roboenv-0.2.0-py3-none-any.whl \
      --out generated/libero --check-images

PREPARED_DEPENDENCIES is an explicitly supplied local interpreter/asset installation,
not something fetched by the CPU bootstrap. The image builder takes source code
from the locked submodules here, and reads prepared interpreters/assets separately.
See [task generation](docs/harbor-tasks.md) for the deployment map, selection,
references, format guarantees, and execution.

No solution is included by default. Example reference programs are project-authored,
not official benchmark expert policies. Enabling references does not certify them.

## Integrate and operate

- [Installation, asset receipts, and reproducibility](docs/setup.md)
- [GPU validation and evidence boundaries](docs/gpu-validation.md)
- [Add a simulator](docs/adding-simulators.md)
- [Add a device binding](docs/adding-devices.md)
- [Change simulators or devices](docs/simulator-device-recipes.md)
- [Device service contract](docs/device-service.md)
- [Source dependencies](third_party/README.md)
- [Quality and release gates](docs/quality-audit.md)

The server starts before the agent. Agents discover devices with roboenv devices
and use the SDK or CLI; exported drivers are thin clients. Simulator control,
state, budgets and scoring stay in the environment container. The current
observation condition is benchmark ground truth; BEHAVIOR uses symbolic
primitives, not continuous motor control.

This is pre-release software. Source/format/CPU tests are not evidence that every
GPU task or embodiment succeeds. Public packages and simulator images have not
been released; see the audit for remaining gates.
