# Design of RoboEnv

RoboEnv makes robotics simulators available as environments with discoverable
devices. It is an independent library that integrates with Complex and other
agent harnesses. Agents can reuse device-based programs across simulators when
the required capabilities are available.

This document describes the intended architecture. Backend capability and
validation coverage are documented separately; the design is not a claim that
every simulator already implements every device interface.

## Three user-facing concepts

A **simulator** creates a world and exposes its devices. A **task** combines a
simulation configuration with an objective and evaluation rules. A **running
environment** is an initialized task that clients can connect to.

Users prepare one simulator, select a task, and start an environment. Common
tasks select their simulator and robot defaults, so users need not assemble
deployment files or understand worker processes. The Python API and CLI expose
the same workflow, with explicit configuration available for advanced use.

## Devices and ownership

Each robot has a catalog of device profiles describing intrinsics, mounting
frames, units, and capabilities. Facts come from pinned simulator models;
scene-dependent calibration is resolved against the initialized world. Unknown
facts remain unknown. A device is only advertised as calibrated when the
required facts have been established.

Backend device implementations connect these profiles to real simulator
entities. Devices follow their type's interface, including observation,
actuation, kinematics, and lifecycle where supported. Missing capabilities fail
explicitly. Symbolic environments retain their own declared action abstraction.

One world owns simulation state, lifecycle, execution coordination, and budgets.
Device bindings share that world. Attaching or unloading a client does not reset
the episode or close another client's devices. RoboEnv owns benchmark verdicts
and execution evidence; clients cannot establish success by declaring it.

## Integration with Complex

Complex owns the agent, perception tools, controller programs, and skills.
RoboEnv provides calibrated profiles and thin device drivers compatible with
Complex's device interface. Drivers call environment services; simulator
implementation stays in RoboEnv. Integration installs these device exports into
the harness workspace without coupling the environment to a particular agent.

The SDK, CLI, and exported drivers reach the same device service. Devices are
the primary API. Simulator-specific primitives are explicit capabilities when
an environment requires a specialized action abstraction.

## Setup and execution

Each simulator uses pinned Git submodules with recorded, checksum-verified
patches, an isolated dependency environment, and explicitly acquired assets.
The workflow is bootstrap, prepare, then start. A shell bootstrap installs the
CLI virtual environment. `roboenv prepare` installs the selected simulator in
an automatically located isolated environment and reports actionable failures.
`roboenv start` launches a task using that prepared interpreter. Users do not
need to build wheels or run developer packaging checks. `doctor` is read-only;
`prepare` explicitly requests installation. Repeated preparation can reuse
registered interpreters without overwriting them. Assets and GPU validation
remain separate, explicit responsibilities.
Completed installations and diagnostics are preserved.

Local execution starts the simulator service and waits for device readiness.
Each running environment listens on a separate port, chosen explicitly or
assigned automatically. Clients connect to `http://HOST:PORT`; no run ID is
required. Multiple environments have independent worlds, budgets, and lifetimes.
Connecting to an existing environment and owning its lifetime are separate
operations. Heavy simulator dependencies are isolated from the lightweight
client library.

Harbor export uses the same resolved task and device configuration as local
execution. It packages reproducible environment artifacts and automatically
starts services before the agent. Host deployment choices and agent settings
remain separate from portable task definitions.

## Simplicity and correctness

Source packages follow domain responsibilities: devices, tasks, provisioning,
Harbor integration, command-line dispatch, and optional control helpers.
Connection and local lifecycle remain the public entry points. Domain code does
not depend on the CLI or import simulator implementations. Task definitions and
artifact handling are independent of Harbor; integration depends on them.
Runtime adapters and device handlers are grouped by simulator. Robot variants
share identical handler implementations while retaining distinct profiles and
binding identities. Package builds must contain current source files only.

Maintain one canonical configuration model, one device execution path, and one
owner of each piece of state. Generate derived configuration instead of keeping
parallel handwritten copies. Group backend-specific behavior together and remove
superseded implementations. Backward compatibility is not a constraint during
this development refactor.

The quickstart covers preparation, launch, device access, and Harbor export.
Developer contracts, packaging internals, and historical validation evidence
belong in reference documentation.

Refactoring preserves benchmark success rules, horizons, units, observation
restrictions, retry semantics, and final-state evidence. Setup, service boot,
device conformance, task success, and fresh-process replay are separate
validation results.
