# Roboenv device service v1

The generic service lives in `src/roboenv/devices/service.py`; backend bindings implement
device actions without loading executable catalog drivers. For extension points and assembly
configuration, see [simulators](adding-simulators.md) and [devices](adding-devices.md).

Status: implemented on the native Harbor path. The wire envelope remains
`roboenv/1`; this additive capability is `roboenv/devices/1`. CPU conformance is
tested with actual supervisor sockets and fresh worker processes. This is not a
claim of GPU replay or real-hardware safety. Current evidence and remaining gates
are recorded in the [quality audit](quality-audit.md). Earlier integration tests
covered Codex, Claude Code and original Complex on one synthetic CPU task;
those do not certify every simulator.

## The contract an agent receives

A Harbor task starts one environment service and waits for both its worker and
device registry to be ready. The task provides its instructions, a read-only
environment description and client device profiles. Any shell-capable agent can
start work using the CLI or Python SDK. No agent package, credentials, simulator
installation, or agent-authored environment driver is required by Roboenv.

```sh
roboenv describe
roboenv devices
roboenv device arm read '{}'
roboenv device arm write '{"target":[0.3,0.1,0.2]}' --request-id approach-001
roboenv operation approach-001
```

The arm example is conditional on that device advertising `write`. A symbolic
task may expose only a controller and named primitives. Agents must inspect the
actual capability description, control mode, units and bounds before commanding.

```python
from roboenv.client import Client, RemoteError, UncertainResult

env = Client()  # ROBOENV_SOCKET, ROBOENV_SESSION, ROBOENV_GENERATION
print(env.describe())
print(env.devices())
arm = env.device("arm")
print(arm.describe())
observation = arm.read()
try:
    reply = arm.call("write", {"target": [0.3, 0.1, 0.2]}, request_id="approach-001")
except UncertainResult as error:
    print(env.operation(error.request_id))  # inspect; never blindly issue a new motion
except RemoteError as error:
    print(error.code, error.details)
```

Complex/Robodex receive the same service through the existing device-driver/1
interface. The exported `driver.py` is a one-shot, thin SDK client. Roborl binds
the environment profiles into the agent's workspace; Roboenv does not depend on
an agent or a Harbor Python library for device access. The optional task generator
uses Harbor's real loader for format validation. Claude Code/Codex-style shell agents need no
device-framework integration at all. Installing/running an agent is still the
responsibility of its Harbor integration, not the environment image.

## Operations

All requests retain protocol, session, generation, request_id, deadline_s, op,
and object args. Messages are one NDJSON object per connection, at most 8 MiB.
Harbor uses the local Unix socket. Local `roboenv start` also exposes a loopback
HTTP endpoint at `http://127.0.0.1:PORT`. The SDK discovers internal session and
generation metadata with a GET to `/`; users need only the endpoint. Established
clients remain fenced to their original world if the port is reused.
Both transports use the same
request envelope, execution lane, and journal; a request ID deduplicates across
transports. HTTP accepts JSON POST requests and public client operations only.
Owner shutdown/reset/submit operations remain on the Unix transport. There is
no WebSocket or MCP transport.

| Operation | Exact args | Result |
|---|---|---|
| `environment.describe` | `{}` | Task facts, identity, device contract and execution model |
| `devices.list` | `{}` | `{schema_version, devices: [descriptor, ...]}` |
| `devices.describe` | `{device}` | One descriptor, including supported/unsupported actions |
| `devices.call` | `{device, action, args: {…}}` | A synchronous device result with data and execution evidence |
| `operations.get` | `{request_id}` | Current state of a journaled request in this session/generation |

`api`, `call`, `observe`, `score.peek`, `capture` and `vla` remain available.
Native primitive parameters are unchanged. Device calls always use the service;
there is no alternate local driver execution path. Read exported profile files
directly when inspecting an offline definition.

The [JSON Schema](../src/roboenv/specs/device-service.schema.json) describes
discovery, descriptors and successful call results. New optional fields may be
added within v1; clients should ignore unknown response fields. Changes to action
meaning, units, frames or required fields require a new contract/descriptor, not
silent reinterpretation.

## Device facts and capability honesty

Descriptors include `name`, `type`, `intrinsics`, `extrinsics`, `control_mode`,
`actions`, and `execution`. Device parameters belong to the environment. Unknown
catalog calibration remains explicitly unknown; no missing joints or transforms
are synthesized. Current migrated tasks expose benchmark ground truth, not a
vision-only evaluation condition.

An action has `supported`, `read_only`, and `input_schema`; unsupported actions
also have a reason. The compatibility backend currently has object-valued inputs
whose detailed native primitive signatures are discoverable with `api`. It does
not yet provide fully typed per-argument schemas for every migrated action.
`supported` means an implementation is registered, not that this hardware has
passed a calibration or replay test. `validation` makes that distinction explicit.

For current simulator-controller bindings, `stop`, general `ik`/`fk`, and `unload` are unsupported.
In particular, killing a simulator worker on timeout is NOT an emergency stop
for physical hardware. `load` is a readiness check, not a world reset. `policy_run`
is not a device-server action: use `Client.vla` so policy execution retains its
separate journal and verifier accounting.

`read`, `state`, `status`, `load`, `catalog`, `api`, and `ping` cannot dispatch
mutating primitive calls. All other actions are conservatively treated as
mutations, including `invoke` even when it invokes an observation primitive.
The implementation enforces this at the worker bridge and freezes mutating
device actions after submit. Simulator-native tool poses are not labeled as
calibrated arm-base-to-flange poses; legacy `state` is normalized like `read`.

## Execution and uncertainty

One worker owns the world. Device actions and ordinary primitive calls share the
same serialized execution lane, so concurrent requests cannot independently step
the same simulator. There is no async motion-acceptance API in v1: `devices.call`
returns after the action finishes. `operations.get` and supervisor health do not
wait for that lane. Discovery/read calls can wait behind an action.

A successful device response includes:

- `data`: backend-specific observations/results; completion is not task success.
- `sequence`: worker-local device-call attempt number, not a simulation timestep.
- `started_unix_ns` / `finished_unix_ns`: wall-clock timestamps, not hardware sync.
- `budget`: the environment's post-action budget snapshot.
- `calls`: executed underlying public primitive requests and success flags.

The supervisor journals the complete device intent and terminal response. Its
component calls run within that atomic device operation, not as separate network
requests. A crash before a terminal response leaves the entire operation
indeterminate; the system does not reconstruct or replay a partially executed
macro. The verifier rejects failed mutations with unknown effects.

For journaled operations, deduplication keys are `(session, generation,
request_id)`. Retrying the same semantics returns the original result even after
memory-cache eviction. Reusing the id with different semantics is a conflict.
An intent without a durable terminal event is fenced on recovery. There is no
claim of general exactly-once physical execution or automatic world recovery.
A newly initialized world requires a fresh run/session or generation.

Operation states are `unknown`, `running`, `succeeded`, `failed`, or
`indeterminate`. `unknown` is NOT proof that no request can execute: it may not
have reached dispatch yet. `succeeded` means the requested operation completed,
not that the benchmark was solved. Query again with a fresh query request id;
reuse only the target command id. Status history is retained in the run journal.

The SDK retries connecting only before sending. Once sending starts it does not
automatically resend. An `UncertainResult` carries the original request id, even
when the SDK generated it. A concurrent retry that times out waiting for its
original receives `request_in_progress`; it does not overwrite the original
result or generate a second journal intent. A dispatched worker timeout fences
the command and terminates that simulator process group.

Stable rejection codes include `invalid_args`, `unknown_device`,
`unsupported_action`, `permission_denied`, `already_submitted`, and
`request_id_conflict`. `device_failed` may describe partial effects;
`error.details.effects` distinguishes `none`, `read_only`, and `unknown` when
known. Transport/worker failures must be treated conservatively.

## Deployment and ownership

The native task mounts the server catalog at `/opt/serve/robot` and the SDK wheel
read-only at `/opt/roboenv/sdk`. The worker loads the SDK directly from the wheel;
there is no startup download or dependency installation. Rebuild the runtime
image and SDK wheel together before exporting upgraded tasks. A stale image
does not pass the new device-registry readiness probe.

`ROBOENV_SERVER_ROBOT` and `ROBOENV_SERVER_MANIFEST` are owner-only startup
configuration, never request arguments. Bindings are registered environment
implementation code. Task catalogs provide facts and explicit binding selectors;
they cannot name arbitrary Python imports or executable drivers. New backends implement real device
semantics server-side rather than growing agent-side translation logic.

Images are currently returned as paths in the shared read-only run volume.
Remote artifact download, synchronized telemetry, native FK/IK, interruptible
motion and hardware-specific safety interlocks remain separate work.

The deployment is task-local and assumes trusted environment ownership. This
version does not add authenticated multi-tenant networking or a separate
authenticated owner control plane. Historical owner wire operations remain;
hiding them in the public SDK is not an authorization boundary. The independent
verifier still detects non-verifier submission and prohibited episode changes.
Do not expose this Unix service as an unauthenticated remote robot endpoint.

## Validation

From the three-project workspace:

```sh
.venv/bin/python -m pytest -c pyproject.toml --confcutdir=. tests/test_device_service.py
bash tools/check.sh
```

Tests exercise real sockets and independent CPU workers, ordinary CLI execution,
thin drivers, persistence/deduplication, in-flight status, unsupported actions,
read-only enforcement, post-submit freezing, and independent verifier evidence.
Each simulator/embodiment and each named agent still needs its own end-to-end
validation before being advertised as supported in production.
