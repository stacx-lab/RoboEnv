# Device service contract

New integrations follow [DESIGN.md](../DESIGN.md). RoboCasa implements this surface;
unmigrated backends may still have internal primitive bridges. Those bridges are
not additions to the new public device contract.

## Agent surface

The owner runs `roboenv launch TASK --port PORT`. After readiness, an agent connects:

```python
import os
import roboenv

env = roboenv.connect(os.environ["ROBOENV_ENDPOINT"])
print(env.task.instruction)
print(env.devices())
state = env.device("arm").read()
image = env.device("camera_left").read()
```

`env.task` contains immutable public task information. RoboCasa context is empty;
object poses, fixture geometry, scene layout, seed, and verdict are not observations.
`device.profile` describes its supported basic interface. Unknown devices/actions fail.
Device operations return useful data directly; journal and budget receipts remain
internal. The server uses the original benchmark budget and success rules.

Use actual per-type interfaces: sensors read, actuators also write/stop, manipulators
also provide IK/FK, and world has no driver. See [device authoring](adding-devices.md)
and [RoboCasa action semantics](robocasa.md). A device does not expose arbitrary
simulator calls, task macros, or model-policy execution.

## Static configuration and transport

`catalog/robots/ROBOT/` contains only `workspace.yaml` and `devices/`. Profiles and
thin drivers are pre-generated from stable hardware facts and copied directly to
Complex. `catalog/bindings/ROBOT.yaml` is owner-side configuration. Launch reads
these files without modifying the catalog. Endpoints are selected at deployment,
not by rewriting robot configuration for every task.

The wire protocol is `roboenv/1`, with `roboenv/devices/1` device semantics. Requests
carry session, generation, request ID, deadline, op, and object arguments. HTTP and
owner Unix transport share the same world, execution lane, and durable motion results.
HTTP GET `/` discovers episode identity; it does not initialize a simulator.

| HTTP operation | Purpose |
|---|---|
| `environment.describe` | Permitted public task and device information |
| `devices.list`, `devices.describe` | Device discovery |
| `devices.call` | One declared device action |
| `operations.get` | Status/result of an allowed public device/discovery request |
| `health` | Sanitized readiness and liveness |

Owner observe, score, reset, finalization, and private history remain outside HTTP.
An operation-status request must not unwrap an owner result, including a cached
result. HTTP health excludes scene seeds and private worker diagnostics. New backends
must also implement safe task-description projection; a generic transport alone
does not sanitize an adapter's payloads.

Use the SDK for request construction and errors. `roboenv.client.Client` remains
the lower-level transport for CLI/admin code; it is not the recommended agent API.
The CLI can address the same devices:

```sh
roboenv devices --endpoint http://127.0.0.1:8123
roboenv device arm read '{}' --endpoint http://127.0.0.1:8123
```

## Effects, retries, and lifecycle

Inputs are validated before effects. An accepted write executes the declared bounded
native interval, with explicit neutral/hold behavior for other parts. Reads, status,
profile access, recording, and IK/FK do not advance physics. Device close or unload
never closes the world. Stop semantics must be truthful; RoboCasa's in-flight
synchronous interval completes before stop resets the servo goal.

Established SDK clients retain episode identity. Reusing a port for another task
cannot redirect their commands. New one-shot drivers discover the selected service;
runner-supplied identity can pin attachment when necessary.

A timeout may have occurred after motion. Uncertain results carry the request ID;
query its status rather than issuing the motion with a fresh ID. Public status reads
remain available during motion and do not gain access to verifier requests. Do not
hide lost outcomes with automatic motion retries.

## Image artifacts and observation isolation

Camera reads return a client-accessible file and image metadata, not an inaccessible
simulator-local filename. HTTP artifact downloads are episode-scoped. SDK-owned
image files last until explicit client cleanup; one-shot drivers leave returned
images available for the caller/Complex recorder. Reads work before the first write.

Only sensor artifacts are agent-readable. Privileged journals, snapshots, source
assets, and private task definitions are not observation channels. Owner socket
permissions and evidence directory permissions must be enforced in deployment;
read-only mounts still expose their contents. A local agent sharing the simulator's
Unix identity is not isolated from that user's files.

See [simulator integration](adding-simulators.md) for boundary tests,
[Harbor tasks](harbor-tasks.md) for deployment, and
[Complex recording](complex-local.md) for the local functional test plan.
