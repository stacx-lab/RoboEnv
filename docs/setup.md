# Setup and local use

Use Linux and uv. From a source checkout:

```sh
bash tools/bootstrap.sh
source .venv/bin/activate
```

This installs the CLI into the development virtual environment. Activate it in
each new shell, or use `.venv/bin/roboenv` directly.

## Prepare a simulator

```sh
roboenv prepare robotwin
```

This synchronizes RoboTwin sources, installs its isolated interpreter, and
reports setup checks. Substitute another simulator as needed.
Progress is printed to stderr as each stage starts and finishes, with elapsed-time
updates every 15 seconds during long stages. Installation output stays in the
private log whose path is printed before installation; use `tail -f LOG_PATH`
to follow its details. `--json` keeps stdout machine-readable.
No manual source sync, wheel build, clean-install verification, or separate
doctor command is required. Each simulator runs in its own interpreter; the SDK
does not import simulator dependencies. This may download many GB.

## Optional setup controls

```sh
roboenv doctor robotwin
roboenv prepare robotwin --prefix .venv-sim/robotwin-fresh
```

`prepare` defaults to `.venv-sim/robotwin`, checks dependencies and import origins, and
records the interpreter for launch. Repeating it reuses a registered interpreter
if its executable still exists; it does not upgrade or certify that environment
again. For RoboCasa, each preparation also verifies downloaded kitchen assets
and downloads missing or damaged bundles, including with a reused interpreter.
`doctor` provides read-only diagnostics. Existing environments are never
overwritten. Failed installations
retain their environment and private logs. Use `--prefix NEW_DIRECTORY`
for a fresh installation or recovery after failure.

For RoboCasa, `roboenv prepare robocasa` downloads the kitchen textures, fixtures,
and objects listed by the pinned upstream (about 10 GB) automatically. Progress
is reported on stderr; completed bundles have local content receipts and are
reused after verification. If a download fails, rerun the same command; the
interpreter and completed bundles are preserved. This also works for interpreters
prepared before automatic asset downloads were added. Asset verification does
not certify simulator boot or GPU replay.

Other simulators require separate asset acquisition using the pinned upstream's
instructions. Setup does not accept licenses or download model checkpoints. `doctor` reports recipe
and source blockers and whether an interpreter has been registered; it does not
certify GPU, asset, or task readiness. Add `--json` for automation.

For explicit wheels, mixed package inputs, dependency recipes, asset manifests,
and historical evidence, see [installation reference](installation-reference.md).

## Start a task

```sh
roboenv tasks list --simulator robotwin
roboenv launch TASK_ID
```

Choose a task ID from the list. With `.venv` activated, you can use `roboenv`
instead of `.venv/bin/roboenv`. Startup uses the registered interpreter and waits
for the live device registry.
It assigns a free loopback HTTP port and prints an endpoint and directory
containing profiles and thin drivers. Select a fixed port with `--port 8123`;
an occupied port fails startup instead of attaching to another environment. Keep
the command running; Ctrl-C closes the environment. Use `--detach` to leave it
running and `roboenv stop http://127.0.0.1:8123` to close it later from the same
project. `roboenv status http://127.0.0.1:8123` reports service health.

An existing prepared interpreter can be selected with `--interpreter ENV/bin/python`.
Custom tasks accept a task.yaml path and `--catalog CATALOG`. No simulator
installation or asset download occurs during launch. Local launch is intended
for development; Harbor provides the container boundary for agent evaluation.

## Connect and use devices

In another terminal with the development environment active:

```sh
roboenv devices
roboenv device arm read '{}'
roboenv devices --endpoint http://127.0.0.1:8123
```

With one ready local run, discovery selects it. With several, specify
`--endpoint`. `ROBOENV_ENDPOINT` also selects an HTTP service; `ROBOENV_SOCKET`
selects the existing Unix transport used by Harbor. HTTP binds to loopback only.
Remote networking and remote image-file transfer are not provided by this listener.

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
print(env.devices())
print(env.device("arm").read())
```

`roboenv.launch(task="TASK_ID")` returns an owner context manager. Launch
accepts `port=8123` or defaults to automatic port assignment. The endpoint is
available as `env.record["endpoint"]`. Concurrent runs use separate ports. After
a run ends, its port can be reused and a fresh client can connect at the same URL.
Context exit stops the
environment; `connect` only creates a client. Devices support
`read`, `write`, `stop`, `ik`, and `fk` only when their live descriptors advertise
them. Results retain `data`, budget, and execution evidence. Unsupported
kinematics or calibration are not synthesized.

Run definitions, device exports, and logs remain in the project's ignored cache.
Exported local drivers carry a connection file, so they work outside the source
directory with the SDK installed. These local connection files are not portable
Harbor task inputs.

## Export to Harbor

After preparing a native image deployment:

```sh
roboenv export libero-mug-into-microwave --out generated/example
```

The default deployment is `.cache/images/libero/deployment.json`; use
`--deployment FILE` to select another. The command builds a fresh SDK wheel
unless `--wheel FILE` is supplied, and validates the task with the real Harbor
loader. Local launch and export use the same task resolver and device binding
configuration. Image and asset preparation are described in
[Harbor tasks](harbor-tasks.md).

For development verification run `bash tools/check.sh`. CPU checks, container
boot, device conformance, benchmark success, and fresh-process replay are
separate results.
