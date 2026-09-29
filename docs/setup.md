# Setup and local use

Use Linux and uv. From a source checkout:

```sh
bash tools/bootstrap.sh
source .venv/bin/activate
```

This installs the CLI into a Python 3.12 development environment. Activate it in
each new shell, or use `.venv/bin/roboenv` directly. Simulator dependencies use
separate interpreters defined in [runtime/environments.json](../runtime/environments.json).

## Prepare a simulator

```sh
roboenv prepare robotwin
```

This synchronizes pinned RoboTwin sources, installs its isolated interpreter,
downloads and verifies its public embodiment/object assets, and reports setup
checks. RoboTwin and RoboDojo run from their source checkouts directly.
Substitute another simulator as needed.
Progress is printed to stderr as each stage starts and finishes, with elapsed-time
updates every 15 seconds during long stages. Installation output stays in the
private log whose path is printed before installation; use `tail -f LOG_PATH`
to follow its details. `--json` keeps stdout machine-readable.
The SDK does not import simulator dependencies. Preparation may download many GB.

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
interpreter and completed bundles are preserved. Asset verification does
not certify simulator boot or GPU replay.

RoboTwin preparation verifies the pinned upstream archives and preserves existing
asset directories. RoboDojo preparation downloads its pinned public `Assets` tree
(about 41 GB) and verifies every file against its source hash. Missing
files can be resumed; conflicting existing files produce an error without being
overwritten. Set `ROBOENV_ASSET_CACHE=/path/to/data-depot` before preparation to
place these assets on a larger filesystem. Source checkout links point into that
depot, so it must remain available for launch.

RoboDojo requires Isaac Sim and supports the upstream ARX X5 robot.

BEHAVIOR preparation acquires the pinned robot assets, encrypted scene/object
bundle and 2026 challenge instances after explicit acceptance is supplied through
`OMNI_KIT_ACCEPT_EULA=YES` and `ROBOENV_BEHAVIOR_ACCEPT_LICENSE=YES`. Set these only
after reviewing and accepting the respective upstream agreements. The upstream
OmniGibson key installer runs within its own interpreter; keys and native data
remain in the private depot. Existing bundles are verified and preserved. Source
availability alone does not grant access to these assets. Setup does not accept
licenses or download model checkpoints. `doctor` reports recipe
and source blockers and whether an interpreter has been registered; it does not
certify GPU, asset, or task readiness. Add `--json` for automation.

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
running and `roboenv stop RUN_ID --project PROJECT` to close that local run later.
For HTTP clients, select the local service with `roboenv status --port 8123 --json`
or `roboenv stop --port 8123 --json`. The port selects `127.0.0.1`; use
`--endpoint http://HOST:PORT` for remote or container services. HTTP stop requires
`--allow-stop` at launch. Explicit port and endpoint selections override
`ROBOENV_ENDPOINT`, including when other simulations are listening on different
ports. Combining a run reference, `--port`, or `--endpoint` is rejected.
Without a selected run or endpoint,
status rejects ambiguous discovery and stop requires an explicit selection.

An existing prepared interpreter can be selected with `--interpreter ENV/bin/python`.
Custom tasks accept a task.yaml path and `--catalog CATALOG`. Task files default to
`execution_mode: realtime`; `execution_mode: stepped` opts into stepped runs.
`--execution-mode` overrides
that setting for a local launch. Realtime disables configured and native episode
step caps and reports `budget.max: null`; stepped retains the caps. Non-timeout
termination conditions still apply. No simulator
installation or asset download occurs during launch. Local launch is intended
for development; Harbor provides the container boundary for agent evaluation.

## Connect and use devices

In another terminal with the development environment active:

```sh
roboenv devices
roboenv device left_arm read '{}'
roboenv devices --endpoint http://127.0.0.1:8123
```

With one ready local run, discovery selects it. With several, specify
`--endpoint`. `ROBOENV_ENDPOINT` also selects an HTTP service; `ROBOENV_SOCKET`
selects the owner-side Unix transport. HTTP binds to loopback by default; use
`--host 0.0.0.0` for an isolated container network. Camera files are downloaded
through episode-scoped HTTP artifact routes.

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
print(env.devices())
print(env.device("left_arm").read())
```

Python owners use `with roboenv.launch(task="TASK_ID", port=8123) as run:` and
connect to `run.endpoint`. Context exit stops the world; closing an attached
client only releases client resources. Devices expose the operations declared by
their live descriptors and return device data directly. Budgets and execution
receipts remain in owner evidence. Exported joint drivers provide local
FK/IK; the service does not synthesize unsupported operations.

Run definitions, device exports, and logs remain in the project's ignored cache.
Exported local drivers carry a connection file. Copy complete device directories
into the agent workspace; their dependencies are described in
[local Kinex use](kinex-local.md). These episode-specific connection files are
not portable Harbor task inputs.

## Export to Harbor

After preparing a native image deployment:

```sh
roboenv export libero/mug-into-microwave --out generated/example
```

The default deployment is `.cache/images/libero/deployment.json`; use
`--deployment FILE` to select another. The command builds a fresh SDK wheel
unless `--wheel FILE` is supplied, and validates the task with the real Harbor
loader. Local launch and export use the same task resolver and device binding
configuration. Image and asset preparation are described in
[Harbor tasks](harbor-tasks.md).

RoboTwin and RoboDojo tasks expose typed devices and do not ship primitive
reference solutions.
See [simulator validation](gpu-validation.md) for native checks and
[Kinex Harbor integration](../integrations/kinex/README.md) for controller
development and frozen trials.

## Advanced installation and assets

Sources and managed patches are pinned in [upstreams.lock.json](../upstreams.lock.json).
`roboenv sources status --backends BACKEND` checks their exact revisions and tracked
contents without changing the working tree. Recipes and dependency locks are in
`runtime/environments.json` and `runtime/locks/`.

For explicit package inputs, use the low-level installer after source sync:

```sh
roboenv sources sync --backends libero
python -m roboenv.provisioning.setup libero
python -m roboenv.provisioning.setup libero \
  --install --prefix .venv-sim/libero-wheels --wheels build/wheels
```

Without `--install` this only inspects the recipe. Installation requires a new
prefix, uses normal dependency resolution, verifies dependencies and import origins,
and registers the interpreter for launch. It does not acquire assets; use `prepare`
for that step. With no package arguments it uses this checkout's SDK and runtime.
`--sdk PATH --runtime PATH` selects source directories or individual wheels and
cannot be combined with `--wheels`. `--project` selects another checkout;
`ROBOENV_ROOT` identifies the source root when running its simulator elsewhere.
See [release checks](releasing.md) for independent wheel installation.

To inventory independently acquired data-only assets:

```sh
python -m roboenv.provisioning.assets create "$ASSET_DEPOT" \
  --asset-id example-assets --source-url "$ASSET_SOURCE_URL" \
  --source-revision "$ASSET_VERSION" --license-url "$ASSET_LICENSE_URL" \
  --out deployments/local/example-assets.json
python -m roboenv.provisioning.assets verify "$ASSET_DEPOT" \
  deployments/local/example-assets.json
```

Use actual upstream provenance and a new receipt outside the depot. Verification
reads all contents and checks relative paths, sizes, hashes and internal symlinks;
keep the depot quiescent. Content verification does not establish redistribution
rights or native execution. Image interpreter/depot options are described in
[Harbor tasks](harbor-tasks.md).

## Local storage

Keep `.cache/roboenv/runs/` logs and recordings, installation/asset receipts,
simulator assets and failed setup diagnostics until reviewed or archived. `.cache/`
is not wholly disposable. Interpreters default to `.venv-sim/`; package downloads
use uv's shared cache or `UV_CACHE_DIR`. Asset depots selected with
`ROBOENV_ASSET_CACHE` must remain accessible through their source-checkout links.

## Reusing RoboCasa assets when packaging

For orchestrators that install native Python inside an image, acquire assets
separately in a persistent RoboEnv project:

```python
from roboenv.provisioning.sources import sync
from roboenv.provisioning.robocasa_assets import prepare_assets

sync(project, backends=("robocasa",))
prepare_assets(project, reuse_from=previous_project, progress=print)
```

`reuse_from` is optional. Existing destination bundles are checked first. A donor
bundle is reused only when its URL matches the pinned source and every file
matches its receipt hash. Files are copied independently and a receipt is
published after copying completes; donor files and interpreters stay untouched.
Only missing or damaged bundles without a verified donor are downloaded.

Package `third_party/robocasa/robocasa/models/assets` together with the JSON
receipts in `.cache/roboenv/assets/robocasa`, preserving those project-relative
paths. Synchronize image-local sources before copying assets into their checkout.
Subsequent `roboenv prepare robocasa` verifies these files and skips their downloads.
Do not include private diagnostic logs with the receipts.
