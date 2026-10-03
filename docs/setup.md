# Setup and local use

Use Linux and uv. From a source checkout:

```sh
bash tools/bootstrap.sh
source .venv/bin/activate
```

This installs the CLI into a Python 3.12 development environment. Activate it in
each new shell, or use `.venv/bin/roboenv` directly. Simulator dependencies use
separate interpreters defined in [runtime/environments.json](../runtime/environments.json).
Isaac rendering also needs host graphics libraries outside those Python
environments, including `libGLU.so.1` (`libglu1-mesa` on Ubuntu). Python dependency
and import checks do not certify native renderer startup. Use a supported RTX GPU;
NVIDIA lists GPUs without RT cores, including A100 and H100, as unsupported in its
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

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
records the interpreter for launch. Every preparation discards local edits to the
selected third-party sources, restores their locked upstream commits, and reapplies
managed patches. Keep intended source changes in checksum-locked patch files;
downloaded assets, environments, caches, recordings and nested checkouts are preserved.
Repeating preparation reuses a registered interpreter if its executable still
exists; refreshing source does not reinstall dependencies or certify that environment
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
for development; external orchestrators provide container boundaries for evaluation.

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
not reusable across episodes without refreshing their identity.

## Discover tasks and connect an agent

```sh
roboenv tasks discover --backends libero --out generated/libero-source
roboenv tasks list --source generated/libero-source/tasks
roboenv launch libero/mug-into-microwave --port 8123
```

Task discovery reads pinned definitions without importing simulator engines.
Generated catalogs retain source revisions, hashes, patches and exclusions, and
stay outside Git. Agents connect through the SDK or CLI; use the
[public device reference](device-documentation.md) to supply their documentation.
See [simulator validation](gpu-validation.md) for native checks and
[local Kinex use](kinex-local.md) for optional Kinex device setup.

## Standalone simulator images

Build SDK/runtime wheels with `bash tools/check.sh`, prepare the selected simulator,
then use `tools/build_simulator_image.py --help`. The builder requires a backend,
`--dependencies`, `--runtime-wheel`, `--out`, and `--tag`. It uses the registered
interpreter unless `--interpreter ENV/bin/python` is supplied.

`--interpreter-mode image` embeds a relocated interpreter; `depot` creates a
separate read-only, content-verified copy. `--asset-mode` selects reference or copy
staging for assets. Never mount the original project or interpreter. Staging and
Docker storage both need room; `--stage-only` does not produce a runnable image.

The output `deployment.json` uses `roboenv/deployment/1`. External orchestrators
can validate it with `roboenv.provisioning.deployment.load_deployment(path, backend)`.
It requires an immutable image digest, independent read-only depots, and protected
episode settings. Local paths stay private. Runtime options describe shared memory
and readiness grace periods; the external container launcher applies those options.

Use the generated image with its default entrypoint, mounting each declared depot
read-only at its declared target. The entrypoint verifies executable depots using
image-owned `/usr/bin/python3 -I` before executing any simulator interpreter.
Do not bypass it. Mount the matching SDK wheel and owner task/catalog files
read-only, outside the agent container. A launch command inside the image is:

```sh
/opt/sim-env/bin/python -m roboenv.cli.main launch /opt/task/task.yaml \
  --catalog /opt/task/catalog --runtime-root /opt/roboenv/runtime \
  --interpreter /opt/sim-env/bin/python --run-dir /run/roboenv/run \
  --host 0.0.0.0 --port 8123 \
  --python-path /opt/roboenv-runtime --python-path /opt/roboenv-contract-deps \
  --python-path /opt/sdk/roboenv.whl
```

For depot images, set `PYTHONPATH` to include `/opt/sdk/roboenv.whl` alongside
`/opt/roboenv-runtime` and `/opt/roboenv-contract-deps` before invoking that command.
For embedded-interpreter images, omit `/opt/roboenv-runtime` from both `PYTHONPATH`
and `--python-path` (the runtime wheel is installed in the interpreter), and include
the SDK wheel and contract dependencies in both.
Use the actual mounted SDK wheel filename. The owner chooses GPU allocation,
network/port mapping, readiness deadlines and cleanup; no agent runner is bundled.
Agent containers receive only the public endpoint and intended sensor/configuration
resources. A read-only mount does not hide private contents.

## Removed orchestration

Harbor support, `integrations/kinex`, and Kinex agent packaging/reporting were
removed without compatibility stubs. `roboenv export` and `roboenv tasks generate`,
`validate`, and `run`, the `harbor` dependency extra, and the unified Harbor GPU
workflow are no longer available. Use task discovery, local or container launch,
and an external coding-agent harness instead. Existing generated Harbor bundles
and historical receipts are preserved artifacts, not supported execution inputs.
Kinex device catalogs, local workspace setup and compatibility checks remain.
The low-level socket client fallback session is now `roboenv`; use an explicit
session or `ROBOENV_SESSION` for an existing owner. HTTP identity discovery and
launch-generated episode identities are unchanged.

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
[standalone simulator images](#standalone-simulator-images).

## Environment notes

`roboenv.launch` removes every `ROBOENV_*` variable from the simulator
environment before starting the supervisor and sets its own. Backend knobs that
must reach a worker need another prefix (for example `SIMPLE_ISAAC_MOUNT_ATOL`).
Isaac Lab workers and replays log under `$TMPDIR/isaaclab`; on shared hosts where
`/tmp/isaaclab` belongs to another user, point `TMPDIR` inside the project. Replay
tools refuse to overwrite an existing receipt; pass a fresh `--out`.

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
