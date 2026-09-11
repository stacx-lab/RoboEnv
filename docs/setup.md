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

This synchronizes pinned RoboTwin sources, installs its isolated interpreter,
downloads and verifies its public embodiment/object assets, and reports setup
checks. RoboTwin and RoboDojo run from their source checkouts without cap-x.
Substitute another simulator as needed.
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

RoboTwin preparation verifies the pinned upstream archives and preserves existing
asset directories. RoboDojo preparation downloads its pinned public `Assets` tree
(about 41 GB) and verifies every file against its source hash. It also prepares the
YAM URDF, meshes and optical frame from a pinned manufacturer revision. Missing
files can be resumed; conflicting existing files produce an error without being
overwritten. Set `ROBOENV_ASSET_CACHE=/path/to/data-depot` before preparation to
place these assets on a larger filesystem. Source checkout links point into that
depot, so it must remain available for launch.

RoboDojo requires Isaac Sim. YAM additionally requires native USD conversion using
the installed Isaac Lab importer. Once the user has accepted NVIDIA's applicable
Isaac Sim license, preparation recognizes `OMNI_KIT_ACCEPT_EULA=YES` and performs
that conversion, recording the source and output hashes. Without that explicit
acceptance, it reports the remaining conversion requirement. Existing conversions
are verified before reuse. The current conversion uses convex decomposition for
the two YAM fingers; a single hull fills their concave shape and can eject a
gripped object. Upgrading a verified older conversion retains its complete USD
tree and receipt under `native-usd-history/` before publishing the new layers.
Quiesce native jobs before preparing or upgrading their asset depot. Changed
collision geometry requires new native, Kinex and replay evidence. Conformance
checks both loaded finger contact and unsupported dynamic retention followed by
release. Conversion and asset checks alone do not establish motion or task success.

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
roboenv device left_arm read '{}'
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
print(env.device("left_arm").read())
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
roboenv export libero/mug-into-microwave --out generated/example
```

The default deployment is `.cache/images/libero/deployment.json`; use
`--deployment FILE` to select another. The command builds a fresh SDK wheel
unless `--wheel FILE` is supplied, and validates the task with the real Harbor
loader. Local launch and export use the same task resolver and device binding
configuration. Image and asset preparation are described in
[Harbor tasks](harbor-tasks.md).

Historical RoboTwin and RoboDojo primitive `solution.py` programs remain in the
catalog with `retired: true` provenance. Reference export and explicit reference
validation reject them because the default tasks now expose typed devices.
Use `tools/validate_simulators.py --modes probe` for device connectivity checks;
full Kinex controller development and frozen task acceptance use Kinex's
`examples/simulator-acceptance/` harness.

`tools/check_native_transport.py` exercises two concurrent native episodes,
request deduplication, lost-reply recovery, public/owner separation and endpoint
reuse. It discovers an RGB or RGB-D camera from the live inventory; `--camera`
selects a specific sensor. `--seeds FIRST SECOND` selects the two native layouts
for RoboTwin and RoboDojo. For BEHAVIOR, also supply `--instances FIRST SECOND`
with distinct activity instance IDs: changing random seeds does not select a
different presampled activity instance. The checker compares the loaded native
states and rejects a pair that is effectively identical. It requires prepared
native environments and assets; passing its planning tests alone does not certify
BEHAVIOR or RoboDojo execution.

The published RoboDojo layout archive currently contains ARX-X5 scene layouts.
Generated YAM practice tasks explicitly set `env_cfg_type: yam` and
`layout_config: arx_x5`: the native YAM robot operates on those unchanged object
and fixture layouts. Receipts record both settings and the embodiment adaptation;
these tasks are not an official YAM benchmark collection. A private YAM layout
collection can instead select `layout_config: yam` when installed.

For a single GPU on an Isaac host, select the physical device with
`CUDA_VISIBLE_DEVICES` and leave `OMNIGIBSON_GPU_ID` unset so Kit can select the
matching renderer and PhysX can use logical CUDA device 0. A physical index passed
as a logical index after masking can prevent startup. Verify the actual selected
device in native logs; source or import checks do not establish GPU readiness.

For development verification run `bash tools/check.sh`. CPU checks, container
boot, device conformance, benchmark success, and fresh-process replay are
separate results.


## Local cache and disk usage

`.cache/` contains several kinds of local data:

- `pytest/`: temporary test fixtures and copied workspaces. Pytest removes passing
  test directories and retains failed-test artifacts from up to three runs.
  An explicit `--basetemp` overrides pytest’s automatic retention.
- `roboenv/runs/`: simulation logs, recordings, and evidence. Keep these until you
  have reviewed or archived the runs; they are not automatically deleted.
- `roboenv/installations/` and `roboenv/assets/`: interpreter registration and
  asset verification receipts. Keep them with the installed simulator assets.
- Simulator asset folders, setup logs, schema dependencies, and build diagnostics.

Install commands honor `UV_CACHE_DIR` and otherwise use uv’s shared default cache,
so each RoboEnv checkout does not retain its own large set of package downloads.
Older checkouts may still contain `.cache/uv`. To prune its disposable downloaded
wheel entries while retaining expensive source builds:

```sh
du -h --max-depth=1 .cache
uv cache prune --ci --cache-dir .cache/uv
```

Pruning uses uv’s cache locking and leaves installed environments in place. Packages
removed from the cache may need downloading again for a future installation. Do
not delete all of `.cache/`: run evidence, asset files, and registrations also live
there. Actual simulator interpreters are separate under `.venv-sim/` by default.
