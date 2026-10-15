# Setup and local use

## Install the CLI

Use Linux, Git, and uv:

```sh
git clone https://github.com/stacx-lab/RoboEnv.git
cd RoboEnv
bash tools/bootstrap.sh
source .venv/bin/activate
```

Bootstrap creates a Python 3.12 development environment. Activate it in each shell
or use `.venv/bin/roboenv`. Each simulator gets a separate interpreter; see
[runtime/environments.json](../runtime/environments.json).

## Prepare a simulator

```sh
roboenv prepare robotwin
```

Preparation synchronizes pinned sources, installs an isolated interpreter,
acquires assets, and registers the environment for launch. Progress and the private
log path appear on stderr. `--json` keeps stdout machine-readable.

Repeat preparation to reuse a registered interpreter. It restores selected
third-party sources to their locked commits and reapplies managed patches,
discarding local source edits. Keep intended changes in checksum-locked patches.
Assets, environments, caches, recordings, and nested checkouts are preserved.
Source refresh alone does not revalidate reused dependencies.

### Optional setup controls

```sh
roboenv doctor robotwin
roboenv prepare robotwin --prefix .venv-sim/robotwin-fresh
```

`doctor` is read-only and does not certify GPU or task readiness. Existing
environments are never overwritten. After a failed installation, inspect the
printed log and use a new `--prefix` if a fresh environment is needed.

Set `ROBOENV_ASSET_CACHE` before preparation to place supported assets on a larger
filesystem. Keep that depot available: source checkouts link to it.

| Backend | Preparation requirements |
| --- | --- |
| RoboTwin | Pinned embodiment and object archives; existing assets are preserved |
| RoboCasa | About 10 GB of kitchen assets; verified bundles are reused and interrupted downloads can resume |
| RoboDojo | About 41 GB of ARX X5 assets and Isaac Sim; existing conflicting files fail verification |
| BEHAVIOR | Robot assets, encrypted scene/object data, challenge instances, and upstream agreement acceptance |
| LIBERO | Isolated robosuite dependencies and scene assets |

Isaac renderers need host graphics libraries, including `libGLU.so.1`, and a
supported RTX GPU. See [Isaac Lab setup](isaaclab.md) and
[GPU and license defaults](#gpu-and-license-defaults). Import checks alone do not
establish renderer readiness.

## Start a task

```sh
roboenv tasks list --simulator robotwin
roboenv launch robotwin/handover-block-aloha-agilex --port 8123
```

Launch uses the registered interpreter, validates live devices, and prints the
endpoint and run directory. Keep it running; Ctrl-C closes the world. Omit
`--port` for an automatically assigned port. An occupied port fails startup.
Launch does not install dependencies or download assets.

| Option | Behavior |
| --- | --- |
| `--execution-mode stepped` | Physics advances through world waits; native or configured step caps apply |
| `--execution-mode realtime` | Default; physics continues while idle and `budget.max` is null |
| `--stop-on-success` / `--no-stop-on-success` | Override the task's success-stop policy |
| `--record` | Save [spectator videos](spectator.md) |
| `--interpreter ENV/bin/python` | Use a prepared interpreter explicitly |
| `--detach` | Return after startup; close later with `roboenv stop RUN_ID` |
| `--json` | Machine-readable stdout; simulator logs remain on stderr |

The synced catalogs enable `stop_on_success`. At success or a stepped horizon,
physics freezes and pending work ends with `episode_finished`; final reads remain
available. Other native termination rules still apply. See the
[task contract](simulator-integration.md#task-format).

Custom tasks accept a `task.yaml` path and `--catalog CATALOG`. For status, use
`roboenv status --port 8123 --json`. HTTP stop requires `--allow-stop` at launch;
then `roboenv stop --port 8123` closes that service. Port selection uses loopback;
`--endpoint` selects a remote service. Run reference, port, and endpoint are
mutually exclusive; explicit selections override `ROBOENV_ENDPOINT`.

## Connect and use devices

In another terminal with `.venv` active:

```sh
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
roboenv devices
roboenv device left_arm read '{}'
```

```python
import roboenv

with roboenv.connect("http://127.0.0.1:8123") as env:
    print(env.task.instruction)
    print(env.device("left_arm").read())
```

Closing an attached client leaves the world running. Python owners can use
`with roboenv.launch("robotwin/handover-block-aloha-agilex", port=8123) as run:`;
exiting that context closes the world. See [device access](device-documentation.md)
for action schemas, cameras, and frames, or [Kinex setup](kinex-local.md).

Without an endpoint, the CLI selects a single ready local run and rejects
ambiguous discovery. HTTP binds to loopback; `--host 0.0.0.0` exposes it to a
container network. Parallel worlds need distinct ports in a shared network.
Evaluation harnesses own agent isolation and GPU allocation.

## Discover tasks and connect an agent

The built-in catalogs mirror RLE-Bench-inhouse task01 (10 RoboCasa tasks),
task02 (40 LIBERO tasks), task03 (50 RoboTwin Aloha AgileX tasks), and task04
(42 RoboDojo tasks). Their YAML is unchanged except that `max_steps` is omitted
and `execution_mode` is `realtime`.

For the larger upstream task collection:

```sh
roboenv tasks discover --backends robotwin --out generated/robotwin-source
roboenv tasks list --source generated/robotwin-source/tasks
```

Discovery reads pinned definitions without starting simulators. Generated catalogs
include source revisions, hashes, patches, and exclusions; keep them outside Git.
Agent harnesses own task selection and scheduling. They connect through the
public endpoint using the SDK, CLI, or exported device references.

## LIBERO tasks

The built-in catalog has ten tasks each from `libero_10`, `libero_spatial`,
`libero_object`, and `libero_goal`. IDs use `libero/libero-SUITE-NN`, numbered
01–10. Discovery covers all 130 native tasks. Each task selects the Panda catalog
for its tabletop, floor, kitchen, living-room, or study scene.

Absolute joint control is a practice protocol, distinct from the published OSC
action space. `tools/accept_libero.py --out OUTPUT` checks native devices,
transport, Kinex recording, and fresh-process replay without model calls.

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

## Advanced installation and assets

Sources and managed patches are pinned in [upstreams.lock.json](../upstreams.lock.json).
`roboenv sources status --backends BACKEND` checks their exact revisions and tracked
contents without changing the working tree. Recipes and dependency locks are in
`runtime/environments.json` and `runtime/locks/`.

For explicit package inputs, use the low-level installer after source sync:

```sh
roboenv sources sync --backends robotwin
python -m roboenv.provisioning.setup robotwin
python -m roboenv.provisioning.setup robotwin \
  --install --prefix .venv-sim/robotwin-wheels --wheels build/wheels
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

## GPU and license defaults

All eight simulator backends use the first CUDA-visible GPU for compute and rendering:

```sh
export CUDA_VISIBLE_DEVICES=6
roboenv launch robotwin/handover-block-aloha-agilex
```

An unset mask selects logical `cuda:0` according to the CUDA driver's default order.
A list such as `6,2` selects its first visible device. GPU UUIDs and CUDA-supported
UUID prefixes work as masks. Empty or disabled visibility fails before a GPU-backed
launch; CPU-only checks do not probe GPUs. MIG instances are unsupported; these
renderers require a full graphics-capable GPU.

RoboEnv queries the selected CUDA device's identity in a bounded child process,
then matches the graphics device required by the backend. Worker subprocesses expose
only that card to CUDA. Vulkan and EGL ordinals need not equal CUDA ordinals; multi-GPU
rendering is disabled. RoboEnv derives internal renderer settings, including
`MUJOCO_EGL_DEVICE_ID`, `ROBOTWIN_RENDER_DEVICE`, and Isaac GPU settings. Set
`CUDA_VISIBLE_DEVICES` rather than supplying these backend-specific overrides.
The parent shell and other running services are unchanged.

This selects one GPU for simulation and rendering; it is not hardware isolation.
The NVIDIA Vulkan driver may retain a small initialization context on another GPU
even when the renderer explicitly selects the requested device. Use container
GPU isolation when access to other physical cards must be prevented.

In containers, the NVIDIA runtime has already applied `NVIDIA_VISIBLE_DEVICES`.
RoboEnv queries devices actually visible inside the container and does not compare
that variable's host indices against renumbered container indices. Private run
metadata and worker logs retain the selected UUID and renderer indices. Replay
selects the current process's visible GPU, allowing another machine or card.

The SIMPLE `--gpu` / Python `gpu=` selectors, Go2 `device` selector, and ABC
`render_gpu_id` task field have been removed. Remove them from custom tasks and
scripts, and export `CUDA_VISIBLE_DEVICES` before launch or a native/replay tool.
ABC's Blender subprocess inherits the same mask.

Preparation and Isaac startup default `OMNI_KIT_ACCEPT_EULA=YES`; BEHAVIOR
preparation also defaults `ROBOENV_BEHAVIOR_ACCEPT_LICENSE=YES`. These defaults
accept the respective upstream agreements. Explicit values such as `NO` or an
empty string are preserved; asset acquisition remains blocked when acceptance is
declined. Defaults are passed to owned subprocesses rather than exported into the
calling shell. GPU selection and license defaults do not install missing assets.
