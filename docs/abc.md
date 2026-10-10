# ABC

ABC exposes all 33 canonical tasks through one native CPU MuJoCo world, two YAM
arms, two grippers, three RGB cameras, and world. Training, policy inference,
and MJWarp batching are outside this integration.

All 33 task definitions are checked into `catalog/tasks/abc` and available through
`roboenv tasks list --simulator abc`.

```sh
roboenv prepare abc
roboenv tasks discover --backends abc --out generated/abc-source
roboenv launch abc/put-plastic-bottles-in-bin
```

Preparation installs an isolated Python 3.12 environment and verifies the upstream
asset manifest (approximately 640 MB downloaded, 1.72 GB extracted). Assets use
`ROBOENV_ASSET_CACHE` when supplied. `roboenv doctor abc` audits preparation;
it does not certify rendering or task success.

All tasks share `abc-yam`, with both arm bases at 0.76 m. The managed
`abc-unified-arm-mounts.patch` raises the mounts by 1 cm in the six source scenes
used by the ten tasks that originally mounted at 0.75 m. This is an intentional
RoboEnv practice-scene change. Devices are `left_arm`, `right_arm`, `left_gripper`, `right_gripper`,
`camera_top`, `camera_left`, `camera_right`, and `world`. Cameras return native
640×480 RGB. Arm commands are absolute radians; normalized gripper commands use
0 for closed and 1 for open. Native limits and gains remain in effect. Gripper
stop retains its applied target; explicit opening releases it.

Stepped mode defaults to `floor(300 / control_interval_s)` action frames, where
`control_interval_s` is the native physics timestep times task control decimation.
Current ABC tasks use 0.034 seconds, giving 8,823 actions (299.982 seconds).
An explicit owner `backend_kwargs.max_steps` can shorten a test episode.
Realtime is uncapped. The shared `stop_on_success` option defaults to false;
for the balancing maintenance task, native success means the ball has not yet
been dropped, so early success stopping does not demonstrate sustained balance.
These device runs are practice protocols, not published policy-evaluation results.

Seed and optional `backend_kwargs.randomization` select native randomization.
Only the current native language instruction is public; scene truth, future
instructions and evaluator details remain owner-only. For changing instructions,
particularly multi-drawer search, call `env.refresh_task()` between stages.
Copied Kinex workspaces expose the same instruction through `python task.py`.
Practice reset starts a fresh native world. Intermediate-state checkpoint/restore
is no longer supported; reset or start a fresh run to retry a task. Command
recording and fresh-process replay verification remain available.

## Blender rendering

MuJoCo remains the default. Optional Blender Cycles rendering improves both
device images (including Kinex recordings) and RoboEnv spectator videos:

```sh
roboenv prepare abc --renderer blender --prefix .venv-sim/abc-blender
```

Use a new prefix when upgrading an older interpreter without the rendering
dependencies. Once prepared, omit `--prefix` to reuse the registered interpreter.
Preparation verifies Blender 4.5.3 Linux x64 and Photo Studio 01 2K HDRI downloads
in the managed asset cache. Blender assets are not included in wheels.

Set these fields in an owner task YAML copied from any ABC task:

```yaml
execution_mode: stepped
backend_kwargs:
  task_name: put_plastic_bottles_in_bin
  camera_backend: blender
```

Launch that YAML with `roboenv launch PATH --record`. Select a card externally with
`CUDA_VISIBLE_DEVICES`; both MuJoCo and Blender use the same selected GPU. The complete task still needs
its original `id`, `robot`, instruction and budget fields. The same `abc/abc-yam`
catalog serves both renderers. Spectator source views and custom fixed poses keep
their declared resolution and field of view; device cameras remain 640×480 RGB.

Defaults are 16 Cycles samples with OIDN denoising. `ABC_BLENDER_SAMPLES` and
`ABC_BLENDER_DENOISER` (`OPENIMAGEDENOISE`, `OPTIX`, `NONE`) override quality;
`BLENDER` and `ABC_HDRI` select external installations. Effective settings and
binary/lighting hashes are included in owner records for replay.

One Blender process serves the device cameras; recording uses an independent
process. Launch/reset warms the cameras before becoming ready, and repeated reads
within a control frame reuse the rendered images. Startup, render, and shutdown
waits are bounded; failures never silently fall back to MuJoCo. A spectator failure
marks recording incomplete through the existing recorder lifecycle.

Rendering does not advance physics or consume the stepped budget. Cycles can slow
realtime execution below wall-clock speed; use stepped mode when this matters.

## Success rules

The 29 existing native evaluators remain unchanged. Four tasks use explicitly
versioned RoboEnv-defined rules:

- Marker drawers: all five markers wholly contained in the named drawer and
  released, with the drawer within 5 mm of closed for one simulation second.
  Collision geometry determines containment, with 2 mm tolerance.
- Wood-block tower: at least five distinct blocks in an upward load-bearing
  contact chain supported by the table, without robot or fixture support, for
  one simulation second. Linear speed must remain below 0.01 m/s and angular
  speed below 0.1 rad/s.

Their public instructions state these additional requirements. Scoring is cached
per control interval; reads do not advance evaluator history.

## Maintenance and validation

Regenerate catalogs using the ABC interpreter:

```sh
python tools/generate_abc_config.py --out catalog \
  --bindings-out src/roboenv/devices/bindings/defaults \
  --mapping-out src/roboenv/assets/abc-rigs.json
```

Run the small native/Kinex acceptance after preparing ABC and building Kinex
with Node >=22.19:

```sh
.venv/bin/python tools/accept_abc.py --out .cache/abc-acceptance \
  --kinex ../kinex --node node
```

Add `--renderer blender` for the bounded Blender device/spectator/replay probe.
It records named and custom spectator views at two simulation-time frames per
second. `tests/abc_blender.py --out NEW_DIRECTORY`, run with the ABC interpreter,
checks optical projection, moving cameras, renderer failure cleanup, state
invariance, reset and MuJoCo/Blender image comparisons.

It uses GPU 0 for EGL rendering and makes no model calls. Native fixtures,
measured motion, stepped and realtime execution, recording and fresh-process replay are
reported separately from observed task success. Output directories must be new.
CPU tests and catalog checks alone do not certify native acceptance.
