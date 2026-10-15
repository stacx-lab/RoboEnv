# Local RoboTwin and Kinex

Use the Aloha AgileX catalog with a prepared RoboTwin environment and a working
[Kinex](https://github.com/stacx-lab/kinex) installation. RoboEnv owns the simulator;
Kinex connects through exported device drivers.

## Select the task and its paired devices

```sh
roboenv tasks list --simulator robotwin
roboenv robots list --simulator robotwin
```

Each listing prints one ID per line; add `--json` for structured output.
`launch` accepts a task ID, while `setup` accepts a robot ID. Read `robot:` in
`catalog/tasks/<task-id>/task.yaml` to select the matching device collection.
For this guide, `robotwin/handover-block-aloha-agilex` pairs with
`robotwin/robotwin-aloha-agilex`, the supported RoboTwin robot. For simulators
with multiple robots or scenes, use the task YAML to select the exact match. See
[other pairs](setup.md#choose-a-paired-task-and-robot).

## Launch RoboEnv

From the RoboEnv checkout:

```sh
source .venv/bin/activate
roboenv prepare robotwin
roboenv launch robotwin/handover-block-aloha-agilex --port 8123
```

Wait for `Ready` and keep the process running. Realtime physics continues while
idle, with no episode step cap. Use `--execution-mode stepped` to advance through
world waits and enforce the native horizon. The catalog stops on native success;
final device reads remain available.

## Set up the agent workspace

In another terminal, activate the RoboEnv environment, then change to your agent
workspace:

```sh
source /path/to/RoboEnv/.venv/bin/activate
mkdir -p kinex-sim-workspace
cd kinex-sim-workspace
roboenv setup robotwin/robotwin-aloha-agilex
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
roboenv describe --endpoint "$ROBOENV_ENDPOINT"
```

Setup writes `.kinex/workspace.yaml` and `.kinex/devices/`. It works without a
running simulator. The catalog contains eight devices: `left_arm`, `right_arm`,
`left_gripper`, `right_gripper`, `head_camera`, `left_camera`, `right_camera`, and
`world`. Give the agent the initialized `task.instruction` and `task.context`
returned by `describe`. For changing goals, refresh them during the episode.

`roboenv robots list` (or `roboenv setup --list`) lists robot catalogs. Use
`--project` for another RoboEnv checkout or `--catalog` for a standalone catalog.
These select the source; the destination is always the current workspace's `.kinex/`.

Existing workspace or device files block setup. To replace them, save custom
configuration first, then run:

```sh
roboenv setup robotwin/robotwin-aloha-agilex --force
```

This replaces the manifest and entire device collection, preserving other
`.kinex/` content such as recordings and Python environments. Reload devices in
Kinex after replacement. Persistent drivers pin episode identity; reload them
when switching episodes, and update any episode-specific `connection.json`.

## Read and control devices

Start `kinex` from the workspace and run `/devices`, `/load all`, and `/ping all`.
Kinex supplies its Python helpers to the drivers; its workspace Python also needs
PyYAML. Simulator
packages are not needed in the agent environment. Keep the shared
`devices/_roboenv.py` with the device collection when copying it.

Read `left_arm` to obtain measured joint positions. A hold command writes those
positions as one `actions` vector with `duration_s`, then waits through `world`
and reads feedback. Joint targets are absolute radians in profile order. Write
acceptance and elapsed time do not prove arrival; check measured positions.
Device `stop` clears pending references within that device's scope. Gripper stop
preserves its applied target; explicit open releases it. The world clock continues.

Kinex `servo` plans and submits a joint trajectory, then waits on the loaded world
clock. `block: false` submits without waiting. It has no `duration_s`, `ik`, or
`max_joint_delta` argument. Direct device IK accepts an optional seed.
See [device operations and frames](device-documentation.md#device-operations-and-frames)
for units, frame conventions, and FK/IK.

`/unload DEVICE` stops an actuator and releases its connection; `/load DEVICE`
reloads it after edits. EOF or SIGTERM also attempts a bounded actuator stop.
Lost replies are reported without resending commands. A failed cancellation
requires recovery before further writes.

## Record and inspect

Install FFmpeg for camera videos. In the Kinex TUI:

```text
/record snap head_camera left_camera right_camera initial scene
/record start head_camera=2 left_camera=2 right_camera=2 left_arm=5 right_arm=5 device smoke test
/record status
```

Use Kinex's device tool for reads, writes, waits, and stop so commands enter its
dispatch log. Direct shell driver calls bypass that log. Recording continues
between turns until stopped:

```text
/record snap head_camera left_camera right_camera final scene
/record stop partial device and recording smoke test only
/record status
```

Open `.kinex/recordings/RECORDING_ID/recording.json` for the camera video paths;
each camera has `devices/NAME/stream/video.mp4`. Requested sampling rates are
targets; inspect effective rates and skipped samples. These videos show sampled
observations, not a fresh-process command replay. `partial` is an annotation,
not a benchmark verdict.

Stop RoboEnv with Ctrl-C and preserve recordings locally. Report boot, device
behavior, recording, native task success, and fresh-process replay separately;
see [validation](gpu-validation.md). Use different ports and workspaces for
parallel episodes. Evaluation harnesses must isolate agents from owner sockets,
assets, and verifier evidence; a local TUI is not that isolation boundary.
