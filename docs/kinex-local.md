# Local RoboCasa and Kinex

The checked-in `catalog/robots/robocasa/robocasa-panda-omron` directory is already ready
to copy into Kinex. It contains a fixed robot manifest and eight devices:
arm, base, torso, gripper, camera_left, camera_right, camera_wrist, and world. Configuration is independent
of the task, seed, kitchen layout, service port, and episode identity. Launch
never writes to the catalog. See [RoboCasa catalog design](robocasa.md#catalog-and-generation).

Robot-mounted camera calibration, controller scaling, joint limits, and rigid
mounts are pre-generated. Object positions and fixture geometry must be perceived
from camera images; no simulator scene ground truth is sent to the agent.
The mobile `base` has no static world mount. Its declared `feedback.pose_frame: world`
and measured pose supply the live world-to-base edge.

## 1. Launch the service (terminal A, RoboEnv checkout)

```sh
source .venv/bin/activate
roboenv prepare robocasa
roboenv launch robocasa/pick-place-microwave --port 8123
```

Preparation is needed only if the simulator is not already prepared. Wait for
`Ready` and keep this process running. Launch initializes the kitchen and the
simulation service; it neither generates nor updates the catalog configuration.

Realtime runs have no episode step cap, so leaving the service idle while starting
Kinex does not exhaust a native action-frame budget. Frames are still counted;
`budget.max` is null. Use `--execution-mode stepped` when you want idle physics
paused and the task's configured/native step limit enforced. World `read` and `status`
results carry the episode `budget` in the shape Kinex reads, `{elapsed_s, max_s}` in
seconds (`max_s` is null without a cap); `wait` results and other device readings omit
it. Restart the service to pick up runtime changes, then `/reload all` in Kinex to
refresh its connections.

For another task use another task ID. For parallel trials use different ports
and separate Kinex processes with different `ROBOENV_ENDPOINT` values. Both
can use byte-identical copies of the same robot configuration.

## 2. Copy into a fresh Kinex workspace (terminal B, RoboEnv checkout)

The following variables resolve the sibling Kinex source checkout and create
an isolated workspace for this run. Keep this terminal open for the later steps.

```sh
source .venv/bin/activate
roboenv_repo="$PWD"
kinex_repo="$(realpath ../kinex)"
trial_workspace="$(mktemp -d "$roboenv_repo/.cache/kinex-local.XXXXXX")"
cp -R catalog/robots/robocasa/robocasa-panda-omron "$trial_workspace/.kinex"
cd "$trial_workspace"
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
```

Use a working Kinex installation with its TUI dependencies already installed
(see the sibling README). Exported profiles use Kinex's workspace Python. Remote device access
uses standard-library transport in the shared `devices/_roboenv.py`; profile loading needs
PyYAML and typed actuator operations need Kinex’s Python helpers. Kinex supplies
its Python library to drivers; for direct shell calls, set
`export PYTHONPATH="$kinex_repo/packages/core/assets/python${PYTHONPATH:+:$PYTHONPATH}"`.
The agent environment does not need RoboEnv or simulator dependencies. There is
one transport module per device collection, outside the individual device folders.
Drivers resolve it relative to their own file, so the working directory is irrelevant.
When copying selected devices, include the collection's `_roboenv.py` once.
Drivers use `persistent: true`: Kinex starts `driver.py --serve` once and reuses
its configuration, kinematics model, and HTTP connection. `/reload <device>`
restarts the process after driver/profile/connection edits. Session and generation
stay fixed until reload; a pinned `connection.json` must be updated when switching
episodes. HTTP idle connections are recreated before the next request; lost replies
are reported without resending commands.

`/unload` stops an actuator within its declared scope and releases its connection.
On EOF or SIGTERM, a connected persistent actuator also attempts a bounded stop;
a failed stop is logged, not treated as success. This does not stop the world.
Direct single-call scripts retain their targets after exiting.

Camera stream recording requires `ffmpeg`: Kinex encodes each recorded camera to MP4.

Confirm `devices/` exists and contains arm, base, torso, gripper, camera_left, camera_right,
camera_wrist, and world for this example. There is no public composite robot device.
The task instruction is available through `roboenv.connect(ROBOENV_ENDPOINT).task`.
For these direct calls before starting Kinex, prepare its separate Python project
explicitly. Kinex uses `.kinex/python/.venv`; ordinary shell environments remain
independent. Existing environments need explicit dependency updates.

```sh
uv init --bare .kinex/python
uv add --project .kinex/python numpy scipy pillow pyyaml 'opencv-python-headless>=4.7,<5'
export PYTHONPATH="$kinex_repo/packages/core/assets/python${PYTHONPATH:+:$PYTHONPATH}"
```

Check the driver without starting a model request:

```sh
uv run --project .kinex/python python .kinex/devices/arm/driver.py status '{}'
uv run --project .kinex/python python .kinex/devices/camera_left/driver.py read '{}'
```

Expect `ok: true`; camera output must include an existing local RGB PNG `file`
and aligned 16-bit PGM `depth_file`, with acquisition Unix timestamps. The profile declares
`intrinsics.depth.scale_m_per_unit: 0.001`.
Open that image to confirm that it depicts the initialized kitchen. Neither
operation requests physics steps. A realtime world keeps running while idle;
a stepped one advances only during world waits.

## 3. TUI recording smoke test

```sh
kinex
```

Run these built-in slash commands in the TUI:

```text
/ping all
/record snap camera_left camera_right camera_wrist initial kitchen
/record start camera_left=2 camera_right=2 camera_wrist=2 arm=5 base=2 local device smoke test
/record status
```

Recording continues between turns and while the user pauses until `/record stop`.
The thin drivers do not implement native video streaming; Kinex samples camera
reads and encodes the frames. Requested Hz is a target; inspect effective rates
and skipped samples. Reads do not request physics steps. Realtime continues
during recording; stepped advances only during world waits.

With the recording running, use a separate terminal with the same copied workspace, `ROBOENV_ENDPOINT`, and
`PYTHONPATH`. Read the current joint configuration and send it back as a hold target:

```sh
target=$(uv run --project .kinex/python python .kinex/devices/arm/driver.py read '{}' | uv run --project .kinex/python python -c \
  'import json,sys; r=json.load(sys.stdin); assert r["ok"], r; print(json.dumps({"actions":[r["positions"]],"duration_s":0.5}))')
uv run --project .kinex/python python .kinex/devices/arm/driver.py write "$target"
uv run --project .kinex/python python .kinex/devices/world/driver.py wait '{"duration_s":0.5}'
uv run --project .kinex/python python .kinex/devices/arm/driver.py read '{}'
uv run --project .kinex/python python .kinex/devices/arm/driver.py stop '{}'
```

`write` submits an action buffer and returns immediately. The executor consumes
the references over `duration_s` world seconds and retains the final reference.
Each target contains seven absolute joint positions in radians, in the profile’s
model joint order. `world.wait` allows execution time; compare the subsequent
measured positions with the target to check arrival. Neither acceptance nor elapsed
duration proves arrival. Small measured soft-limit overshoots can make a write fail;
consult the joint-limit recovery notes before commanding another target. `stop`
clears pending references and holds the arm; it does not stop the world's clock.

Direct shell driver calls bypass Kinex's dispatch event log; the background
samplers still record resulting images/state. To verify complete command
tracing, perform a second test through Kinex's `device` tool: ask the agent to
read arm state, issue one such write, wait through world, read feedback, and stop. That step
requires your configured model and normal motion approval. Start a fresh
simulation episode before attempting the full task.

Finish the smoke recording in the TUI:

```text
/record snap camera_left camera_right camera_wrist final kitchen
/record stop partial device and recording smoke test only
/record status
```

`partial` is a recording annotation, not a benchmark verdict. Use the reported
recording ID in the next step. `/load all` is available if the user previously
unloaded devices; checked-in profiles default to loaded in Kinex.

## 4. Visualize the recording

After leaving the TUI, in terminal B:

```sh
recording_dir="$trial_workspace/.kinex/recordings/RECORDING_ID"
ls "$recording_dir"/devices/*/stream/video.mp4
```

Replace `RECORDING_ID` with the stream ID from `/record status` (not a separate
snap ID). Kinex encoded one MP4 per recorded camera; `recording.json` lists each
under `devices.<name>.stream.video`. Open `devices/camera_left/stream/video.mp4`, and
likewise for camera_right and camera_wrist, with your desktop viewer.
These show wall-clock samples, not a
frame-perfect simulation replay. The TUI controls recording; these artifacts
provide visual playback. No new visualization server is required.

## Verification

| Check | Required evidence |
|---|---|
| Kitchen boot | Ready service, actual kitchen image, unchanged catalog |
| Configuration | Kinex sees the eight standard devices; driver status/read succeed |
| Basic control | Write submission returns immediately; world wait allows execution and measured feedback checks the hold target; realtime physics continues after device stop |
| Recording | Nonempty camera/state events, readable saved images, completed manifest |
| Visualization | Playable per-camera MP4s show the captured kitchen |
| Dispatch tracing | Kinex-issued read/write/stop appear in device events |
| Episode isolation | Existing SDK clients reject replacement episodes; a fresh driver attaches without rewriting configs |
| Task success | Separate full attempt with the original RoboEnv verifier verdict |
| Replay | Separate fresh-process replay with the same task/seed and recorded actions |

Finish terminal A with Ctrl-C. Preserve recordings and run evidence locally.
A recording smoke test does not certify task success or reproducible replay.

For observation-restricted evaluations, use an isolated agent process/container
without access to simulator assets, owner sockets, or verifier evidence. A local
TUI running as the simulator's Unix user is a functional test, not a filesystem
security boundary. A benchmark deployment runs the agent as a separate unprivileged
user; private run evidence is owner-only.
