# Local RoboCasa + Complex test plan

The checked-in `catalog/robots/robocasa-panda-omron` directory is already ready
to copy into Complex. It contains a fixed robot manifest and eight devices:
arm, body, torso, gripper, camera_left, camera_right, camera_wrist, and world. Configuration is independent
of the task, seed, kitchen layout, service port, and episode identity. Launch
never writes to the catalog. See [research and provenance](robocasa-config.md).

Robot-mounted camera calibration, controller scaling, joint limits, and rigid
mounts are pre-generated. Object positions and fixture geometry must be perceived
from camera images; no simulator scene ground truth is sent to the agent.
The mobile `body` is the robot's frame root, with no declared world mount.
`body.read()` also provides its simulator world pose, without adding a profile edge.

## 1. Launch the service (terminal A, RoboEnv checkout)

```sh
source .venv/bin/activate
roboenv prepare robocasa
roboenv launch robocasa-pick-place-microwave --port 8123
```

Preparation is needed only if the simulator is not already prepared. Wait for
`Ready` and keep this process running. Launch initializes the kitchen and the
simulation service; it neither generates nor updates the catalog configuration.

For another task use another task ID. For parallel trials use different ports
and separate Complex processes with different `ROBOENV_ENDPOINT` values. Both
can use byte-identical copies of the same robot configuration.

## 2. Copy into a fresh Complex workspace (terminal B, RoboEnv checkout)

The following variables resolve the sibling Complex source checkout and create
an isolated workspace for this run. Keep this terminal open for the later steps.

```sh
source .venv/bin/activate
roboenv_repo="$PWD"
complex_repo="$(realpath ../complex)"
trial_workspace="$(mktemp -d "$roboenv_repo/.cache/complex-local.XXXXXX")"
mkdir -p "$trial_workspace/.complex/robots"
cp -R catalog/robots/robocasa-panda-omron "$trial_workspace/.complex/robots/"
cd "$trial_workspace"
export ROBOENV_ENDPOINT=http://127.0.0.1:8123
cat > .complex/complex.json <<'JSON'
{"robot":{"active":"robocasa-panda-omron"}}
JSON
```

Use a working Complex installation with its TUI dependencies already installed
(see the sibling README). Keep the activated RoboEnv venv on PATH: exported
profiles invoke `python3`, which needs the RoboEnv SDK, not simulator dependencies.
Video export requires `ffmpeg`; the keyframe script requires Pillow.

Confirm `devices/` exists and contains arm, body, torso, gripper, camera_left, camera_right,
camera_wrist, and world for this example. It should not contain the optional composite robot.
The task instruction is available through `roboenv.connect(ROBOENV_ENDPOINT).task`.
Check the driver without starting a model request:

```sh
python3 .complex/robots/robocasa-panda-omron/devices/arm/driver.py status '{}'
python3 .complex/robots/robocasa-panda-omron/devices/camera_left/driver.py read '{}'
```

Expect `ok: true`; camera output must include an existing local RGB PNG `file`
and aligned 16-bit PGM `depth_file`, with `depth_scale: 0.001` (metres per unit).
Open that image to confirm that it depicts the initialized kitchen. Neither
operation should consume simulation action frames.

## 3. TUI recording smoke test

```sh
complex
```

Run these built-in slash commands in the TUI:

```text
/ping all
/record snap camera_left camera_right camera_wrist initial kitchen
/record stream camera_left=2:all camera_right=2:all camera_wrist=2:all arm=5:all body=2:all local device smoke test
/record status
```

`:all` is deliberate: it records while the user pauses, including during this
manual smoke test. The thin drivers do not implement native video streaming;
Complex uses its supported stills sampler. Requested Hz is a target; inspect
effective rates and skipped samples. Reads do not advance physics, so a kitchen
with no writes will remain still even while wall-clock recording continues.

With the recording running, use a separate terminal with the same activated
venv and copied workspace. Exercise only one small arm interval:

```sh
python3 .complex/robots/robocasa-panda-omron/devices/arm/driver.py write '{"target":[0,0,0.01,0,0,0]}'
python3 .complex/robots/robocasa-panda-omron/devices/arm/driver.py stop '{}'
```

This is a normalized controller delta, not a position in metres. Check the live
profile for scaling. A single interval can be visually subtle; compare arm state
as well as camera frames. `stop` resets the servo goal without another physics
step. Do not infer task success from a successful command.

Direct shell driver calls bypass Complex's dispatch event log; the `:all`
samplers still record resulting images/state. To verify complete command
tracing, perform a second test through Complex's `device` tool: ask the agent to
read arm state, issue one such bounded write, stop, and read again. That step
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
unloaded devices; checked-in profiles default to loaded in Complex.

## 4. Visualize the recording

After leaving the TUI, in terminal B:

```sh
recording_dir="$trial_workspace/.complex/robots/robocasa-panda-omron/recordings/RECORDING_ID"
python3 "$complex_repo/builtin/skill/reflect/scripts/keyframes.py" "$recording_dir" --device camera_left
python3 "$complex_repo/builtin/skill/reflect/scripts/video.py" "$recording_dir" --device camera_left
```

Replace `RECORDING_ID` with the stream ID from `/record status` (not a separate
snap ID). Open `analysis/keyframes-camera_left.jpg` and `analysis/camera_left.mp4` under
that directory with your desktop viewer. Repeat with `--device camera_right` and `--device camera_wrist` for the other views.
These show wall-clock samples, not a
frame-perfect simulation replay. The TUI controls recording; these artifacts
provide visual playback. No new visualization server is required.

## Acceptance and follow-up

| Check | Required evidence |
|---|---|
| Kitchen boot | Ready service, actual kitchen image, unchanged catalog |
| Configuration | Complex sees the eight standard devices; driver status/read succeed |
| Basic control | One write advances one native interval; reads and stop consume none |
| Recording | Nonempty camera/state events, readable saved images, completed manifest |
| Visualization | Keyframe sheet and playable MP4 show the captured kitchen |
| Dispatch tracing | Complex-issued read/write/stop appear in device events |
| Episode isolation | Existing SDK clients reject replacement episodes; a fresh driver attaches without rewriting configs |
| Task success | Separate full attempt with the original RoboEnv verifier verdict |
| Replay | Separate fresh-process replay with the same task/seed and recorded actions |

Finish terminal A with Ctrl-C. Preserve recordings and run evidence locally.
A recording smoke test does not certify task success or reproducible replay.

For observation-restricted evaluations, use an isolated agent process/container
without access to simulator assets, owner sockets, or verifier evidence. A local
TUI running as the simulator's Unix user is a functional test, not a filesystem
security boundary. Harbor runs the agent as a separate unprivileged user; private
run evidence is owner-only.
