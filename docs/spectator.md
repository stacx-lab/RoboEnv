# Spectator recording (experimental)

```sh
roboenv launch libero/bowl-onto-plate --record
roboenv launch libero/bowl-onto-plate --record ./recordings/episode-1
roboenv launch libero/bowl-onto-plate --record --record-fps 5 --record-views views.json
```

Recording saves one H.264 MP4 per view under the printed owner-only recording
directory (`evidence/recording/` inside the run by default). `--record DIR` writes
the videos and metadata directly into a new or empty directory; relative paths
are resolved from your current directory. Existing recordings are never overwritten.
FFmpeg with `libx264` must be on
PATH; it is optional when recording is disabled. Python owners use
`roboenv.launch(task, recording={})`, or pass `recording={"fps": 5, "views": [...]}`.
Use `recording={"directory": "./recordings/episode-1"}` to select the output folder.
Recording is experimental on all five backends. LIBERO, RoboCasa and RoboTwin
have native spectator isolation checks; RoboDojo and BEHAVIOR remain unverified.
There is no browser preview or offline replay fallback.

Defaults are **10 simulation-time FPS**, height **720**, and the source camera's
aspect ratio. LIBERO and RoboCasa select `third_person` and `wrist`; RoboTwin,
RoboDojo and BEHAVIOR select `third_person`, `left_wrist` and `right_wrist`.
These are separate spectator cameras; changing their resolution does not resize
agent observations. For example, `views.json` for LIBERO can contain:

```json
[
  {"name": "overview", "source": "third_person", "height": 720},
  {"name": "hand", "source": "wrist", "width": 480, "height": 480}
]
```

Video dimensions must be even. Views and FPS stay fixed for the episode.
Unknown views, invalid settings and unavailable encoders fail requested recording
startup before the service becomes ready.

## Timing and lifecycle

In **stepped** mode, frames follow simulation advances during `world.wait`;
time spent thinking while the world is idle is omitted. In **realtime** mode,
the world advances and records while the agent thinks. Rendering adds owner-thread
work, so simulation can run slower than wall time. Requested FPS sets video timing;
new images can only be sampled at native control boundaries.

Encoding and disk writes run in a separate process behind a bounded queue.
A slow encoder drops samples without blocking control; playback holds the last
image across missing samples instead of speeding up time. Videos include the
initial state and a final sample, quantized to video frame intervals; even a world
that never advances produces one frame. `frames.jsonl` maps output frame ranges
(exclusive `frame_stop`) to source simulation timestamps. `manifest.json` records resolved views, counts,
drops, completion status and errors; `encoder.log` contains FFmpeg diagnostics.

Submit, stop and normal shutdown finish recording, with a five-second encoder
drain deadline. Use Ctrl-C for a foreground launch, or `roboenv stop RUN_ID` after
`--detach`. Reset or restore marks the recording `interrupted` and disables it for
that run; launch again for a new recording. Encoder or capture errors mark it
`incomplete` without changing the native task verdict. Forced termination may leave
partial videos and an incomplete manifest. Fragmented MP4 retains completed
fragments where possible. Recording files and cameras are not exposed through
agent device discovery or health responses.

## One interface, three engines

```python
from roboenv_runtime.spectator import ViewSpec, create_spectator_renderer

# adapter is an already initialized native adapter, on its owning thread.
renderer = create_spectator_renderer(adapter, [
    ViewSpec("overview", source="third_person", height=720),
    ViewSpec("hand", source="wrist", height=480),  # single-arm MuJoCo backends
])
try:
    frames = renderer.capture()  # {view_name: independent uint8 RGB array}
finally:
    renderer.close()
```

Omitting views selects the available third-person and wrist views, at height 720
with the source aspect ratio. Dual-arm backends use `left_wrist` and `right_wrist`.
BEHAVIOR resolves these roles from mounting metadata, not numbered camera names.
The `third_person` role follows the benchmark's existing overview/head camera;
it does not imply a fixed camera outside the robot.

A view with `pose=(x, y, z, qx, qy, qz, qw)` instead of `source` defines an
additional fixed spectator camera. The pose is world-relative, in metres, with
OpenCV optical axes (+X right, +Y down, +Z forward). `fov_y` is vertical FOV in
degrees and applies only to fixed views. Explicit `width` and `height` select the
render resolution. Defaults for fixed views are 1280×720 and 45° vertical FOV.

MuJoCo uses an independent GL context and framebuffer without resizing the
agent's framebuffer or changing `model.vis`. SAPIEN uses independent cameras and
reads wrist mount poses directly, bypassing task hooks that may randomize lighting.
Isaac uses separate camera prims and render products while sharing the world's
native render graph. Its products remain allocated until world shutdown; do not
repeatedly create renderers in a live world.

Create one renderer after reset and close it before replacing or destroying the
world. Creation, capture and cleanup belong on the world's owner thread. Captures
read live state without advancing physics. Configuration stays fixed for the
episode, and spectator cameras are never added to agent catalogs or discovery.

## Run the native probe

Use existing prepared environments and assets. Select a GPU and run each backend
sequentially, in both clock modes, into fresh private output directories:

```sh
.venv/bin/python tools/check_spectator.py libero/bowl-onto-plate \
  --gpu 0 --clock stepped --seconds 2 --timeout 180 \
  --out .cache/spectator/libero-stepped
```

Other initial cases are `robocasa/pick-place-microwave`,
`robotwin/stack-blocks-three`, `robodojo/stack-bowls`, and
`behavior/turning-on-radio`. RoboDojo fluid and deformable scenes need additional
explicit cases before their coverage can be claimed. `--interpreter` overrides
the registered interpreter. `--views` accepts a JSON list of `ViewSpec` fields
for smaller/larger resolutions or additional fixed views.

The probe uses the real worker's dispatcher and clock, stages a small arm motion,
and interleaves agent camera reads with spectator capture. It compares native
state, RNG state, budgets, verdicts and device descriptors around setup/capture/
cleanup, checks agent image changes against baseline renderer variability, and
retains sample PNGs, capture latency, a run specification and a result receipt.
A native hang is bounded by the parent process timeout. Always inspect
`result.json`; process exit alone is not a passing receipt.

This renderer probe does **not** establish HTTP concurrency, recording throughput,
encoder failure isolation, task success, or fresh-process replay. CPU encoder and
lifecycle checks are in `tests/test_recording.py`. Missing environments and skipped
native checks are not acceptance evidence. Shared native render failures can still
affect the simulator even when encoding runs in a separate process.
