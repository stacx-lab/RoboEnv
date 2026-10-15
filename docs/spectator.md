# Spectator recording (experimental)

```sh
roboenv launch robotwin/handover-block-aloha-agilex --record
roboenv launch robotwin/handover-block-aloha-agilex --record ./recordings/episode-1
roboenv launch robotwin/handover-block-aloha-agilex --record --record-fps 5 --record-views views.json
```

Recording saves one H.264 MP4 per view under the printed owner-only recording
directory (`evidence/recording/` inside the run by default). `--record DIR` writes
the videos and metadata directly into a new or empty directory; relative paths
are resolved from your current directory. Existing recordings are never overwritten.
FFmpeg with `libx264` must be on
PATH; it is optional when recording is disabled. Python owners use
`roboenv.launch(task, recording={})`, or pass `recording={"fps": 5, "views": [...]}`.
Use `recording={"directory": "./recordings/episode-1"}` to select the output folder.
Recording is experimental on supported backends. LIBERO, RoboCasa and RoboTwin
have native spectator isolation checks; RoboDojo and BEHAVIOR remain unverified.
SIMPLE's MuJoCo checks cover all three example tasks in both clock modes.
Its stepped Isaac XMove frame/state checks pass, but a standalone Kit shutdown crash
prevents a complete native isolation certificate. The normal Isaac XMove CLI
recording and owner-stop path passes in both clocks; neither establishes task success.
There is no browser preview or offline replay fallback.

Defaults are **10 simulation-time FPS**, height **720**, and the source camera's
aspect ratio. LIBERO and RoboCasa select `third_person` and `wrist`; RoboTwin
and RoboDojo select `third_person`, `left_wrist` and `right_wrist`. BEHAVIOR
selects `left_wrist`, `right_wrist` and `overview`: an external 65-degree view
following the base position with a world-upright horizon. Read-only scene rays
shorten the camera distance at walls and ceilings and select another bearing
when it offers at least 0.5 m more clearance. Clutter can still obscure objects;
the wrist views retain manipulation detail. This replaces only BEHAVIOR's
default trunk recording view. Explicit `source: "third_person"` still selects
its trunk camera; agent devices and other backends retain their original views.
These are separate spectator cameras; changing their resolution does not resize
agent observations. SIMPLE instead records the existing `camera_head_left` and
`camera_head_right` frames at their native 640×360 resolution in both
`mujoco_isaac` and diagnostic `mujoco` modes. Its views therefore have exactly
the device cameras' positions, orientations and optical parameters. Explicit
output sizes resample those frames on the CPU, without adding scene detail.
SIMPLE does not support fixed-pose spectator views. For example, `views.json`
for RoboTwin can contain:

```json
[
  {"name": "overview", "source": "third_person", "height": 720},
  {"name": "hand", "source": "left_wrist", "width": 480, "height": 480}
]
```

For SIMPLE, record both head views or select just the left view:

```sh
roboenv launch simple/xmove-pick --record
```

```json
[
  {"name": "left", "source": "camera_head_left", "height": 360}
]
```

Video dimensions must be even. Views and FPS stay fixed for the episode.

Every sample received by the encoder is represented in the video, including an
off-grid terminal capture that shares a frame interval with the previous sample.
This can add one real sample frame to the constant-rate video; no artificial
terminal hold is added. `frames.jsonl` records the authoritative simulation sample
times and their encoded frame ranges. `represented_samples` counts distinct
encoded samples, whereas `encoded_frames` also includes holds across sampling gaps.

In stepped mode, sampling applies up to two seconds of encoder backpressure:
physics stays frozen while the bounded queue drains. An encoder that does not
recover marks recording incomplete instead of silently losing a sample. Realtime
sampling remains nonblocking and reports dropped samples in the manifest. Smooth
video timestamps alone do not prove that every source sample was captured.
Unknown views, invalid settings and unavailable encoders fail requested recording
startup before the service becomes ready.

ABC uses its selected camera backend for spectator recording; see
[ABC rendering setup](abc.md#blender-rendering).

## Isaac Lab views

All three Go2–D1 tasks and all three YAM calibration exercises support `--record`.
They default to one dedicated `third_person` camera at 1280×720, with a 50°
vertical field of view and 10 simulation-time FPS:

```sh
roboenv launch isaaclab/yam-calibrate-wrist-cameras --record
roboenv launch isaaclab/go2-d1-control --record
```

YAM looks through the enclosure's open front from `(-1.1, -0.5, 1.85)` toward
`(0.5, 0, 1.05)` in world metres. This keeps both arms and the tabletop in view
without looking through the opaque back or side walls. Go2 control follows the
base's XY translation with a level horizon, from offset `(1.6, -2.2, 1.4)` toward
`(0.3, 0, 0.4)`. It does not rotate with the base. Reach and pick-and-place use
fixed oblique workspace views, starting respectively at `(2.7, -3.2, 2.1)` and
`(3.6, -4, 2.6)`, looking toward `(0.65, 0, 0.45)` and `(1.2, 0, 0.45)`.
These views back away as needed to fit the task's configured interaction regions
inside a 10% image-edge margin.

Custom fixed-pose `ViewSpec` entries are supported through `--record-views`.
Their `pose` is world XYZ plus XYZW quaternion with OpenCV optical axes (right,
down, forward). Spectator cameras have separate USD prims and render products;
they are absent from the public device catalog and do not supply calibration
measurements. Capture calls render without advancing physics. One renderer owns
all views for an episode, and its native products stay alive until world shutdown.

### Isaac Lab validation limits

The matrix below records the initial spectator validation with the original
YAM renderer. YAM calibration cameras subsequently switched to supersampling,
first with offline path tracing and then with real-time RTX/DLAA for live-session
performance; see [camera image quality](yam-calibration.md#camera-image-quality).
Historical recordings and receipts retain their original renderer settings.

The intermediate offline-path-tracing configuration passed a stepped wrist-task
state/image-isolation and motion probe, but its spectator captures averaged 7.6
seconds on the shared validation GPU. That configuration has been replaced.
The full task/clock matrix and CLI recordings have not been repeated with the
current renderer.

After the real-time RTX/DLAA change, a stepped wrist-calibration probe passed
state and device-image isolation, produced measured arm motion, and exited cleanly.
Seven 1280×720 spectator captures average 0.102 seconds (maximum 0.109 seconds).
The images pass the existing variability tolerance; temporal rendering does not
promise pixel-identical output. See the YAM camera documentation for whole-loop
timings and the remaining gap from wall-clock realtime.
That receipt predates the YAM inertial correction. The corrected model has separate
buffer/hold and raw PhysX state checks in `tests/yam_hold_native.py`; the full
spectator image-isolation matrix has not been repeated for it.

The final review added corrected-model probes for all three YAM layouts:
overhead and dual-arm in stepped mode, and wrist calibration in realtime mode.
All passed state/image isolation and small-motion checks with clean native exits;
capture times averaged 0.105–0.110 seconds. The overhead retest also includes the
corrected board orientation. The latest wrist and overhead checks inspect raw
PhysX joint and root state in addition to cached sensor buffers. Resolved view
metadata now reports the actual 50-degree vertical field of view; this metadata
correction does not change the lens or relabel historical recordings.

A separate 79.526-second recorded-command replay produced 796 frames with no
drops and no render-induced raw joint-state changes. Its strict arm-settling
gate failed at some poses; see [final comparison](yam-calibration.md#final-comparison).
A fresh Go2 control probe again preserved state and completed motion, but failed
wrist-RGB isolation (mean absolute change 3.15 versus a 2.81 tolerance). The
known Go2 image-isolation limitation remains.

Bounded GPU probes exercise all six tasks in stepped and realtime modes with
small arm motions. All 12 state/lifecycle checks pass with clean native exits. All six YAM cases
pass the device-image checks. Go2 pick-and-place passes in the final two runs;
control and reach exceed the wrist-RGB threshold in both clocks (8/12 image
checks pass overall).
Go2 preserves physical state, RNG, budgets, device inventory, camera poses and
intrinsics, and depth images, but its wrist RGB sometimes exceeds the probe's
image-change tolerance. **Go2 recording is not image-isolation certified.**
The shared RTX graph can change rendered appearance even when the camera and
physics state are unchanged; do not assume recorded and unrecorded RGB are
interchangeable for pixel-sensitive experiments.

The probe warms device cameras and compares equal render-update gaps before and
after spectator creation, keeping the existing baseline MAE + 1 threshold.
Image failures remain failures in its receipt; motion and state checks continue
so their results can be reported separately. Native products receive a bounded
64-update warmup at startup, which adds startup latency without advancing physics.
The final render path uses the pinned Isaac Lab tiled-product pipeline.

Four normal CLI recordings (Go2 and YAM, each in both clocks) complete with zero
dropped samples and decodable MP4s, using both the overview and a custom
640×360 fixed view. These are framing, lifecycle, and recording checks;
they do not certify task success, calibration accuracy, or trajectory replay.
If CUDA reports that its temporary-directory name is too long, launch with a
short `--run-dir` path; the private CLI probes used a short temporary run root.

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
`--detach`. Reset marks the recording `interrupted` and disables it for
that run; launch again for a new recording. Encoder or capture errors mark it
`incomplete` without changing the native task verdict. Forced termination may leave
partial videos and an incomplete manifest. Fragmented MP4 retains completed
fragments where possible. Recording files and cameras are not exposed through
agent device discovery or health responses.

## Shared recording interface

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
with the source aspect ratio, or both native 640×360 head-camera views for SIMPLE.
RoboTwin, RoboDojo, and BEHAVIOR also use `left_wrist` and `right_wrist`;
Isaac Lab defaults to its dedicated `third_person` overview only.
BEHAVIOR resolves these roles from mounting metadata, not numbered camera names.
On backends other than Isaac Lab, the `third_person` role follows the benchmark's
existing overview/head camera; it does not imply a fixed camera outside the robot.

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
SIMPLE reads its existing Isaac sensor buffers without another render update.
Independent high-resolution Isaac products were excluded because native probes
found that their initialization changed the agent images through the shared
renderer. MuJoCo mode reads its existing camera renderer at its native resolution.

Create one renderer after reset and close it before replacing or destroying the
world. Creation, capture and cleanup belong on the world's owner thread. Captures
read live state without advancing physics. Configuration stays fixed for the
episode, and spectator cameras are never added to agent catalogs or discovery.

## Run the native probe

Use existing prepared environments and assets. Select a GPU and run each backend
sequentially, in both clock modes, into fresh private output directories:

```sh
.venv/bin/python tools/check_spectator.py robotwin/handover-block-aloha-agilex \
  --clock stepped --seconds 2 --timeout 180 \
  --out .cache/spectator/robotwin-stepped
```

Other initial cases are `robocasa/filter-microwavable-item`,
`robotwin/stack-blocks-three-aloha-agilex`, `robodojo/stack-bowls`, and
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
`result.json`; process exit alone is not a passing receipt. The receipt distinguishes
`isolation_passed` from `native_exit_code`; overall `passed` requires both the
isolation checks and a clean native exit.

SIMPLE uses the same probe, with `--sim-mode mujoco_isaac` (the task default) or
`--sim-mode mujoco`. For example:

```sh
.venv/bin/python tools/check_spectator.py simple/xmove-pick \
  --clock stepped --sim-mode mujoco_isaac --seconds 2 \
  --out .cache/spectator/simple-xmove-isaac-stepped
```

Its probe forces fresh stereo reads and compares controller state, native counters
and camera calibration as well as physics state. Capture latency is reported
separately from configured video FPS. GPU selection follows the normal SIMPLE
launch path's CUDA/Vulkan identity checks.

This renderer probe does **not** establish HTTP concurrency, recording throughput,
encoder failure isolation, task success, or fresh-process replay. CPU encoder and
lifecycle checks are in `tests/test_recording.py`. Missing environments and skipped
native checks are not acceptance evidence. Shared native render failures can still
affect the simulator even when encoding runs in a separate process.

ABC uses the shared MuJoCo spectator renderer with `third_person` (top),
`wrist` (left), and `right_wrist` views, independent of agent RGB cameras.
