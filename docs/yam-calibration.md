# YAM calibration practice in Isaac Lab

These three open-ended exercises recreate the ABC dual-YAM tabletop hardware in
Isaac Lab, using Isaac Sim / PhysX. They do not run ABC's MuJoCo backend.

| Task ID | Extrinsics initially unknown | Fixture |
|---|---|---|
| `isaaclab/yam-calibrate-overhead-camera` | `camera_top` in world | ChArUco board attached to the left flange |
| `isaaclab/yam-calibrate-wrist-cameras` | Both wrist cameras in their respective flanges | One fixed tabletop ChArUco board |
| `isaaclab/yam-calibrate-dual-arm` | Right base in left base | Two calibrated probes and four noncoplanar contact fiducials |

The wrist task calibrates both cameras in one exercise. Captures may be collected
independently or interleaved; simultaneous arm motion is not required. There are
no separate arm-to-world mount exercises.

## Prepare and launch

Bootstrap RoboEnv and prepare its isolated Isaac interpreter following
[Isaac Lab setup](isaaclab.md). Acquire ABC's source and mesh assets following
[ABC setup](abc.md). Asset preparation below only reads the task-free ABC MJCF
and copies the required meshes; it does not import MuJoCo or start a simulator.

```sh
source .venv/bin/activate
roboenv prepare isaaclab
roboenv prepare abc
python tools/prepare_yam.py
roboenv launch isaaclab/yam-calibrate-wrist-cameras
```

`prepare_yam.py` defaults to ABC's `yam_bimanual_empty.xml` and writes
`.cache/isaaclab/yam/assets.json`. Use `--source` and `--out` for another prepared
source location or a new bundle. Existing bundles are never overwritten. When
using another output location, set the task's `backend_kwargs.asset_manifest`.

The first native launch imports the three prepared MJCF files through Isaac Lab's
MJCF-to-USD converter. USDs are cached under the bundle, keyed by its manifest
hash. Source meshes, generated USDs, and the physical manifest stay outside the
public workspace. Refresh the bundle after changing its source hardware or
profiles. Do not run source preparation inside a live Isaac worker.

## Workspace and calibration

Device names match ABC: `left_arm`, `right_arm`, `left_gripper`, `right_gripper`,
`camera_top`, `camera_left`, `camera_right`, and `world`. Dual-arm practice also
includes read-only `left_contact` and `right_contact` force/torque devices.
Export or copy the selected task's robot bundle using the normal
[workspace preparation flow](kinex-local.md), then connect its drivers to the
running service. Do not copy the fully calibrated ABC robot bundle instead.

Connections, arm kinematics, RGB intrinsics, gripper mounts, and TCPs are complete.
Only the exercise's target mount translations and rotations use `"???"`, with
`needs_calibration` metadata. Other extrinsics remain known prerequisites.
The simulator holds its physical geometry separately. Editing a workspace
extrinsic never changes the simulated hardware pose.

Use the existing camera-extrinsics skill for eye-to-hand or eye-in-hand sampling.
The boards use `DICT_4X4_50`, 7×5 squares of 30 mm, and 22.5 mm markers. For the
overhead task, compose camera-in-left-base with the known left-base world mount.
For wrist cameras, save camera-in-flange for each corresponding arm. Images and
joint readings are sequential observations; stop and settle before pairing them.

For dual-arm alignment, keep the probes' grippers open and touch the same marked
top center on each of A (red), B (green), C (blue), and D (yellow). Move one arm at
a time, park the other, and retract before switching. Probe tips coincide with
the existing calibrated gripper TCPs. Record TCP positions in each arm's own base
frame and solve right-base-in-left-base. Save the right arm's mount using parent
`left_arm`, parent frame `left_arm`.

Contact devices report fixture-filtered normal forces on the probe-bearing
flange, expressed in its gripper TCP axes, in newtons. They do not report contact
positions, fixture coordinates, or the missing base transform. Contact confirms
physical touching, not exact centering; avoid touching fixtures with another part
of the flange. These readings replace operator contact confirmation for practice.

Save calibration profiles and evidence locally. There is no submission device,
calibration evaluator, or automatic completion. Both stepped and realtime modes
have no episode horizon. The existing owner score interface returns a neutral,
incomplete report; it does not assess the calibration. Skill-side fit diagnostics
can still guide the learner. Explicit shutdown and ordinary runtime faults retain
their usual behavior.

## Camera image quality

Calibration cameras render at 1280×960 internally and integrate each 2×2 block
into the public 640×480 RGB image. The public intrinsics and field of view stay
unchanged: rendering scales focal lengths and principal points with the correct
pixel-center offset before downsampling.

These scenes use real-time RTX with full-resolution DLAA and the renderer's
denoiser, with frame generation and motion blur disabled. DLAA still uses temporal
reconstruction, but avoids the low-resolution input of DLSS performance mode,
which visibly smeared small ChArUco cells and gripper edges.
Each camera read now pumps three actual RTX updates after setting its pose,
without advancing physics. Calling `SimulationContext.render()` alone did not
pump the headless RTX renderer in the pinned Isaac Lab version.

An earlier fix used 64-sample offline path tracing. It improved marker edges,
but that global setting also made every spectator frame expensive and slowed
live simulation substantially. Real-time RTX keeps the same supersampling and
public camera geometry without imposing offline path tracing on spectator views.
First-render shader compilation still adds startup latency. Distant boards need
adequate image coverage, and rendering checks do not certify calibration accuracy.
Restart an existing simulator process to apply the new rendering configuration.

In a same-process comparison, advancing 1.02 simulated seconds with ten spectator
frames took 35.0 wall seconds with offline path tracing and 6.4 seconds with
real-time RTX/DLAA. The simulation/control loop alone took about 5.3 seconds.
A fresh process with the current configuration detected 16 markers in each
wrist camera's first image, with spectator captures around 0.1 seconds each.
These are bounded measurements on a shared machine, not a realtime guarantee;
the initial shader compilation and spectator warmup are excluded from those
steady-state timings.

## Implementation and validation boundary

Reusable articulation, camera, control, and asset code lives under
`adapters/isaaclab/robots/yam/`. Separate registered task factories select shared
calibration fixtures under `tasks/yam_calibration/`. Future YAM tasks can reuse the
robot implementation; unrelated robots need not depend on calibration fixtures.
The private binding implementation is `isaaclab.yam/1`.

`tools/generate_yam_calibration.py` deterministically generates the three public
robot bundles, task instructions, and private binding defaults from ABC's shared
hardware profiles. Regenerate them together when changing hardware facts.

The physical setup preserves ABC's normalized 0.76 m arm-base height, RGB camera
mounts and 640×480 pinhole intrinsics. The shared control interval is 34 ms with
17 PhysX steps of 2 ms. Source arm-drive gains, full link gravity compensation,
and opposing gripper coupling are set explicitly. The infinite MuJoCo tabletop
collision plane is represented by a finite tabletop collider in Isaac.
PhysX dynamics are not claimed to reproduce MuJoCo trajectories exactly.

Before initializing PhysX, each imported arm receives the masses, centers of mass,
principal inertias and gravity-compensation flags from the compiled source MJCF.
This preserves inferred inertials and the source compiler's `boundmass` and
`boundinertia`. The pinned USD importer alone omitted those bounds: each empty
wrist camera frame became a 1 kg PhysX fallback instead of the source 0.001 kg.
Both arms consequently sagged under constant position targets. Restoring source
inertials reduced the measured home-pose error from about 0.09 rad to 0.00026 rad,
and maximum residual speed from about 0.021 rad/s to below 0.00018 rad/s.
This correction is applied over cached USD assets at startup; restart the
simulator to apply it.

Constant joint targets are staged once per control interval; actuator writes,
physics steps, articulation updates and contact updates still run at every
2 ms substep.
The earlier target-staging optimization was checked against its preceding loop
with identical joint positions at 30 control boundaries, before the inertial
correction above. It is not a dynamics equivalence claim for the corrected model.

`tests/yam_hold_native.py`, run with the isolated Isaac interpreter and `--out`
pointing to a new private directory, checks imported masses against compiled MJCF,
buffered motion of each arm while its peer holds, retained final targets, settling,
explicit stop scope, and raw PhysX state preservation across camera/spectator
reads. It exercises a bounded wrist-task trajectory, not arbitrary pose reachability.

CPU checks cover catalog resolution, incomplete profiles, geometry separation,
shared-clock behavior, retained gripper preload, and portable asset integrity.
The three prepared MJCF files were compiled on CPU in an isolated MuJoCo
interpreter, and the ChArUco bitmap was checked using OpenCV without rendering.
Bounded GPU spectator probes now cover native USD import, device-camera reads,
small arm motion, and third-person framing of all three fixture layouts. They
also compare state, RNG, budgets, camera configuration, and device images across
spectator setup, capture, and shutdown; see [spectator validation](spectator.md).
The pinned Isaac Lab runtime uses XYZW poses; the native implementation and
contact-reporter setup follow that API.

The overhead exercise mounts the board with its printed face along flange -Z.
The previous +Z mounting exposed the blank backing to the head camera at home.
A fresh rendered-image check detects three markers at home after the correction;
the oblique starting view still requires diverse, board-visible calibration poses.

### Final comparison

A fresh process replayed a saved wrist-calibration session's 100 device calls,
including 13 arm moves, over 79.526 simulation seconds. The original cancelled
wait was reconstructed from its observed simulation endpoint; explicit stops were
preserved. Recording completed with 796 frames and zero dropped samples. Camera
reads and spectator captures produced no raw PhysX joint-position or velocity
changes. Runtime was 472.8 seconds, excluding startup: about 5.94 wall seconds per
simulation second on the validation machine, still slower than realtime.

At matching recorded arm reads, median maximum joint speed fell from 0.0192 to
0.000718 rad/s; maximum speed fell from 0.0431 to 0.00316 rad/s. The strict replay
settling gate **did not pass**: some samples exceeded 0.002 rad target error or
0.003 rad/s speed. Maximum target error was 0.00456 rad (0.26 degrees), concentrated
in joint 1 after larger moves. Queue completion alone does not certify settling.
These receipts are retained as failures rather than relaxing their thresholds.
In a separate fresh-process hold initialized from the replay's final joint
positions, velocities and targets, right-arm error fell from 0.00445 rad to below
0.002 rad after 6.12 additional simulation seconds. This supports slow settling
at that pose; it is not an exact native-checkpoint continuation or a guarantee
for every pose. Controller gains were left unchanged.

Both wrist cameras detect 16 markers at the initial pose. Detection varies from
zero to 17 across the saved commands; some original poses also had no detected
markers. This is a command replay using poses planned against the earlier model,
not a rerun of the calibration agent against corrected feedback.

Probe reachability, contact stability under deliberate touching, and
calibration-agent execution remain unverified. No calibration accuracy or completed
calibration claim is made.
