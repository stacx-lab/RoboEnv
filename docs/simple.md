# SIMPLE humanoid devices

SIMPLE is an experimental RoboEnv backend for G1 locomotion and manipulation.
It uses SIMPLE's decoupled teleop task family with an in-process GR00T WholeBodyControl
policy. MuJoCo owns physics and native task evaluation; Isaac Sim supplies the default
RGB images. `sim_mode: mujoco` is a diagnostic rendering option.

## Preparation and launch

```sh
roboenv prepare simple
roboenv tasks list --simulator simple
roboenv launch simple/g1-xmove-pick
roboenv launch simple/g1-transfer-between-tables --realtime
```

Preparation installs the pinned Python 3.10 recipe into a separate interpreter and
acquires checksum-verified public assets, about 4.3 GB. The default image path also
requires Isaac Sim 4.5, a compatible NVIDIA GPU/driver and Vulkan. Accept NVIDIA's
Omniverse license before setting `OMNI_KIT_ACCEPT_EULA=YES` in the launch environment.
RoboEnv does not set license acceptance automatically. To use MuJoCo images, copy a
representative `task.yaml` and change `backend_kwargs.sim_mode` to `mujoco`, then launch
that file with this checkout's catalog. Launch verifies source patches, extracted
asset inventories, robot geometry and camera facts; it does not download assets.

The three checked-in examples select:

| RoboEnv task | Native task | Synchronous horizon |
| --- | --- | --- |
| `simple/g1-xmove-pick` | `G1WholebodyXMovePickTeleop-v0` | 800 frames / 16 s |
| `simple/g1-transfer-between-tables` | `G1WholebodyLocomotionPickBetweenTablesTeleop-v0` | 1200 frames / 24 s |
| `simple/g1-bend-pick` | `G1WholebodyBendPickTeleop-v0` | 800 frames / 16 s |

Discovery reads all 12 decoupled G1 teleop registrations from the pinned upstream
without importing the simulator. Only the three representative examples live in Git.
The pinned asset set covers these three examples; other discovered tasks may require
additional task-specific assets before native launch.
Domain randomization defaults to upstream level 0; levels 1 and 2 can be selected in
the task definition. Instructions and success predicates remain native SIMPLE.

## Control contract

The eight devices are `base`, two arms, two seven-joint hands, two head RGB
cameras and `world`. The world is a coordinate frame with no driver.

```python
import roboenv

with roboenv.connect() as env:
    env.device("base").write(
        mode="locomotion",
        velocity={"vx": 0.15, "vy": 0.0, "wz": 0.1},
        posture={"height": 0.72, "roll": 0.0, "pitch": 0.05, "yaw": 0.0},
    )
```

Base velocities are pelvis-frame metres/second and yaw radians/second; height is a
pelvis-height reference in metres. Bounds are ±0.5 m/s, ±1 rad/s and 0.2–0.74 m.
Posture also requires torso roll, pitch and yaw references in radians, in WBC's
heading-aligned frame. Torso yaw is relative to that heading; `velocity.wz` commands
turning. The exported profile declares all posture defaults and bounds. Orientation
bounds use native waist joint limits as input bounds, not guaranteed reachable torso
orientations. These are policy references, not guaranteed measured motion.
A managed WBC hook sends metric yaw rate directly into the native policy command and
observation history. SIMPLE's original teleop navigation vector instead carries a
turn flag and heading target; forwarding `[vx, vy, wz]` to that interface would give
the wrong semantics. No joystick, DDS transport, external WBC process or policy
server is required for this backend.

WBC owns all leg and waist targets. There is no separate writable waist device:
base posture feeds WBC's torso-orientation input directly. Arms and hands accept
absolute joint targets in their exported native order. Their native interpolation
supplies the upper targets; lower-body inference overwrites only leg and waist slots
in the combined command. Kinex solves arm FK/IK locally from the portable geometry.
Arms and head cameras mount to the measured `base:torso` frame. Base reads expose
`frame_poses.torso` (torso pose relative to pelvis) and `waist_positions` (named native
joint angles in radians), so arm frame transforms follow actual waist motion.
Hand feedback contains joint positions and velocities; unsupported
force and object-detection fields are null. No grasp confirmation is synthesized.

One synchronous write advances four 5 ms physics substeps (20 ms). Coordinated
`base.batch_write` validates every component before applying any target and advances
once. Other goals remain latched. Realtime writes latch goals; the existing worker
scheduler steps at a best-effort 50 Hz on the native owner thread. Realtime has no
frame horizon and zeroes base velocity after 0.5 s without refresh, retaining posture.
Native success/termination remains active. Reads, local geometry, device loading
and stop do not request physics. Stop holds a component; stopping base zeroes velocity
and retains height, roll, pitch and yaw references. Native standing initialization has a bounded settling
phase outside the episode budget and is included in private replay identity.
Each episode uses a fresh worker; a second in-process reset is rejected.

## Kinex and evidence

Prepare a new workspace with the matching clock:

```sh
python tools/prepare_kinex_workspace.py simple/g1-xmove-pick --out PATH
# Add --realtime when launching realtime.
```

The exported drivers use Kinex's ordinary lifecycle, local manipulator geometry,
normal Python tools, synchronized device operations and recording. The existing
`integrations/kinex/g1-motion` example also works with this rig. It demonstrates
composition of base references and arm IK; it is not a task solver.

Native state, task restoration, controller history, scoring and command records
remain owner-only. Success is cached from stepping: reads and scoring never call
stateful native reward predicates again. Fresh-process replay reconstructs the initial
native task, repeats settling, applies accepted writes/stops at their recorded physics
boundaries, and compares full MuJoCo integration state, WBC history and native verdict.
It never restores the final state.

Executable checks, run with the prepared simulator interpreter:

```sh
python tests/simple_native.py --out NEW_NATIVE_DIR
python tools/replay_simple_devices.py NEW_NATIVE_DIR/record.json --out NEW_REPLAY.json
python tests/simple_kinex.py --out NEW_KINEX_DIR --kinex PATH_TO_KINEX
# Both probe helpers accept --realtime and --sim-mode mujoco_isaac.
```

`tools/accept_simple.py --python SIMULATOR_PYTHON --out NEW_DIR` runs all six
native task/clock cases, their independent replays, tampered-record rejection,
the same six Kinex cases and a public-service watchdog check. Use
`--native-only` for native checks, `--tasks xmove` for a subset, and `--jobs 2`
when resources permit. Its child processes clear inherited Python search paths.

These are no-model infrastructure checks. They do not attempt benchmark task success.
The metric device interface, local IK, initialization and realtime horizon are an
adapted practice protocol, `simple-g1-metric-wbc/2`, not an official SIMPLE baseline.
Version 2 folds waist control into base posture; version 1 command records and
workspaces with a separate waist device must be regenerated for this interface.
Live model evaluation, task-performance trials and Harbor deployment certification
are separate work.

## Reproducibility

`upstreams.lock.json` pins SIMPLE and the three required nested controller/SDK sources;
managed patches have checksums. `runtime/locks/simple-resolved.txt` pins dependencies; `runtime/locks/simple.in`
records its direct inputs.
`src/roboenv/assets/simple-g1.json` pins the Hugging Face asset revision, archive hashes
and extracted inventories. `tools/generate_simple_config.py` reads the pinned MJCF and
camera constants to regenerate the complete catalog and runtime facts; provenance is
in `docs/simple-config.provenance.json`.

The source patches expose metric WBC commands and a simulation clock, skip DDS for
in-process control, make unrelated CuRobo/OpenPI integrations optional, fix the split
WBC package metadata, and route assets through an offline verified depot. Native
reward predicates, task definitions and PD/ONNX controller weights are preserved.
