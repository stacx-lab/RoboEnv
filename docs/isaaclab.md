# Isaac Lab: Unitree Go2 (experimental)

Isaac Lab is not officially supported by RoboEnv. This integration is an
experimental prototype; the setup and device descriptions below do not establish
simulator validation or benchmark support. The Go2 catalog uses Kinex's `legged`
device contract for `body`, with device-owned multi-chain FK/IK and separate
RGB-D, IMU and foot-contact devices. CPU schema checks do not establish native
physics, walking or balance performance.

RoboEnv exposes one free-base Go2 as a device service. Kinex commands absolute
joint positions through native DC motor control. This is a posture demonstration,
not a walking policy or an upstream locomotion benchmark.

## Prepare and launch

```sh
roboenv prepare isaaclab
roboenv doctor isaaclab
roboenv launch isaaclab/go2-posture --sync
```

Launch streams simulator logs to the terminal and also saves them under the run's
`evidence/logs/worker.log`. The IsaacLab adapter sets `OMNI_KIT_ACCEPT_EULA=YES`
before starting Kit, accepting NVIDIA's Omniverse license agreement without an
interactive prompt.

The source revision and package versions are pinned in
[upstreams.lock.json](../upstreams.lock.json) and
[runtime/environments.json](../runtime/environments.json). Simulation packages
live in an isolated interpreter. Installation requires the NVIDIA package index;
Go2 USD assets are downloaded and checksum-verified separately. Source discovery
and SDK imports do not start Kit.

Use an RTX-capable GPU and compatible Vulkan driver. NVIDIA lists H100/A100 as
unsupported for Isaac Sim rendering. CPU catalog checks do not validate native
physics, rendered sensors, or task success. Asset and binary license terms remain
those of their upstream providers.

Go2 assets live under `.cache/isaaclab/assets`; `ROBOENV_ISAACLAB_ASSETS` may select
an existing deployment depot. Preparation refuses changed existing assets. Launch
verifies their hashes and never downloads assets or rewrites the catalog.

## Devices and timing

| Device | Interface |
|---|---|
| `body` | Type `legged`; twelve measured joint positions/velocities; absolute joint targets in radians; named foot poses relative to the base |
| `camera_front` | Synthetic forward RGB-D, 256×256, 60° FOV |
| `imu` | Sensor-frame angular velocity (rad/s), proper acceleration (m/s², +g at rest) |
| `contact_fl_foot`, `contact_fr_foot`, `contact_rl_foot`, `contact_rr_foot` | Net foot-frame contact force (N), type `force_torque`; no invented torque channels |
| `world` | Reference profile only; no implicit body-to-world localization |

The body joint order is FL, FR, RL, RR, with hip, thigh, calf for each leg.
A write replaces the persistent targets and advances one 20 ms control interval
(four 5 ms physics steps). Acceptance is not arrival. Repeated targets continue
motion. Stop requests a hold at measured joints without advancing time; it does
not promise balance. Read, load, unload, status, and recording never advance time.
Realtime is not implemented. Native startup primes inertial/contact sensors for
one interval before episode time starts at zero.

Reads include `sample_time_s`. RGB and optical-axis metric depth are rendered
only when requested for a new simulation frame. Depth uses the shared 16-bit PGM
millimetre encoding: zero means invalid. The profile declares the depth scale;
reads report both image dimensions and paired acquisition timestamps. Downloaded
image paths survive a one-shot
driver's exit. Camera and contact configurations describe synthetic sensors, not
claims of matching Unitree's physical camera or force sensors.

All native poses in this pinned Isaac Lab version are xyzw. The camera uses
OpenCV optical axes. Named frames are advertised in `extrinsics.frames` and read
through `frame_poses`, relative to the canonical device frame. Sensors mount on
the corresponding `parent_frame`; world/object poses are owner-only.

## Catalog and demonstration

The profile generator reads the actual USD joint limits and IMU mount on CPU,
and reads initialization defaults from the pinned upstream configuration AST.
It extracts body-to-foot chains from USD physics joint frames and axes, including fixed joints. The generated multi_chain model owns the ordered joint definitions; there is no intrinsics.joints field. It does not approximate the simulated model with a different URDF. Its explicit
camera configuration is shared with runtime. Native initialization checks the
joint layout, limits, and home configuration against generated facts.

`adapters/isaaclab/profiles.py` supplies both catalog generation and binding
validation. Bindings verify operational fields against these shared facts while
allowing sensor calibration metadata and bias to change. Native observation
conversion lives in `observations.py`; the backend owns stepping and frame caching.

```sh
# A separate CPU interpreter needs usd-core==25.11, numpy, and PyYAML.
python tools/generate_isaaclab_config.py --out .cache/go2-catalog
```

To update checked-in facts intentionally, omit `--out` and supply
`--facts-out runtime/src/roboenv_runtime/adapters/isaaclab/go2.json` and
`--provenance docs/isaaclab-config.provenance.json`.

Copy `catalog/robots/isaaclab/isaaclab-go2/workspace.yaml` and its `devices/` directory into
Kinex's `.kinex/` workspace. Device profiles default to
`access.connection.url: http://127.0.0.1:8123`; edit that URL for a different server.
An explicit `connection.json` takes precedence, followed by `ROBOENV_ENDPOINT`
or `ROBOENV_SOCKET`, then the profile connection URL. Catalog profiles are
registered unloaded; use `/load` in Kinex to activate the devices you need.
Keep each copied `_roboenv.py` support file beside its driver. IMU and contact
devices share the same read-only sensor driver. Reading the YAML profile requires
PyYAML in the driver interpreter. Bindings stay in RoboEnv; the copied drivers
need no simulator or RoboEnv package.

`isaaclab/go2-posture` publishes a slightly crouched target derived from the
initial stance. Success requires every joint to remain within 0.05 rad for 25
consecutive control intervals (0.5 simulated seconds). The budget is 500 intervals.
Only the owner/verifier sees the verdict. A caller can repeatedly write the same
target to allow settling. Completion is evaluated at submission, not latched forever.

Deployment uses the existing simulator-image builder and Harbor export path.
Build a real image and deployment manifest before exporting; no image digest or
GPU validation receipt is fabricated. On an RTX host, set
`ROBOENV_ISAACLAB_TEST_PYTHON` to the prepared interpreter to run the native smoke
check. CPU tests and rendered replay are reported separately.

## Controller and arm composition

The device accepts joint positions, not velocity or walking commands. Exported Kinex drivers also provide local `fk({joints}) -> {pose}` and `ik({pose,seed?}) -> {joints}` without stepping or changing native targets; the public environment service continues to expose its basic actions only.

Legged `pose` is a complete array in FL/FR/RL/RR order. Every entry is
`{position:[x,y,z],orientation:null|[qx,qy,qz,qw]}` relative to body. Null orientation
requests position-only IK. All feet are required; FK returns full orientations.
IK uses measured joints once when seed is omitted and verifies its result with FK.
The profile declares `control.speed:null`; speedFraction is unsupported.

`servo_to({device:"body",pose:[...]})` passes the target to this driver's IK, then
runs the same joint-arrival loop used for arms. Its only alternative input is
`joints`; there is no feet option. Convert frames and TCP offsets before calling;
frame/end_effector and Cartesian-tolerance tool parameters are not supported.
Joint arrival does not imply foot clearance, fixed body attitude or balance.

The model extraction receipt is `isaaclab-kinematics.provenance.json`; it is CPU
evidence, not native validation. The native test separately compares FK against
measured body-relative feet and exercises the exported driver with servo_to.
 Controllers
own policy inference and support behavior. `stop` holds measured joint positions;
it does not certify support while an arm operates. An independently controlled
arm mounts to `body` through its own calibrated `extrinsics.mount`; the body has
no static world mount. Body-relative camera/arm transforms need no world pose.
World-relative manipulation requires localization, which this prototype does not
expose. A physical arm, payload-adapted controller and coordinated simulation
scheduling are separate work; concurrent device writes are not atomic steps.
