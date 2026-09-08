# Isaac Lab: Unitree Go2 (experimental)

Isaac Lab is not officially supported by RoboEnv. This integration is an
experimental prototype; the setup and device descriptions below do not establish
simulator validation or benchmark support. The Go2 catalog still uses the legacy
`body` device type, which the current Complex schema rejects; full catalog
compatibility is not validated by the passing sensor and binding tests.

RoboEnv exposes one free-base Go2 as a device service. Complex commands absolute
joint positions through native DC motor control. This is a posture demonstration,
not a walking policy or an upstream locomotion benchmark. G1 is deferred in
[plan.md](../plan.md).

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

The source lock pins Isaac Lab's current default branch at
`99f1423e5d4a26216c0eedeb2aa78099a8c3a7d1` (3.0 beta2), paired with Isaac Sim
6.0.1.0 and Python 3.12. Simulation packages live in an isolated interpreter.
A checksum-tracked source metadata patch relaxes Isaac Lab’s test-only coverage
pin to accept the version required by Isaac Sim. The complete dependency lock
and editable-package installation recipe have been resolved together in a dry run;
that is not a completed simulator installation.
Installation requires the NVIDIA package index; Go2 USD assets are downloaded and
checksum-verified separately. No source discovery or SDK import starts Kit.

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
| `body` | Twelve measured joint positions/velocities; absolute joint targets in radians; named foot poses relative to the base |
| `camera_front` | Synthetic forward RGB-D, 256×256, 60° FOV |
| `imu` | Sensor-frame angular velocity (rad/s), proper acceleration (m/s², +g at rest) |
| `contact_fl_foot`, `contact_fr_foot`, `contact_rl_foot`, `contact_rr_foot` | Net foot-frame contact force (N), type `other`; no invented torque channels |
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
millimetre encoding: zero means invalid. Downloaded image paths survive a one-shot
driver's exit. Camera and contact configurations describe synthetic sensors, not
claims of matching Unitree's physical camera or force sensors.

All native poses in this pinned Isaac Lab version are xyzw. The camera uses
OpenCV optical axes. Named frames are advertised in `extrinsics.frames` and read
through `frame_poses`, relative to the canonical device frame. Sensors mount on
the corresponding `parent_frame`; world/object poses are owner-only.

## Catalog and demonstration

The profile generator reads the actual USD joint limits and IMU mount on CPU,
and reads initialization defaults from the pinned upstream configuration AST.
It does not approximate the simulated model with a different URDF. Its explicit
camera configuration is shared with runtime. Native initialization checks the
joint layout, limits, and home configuration against generated facts.

```sh
# A separate CPU interpreter needs usd-core==25.11, numpy, and PyYAML.
python tools/generate_isaaclab_config.py --out .cache/go2-catalog
```

To update checked-in facts intentionally, omit `--out` and supply
`--facts-out runtime/src/roboenv_runtime/adapters/isaaclab/go2.json` and
`--provenance docs/isaaclab-config.provenance.json`.

Copy `catalog/robots/isaaclab-go2/workspace.yaml` and its `devices/` directory into
Complex's `.complex/` workspace. Device profiles default to
`access.url: http://127.0.0.1:8123`; edit that URL for a different server.
An explicit `connection.json` takes precedence, followed by `ROBOENV_ENDPOINT`
or `ROBOENV_SOCKET`, then the profile URL. Reading the YAML profile requires
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
