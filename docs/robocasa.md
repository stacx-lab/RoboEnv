# RoboCasa devices

RoboCasa exposes one initialized task through an HTTP endpoint. Complex is an
agent harness using that service; Harbor owns trial scheduling and lifecycle.
Different trials use independent worlds, budgets, journals, and ports.

## Start and connect

After `roboenv prepare robocasa`, start a catalog task:

```sh
roboenv launch robocasa-pick-place-microwave --port 8123
```

Use another port for a concurrent task,
`--port 0` for automatic assignment, or `--detach` for owner-managed background
execution. `roboenv stop ENDPOINT` stops a locally owned run. `--host` defaults
to `127.0.0.1`; an isolated container service can bind `0.0.0.0`.

```python
import roboenv

env = roboenv.connect("http://127.0.0.1:8123")
print(env.task.instruction)
print(env.devices())
arm = env.device("arm")
print(arm.profile)
state = arm.read()
arm.write([0, 0, 0, 0, 0, 0])  # Set a hold target at the current pose.
arm.stop()
image = env.device("camera_left").read()
print(image["file"])
```

A Python owner uses `with roboenv.launch(task=TASK, port=8123) as run:` and connects
to `run.endpoint`. Closing an attached client releases only its local resources.
Device responses contain their data directly; execution receipts stay in the
service journal. Transport uncertainty raises an exception carrying a request ID
and never automatically resubmits motion.

## Device contracts

The default Panda–Omron catalog contains these devices. Every driven device also
provides `load`, `unload`, and `status`; those operations do not reset the world.

| Device | Operations | Read result / write target |
|---|---|---|
| `arm` | `read`, `write`, `stop`, `ik`, `fk` | Joint `positions`, `velocities`, `faults`, and `flange_pose`; write six normalized native OSC deltas |
| `torso` | `read`, `write`, `stop`, `ik`, `fk` | Lift joint state and `flange_pose`; write one normalized native joint delta |
| `body` | `read`, `write`, `stop` | Planar joint state and world `pose`; write normalized forward, left, yaw-rate commands |
| `gripper` | `read`, `write`, `stop` | Finger joint state; write one signed native command, positive closes and negative opens |
| `camera_left`, `camera_right`, `camera_wrist` | `read` | `file`, `depth_file`, `depth_scale`, `width`, `height`, `channels`; aligned upright RGB-D |
| `world` | Profile only | Root frame and declared space bounds; no state or score shortcut |

All native write components lie in `[-1, 1]`; they are not metre/radian targets.
Profiles record the actual controller scaling, component order, joint limits,
and control interval. The world starts a background clock when the worker is
ready, paced at the native control frequency (normally 20 Hz). Physics continues
while the agent thinks or produces code: unsupported objects fall and existing
controller targets remain active. Reads take consistent snapshots; they do not
request extra simulation steps.

A write updates a device command and returns `accepted: true, completed: false`;
this acknowledges acceptance, not arrival at the target. Arm and torso deltas
are converted to a target once and the native servo keeps pursuing that target.
Base velocity and signed gripper commands persist until replaced or stopped.
Writing one device preserves other devices' goals. Arm writes still use native
OSC deltas; absolute joint/EEF target APIs and a servo tool are separate work.

`stop` holds the arm/torso/gripper or commands zero base velocity. It does not
pause world physics or teleport joints. It is not a physical emergency stop.
Submit atomically freezes the terminal state. Both clock modes disable the
original horizon and legacy `max_steps` limits. Realtime physics continues until
submission or service shutdown (or a simulator failure). `action_frames`
still counts every interval, including thinking time, with `max: null`. There is
no horizon-based success rejection in this setting.

Pacing is best effort, not a hard realtime guarantee: simulation cost and serialized
camera/kinematics requests can make simulated time lag wall time. Physics steps
are never skipped. Replaying only command values without their simulation-frame
timing does not reproduce a realtime episode.

Choose the clock when starting an episode, using `backend_kwargs` in its task YAML:

```yaml
backend_kwargs:
  task_name: PickPlaceCounterToMicrowave
  realtime: false  # false = synchronous; true = realtime (default)
```

Start that task with `roboenv launch /path/to/task.yaml` or
`roboenv.launch(task="/path/to/task.yaml")`. Restart the service after changing the
setting. The live actuator profile reports `intrinsics.clock_mode` as `realtime`
or `stepped` (synchronous); exported Harbor instructions reflect the chosen mode.

In synchronous mode, the world is paused between writes. Each arm, torso, gripper,
or base write runs one native control interval (normally 0.05 simulated seconds)
and then returns `accepted: true, completed: true`. Completion means that interval
finished, not that the arm reached its target. Other parts receive neutral native
commands for that interval. Reads, camera captures, FK/IK and `stop` do not advance
physics. There is no background clock and no horizon limit in this mode either.


`flange_pose` and FK/IK poses use metres and quaternion xyzw in the manipulator's
base frame. FK accepts `joints`; IK accepts `pose` and optional `seed`. Both use
scratch MuJoCo data without changing the live state. IK reports failure when its
bounded numerical search cannot find a solution; it is not a motion planner.
Joint limits are soft physical constraints: reads preserve measured overshoot,
while explicit FK joints and IK seeds are validated against nominal ranges.
Implicit IK seeds are projected into those ranges on scratch data only. See
[joint limits and recovery](robocasa-config.md#arm-joint-limits-and-recovery).
The gripper mounting pose is its grip-site/TCP offset from the flange. Camera
intrinsics and static mounts are resolved from the model. The mobile chassis
(`body`, type `body`, mobility `mobile`) is an independent frame root: its
profile omits `extrinsics.mount` entirely. Torso, arm and cameras remain connected
through their actual mounts without a required path to `world`. `body.read()`
still provides its current simulator world pose; this does not create a mount edge.

Camera files are downloaded to a client-local directory and remain available
until that client is explicitly closed. The thin one-shot driver leaves its
returned image file available to the caller. Artifacts are scoped to the episode;
an old client cannot download an image from a replacement run on the same port.

The three standard camera profiles have type `rgbd`. A read renders RGB and depth
together without stepping physics. Color is PNG; `depth_file` is a 16-bit binary
PGM (P5, big-endian, maxval 65535), aligned to color. Multiply its raw pixels by
`depth_scale` (0.001 m/unit) to obtain optical-axis depth in metres. Zero means
invalid; far-plane/background pixels and values outside the representable
0.001–65.535 m range are invalidated rather than clipped. This scale is the
RoboEnv output encoding, not the native nonlinear depth-buffer scale.

## Shared RoboCasa devices

Standard RoboCasa tasks reuse the checked-in Panda–Omron device configuration.
Changing a task or layout does not rewrite profiles. Scene state is not exposed
to the agent; it must be perceived from the cameras. See [static configuration research](robocasa-config.md).

The general binding mechanism also supports custom task definitions:

A task's `device_bindings` is the complete device selection, not a patch over a
fixed simulator inventory. Bindings can rename devices, choose existing profiles,
and select a camera through `config.camera`. Camera names must also be enabled
in `backend_kwargs.camera_names`. Different tasks may choose different lists.
Only resolved task profiles and drivers are published to the agent.

The service provides no `pick`, `place`, `move`, `invoke`, primitive catalog,
scene RPC, policy runner, or score API to HTTP clients. Perception and control
programs belong to Complex or another harness. Task context contains no object poses, fixture geometry, or layout metadata.
Simulator ground truth remains private to the owner and verifier.

## Harbor

Export uses the same task resolver as local launch. Each trial gets an isolated
Compose service network and `ROBOENV_ENDPOINT=http://sim:8123`; internal ports may
repeat across networks without publishing a conflicting host port. Readiness
includes device binding and publication of live profiles and thin drivers under
`/run/roboenv/robot/devices`. The Complex integration can copy these into its
trial-local robot workspace.

The agent runs as `agent`; the verifier runs as `root` through Harbor's per-phase
user configuration. The owner Unix socket is mode 0600 and owned by the service
user (root in the generated deployment). Agent containers drop capabilities and
prevent privilege escalation. HTTP permits only device access/discovery and
administrative health/request-status reads; it rejects owner and primitive RPCs.
The verifier retains access to the owner socket and read-only execution evidence.
Each service start creates a fresh episode identity, including container restarts.
Harbor stops the environment during trial cleanup, even if the agent fails.

## Validation

`bash tools/check.sh` runs CPU, format, privacy, and packaging checks. When a
prepared RoboCasa interpreter is available, the CPU suite additionally exercises
real Panda–Omron controllers in robosuite's Lift scene: joint reads, one-interval
writes, stop, FK/IK, invalid commands, and unchanged live state during kinematics.
Separate realtime and synchronous CPU workers check paused state between writes,
one interval per synchronous write, advancement beyond the old horizon,
free fall without requests in realtime mode, continued arm motion,
goal preservation across device writes, device stop, and a frozen submission.
Clock tests check unbounded advancement, finite budgets for other settings, failure,
and thread cleanup.
Two real CPU instances also exercise HTTP task/device isolation and image retrieval.
When the sibling Complex checkout is installed, its actual schema and required
driver interfaces validate exported profiles. Set `ROBOENV_ROBOCASA_TEST_PYTHON`
to select the isolated simulator interpreter explicitly.
Camera transport tests use synthetic pixels. These checks do not certify a
RoboCasa kitchen boot, rendered image calibration, task success, or GPU replay.

For a populated Complex workspace, TUI recording, and MP4 playback, follow
[the local test plan](complex-local.md).

The body owns the planar mobile base. The torso remains a separate one-joint
`manipulator` so Complex can resolve its live base-to-flange transform. The arm
and external cameras mount on that flange; the wrist camera mounts on the arm
flange. Merging the lift into a `body` would require named moving frames in
Complex's body interface and frame graph.
