# Simulator command-limit audit

The execution path, not a URDF rating or an action-space declaration, determines
whether a limit applies. Joint and velocity profiles use
`control.limit_handling: native`: shared preflight, backend setters and portable
Kinex helpers validate types, finiteness, dimensions and supported modes, then
leave numeric saturation to the simulator. Physical constraints remain active.

Scalar grippers deliberately retain Kinex's normal **rejection** behavior:
normalized position commands must be in `[0, 1]`; direction commands must be in
`[-1, 1]` and are reduced to their sign. The entire sequence is checked before
any effect. This uses the native normalized range but rejects instead of silently
clipping. Panda also accepts larger finite direction magnitudes natively; keeping
its canonical profile range is an intentional consistency restriction requested
for grippers. No new scalar validation mechanism was added to Kinex.

## Enforced and not enforced

| Integration | Enforced by the API or retained in native execution | Not imposed by RoboEnv / Kinex |
| --- | --- | --- |
| RoboTwin Piper / ALOHA | Gripper profile `[0, 1]` rejection. Actual `Robot.set_gripper` runs each physics substep, retaining normalized clipping, incremental targets and drive-velocity computation. SAPIEN physical joint constraints remain. | No arm target rejection at URDF position ranges. No command speed cap from URDF ratings of 3 / 1000 rad/s, or effort cap from the URDF's 100 N m rating; these unused ratings are now unspecified. |
| SIMPLE G1 | Physical joint constraints and PD torque clipping in the active `g1_sonic` decoupled WBC path. | No added translation ±0.5 m/s, yaw ±1 rad/s, height 0.2–0.74 m, torso roll/pitch ±0.52 rad or yaw ±2.618 rad restrictions. No arm/hand write rejection at model ranges. The metric hook bypasses the separate teleoperation path and its yaw clipping. |
| RoboCasa Panda / Omron | Gripper direction `[-1, 1]` rejection, sign semantics and native internal gripper saturation. Native wheeled-robot control clips actuator inputs after rotation/scaling: base translation ±1 m/s and yaw ±1.5 rad/s. MuJoCo also applies actuator control/force limits; base force ranges are ±600 N / N m. Native arm/torso torque clipping and physical constraints remain. | No independent body-command ±1 box. No absolute arm/torso target rejection at model ranges; the native absolute controller assigns `goal_qpos` directly. Base actuator limits are not guarantees on measured speed. |
| LIBERO Panda | Gripper direction `[-1, 1]` rejection, sign semantics and native internal gripper saturation. Physical joint constraints and native torque clipping remain. | No absolute joint target rejection. The selected absolute-goal path has no `qpos_limits`; model ranges remain geometry. |
| BEHAVIOR R1Pro | Gripper profile `[0, 1]` rejection. Native controllers clip limited joint-position targets, including fingers. Base velocity output clips to native joint limits after coordinate conversion: translation ±1.5 m/s and yaw ±pi rad/s; native base joint max effort is configured to 1000 N / N m. | No extra joint target rejection or body-frame translation ±0.5 / yaw ±1 box. No unverified URDF arm/trunk velocity or effort ratings. Native controller and physics limits still apply; unspecified metadata does not disable them. |
| RoboDojo ARX X5 | Gripper profile `[0, 1]` rejection. Native EvalEnv also clips normalized gripper commands and interpolated finger targets; MetaControl retains its finger-goal increment cap. X5 simulator arm limits of 5 rad/s and 100 N m remain in the actuator configuration and profile facts. Physical joint constraints remain. | No arm target-range rejection or additional speed/effort cap. Only upstream X5 is supported; the local YAM integration and patch are removed. |
| Isaac Lab | Simulator source management, preparation and diagnostics remain. | G1 task/device integration removed. SIMPLE G1 is independent. |

RoboTwin previously used a direct gripper setter that reproduced only `[0, 1]`
saturation. The review replaced it with the upstream setter at 250 Hz. Its default
`gripper_eps=0.1` limits increasing normalized targets by 0.1 per call; decreasing
targets are not symmetrically rate-limited. Native drive velocity is
`clip(new_target - old_target, -1, 1) * 0.05`. This is a drive target, not a measured
velocity guarantee. Explicit stop holds preserve measured finger positions,
including soft-limit overshoot, without ramping peer grippers.

RoboDojo clipping occurs above the robot manager, in EvalEnv's `take_action_batch`
and `process_control_info`. Inspecting only the manager misses it. MetaControl
additionally caps finger goals relative to measured positions to 20% of the
finger span; the adapter still executes that native path.

Model position ranges remain available for FK/IK and planning. Kinex's optional
servo planner retains planning rates; those are not hard device-write caps.
`null` command bounds mean no independent bound is specified in that command
frame, not unlimited physical motion. User-configured strict profiles keep their
existing checks. Existing workspaces contain copied profiles and must be refreshed
and reloaded to pick up the changed facts; restart the service for runtime changes.

## Execution sources inspected

- RoboTwin: [`Robot.set_arm_joints` and `Robot.set_gripper`](../third_party/RoboTwin/envs/robot/robot.py), [evaluation `take_action`](../third_party/RoboTwin/envs/_base_task.py), and installed SAPIEN 3 URDF loader `_parse_articulation`. The loader applies lower/upper position ranges but not these URDF velocity/effort ratings to drives. [Adapter](../runtime/src/roboenv_runtime/adapters/robotwin/devices.py).
- SIMPLE: [metric hook](../third_party/patches/simple-decoupled-control.patch), [active robot PD control](../third_party/SIMPLE/src/simple/robots/g1_sonic.py), and [device control](../runtime/src/roboenv_runtime/adapters/simple/control.py).
- RoboCasa: [absolute joint controller](../third_party/robosuite/robosuite/controllers/parts/generic/joint_pos.py), [base controller](../third_party/robosuite/robosuite/controllers/parts/mobile_base/joint_vel.py), [wheeled-robot actuator clipping](../third_party/robosuite/robosuite/robots/wheeled_robot.py), [Omron actuators](../third_party/robosuite/robosuite/models/assets/bases/omron_mobile_base.xml), and [Panda gripper](../third_party/robosuite/robosuite/models/grippers/panda_gripper.py). The base's computed `scaled_delta` does not replace the action subsequently transformed into `goal_qvel`.
- LIBERO: [absolute-goal binding](../runtime/src/roboenv_runtime/bindings/backends/libero/devices.py), and pinned robosuite 1.4 position-controller and single-arm actuator code in its isolated interpreter.
- BEHAVIOR: pinned [`BaseController.clip_control` and `step`](https://github.com/StanfordVL/BEHAVIOR-1K/blob/e008fcb7b2c52cc3c1209287babedd9018dc3202/OmniGibson/omnigibson/controllers/controller_base.py), [holonomic controller](https://github.com/StanfordVL/BEHAVIOR-1K/blob/e008fcb7b2c52cc3c1209287babedd9018dc3202/OmniGibson/omnigibson/controllers/holonomic_base_joint_controller.py), [robot limit configuration](https://github.com/StanfordVL/BEHAVIOR-1K/blob/e008fcb7b2c52cc3c1209287babedd9018dc3202/OmniGibson/omnigibson/robots/robot.py), and [selected controllers](../runtime/src/roboenv_runtime/adapters/behavior/devices.py).
- RoboDojo: pinned [EvalEnv action processing](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/src/eval_client/eval_env.py), [MetaControl](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env/robot_manager/control_manager.py), [X5 actuator configuration](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env/robot_manager/robot_config/x5.py), and [X5 class](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env/robot_manager/robot_class/x5.py).

## Verification scope

[Regression tests](../tests/test_native_limits.py) check every exported scalar
gripper profile for endpoint acceptance and rejection before effects, and execute
the actual pinned RoboTwin setter through the adapter with CPU test joints.
[Compiled Panda–Omron CPU checks](../tests/robocasa_cpu.py) verify out-of-range
absolute goals are accepted while native actuator clipping remains effective.
Shared preflight, Python helpers and Kinex command tests cover native passthrough
and retained malformed-input rejection.

These checks do not certify benchmark boot, task success or fresh-process GPU
replay. Existing assets, environments, copied workspaces and historical receipts
are preserved. The [new SIMPLE generation receipt](simple-native-limits.provenance.json)
records updated facts; the previous receipt remains historical evidence.
