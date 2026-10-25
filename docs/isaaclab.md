# Isaac Lab

Isaac Lab source management, isolated installation and diagnostics remain available:

```sh
roboenv sources sync --backends isaaclab
roboenv prepare isaaclab
roboenv doctor isaaclab
```

The experimental [Go2–D1 environment](go2-d1.md) provides a bundled task and
device adapter. The [dual-YAM calibration exercises](yam-calibration.md) add
overhead-camera, combined wrist-camera, and dual-arm alignment practice using
ABC-derived tabletop hardware. The former G1 practice task,
profiles, bindings, runtime and task-specific asset tools have been removed.
SIMPLE's G1 integration is separate and remains supported.

The [Booster T2 integration](booster-t2.md) adds whole-body locomotion, two arms,
Revo2 hands and a head RGB-D camera through the same owner, bindings and clock.

Standalone Isaac Lab remains pinned to `99f1423`, with its existing resolved
dependency lock, coverage compatibility patch and opt-in CloudXR patch. The
proposed update to `998bf88` was reverted because its dependencies conflict with
Isaac Sim 6.1 package metadata. `sources check-updates --backends isaaclab` will
therefore report a newer upstream revision; this pin is intentionally retained.

The runtime recipe omits the optional `isaaclab_teleop` package: these device-driven
tasks do not use it, and its current dependency constraints conflict with the
resolved runtime lock. Core Isaac Lab, PhysX, assets, and task packages remain
in the isolated runtime.

Existing environments, cached assets and user workspaces are preserved.
See [setup](setup.md) for installation requirements.

## RoboBallet eight-arm reaching

`isaaclab/roboballet-eight-arm` exposes eight Panda arms, two RGB-D cameras,
`world`, and a read-only `workcell` device. It uses the author's original MuJoCo
kinematics and collision implementation with an Isaac Lab visual scene. Isaac
rendering does not advance another physics world. No learned policy, assignment,
IK solution, or reference trajectory is supplied to the agent.

Prepare the runtime and the separately pinned source/mesh archive:

```sh
roboenv prepare isaaclab
python tools/prepare_roboballet.py
roboenv launch isaaclab/roboballet-eight-arm --record
```

The asset tool requires `zstd` and honors `ROBOENV_ASSET_CACHE`. Use the same cache
setting at launch, or set `backend_kwargs.asset_directory` to its reported
`selected` directory. Existing files are verified rather than overwritten.
Ordinary Isaac Lab preparation does not download RoboBallet assets. The robot
catalog is `isaaclab/isaaclab-roboballet-eight-panda`; its profiles and drivers
export through the existing workspace flow.

The catalog defaults to target set 1, seed 0, stepped execution, and
stop-on-success. It retains 40 oriented targets, 30 obstacles, the 0.1-second
shared control interval and 420-frame (42-second) horizon. Acceptance requires
25 mm / 15-degree accuracy, a five-frame dwell, and eventual return of every arm
to home. Joint samples retain the original velocity/acceleration limits,
collision rollback and final home-capture rules. Explicit realtime execution
follows RoboEnv's uncapped policy and is not the bounded 42-second protocol.

`tools/generate_roboballet_scene.py` generates the frozen scene from the author's
geometry environment. All ten original target sets remain in it. Both targets and the
workpiece receive the same 0.6 m vertical translation. A conservative reach
check rejects impossible target sets before renderer startup; passing it does
not certify a collision-free schedule. `workcell.check` evaluates discrete
candidate configurations without executing them or advancing the clock.

Owner records contain source and asset identities, seeds, target-set selection,
commands, initial/final state, and per-frame motion, dwell and collision telemetry.
Unsafe proposals and rolled-back robot steps are separate counts. `--record`
uses the existing spectator MP4 pipeline; the legacy offline recording/export
API is not part of this integration. Historical evidence requires its original
source and runtime. Native task checks, device transport, rendering isolation and
task success are separate validation results.

Go2 base reads retain `height_m`, measured as root Z above the scene's Z=0
reference plane, and body-frame linear/angular velocities. Height is not terrain
clearance. Reads and write states omit global pose and empty frame maps;
world-relative transforms require explicit estimates.

## Implementation layout

The Isaac Lab backend selects a registered task using `backend_kwargs.task_name`.
The three Go2 tasks share the Go2–D1 robot implementation; the three calibration
exercises share the YAM embodiment. The generic adapter
does not assume that every task has a Go2 base, a six-joint arm, or a binary score.

```text
runtime/src/roboenv_runtime/adapters/isaaclab/
├── adapter.py                 # RoboEnv lifecycle, score and budget delegation
├── registry.py                # Task name -> lazily imported Task factory
├── common/
│   ├── environment.py         # IsaacEnvironment lifecycle protocol
│   ├── observations.py        # RGB-D serialization without physics stepping
│   └── recording.py           # Selected task and shared source identities
├── robots/go2_d1/
│   ├── environment.py         # Retained commands, device reads, journal
│   ├── native.py              # Robot/actuator/camera setup and physical stepping
│   ├── robot.py               # URDF kinematics and mass model
│   ├── controllers.py         # Feedback trot, leg IK and arm-control utilities
│   ├── limits.py              # D1 actuator limits
│   └── task.py                # Shared Go2 task validation and robot measurements
└── tasks/
    ├── go2_d1_control/task.py
    ├── go2_d1_reach/{task.py,scoring.py}
    └── go2_d1_pick_place/{task.py,scoring.py}
```

The GPU/application launcher remains shared with other Isaac-based backends in
`adapters/gpu.py`. Public task YAML stays in `catalog/tasks/isaaclab/`, robot
profiles stay in `catalog/robots/isaaclab/`, and runtime device bindings stay in
`bindings/backends/isaaclab/`. The SDK resolver passes selected device profiles
to Isaac tasks without importing runtime code. Robot-specific geometry validation
and TCP resolution happen in the robot implementation before simulator startup.

## Adding a task

For another layout of an existing task, add a catalog entry that reuses its
`backend_kwargs.task_name` and supplies different `task_spec` parameters. The
catalog ID may differ from the implementation name; no new Python module is needed.

For a new task using Go2–D1:

1. Add `tasks/<task_name>/task.py` and, when useful, `scoring.py`. Define a `Task`
   subclass of `robots.go2_d1.task.Go2Task` with its `name`, exact configuration
   `fields`, `scene_type`, and `state_type`. Implement `validate(spec)` for its
   configuration types and constraints; `validate_numeric_spec` is an optional
   helper for metric scalar/vector fields, as used by reach and pick-and-place.
   Task configuration can also contain asset references or other JSON values,
   validated by that task. The inherited factory reuses `Go2Env`.
2. Put scene objects, asset-loading code and measurements in the task's `Scene`.
   `Go2Scene.build()` runs during native construction; `reset()` initializes episode
   state; `physics_update()` runs after each physics step; `sample()` supplies the
   measurements evaluated after each control interval. Inherit the Go2 robot/TCP
   measurement helper where useful. Set `contact_sensors = True` on `Task` if its
   robot contacts are required. The pick-and-place task is a complete example.
3. Implement the pure evaluator in the task's state class. `Go2TaskState.update()`
   preserves terminal state and handles robot falls before calling `evaluate()`.
   Evaluation must use measured state, and must not move the robot or payload.
4. Add the implementation name and module path to `registry.TASKS`, then add its
   catalog YAML pointing to the existing Go2 robot. Add tests for configuration,
   success/failure rules, and the public device path.

Task-owned asset definitions and loaders belong with the task. Downloaded meshes,
generated USD files and prepared bundles remain outside Git, referenced through
configuration/manifests. All Go2 tasks continue to share the prepared robot bundle.

## Adding another robot or an upstream environment

A task factory for another robot exposes `name`, `embodiment`, and
`create_environment(**kwargs)`. Its returned environment implements
`common.environment.IsaacEnvironment`: `control_interval`, `tick`,
`configure_execution_mode`, `reset`, `close`, `score`, `budget`, and `record`.
Keep reusable robot behavior under `robots/<robot_name>/`; a task factory can
compose it with its scene and evaluator. Add the robot catalog, binding defaults,
and runtime bindings for its actual device inventory. Bindings access the selected
robot environment through `context.native.environment`.

`WorldSession` remains the sole clock owner. One `tick()` advances exactly one
control interval for all devices; reads and writes never request physics steps.
The selected environment supplies its own score and budget semantics. Realtime
execution has no episode step cap. Preserve final observations on termination;
do not silently auto-reset an upstream RL environment. Public device bindings must
keep privileged task state and verdicts private.

Registration is explicit; this backend does not automatically discover or wrap
Isaac Lab Gym/RL tasks. Adapting one requires mapping its action, observation,
termination and reset behavior to the same RoboEnv contracts.

## Refactor and replay compatibility

The former internal `isaaclab.go2_d1.*` modules moved to `isaaclab.robots.go2_d1.*`;
controller math is now `controllers.py`, and `RobotModel` is in `robot.py`.
The combined task module was split into task-owned scene and scoring modules.
Existing public task IDs, bindings, commands, timing and success rules are retained.

Go2 records now hash backend-relative source paths for the selected task, the shared
robot implementation and common helpers. Changes to an unrelated task implementation
do not invalidate that record; changes to the shared registry do. The dependency
lock and selected upstream source identities remain included. Recordings made
before this refactor fail strict source-identity comparison: retain their original
checkout and collect new native/transport recordings and fresh-process replay
evidence for this implementation. CPU checks do not certify simulator replay.

## Spectator recording

Use `--record` for a dedicated third-person overview on all six bundled tasks.
Robot implementations provide `spectator_sources()`; shared geometry helpers live
in `common/spectator.py`, and the renderer lives in `spectator/isaaclab.py`.
Go2 tasks select their fixed workspace through `scene.spectator_pose()`; returning
`None` uses the robot-following control view. New embodiments should provide their
own sources without exposing physical calibration transforms in device profiles.
See [spectator recording](spectator.md#isaac-lab-views) for poses and custom views.
