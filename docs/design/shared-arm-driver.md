# RFC: a shared Cartesian arm driver

Status: **draft proposal; maintainer decision pending; do not merge automatically**.
This is an executable design prototype, not an approved architecture or a GPU
support announcement. No default robot binding, historical reference program,
simulator controller or benchmark success rule is changed.

## Problem and proposal

Today, robot device facts and binding plans are declarative, but Cartesian arm
handlers are repeated for each backend, embodiment and left/right slot. The
service already shares request validation, serialization and execution evidence.
This proposal also shares Cartesian command orchestration without moving physics
or environment ownership into the driver.

```text
Agent / SDK / generated thin driver
                  |
          existing DeviceService
                  |
             ArmDriver                 shared command logic
                  |
               ArmPort                 typed capability boundary
          /       |       \       \
       LIBERO  RoboCasa  RoboTwin  RoboDojo
                  |
       existing counted world controllers
                  |
          simulator-owned robot entity
```

The initial ports adapt the project's existing benchmark controllers, not raw
MuJoCo/SAPIEN/Isaac APIs. Those controllers already encode task-specific control
and accounting. A future engine-native port could implement the same interface,
but only after preserving the world's execution and evidence boundary.

## Responsibilities

| Component | Owns | Does not own |
|---|---|---|
| Device definition | Logical name, device type, declared hardware facts | Simulator instance, motion execution |
| Binding | Explicit backend/embodiment/entity selection and driver configuration | Arbitrary imports, robot creation |
| ArmDriver | Input checks, orientation preservation, relative-to-absolute commands, measured pose errors | Servo loop, trajectory planner, retries, simulator stepping |
| ArmPort | Existing arm access, native pose conversion, controller invocation | Episode reset, independent budgets or verdicts |
| World and worker | Entity lifetime, serialization, deadlines, step budget, scoring and journal | Agent reasoning |

The implementation lives in `runtime/src/roboenv_runtime/drivers/arm.py`; the
ports and opt-in binding live in `runtime/src/roboenv_runtime/bindings/arm_ports.py`.
The SDK has no new dependency or simulator import. There is no new wire protocol,
server process, permission authority or agent loop.

## Experimental interface

The binding ID is `arm.cartesian-proposal/1`. It is registered but **not selected
by any default catalog plan**. Its name intentionally communicates proposal status.

- `read({})` / `state({})`: position in metres, quaternion in xyzw order, frame,
  control point, orientation frame and motion policy.
- `write({frame, target})`: target is xyz or xyz plus quaternion xyzw. An xyz-only
  command first reads and preserves orientation.
- `displace({frame, delta})`: delta is three metre offsets in the declared frame.
  The common driver reads the current pose and issues one absolute command.
- Motion responses include the requested and observed pose, position/orientation
  errors, `target_reached` and the original backend report. `ok` at the service
  level means a command returned, not that the target or benchmark was solved.

All fields are strict: unknown options, wrong frames, non-finite numbers, Boolean
coordinates, malformed targets and non-unit quaternions are rejected before
backend calls. Tiny quaternion normalization error is tolerated; an arbitrary
quaternion is not silently repaired. Quaternion signs q and -q denote the same
rotation when measuring orientation error.

The configured `position_tolerance_m` and `orientation_tolerance_rad` only govern
the reported pose comparison. They do **not** change the underlying controller's
stopping rules. Defaults are 0.01 m and 0.05 rad. Reaching a pose is not evidence
of collision-free motion, grasp success, task completion or safety certification.
If a backend returns a budget-exhaustion report, it remains visible even when its
final pose happens to lie within the requested tolerances.

## Explicit semantic differences

| Backend | Entity selectors | Frame / position reference | Existing motion |
|---|---|---|---|
| LIBERO Panda | `arm` | Robot base / API tool point, hand orientation | One-shot IK; undershoot is measured, not silently retried |
| RoboCasa Panda/Omron | `arm` | World / grip site | Cartesian servo |
| RoboTwin Piper, ALOHA-AgileX | `left_arm`, `right_arm` | World / native end-effector | Native planner |
| RoboDojo ARX X5, YAM | `left_arm`, `right_arm` | World / fingertip midpoint | Cartesian servo |

Ten selectors share the same ArmDriver. The ports convert the controllers' wxyz
quaternions to/from the proposal's xyzw interface. They never invent a
base-to-world transform or label an unknown tool pose as a flange transform.
LIBERO's TCP, wrist and fingertip positions are different; the port explicitly
selects its API tool point. RoboDojo's ambiguous `arm` alias is rejected.

This is code-level coverage of selectors, **not ten validated robot deployments**.
BEHAVIOR's current symbolic controller has no continuous-arm port and is rejected.
Joint-space control, IK/FK, emergency stop, coordinated dual-arm trajectories,
gripper actions, pick/place strategies and force control are out of scope.

## Opt-in assembly example

Within an explicitly prepared robot definition, provide a profile whose action
list describes this proposal, rather than copying a legacy arm's richer actions:

```yaml
# devices/arm/profile.yaml
name: arm
type: manipulator
access:
  control_mode: cartesian_pose
  driver:
    actions: [read, state, write, displace]
intrinsics: {}
extrinsics: {}
```

Then select a concrete existing entity:

```yaml
schema_version: roboenv/bindings/1
backend: robotwin
embodiment: piper
devices:
  arm:
    binding: arm.cartesian-proposal/1
    entity: left_arm
    profile: devices/arm/profile.yaml
    config:
      position_tolerance_m: 0.01
      orientation_tolerance_rad: 0.05
```

The same profile/driver can bind to another supported backend and entity. Hardware
facts must still describe the actual robot; an empty field means unknown, not a
default calibration. An existing task's `device_bindings` replaces the complete
device mapping, so retain other devices explicitly when trying the proposal.
Do not change shipped catalog profiles just to turn on this prototype globally.

## Execution and failure semantics

Ports use `BackendControllers` and the existing request ExecutionContext. A full
pose command costs one native motion call plus one native pose query. Position-only
commands and displacements additionally read orientation/current pose. Those reads
are visible to the existing world's accounting; no private budget is introduced.

There are no automatic retries or corrective loops. An exception after movement
propagates with the service's unknown-effects trace. Read-only execution contexts
reject motion before invoking the controller. Missing motion controllers leave
read capabilities available; missing pose controllers also disable motion because
the driver could not measure its outcome. Reset/restore replaces bindings, close
invalidates old handles, and submit freezes motion through the existing worker.
These are trusted environment components, not a sandbox for hostile port code.

## Evidence and review gates

`tests/test_shared_arm_proposal.py` exercises all selectors through the real
binding registry and device service, with fake physics. It also checks coordinate
conversion, wrong inputs, unavailable capabilities, no retry after a partial
effect, read-only guards and the actual worker's accounting/reset/submit path.
No paid model or GPU is required for those tests.

Before promoting any default binding:

- Review this design and choose the public API shape.
- Add exact simulator/image/robot/seed receipts for read and motion operations.
- Compare old and new controller calls and budget costs on representative tasks.
- Verify TCP/frame semantics in each real simulator, including both dual-arm sides.
- Decide whether any default change warrants a new contract/version and migration guide.

Existing main-branch GPU checks do not certify this proposal. The initial PR is
intentionally draft, and must not be merged merely because CPU checks pass.

## Decisions still open

1. Keep `write` for the current device convention, or expose a typed `move_pose` API?
2. Keep native frames explicit, or require a separately verified transform service?
3. Return a measured `target_reached=false`, or raise a dedicated error on undershoot?
4. Adapt benchmark controllers first, or invest in engine-native ports immediately?
5. Should planning be another capability/port, or an optional component above ArmDriver?
6. When should robot definitions move from per-backend catalogs into reusable model assets?

Alternatives are to retain the current backend handlers, or to replace them all
at once with native control. Retaining them avoids migration but keeps repetition;
a full replacement may offer a cleaner control boundary but changes substantially
more simulator behavior and needs a much larger validation matrix. This opt-in
prototype makes the shared-driver boundary reviewable without committing to either
final migration strategy.
