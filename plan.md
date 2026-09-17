# Isaac Lab follow-up: Unitree G1

Go2 is the first implementation. G1 remains planned, not implemented or validated.
Reuse the Go2 backend, native device bindings, synchronous clock, portable drivers,
asset preparation, and named device frames. Keep robot differences in rig configuration.

## Initial G1 rig

- Use the exact upstream locomotion G1 asset from the pinned Isaac Lab revision;
  do not substitute the 29-DOF or dexterous-hand models under the same rig name.
- Start fixed-base. Expose legs and torso joints through `body`, and left/right
  arm chains as `manipulator` devices. Leave hand joints at configured rest targets.
- Give each commanded joint one owner. Compose persistent device targets into the
  articulation before each native interval; sensor reads and stop never step.
- Reuse RGB-D, IMU, and per-foot contact bindings. Mount arms/camera on a declared
  moving torso frame and expose only robot-relative named link transforms.

## Kinex changes

- Go2 now uses the joint-controlled `base` type, including driver actions,
  calibration readiness and dynamic frames. Do not revive the legacy `body` type.
  The proposed fixed-start G1 rig still needs an explicit mounting/support design;
  the current base contract describes a free body, not a static world mount.
- Export arm serial-chain geometry from the actual asset for existing local FK/IK.
  A Cartesian request is local IK followed by native absolute joint targets, not
  a simulator solver RPC. Respect actual arm DOFs and report nonconvergence.
- Keep manipulator `base`/`flange` behavior alongside named moving body frames.

## Demonstration and verification

Add `isaaclab/g1-reach` with a published, FK-derived reachable arm pose. Require
joint error below 0.05 rad and flange error below 1 cm for 0.5 simulated seconds,
within 500 synchronous control intervals. Keep scoring private.

Validate exact joint ownership/order, fixed-body actuation, native flange vs local
FK, reachable IK and nonconvergence, moving sensor mounts, persistent targets,
and copied drivers in Kinex. Report CPU checks, RTX boot, device behavior,
task completion, and fresh-process replay separately.

## Deferred

Free-base balance, pretrained walking controllers, dexterous hand control, LiDAR,
and realtime main-thread scheduling are separate follow-ups. A future walking rig
must own every joint controlled by its policy and cannot overlap raw joint devices.
Do not add policy registries, whole-body IK, or runtime control-mode switching now.

## G1 humanoid device composition

- Added generated base, waist, independent arms and TriHand profiles.
- Added shared-worker target retention, snapshot and single-interval batch writes.
- Added local asset/source verification and upstream pick/place joint-action adapter.
- Added a Kinex composition example and CPU control/export tests; the example is
  now a test-only fixture under `tests/fixtures/device-composition/`.
- Native batch/retention/stop and exported Kinex controller checks passed.
- Native task completion remains a separate validation gate; rendering was verified
  on the available H100 with the existing RGB-D probe.

- Complete G1 example is in the default catalog, with torso/scene RGB-D cameras,
  portable task paths and checksum-pinned asset preparation.

- G1 layout follows the catalog boundary: external bindings, package-owned runtime
  facts/asset pins, docs provenance, and test-only Kinex composition fixtures. Robot
  directories contain only workspace.yaml and devices/.

## Unified base contract

- G1 uses a base locomotion mode with named velocity and pelvis-height targets; waist and arms remain manipulators.
- Go2 uses a base with exclusive joint-position control and existing foot IK/FK; no gait policy is implied.
- Wheeled base bindings and portable adapters use the same motion schema; native controller units and timing are preserved.

## Single mobile-base interface revision

- [x] Base writes use velocity plus optional posture, without a mode selector; untracked posture is ignored.
- [x] G1 and wheeled catalogs/bindings use the same flattened control facts.
- [x] Removed Go2 direct-joint demo, catalog, binding, generator and native joint-control tests. No Go2 policy is supplied.
- [x] Kept shared sensor/frame utilities and historical provenance; no vendor or cached asset changes.

Earlier Go2/direct-joint and multi-mode entries are historical and superseded.

## Named motion modes restored

- [x] Restored `control.mode: motion` and named per-mode velocity/posture capabilities in profiles, generators and bindings.
- [x] G1 selects locomotion; wheeled controllers select drive. Per-mode schemas also support posture-only commands.
- [x] Untracked posture remains ignored at the device boundary; Go2 direct-joint control remains removed.

## G1 realtime execution

- [x] Optional main-thread world-clock stepping; synchronous mode remains the default.
- [x] Atomic target replacement, independent physics timestamps, terminal-write rejection and no post-terminal physics.
- [x] Owner-side base expiry to zero travel with retained height (0.5 s default, task-configurable).
- [x] Runtime execution facts, matching Kinex workspace flags, and paced/fresh-feedback Kinex integration fixture.
- [ ] Native GPU pacing, camera-load, balance and fresh-process realtime replay validation.

## Uncapped G1 realtime horizon

- [x] Disable only the native time-out term in realtime mode and expose an unbounded frame budget.
- [x] Keep synchronous horizons, success/failure termination, and command-expiry configuration unchanged.
- [x] CPU regressions cover stepping past 20 seconds and retained non-timeout termination.
