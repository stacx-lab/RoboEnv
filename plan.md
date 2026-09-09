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

## Complex changes

- Go2 now uses the joint-controlled `legged` type, including driver actions,
  calibration readiness and dynamic frames. Do not revive the legacy `body` type.
  The proposed fixed-start G1 rig still needs an explicit mounting/support design;
  the current legged contract describes a free body, not a static world mount.
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
and copied drivers in Complex. Report CPU checks, RTX boot, device behavior,
task completion, and fresh-process replay separately.

## Deferred

Free-base balance, pretrained walking controllers, dexterous hand control, LiDAR,
and realtime main-thread scheduling are separate follow-ups. A future walking rig
must own every joint controlled by its policy and cannot overlap raw joint devices.
Do not add policy registries, whole-body IK, or runtime control-mode switching now.
