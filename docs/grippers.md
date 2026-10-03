# Gripper geometry

RoboCasa, all five LIBERO Panda rigs, both RoboTwin rigs, RoboDojo ARX X5 and
BEHAVIOR R1Pro use the same scalar-gripper contract. SIMPLE G1 retains its
separate dexterous-hand contract.

Position grippers reject commands outside the profile's `[0, 1]` range.
Panda direction grippers reject values outside `[-1, 1]`, then use the sign
(positive closes, negative opens, zero holds). These are Kinex's normal scalar
validation rules, applied to the entire sequence before any effect. Native Panda
controllers accept larger magnitudes too, but the public profile deliberately
retains the canonical range for consistent behavior.

Native motion limits remain in effect: RoboTwin's setter clips normalized goals
and computes incremental finger targets and drive velocities each physics step;
RoboDojo clips normalized goals and interpolated finger targets; BEHAVIOR clips
finger controller output to native joint ranges. Stop holds retain the applied gripper target and actuator preload across all
backends, including Go2–D1. They never replace that target with measured aperture.
Direction stop freezes the accumulated finger target; position stop preserves the
applied position reference. Pending actions are discarded, and explicit open is
required to release. Reset initializes from measured posture independently of stop.
This behavior also applies to cancellation and controller cleanup.

Restart owned simulator services after upgrading; refresh copied Kinex gripper
drivers/runtime helpers before reloading. Existing borrowed services remain under
their owner's lifecycle control.

- `extrinsics.mount` places the gripper base in its arm's `flange` frame.
  `intrinsics.frames.tcp` places the TCP relative to the gripper base.
- `intrinsics.geometry` gives approach and closing axes in gripper-base coordinates.
  Its optional `description` describes the fingers, pads and known limitations.
- Reads and write states report physical `aperture_m`, faults
  and timestamp. They omit `frame_poses`, finger positions and velocities.
- Profiles expose TCP and other fixed references, not moving finger/contact frames.
  Unknown gripper transforms cannot be resolved through a read.

Use `gripper:tcp` rather than assuming the flange is the grasp center.

## Model-specific geometry

Panda keeps its native `grip_site` TCP and two opposing 16 × 16 mm inner pads.
The driver calibrates aperture conversion from the compiled geometry at setup;
reads use measured joint positions without computing finger poses.

Piper, ALOHA AgileX, ARX X5 and R1Pro place the TCP at the midpoint of two
source contact reference points at zero finger displacement, keeping the arm
flange's axes. Their mesh-defined contact surfaces have no calibrated rectangular
pad dimensions. Internal measured finger positions determine aperture, including
asymmetric openings. Aperture is clamped at zero for soft-contact overlap and
is not whole-finger clearance or a guarantee that an object fits.

D1 has two opposing sliding fingers; pad dimensions and physical aperture
calibration remain unspecified. Its aperture feedback is null.

## Source and checks

[`gripper_contacts.json`](../runtime/src/roboenv_runtime/adapters/gripper_contacts.json)
records points and normals in each finger link's coordinates, with source
revisions and checksums. Piper, ALOHA and X5 use their pinned collision STL
surfaces.
R1Pro uses the native USD convex collision surfaces; its old URDF collision
mesh paths are not used to calibrate the contacts.

The numerical fixtures retain source joint transforms and the mesh facets
containing each contact. Tests verify source surface membership, normals, asymmetric aperture,
fixed TCP transforms, both-arm catalogs and native read/stop integration. MuJoCo tests also exercise the LIBERO binding without stepping
physics. These checks do not replace full task replay in each simulator.

Existing workspaces contain copied profiles. Refresh their device profiles and
reload the devices after updating RoboEnv; restart the environment service to
load the corresponding binding changes.
