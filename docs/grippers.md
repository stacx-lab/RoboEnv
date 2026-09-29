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
finger controller output to native joint ranges. Internal stop holds can retain
measured soft-limit overshoot without treating it as a new user command.

- `extrinsics.mount` places the device TCP in its arm's `flange` frame.
- `intrinsics.geometry` describes that TCP and gives approach and closing axes
  in device coordinates.
- `geometry.contacts` references frames declared in `intrinsics.frames`.
  A contact frame's +Z points into the jaws; its local XY plane is tangent to the
  contact surface. `size_m` describes a rectangular patch only when known.
- Every read supplies `frame_poses` for the moving contacts, alongside measured
  finger positions and physical `aperture_m`.

Use the declared frame, for example `gripper:left_pad` or
`left_gripper:pad_1`, instead of assuming the flange is the grasp center.
Mounting, joint command signs and feedback channel signs are independent facts.

## Model-specific frames

Panda keeps its native `grip_site` TCP. `left_pad` and `right_pad` locate the
centers of its inner box-pad surfaces, measured from live MuJoCo transforms.
Each patch is 16 × 16 mm. Both LIBERO's pinned robosuite 1.4.0 model and RoboCasa's
model are checked against their compiled geometry.

Piper, ALOHA AgileX, ARX X5 and R1Pro place the TCP at the midpoint of two
source contact reference points at zero finger displacement, keeping the arm
flange's axes. `pad_1` and `pad_2` correspond to feedback channels 1 and 2.
Their transforms combine the pinned URDF with measured finger positions.
Unequal finger positions therefore shift the contacts independently.

These mesh contacts do not define rectangular patches: `size_m` is `null`.
The aperture model reports contact-reference separation along the closing axis,
clamped at zero for soft-contact overlap. It does not describe whole-finger
collision clearance or guarantee that an object fits at every insertion depth.

## Source and checks

[`gripper_contacts.json`](../runtime/src/roboenv_runtime/adapters/gripper_contacts.json)
records points and normals in each finger link's coordinates, with source
revisions and checksums. Piper, ALOHA and X5 use their pinned collision STL
surfaces.
R1Pro uses the native USD convex collision surfaces; its old URDF collision
mesh paths are not used to calibrate the contacts.

The numerical fixtures retain source joint transforms and the mesh facets
containing each contact. Tests verify surface membership, inward normals,
asymmetric opening, aperture conversion, both-arm catalogs and native read
integration. MuJoCo tests also exercise the LIBERO binding without stepping
physics. These checks do not replace full task replay in each simulator.

Existing workspaces contain copied profiles. Refresh their device profiles and
reload the devices after updating RoboEnv; restart the environment service to
load the corresponding binding changes.
