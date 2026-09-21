# Gripper geometry

RoboCasa, all five LIBERO Panda rigs, both RoboTwin rigs, both RoboDojo rigs and
BEHAVIOR R1Pro use the same scalar-gripper contract. IsaacLab G1 retains its
separate dexterous-hand contract.

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

Contact patches and aperture describe the inner grasping surfaces. They do not
bound the whole fingers, hand, wrist, or arm, and do not establish clearance along
a motion. A nonzero aperture also does not establish that an object remains held;
use measured motion and fresh observations to verify the grasp.

## LIBERO direction and stop behavior

LIBERO retains the last gripper direction across control intervals, including
intervals advanced by arm writes in synchronous mode. `+1` closes, `-1` opens,
and `0` retains the accumulated actuator target. The target changes by 0.25 per
control interval with the native 20 Hz configuration; this is not an aperture
change in metres.

`arm.stop()` holds measured arm joint positions and leaves the gripper direction
and accumulated target unchanged. `gripper.stop()` sets the direction to zero and
replaces the accumulated target with measured finger positions. It can therefore
relax contact preload; writing zero retains the previous target instead. Neither
device stop requests a control step or finalizes the simulation. In realtime
mode, the owner continues advancing physics independently of reads and stops.
These semantics are shared by all five LIBERO Panda catalogs and are included in
the live action descriptions exported to the agent's device guide.

## Model-specific frames

Panda keeps its native `grip_site` TCP. `left_pad` and `right_pad` locate the
centers of its inner box-pad surfaces, measured from live MuJoCo transforms.
Each patch is 16 × 16 mm. Both LIBERO's pinned robosuite 1.4.0 model and RoboCasa's
model are checked against their compiled geometry.

Piper, ALOHA AgileX, ARX X5, YAM and R1Pro place the TCP at the midpoint of two
source contact reference points at zero finger displacement, keeping the arm
flange's axes. `pad_1` and `pad_2` correspond to feedback channels 1 and 2.
Their transforms combine the pinned URDF with measured finger positions.
Unequal finger positions therefore shift the contacts independently.

These mesh contacts do not define rectangular patches: `size_m` is `null`.
The aperture model reports contact-reference separation along the closing axis,
clamped at zero for soft-contact overlap. It does not describe whole-finger
collision clearance or guarantee that an object fits at every insertion depth.
YAM opens with negative finger displacement; its aperture coefficients retain
that sign.

## Source and checks

[`gripper_contacts.json`](../runtime/src/roboenv_runtime/adapters/gripper_contacts.json)
records points and normals in each finger link's coordinates, with source
revisions and checksums. Piper, ALOHA and X5 use their pinned collision STL
surfaces. YAM uses the manufacturer meshes used by RoboEnv's USD conversion.
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
