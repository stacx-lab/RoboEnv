# Go2–D1 model prototype

`build_go2_d1.py` replaces the ARX5 subtree in UMI-on-Legs' Go2–ARX5 URDF with
Unitree's D1-550 description, prefixes arm names, bundles relative mesh paths,
and records source/mesh hashes. These tools produce geometry bundles; the
[Go2–D1 Isaac Lab preparation](../../docs/go2-d1.md) adds dynamics and a registered environment.

Inputs:

- UMI-on-Legs source revision `d75c9c182d8044dadf53043612da2ffbf1936a97`.
- Unitree D1 archive:
  `https://oss-global-cdn.unitree.com/static/90b2525be5d84531ab9814f48b2f86a7.zip`
- Archive SHA256:
  `76cbe7725b1640cf214a6aa97e27d995c33ad5e4a4e7f29d57ec7ff7346a7164`.

Acquire the archive separately, verify its checksum, and extract it locally.
Keep downloaded assets and generated models outside source control.

```sh
python tools/robots/build_go2_d1.py --umi /path/to/umi-on-legs \
  --d1 /path/to/d1_description --out generated/go2-d1
python tools/robots/render_go2_d1.py generated/go2-d1
```

Builder dependencies are standard-library only. Rendering requires numpy,
trimesh, pycollada, mujoco, Pillow, EGL and the system DejaVu Sans font. Install
these in a separate visualization environment. The renderer uses actual visual
meshes and URDF forward kinematics; no physics is stepped. Presentation colors
are neutral overrides. `preview_pose.json` sets the illustrative joint pose.

The model retains the ARX5 mount at `(0.085, 0, 0.094)` metres with zero rotation.
This is a provisional mounting choice; no D1 mounting plate or hardware fit is
verified. D1's source zero effort/velocity limits are preserved, not substituted
with guesses. Actuator data, gripper coupling, collisions, control and native
Isaac Lab acceptance remain separate work. The Go2–ARX5 policy is not validated
for the Go2–D1 morphology.

## RealSense camera mounts

`add_go2_d1_cameras.py` adds D405 and D435 meshes and nominal sensor/optical frames
from the official `realsenseai/realsense-ros` repository, revision
`9a11121700cb4780e273e34141f6402fe184321d` (Apache-2.0).
Acquire `realsense2_description/urdf/{_d405,_d435,_materials,_usb_plug}.urdf.xacro`,
`realsense2_description/meshes/{d405.stl,d435.dae}`, and the repository `LICENSE`
into one local source directory. Add `source.json` containing `repository` and
`revision`. The output retains these sources and their SHA256 hashes.

```sh
python tools/robots/add_go2_d1_cameras.py --model generated/go2-d1 \
  --source .cache/go2-d1/realsense --out generated/go2-d1-cameras
python tools/robots/render_go2_d1.py generated/go2-d1-cameras
```

The camera builder also needs `xacro`. Edit `go2_d1_camera_mounts.json` to refine
placement, then regenerate into a new directory. Mount coordinates refer to the
camera's bottom screw frame, using upstream conventions:

- D405: parent `d1_Link6`, translation `(-0.087, 0, 0.008)` m,
  RPY `(0, -pi/3, 0)`. Camera forward tilts 30 degrees from wrist +Z toward the gripper center.
- D435: parent `Head_upper`, translation `(-0.01, 0, 0.058)` m,
  zero RPY. Camera forward is along robot +X.

Camera optical axes follow ROS conventions: +Z forward, +X right, +Y down.
Mounting blocks are visual placeholders. Upstream camera inertias explicitly
warn that they are not reliable; they are retained as source metadata, not
validated dynamics. Sensor streams and intrinsics are not configured by this
geometry tool. Preview camera housings are highlighted orange/blue for visibility.

## Views from the mounted cameras

After rendering the external robot preview (which also saves the world transforms):

```sh
python tools/robots/render_camera_views.py generated/go2-d1-cameras \
  --out generated/go2-d1-camera-views
```

This renders static RGB pinhole views at the URDF's color optical frames, at
960x540. Nominal product-page RGB FOVs are D405 87x58 degrees and D435 69x42 degrees;
these are visualization assumptions, not per-unit calibration. D435 depth has a
wider FOV than its RGB camera and is not rendered by this helper. The current
scene contains the robot, an empty floor and sky. The current camera's own CAD
housing is excluded because its opaque cover would block rays at the optical
origin; all other robot geometry remains visible. No pose or mount is changed.
`capture.json` records projection, world transforms, sources and axis checks.
