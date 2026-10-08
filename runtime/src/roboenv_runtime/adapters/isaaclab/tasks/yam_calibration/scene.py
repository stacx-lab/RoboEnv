"""Physical calibration fixtures. Never exposed as solved public transforms."""

import numpy as np

from .task import BOARD

# Distinct top heights provide a noncoplanar correspondence set.
FIDUCIALS = {
    "A": ((0.46, -0.10, 0.79), (0.8, 0.05, 0.05)),
    "B": ((0.59, -0.09, 0.84), (0.05, 0.65, 0.05)),
    "C": ((0.59, 0.10, 0.80), (0.05, 0.1, 0.8)),
    "D": ((0.45, 0.10, 0.88), (0.9, 0.75, 0.05)),
}


def board_image():
    """Metric board at an integral number of pixels per square and marker cell."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (BOARD["squares_x"], BOARD["squares_y"]), BOARD["square_m"], BOARD["marker_m"], dictionary
    )
    return board.generateImage((280, 200), marginSize=0, borderBits=1)


def black_rectangles(pixels):
    """Exact black scanline runs; avoids textures, UV mirroring, and filtering."""
    for row, values in enumerate(pixels):
        starts = np.flatnonzero(np.diff(np.r_[False, values == 0, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, values == 0, False].astype(int)) == -1)
        for start, end in zip(starts, ends, strict=True):
            yield start, row, end, row + 1


def create_board(stage, path, position, *, orientation=(0, 0, 0, 1)):
    import isaaclab.sim as sim
    from pxr import UsdGeom, Vt

    backing = sim.CuboidCfg(
        size=(0.23, 0.17, 0.003),
        collision_props=sim.CollisionPropertiesCfg(),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=1.0),
    )
    backing.func(path, backing, translation=position, orientation=orientation)
    pixels = board_image()
    height, width = pixels.shape
    points, indices = [], []
    for x0, y0, x1, y1 in black_rectangles(pixels):
        coordinates = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
        indices.extend(range(len(points), len(points) + 4))
        points.extend(
            [(x / width * 0.21 - 0.105, 0.075 - y / height * 0.15, 0.0016) for x, y in coordinates]
        )
    mesh = UsdGeom.Mesh.Define(stage, path + "/Print")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (len(points) // 4))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([(0.0, 0.0, 0.0)]))
    # An explicit material keeps the board independent of viewport display-color policy.
    black = sim.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=1.0)
    black.func(path + "/Black", black)
    sim.bind_visual_material(str(mesh.GetPath()), path + "/Black")


class CalibrationScene:
    board_parent = None
    board_orientation = (0, 0, 0, 1)

    def __init__(self, native):
        self.native = native

    def build(self):
        native = self.native
        import isaaclab.sim as sim
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import UsdGeom, UsdPhysics

        stage = native.sim.stage
        if self.board_parent == "left":
            create_board(
                stage,
                native.flange_paths["left"] + "/Board",
                (0, 0, 0.16),
                orientation=self.board_orientation,
            )
        elif self.board_parent == "world":
            create_board(
                stage, "/World/Board", (0.56, 0, 0.752), orientation=self.board_orientation
            )
        else:
            fixture_paths = []
            for name, (top, color) in FIDUCIALS.items():
                path = "/World/Fiducials/" + name
                height = top[2] - 0.75
                cfg = sim.CylinderCfg(
                    radius=0.015,
                    height=height,
                    collision_props=sim.CollisionPropertiesCfg(),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=color),
                )
                cfg.func(path, cfg, translation=(top[0], top[1], 0.75 + height / 2))
                # These fixtures are static colliders, not rigid-body actors.
                # PhysX contact filters must name the collision shape itself.
                fixture_paths.append(path + "/geometry/mesh")
                # Small visual center dot, with no additional collision surface.
                dot = sim.CylinderCfg(
                    radius=0.0015,
                    height=0.0001,
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                )
                dot.func(path + "/Center", dot, translation=(0, 0, height / 2 + 0.00005))
            for side in ("left", "right"):
                parent = native.flange_paths[side]
                # A convex pointed tool whose apex is precisely the existing gripper TCP.
                mesh = UsdGeom.Mesh.Define(stage, parent + "/Probe")
                points = [(0, 0, 0.1347)] + [
                    (0.004 * np.cos(a), 0.004 * np.sin(a), 0.055)
                    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)
                ]
                faces = [[0, i + 1, (i + 1) % 12 + 1] for i in range(12)]
                faces += [list(range(12, 0, -1))]
                mesh.CreatePointsAttr(points)
                mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
                mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDisplayColorAttr([(0.65, 0.65, 0.65)])
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                    "convexHull"
                )
                # MJCF retains nested rigid links. The spawn helper stops at
                # the first rigid ancestor, so enable this reporter explicitly.
                sim.activate_contact_sensors(parent)
                native.contacts[side] = ContactSensor(
                    ContactSensorCfg(
                        prim_path=parent, update_period=0.0, filter_prim_paths_expr=fixture_paths
                    )
                )
