"""BEHAVIOR-only external recording view; agent camera products stay untouched."""

import numpy as np

from . import CameraSource
from .isaac import Engine as IsaacEngine


class Engine(IsaacEngine):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.sources["overview"] = CameraSource(None, 640, 480, fov_y=65.0)
        self._overview_direction = 0

    def source_attributes(self, source):
        if source is not self.sources["overview"]:
            return super().source_attributes(source)
        aperture = 20.0
        return {
            "projection": "perspective",
            "verticalAperture": aperture,
            "horizontalAperture": aperture * source.width / source.height,
            "focalLength": aperture / (2 * np.tan(np.deg2rad(source.fov_y) / 2)),
            "horizontalApertureOffset": 0.0,
            "verticalApertureOffset": 0.0,
            "clippingRange": (0.01, 100.0),
        }

    def view_transform(self, view):
        if view.source != "overview":
            return super().view_transform(view)
        robot = self.world.robot
        # R1Pro's virtual base joints move this link, not the articulation root.
        position, _ = robot.links[robot.base_footprint_link_name].get_position_orientation()
        target = position.cpu().numpy().astype(float) + [0, 0, 0.9]
        angles = np.arange(8) * np.pi / 4 + 5 * np.pi / 4
        offsets = np.column_stack((2.6 * np.cos(angles), 2.6 * np.sin(angles), np.full(8, 1.5)))
        radius = np.linalg.norm(offsets[0])
        directions = offsets / radius
        distances = np.array([self._clearance(target, direction, radius) for direction in directions])
        best = int(np.argmax(distances))
        # Keep a world-fixed bearing until another side offers materially more room.
        if distances[best] > distances[self._overview_direction] + 0.5:
            self._overview_direction = best
        eye = target + directions[self._overview_direction] * distances[self._overview_direction]
        backward = eye - target
        backward /= np.linalg.norm(backward)
        right = np.cross([0, 0, 1], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        transform = np.eye(4)
        # USD cameras look along -Z, with +Y up.
        transform[:3, :3] = np.column_stack((right, up, backward))
        transform[:3, 3] = eye
        return transform

    def _clearance(self, target, direction, radius):
        from omnigibson.utils.sampling_utils import raytest

        # Read-only native scene queries keep the eye in front of walls/ceilings.
        hit = raytest(target.tolist(), (target + direction * radius).tolist(),
                      ignore_bodies=[link.prim_path for link in self.world.robot.links.values()])
        return max(0.01, float(hit["distance"]) - 0.2) if hit["hit"] else radius
