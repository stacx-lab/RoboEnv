"""Registered overhead camera practice exercise."""

from ..yam_calibration.scene import CalibrationScene
from ..yam_calibration.task import CalibrationTask


class Scene(CalibrationScene):
    board_parent = "left"
    # XYZW: turn the printed face toward the overhead camera at the home pose.
    board_orientation = (1, 0, 0, 0)


class Task(CalibrationTask):
    scene_type = Scene
    name = "yam-calibrate-overhead-camera"
