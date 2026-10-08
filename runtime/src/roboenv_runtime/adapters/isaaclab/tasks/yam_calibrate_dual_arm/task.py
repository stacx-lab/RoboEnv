"""Registered dual arm practice exercise."""

from ..yam_calibration.scene import CalibrationScene
from ..yam_calibration.task import CalibrationTask


class Scene(CalibrationScene):
    board_parent = None


class Task(CalibrationTask):
    scene_type = Scene
    name = "yam-calibrate-dual-arm"
