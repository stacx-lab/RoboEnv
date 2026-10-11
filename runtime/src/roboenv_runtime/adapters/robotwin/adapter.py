"""RoboTwin typed devices, native verdict and owner-only episode evidence."""

import json

from roboenv_runtime.adapters.native import NativeDeviceAdapter
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import ScoreReport
from roboenv_runtime.paths import robotwin_root


@register_adapter
class RoboTwinAdapter(NativeDeviceAdapter):
    name = "robotwin"
    embodiment = "piper"

    def __init__(
        self, task_name="handover_block", embodiment="piper", *, stop_on_success=False, **kwargs
    ):
        self._set_stop_on_success(stop_on_success)
        from .devices import RoboTwinDevices

        self.embodiment = embodiment
        self._env = RoboTwinDevices(task_name, embodiment, **kwargs)

    def goal(self):
        # An unset native instruction delegates to the explicit public task
        # catalog; a filename cannot replace its object/goal semantics.
        return self._env.task.get_instruction() or ""

    def task_context(self):
        task = self._env.task
        if self._env.task_name == "adjust_bottle":
            side = "left" if task.qpose_tag == 0 else "right"
            path = (robotwin_root() / "description/objects_description/001_bottle"
                    / f"base{task.model_id}.json")
            description = json.loads(path.read_text())
            return {"assigned_arm": side, "target_side": side,
                    "bottle": {"name": description["raw_description"],
                               "description": description["seen"][-1]}}
        if self._env.task_name in {"place_a2b_left", "place_a2b_right"}:
            objects = {}
            for role in ("A", "B"):
                name = getattr(task, f"selected_modelname_{role}")
                model = getattr(task, f"selected_model_id_{role}")
                path = (robotwin_root() / "description/objects_description"
                        / name / f"base{model}.json")
                description = json.loads(path.read_text())
                objects[role] = {"name": description["raw_description"],
                                 "description": description["seen"][-1]}
            return {"objects": objects}
        return {}

    def score(self):
        success = self._env.task_completed()
        return ScoreReport(reward=float(success), raw_score=float(success), completed=success,
                           detail={"control_protocol": "absolute_joint_position_25hz_practice",
                                   "official_action_equivalence": False})
