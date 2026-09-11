"""RoboTwin typed devices, native verdict and owner-only episode evidence."""

from roboenv_runtime.adapters.native import NativeDeviceAdapter
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import ScoreReport

@register_adapter
class RoboTwinAdapter(NativeDeviceAdapter):
    name = "robotwin"
    embodiment = "piper"

    def __init__(self, task_name="handover_block", embodiment="piper", **kwargs):
        from .devices import RoboTwinDevices

        self.embodiment = embodiment
        self._env = RoboTwinDevices(task_name, embodiment, **kwargs)

    def goal(self):
        # An unset native instruction delegates to the explicit public task
        # catalog; a filename cannot replace its object/goal semantics.
        return self._env.task.get_instruction() or ""

    def score(self):
        success = self._env.task_completed()
        return ScoreReport(reward=float(success), raw_score=float(success), completed=success,
                           detail={"control_protocol": "absolute_joint_position_25hz_practice",
                                   "official_action_equivalence": False})
