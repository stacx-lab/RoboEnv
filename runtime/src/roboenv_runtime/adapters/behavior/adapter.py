"""Continuous R1Pro devices and owner-only native BDDL acceptance."""

from roboenv_runtime.adapters.native import NativeDeviceAdapter
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import ScoreReport


@register_adapter
class BehaviorAdapter(NativeDeviceAdapter):
    name = "behavior"
    embodiment = "r1pro"

    def __init__(self, activity_name="picking_up_trash", **kwargs):
        from .devices import BehaviorDevices

        self._env = BehaviorDevices(activity_name, **kwargs)

    def goal(self):
        return self._env.challenge.get("instruction", self._env.activity_name.replace("_", " "))

    def score(self):
        success = self._env.task_completed()
        return ScoreReport(
            reward=float(success),
            raw_score=float(success),
            completed=success,
            detail={
                "control_protocol": "continuous_joint_and_base_velocity_30hz",
                "grasping_mode": "physical",
                "official_action_equivalence": False,
            },
        )
