"""RoboDojo native device lifecycle and owner-only verdicts."""

from roboenv_runtime.adapters.native import NativeDeviceAdapter
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import ScoreReport


@register_adapter
class RoboDojoAdapter(NativeDeviceAdapter):
    name = "robodojo"
    embodiment = "arx_x5"

    def __init__(
        self, task_name="pack_objects_into_box", env_cfg_type="arx_x5", *, stop_on_success=False,
        **kwargs,
    ):
        self._set_stop_on_success(stop_on_success)
        from .devices import RoboDojoDevices

        self.embodiment = env_cfg_type
        self._env = RoboDojoDevices(task_name, env_cfg_type, **kwargs)

    @property
    def native_terminated(self):
        return bool(self._env._env.end_flag[0])

    def recording_source(self):
        if self._env._env.scene_manager._fluid_objects[0]:
            from .recording import RoboDojoSource

            return RoboDojoSource(self._env)
        return None

    def goal(self):
        # The native observation manager already resolved labels and selected a
        # description at reset. Reading it must not rerun description sampling.
        instructions = self._env._env.obs_manager.instruction
        if len(instructions) != 1 or not isinstance(instructions[0], str):
            raise ValueError("Expected one cached native RoboDojo instruction")
        return instructions[0]

    def score(self):
        success = self._env.task_completed()
        cached = getattr(self._env._env.reward_manager, "_gated_score_lst", None)
        raw_score = float(cached[0]) if cached else 100.0 * float(success)
        return ScoreReport(
            reward=raw_score / 100.0,
            raw_score=raw_score,
            completed=success,
            detail={
                "control_protocol": "native_eval_joint_action",
                "official_action_equivalence": False,
            },
        )
