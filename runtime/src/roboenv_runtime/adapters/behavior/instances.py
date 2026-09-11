"""Restore released training instances through native OmniGibson state APIs."""

import hashlib
import json
from pathlib import Path


def restore_instance(env, robot, instance_id):
    """Follow the native evaluator's task-state, robot-pose and settling order.

    See omnigibson/eval/evaluator.py::Evaluator.load_task_instance. Keep policy,
    metric and video dependencies outside normal device setup. This owner-only
    initialization happens before the recorded episode clock starts.
    """
    import omnigibson as og
    from omnigibson.utils.asset_utils import get_task_instance_path
    from omnigibson.utils.bddl_utils import is_system_bddl_inst
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    task = env.task
    filename = task.get_cached_activity_scene_filename(
        task.scene_name, task.activity_name, task.activity_definition_id, instance_id
    )
    relative = f"{task.scene_name}_task_{task.activity_name}_instances/{filename}-tro_state"
    path = get_task_instance_path(task.scene_name, relative, mode="train")
    if path is None:
        raise FileNotFoundError(
            f"No released training instance {instance_id} for {task.activity_name} "
            f"in {task.scene_name}"
        )
    data = Path(path).read_bytes()
    state = recursively_convert_to_torch(json.loads(data))
    poses = {key.lower(): value for key, value in state["robot_poses"].items()}
    available = poses.get("robot", poses.get(robot.model))
    if not available:
        raise ValueError(f"Instance {instance_id} has no start pose for {robot.model}")
    unknown = set(state) - {"robot_poses"} - task.object_scope.keys()
    if unknown:
        raise ValueError(f"Instance contains unknown task entities: {sorted(unknown)}")
    for key, value in state.items():
        if key == "robot_poses":
            robot.set_position_orientation(available[0]["position"], available[0]["orientation"])
            env.scene.write_task_metadata(key=key, data=value)
        else:
            task.object_scope[key].load_state(value, serialized=False)
    og.sim.update_handles()
    for _ in range(25):
        og.sim.step_physics()
        for name, entity in task.object_scope.items():
            if entity is not None and not is_system_bddl_inst(name):
                entity.keep_still()
    env.scene.update_initial_file()
    env.scene.reset()
    return {
        "split": "train",
        "file": relative + ".json",
        "sha256": hashlib.sha256(data).hexdigest(),
        "settling_physics_steps": 25,
    }
