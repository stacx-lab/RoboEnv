"""Build one RoboCasa kitchen at one seed several times on CPU and print what each build placed.

Runs in the prepared RoboCasa interpreter, through the same adapter reset as a launched
world, with no renderer or GPU:

  python tests/robocasa_kitchen_cpu.py TASK SEED BUILDS

Prints one JSON line: the imported RoboCasa package directory and one scene per build.
"""

import json
import os
import sys
from pathlib import Path


def plain(value):
    return value.item() if hasattr(value, "item") else value


def rounded(values):
    return [round(float(v), 6) for v in values]


def scene(env, assets):
    meta = env.episode_meta()
    models = {}
    for cfg in meta.get("object_cfgs", []):
        models[cfg["name"]] = os.path.relpath(cfg["info"]["mjcf_path"], assets)
    poses = {}
    for name in env.list_objects():
        position, quaternion = env.get_object_pose(name)
        poses[name] = rounded(position) + rounded(quaternion)
    base = env.sim.model.body_name2id(env._robot.robot_model.root_body)
    return {
        "layout": plain(meta.get("layout_id")),
        "style": plain(meta.get("style_id")),
        "instruction": meta.get("lang"),
        "models": models,
        "poses": poses,
        "robot_base": rounded(env.sim.data.body_xpos[base]),
    }


def main():
    task, seed, builds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    import robocasa.models
    from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter

    assets = Path(robocasa.models.assets_root)
    adapter = RoboCasaAdapter(task, seed=seed, enable_render=False)
    try:
        scenes = []
        for _ in range(builds):
            adapter.reset(seed)
            scenes.append(scene(adapter._env, assets))
    finally:
        adapter.close()
    print(json.dumps({"robocasa": str(Path(robocasa.__file__).resolve().parent), "scenes": scenes}))


if __name__ == "__main__":
    main()
