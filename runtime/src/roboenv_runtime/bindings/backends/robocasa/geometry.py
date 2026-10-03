"""Physical mount extraction, independent of the selected command contract."""

import numpy as np


def mounts(spec, definition, context):
    world = context.native._env
    robot, sim = world._robot, world.sim
    model, data = sim.model._model, sim.data._data
    arm_joints = robot.part_controllers["right"].joint_index
    arm_base = int(model.body_parentid[model.jnt_bodyid[arm_joints[0]]])
    entity = spec.entity
    extrinsics = dict(definition.extrinsics)
    if entity == "camera":
        camera = spec.config["camera"]
        camera_id = sim.model.camera_name2id(camera)

    def relative_mount(position, rotation, base):
        import mujoco

        base_rotation = data.xmat[base].reshape(3, 3).T
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (base_rotation @ rotation).ravel())
        return {
            "translation": (base_rotation @ (position - data.xpos[base])).tolist(),
            "rotation": [*quat[1:].tolist(), float(quat[0])],
        }

    if entity in {"arm", "torso"}:
        parent_entity = "torso" if entity == "arm" else "base"
        parent = (context.devices or {}).get(parent_entity, parent_entity)
        extrinsics = {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "mount": {
                "parent": parent,
                "parent_frame": "top" if entity == "arm" else parent,
                "translation_m": [0, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1],
            },
        }
    elif entity == "gripper":
        flange = sim.model.body_name2id(robot.robot_model.eef_name["right"])
        site = sim.model.site_name2id(robot.gripper["right"].important_sites["grip_site"])
        body = int(model.body_parentid[model.site_bodyid[site]])
        transform = relative_mount(data.xpos[body], data.xmat[body].reshape(3, 3), flange)
        extrinsics = {
            "source": "introspected",
            "method": "introspection",
            "calibrated_at": None,
            "mount": {
                "parent": (context.devices or {}).get("arm", "arm"),
                "parent_frame": "flange",
                "translation_m": transform["translation"],
                "rotation_xyzw": transform["rotation"],
            },
        }
    elif entity == "camera":
        names = context.devices or {}
        frames = {0: (names.get("world", "world"), None)}
        if "base" in names:
            frames[sim.model.body_name2id(robot.robot_model.base.root_body)] = (names["base"], None)
        if "arm" in names:
            frames[arm_base] = (names["arm"], None)
            frames[sim.model.body_name2id(robot.robot_model.eef_name["right"])] = (
                names["arm"],
                "flange",
            )
        if "torso" in names:
            frames[arm_base] = (names["torso"], "top")

        def rigid_root(body):
            while body and not model.body_jntnum[body]:
                body = int(model.body_parentid[body])
            return body

        # The camera can sit on the lift support ABOVE the named torso flange.
        # Resolve all bodies in the same rigid assembly, including descendants.
        rigid_frames = {rigid_root(body): body for body in frames}
        ancestor = rigid_frames.get(rigid_root(int(model.cam_bodyid[camera_id])))
        if ancestor is not None:
            parent, frame = frames[ancestor]
            # MuJoCo cameras look along -Z with +Y up; optical frames use +Z forward, +Y down.
            rotation = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            transform = relative_mount(data.cam_xpos[camera_id], rotation, ancestor)
            extrinsics = {
                "source": "introspected",
                "method": "introspection",
                "calibrated_at": None,
                "mount": {
                    "parent": parent,
                    "parent_frame": frame or parent,
                    "translation_m": transform["translation"],
                    "rotation_xyzw": transform["rotation"],
                },
            }
    elif entity == "base":
        extrinsics = {"source": "introspected"}
    return extrinsics
