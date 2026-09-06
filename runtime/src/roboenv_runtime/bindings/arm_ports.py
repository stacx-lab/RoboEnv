"""Thin adapters over existing counted controllers; no simulator imports.

The selector is an allowlisted robot arm, not an arbitrary attribute/import path.
No adapter silently switches planners, retries motion or changes world ownership.
"""

from roboenv_runtime.drivers.arm import ArmDriver, ArmSemantics, Pose

from .backend import BackendControllers

PROPOSAL_BINDING = "arm.cartesian-proposal/1"


def _wxyz(pose):
    x, y, z, w = pose.quaternion_xyzw
    return [w, x, y, z]


class ControllerArmPort:
    read_tools = ()
    move_tools = ()

    def __init__(self, context, entity):
        self.backend = BackendControllers(context)
        self.entity = entity

    def unavailable(self, *, motion):
        required = self.read_tools + (self.move_tools if motion else ())
        missing = sorted(set(required) - set(self.backend.tools))
        return "Backend controllers unavailable: " + ", ".join(missing) if missing else None

    def read_pose(self, execution):
        return self.backend.execute(lambda _: self._read(), {}, execution)

    def move_pose(self, pose, execution):
        return self.backend.execute(lambda _: self._move(pose), {}, execution)


class LiberoArmPort(ControllerArmPort):
    read_tools, move_tools = ("get_hand_pose",), ("goto_pose",)
    semantics = ArmSemantics("robot_base", "api_tool_point", "hand", "one-shot-ik")

    def _read(self):
        state = self.backend.call("get_hand_pose")
        return Pose.from_wxyz(state["tcp"], state["quat_wxyz"])

    def _move(self, pose):
        return self.backend.call("goto_pose", list(pose.position), _wxyz(pose))


class RoboCasaArmPort(ControllerArmPort):
    read_tools, move_tools = ("get_gripper_pose",), ("goto_pose",)
    semantics = ArmSemantics("world", "grip_site", "grip_site", "cartesian-servo")

    def _read(self):
        pose = self.backend.call("get_gripper_pose")
        return Pose.from_wxyz(pose[:3], pose[3:])

    def _move(self, pose):
        return self.backend.call("goto_pose", list(pose.position), _wxyz(pose))


class RoboTwinArmPort(ControllerArmPort):
    semantics = ArmSemantics(
        "world", "native_end_effector", "native_end_effector", "native-planner"
    )

    def __init__(self, context, entity):
        super().__init__(context, entity)
        side = {"left_arm": "left", "right_arm": "right"}[entity]
        self.read_tools = ("get_gripper_pose_" + side,)
        self.move_tools = ("goto_pose_" + side,)

    def _read(self):
        pose = self.backend.call(self.read_tools[0])
        return Pose.from_wxyz(pose[:3], pose[3:])

    def _move(self, pose):
        return self.backend.call(self.move_tools[0], list(pose.position), _wxyz(pose))


class RoboDojoArmPort(ControllerArmPort):
    read_tools, move_tools = ("get_gripper_pose",), ("goto_pose",)
    semantics = ArmSemantics("world", "fingertip_midpoint", "native_tool", "cartesian-servo")

    def __init__(self, context, entity):
        super().__init__(context, entity)
        self.side = {"left_arm": "left", "right_arm": "right"}[entity]

    def _read(self):
        pose = self.backend.call("get_gripper_pose", self.side)
        return Pose.from_wxyz(pose[:3], pose[3:])

    def _move(self, pose):
        return self.backend.call("goto_pose", self.side, list(pose.position), _wxyz(pose))


PORTS = {
    ("libero", "panda", "arm"): LiberoArmPort,
    ("robocasa", "panda_omron", "arm"): RoboCasaArmPort,
    **{
        (backend, embodiment, side + "_arm"): port
        for backend, embodiments, port in (
            ("robotwin", ("piper", "aloha_agilex"), RoboTwinArmPort),
            ("robodojo", ("arx_x5", "yam"), RoboDojoArmPort),
        )
        for embodiment in embodiments
        for side in ("left", "right")
    },
}


def bind_cartesian_arm(spec, root, context):
    if spec.binding != PROPOSAL_BINDING:
        raise ValueError("Cartesian arm proposal binding mismatch")
    if set(spec.config) - {"position_tolerance_m", "orientation_tolerance_rad"}:
        raise ValueError("unknown Cartesian arm proposal configuration")
    key = context.backend, context.embodiment, spec.entity
    factory = PORTS.get(key)
    if factory is None:
        raise ValueError("no reviewed Cartesian arm port for this backend/robot/entity")
    definition = spec.definition(root)
    if definition.type != "manipulator":
        raise ValueError("Cartesian arm binding requires a manipulator definition")
    driver = ArmDriver(factory(context, spec.entity), **spec.config)
    return driver.bind(
        definition,
        binding={
            "id": PROPOSAL_BINDING,
            "backend": context.backend,
            "embodiment": context.embodiment,
            "entity": spec.entity,
        },
    )
