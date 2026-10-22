"""The T2 as an Isaac Lab articulation, modeled as the policy was trained: the robot config's URDF
with its collision boxes, the policy's joints driven by joint torque so that Booster's PD runs
outside PhysX, and the finger joints position-driven by PhysX."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

from roboenv_runtime.adapters.isaaclab.robots.booster_t2 import assets
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.robot import RobotConfig

CACHE_DIR = assets.generated_directory() / "t2_isaac"
"""Where the edited URDF and its USD conversion are written."""

JOINT_SPEED_CAP = 100.0
"""PhysX joint speed limit, rad/s: far above every motor's, so that the PD's torque speed curve is
the only limit."""


def edited_urdf(robot: RobotConfig) -> Path:
    """Write the robot config's URDF with its edits into `CACHE_DIR` and return the file: mimic
    joints become independent, mesh paths become absolute, and each link of
    `robot.simulation.collision_boxes` gets that box as its only collision."""
    tree = ET.parse(robot.urdf)
    root = tree.getroot()
    for joint in root.findall("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        if "://" not in name and not Path(name).is_absolute():
            mesh.set("filename", str((robot.urdf.parent / name).resolve()))
    links = {link.get("name"): link for link in root.findall("link")}
    for link_name, box in robot.simulation.collision_boxes.items():
        link = links[link_name]
        for collision in link.findall("collision"):
            link.remove(collision)
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=" ".join(f"{v:.9g}" for v in box.pos), rpy="0 0 0")
        shape = ET.SubElement(ET.SubElement(collision, "geometry"), "box")
        shape.set("size", " ".join(f"{v:.9g}" for v in box.size))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{robot.urdf.stem}_isaac.urdf"
    tree.write(out)
    return out


def t2_articulation_cfg(robot: RobotConfig, joint_names: Sequence[str]) -> ArticulationCfg:
    """The T2 articulation at `/World/Robot`, at its keyframe pose with the fingers open. The
    policy's joints are torque-driven (zero drive gains) with the motor's armature and effort
    limit; the finger joints are position-driven with `robot.fingers`' gains. The URDF and its USD
    conversion are written to `CACHE_DIR`.

    Args:
        robot: Robot config.
        joint_names: The policy's joints: every joint but the fingers, any order.
    """
    sim = robot.simulation
    fingers = robot.fingers
    motors = {name: robot.motor(name) for name in joint_names}
    joint_pos = dict(zip(joint_names, robot.keyframe_joint_pos(joint_names).tolist()))
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(edited_urdf(robot)),
            usd_dir=str(CACHE_DIR / "usd"),
            fix_base=False,
            merge_fixed_joints=True,
            self_collision=sim.self_collision,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                )
            ),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=sim.self_collision,
                solver_position_iteration_count=sim.solver_iterations[0],
                solver_velocity_iteration_count=sim.solver_iterations[1],
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(robot.keyframe_root_pos()),
            joint_pos={**joint_pos, **dict.fromkeys(fingers.joints, 0.0)},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=list(joint_names),
                stiffness=0.0,
                damping=0.0,
                armature={name: m.armature for name, m in motors.items()},
                effort_limit_sim={name: m.effort_limit for name, m in motors.items()},
                velocity_limit_sim=JOINT_SPEED_CAP,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(fingers.joints),
                stiffness=fingers.stiffness,
                damping=fingers.damping,
                armature=fingers.armature,
            ),
        },
    )
