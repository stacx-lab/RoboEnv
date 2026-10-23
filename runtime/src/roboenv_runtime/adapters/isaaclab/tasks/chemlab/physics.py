"""Import released inertias, collision proxies and joints through Isaac Lab."""

from pathlib import Path


class InstrumentPhysics:
    def __init__(self, scene, name, instrument, pose):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
        from scipy.spatial.transform import Rotation

        self.name = name
        self.model = scene.env.projections[name].model
        converter = sim_utils.UrdfConverter(
            sim_utils.UrdfConverterCfg(
                asset_path=str(instrument.root / instrument.row["urdf"]),
                usd_dir=str(Path(scene.textures.name) / name / "physics"),
                fix_base=True,
                merge_fixed_joints=True,
                make_instanceable=False,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    target_type="position",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=1000.0,
                        damping=100.0,
                    ),
                ),
            )
        )
        path = f"/World/{name}/physics"
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        cfg = sim_utils.UsdFileCfg(
            usd_path=converter.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        )
        cfg.func(
            path,
            cfg,
            translation=pose[:3, 3].tolist(),
            orientation=quaternion[[3, 0, 1, 2]].tolist(),
        )
        prims = list(Usd.PrimRange(scene.sim.stage.GetPrimAtPath(path)))
        # The existing material renderer supplies visual meshes and live panels.
        # Hide imported visuals only; retain their physical links and collisions.
        for prim in prims:
            if prim.GetName() in {"visuals", "Geometry"}:
                UsdGeom.Imageable(prim).MakeInvisible()
            # Released screw couplings include prismatic/angular mimic pairs.
            # PhysX cannot resolve some of these as native mimic constraints.
            # Preserve the URDF relation through explicit runtime coordinates.
            for schema in prim.GetAppliedSchemas():
                if schema.startswith("PhysxMimicJointAPI:"):
                    prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, schema.split(":", 1)[1])
        self.evidence = {
            "rigid_bodies": sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),
            "colliders": sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims),
            "joints": sum(p.IsA(UsdPhysics.Joint) for p in prims),
            "authority": "instrument process model",
        }
        self.actor = None
        if not self.model.actuated_joint_names:
            return
        self.actor = Articulation(
            ArticulationCfg(
                prim_path=path,
                spawn=None,
                actuators={
                    "instrument": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=1000.0,
                        damping=100.0,
                    )
                },
            )
        )

    def sync(self, config):
        import torch

        if self.actor is None:
            self.evidence.update(joint_names=[], positions=[])
            return
        position = self.actor.data.default_joint_pos.clone()
        resolved = dict(zip(self.model.actuated_joint_names, self.model.cfg, strict=True))
        resolved.update(config)
        for joint in self.model.robot.joints:
            if joint.mimic is not None:
                resolved[joint.name] = (
                    resolved.get(joint.mimic.joint, 0.0) * joint.mimic.multiplier
                    + joint.mimic.offset
                )
        for index, name in enumerate(self.actor.joint_names):
            if name in resolved:
                position[:, index] = resolved[name]
        # Runtime coordinates are authoritative for instrument-command mode.
        # No contact-derived chemistry or force calibration is invented here.
        self.actor.write_joint_state_to_sim(position, torch.zeros_like(position))
        self.actor.set_joint_position_target(position)
        self.actor.write_data_to_sim()
        self.evidence["joint_names"] = self.actor.joint_names
        self.evidence["positions"] = position[0].tolist()
