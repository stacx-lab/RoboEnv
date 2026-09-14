"""Construct the pinned upstream task with explicit upper-body joint targets."""


def create(paths, model, device, *, realtime=False):
    import torch
    import warp as wp
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
        LocomanipulationG1EnvCfg,
    )

    class UpperBodyAction(JointPositionAction):
        def apply_actions(self):
            super().apply_actions()
            # Preserve the upstream Pink action's gravity compensation while
            # accepting joint goals. The free-base generalized force offset is 6.
            ids = torch.tensor(self._joint_ids, device=self.device)
            forces = wp.to_torch(self._asset.root_physx_view.get_gravity_compensation_forces())[
                :, ids + 6
            ]
            self._asset.set_joint_effort_target_index(target=forces, joint_ids=self._joint_ids)

    class Episode(ManagerBasedRLEnv):
        allow_reset = True

        def _reset_idx(self, env_ids):
            # Keep the terminal state available to the owner/verifier. Explicit
            # reset is still supported; reaching a terminal state never resets it.
            if self.allow_reset:
                super()._reset_idx(env_ids)

    from .g1_cameras import configure

    cfg = LocomanipulationG1EnvCfg()
    if realtime:
        # Interactive runs have no time horizon; keep success/failure terms intact.
        cfg.terminations.time_out = None
    configure(cfg.scene, model["cameras"])
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.scene.robot = cfg.scene.robot.copy()
    cfg.scene.robot.spawn.usd_path = paths["robot"]
    cfg.scene.packing_table.spawn.usd_path = paths["table"]
    cfg.scene.object.spawn.usd_path = paths["object"]
    cfg.actions.lower_body_joint_pos.policy_path = paths["policy"]
    names = [joint["name"] for name, part in model["parts"].items() for joint in part["joints"]]
    cfg.actions.upper_body_ik = JointPositionActionCfg(
        class_type=UpperBodyAction,
        asset_name="robot",
        joint_names=names,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )
    if abs(cfg.sim.dt * cfg.decimation - model["control"]["execution"]["interval_s"]) > 1e-9:
        raise ValueError("G1 control period differs from the generated catalog")
    return Episode(cfg=cfg), names
