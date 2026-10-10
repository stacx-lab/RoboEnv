"""Create the native RoboDojo environment; device control lives in devices.py."""

from __future__ import annotations

import os
import sys
import secrets

# --------------------------------------------------------------------------
# RoboDojo checkout + Isaac boot (both are process-global and one-shot)
# --------------------------------------------------------------------------
from roboenv_runtime.paths import robodojo_root

ROBODOJO_ROOT = str(robodojo_root())  # $ROBODOJO_ROOT or <repo>/RoboDojo

_APP = None  # the SimulationApp handle; Isaac allows exactly one per process


def _boot_isaac(headless: bool = True, enable_cameras: bool = True):
    """Start Isaac Sim once. Must happen before any `env.*`/`task.*` import."""
    global _APP
    if _APP is not None:
        return _APP

    if not os.path.isdir(ROBODOJO_ROOT):
        raise RuntimeError(f"RoboDojo checkout not found at {ROBODOJO_ROOT}; set $ROBODOJO_ROOT")
    # The native EvalEnv accepts an injected client; no policy checkout is needed.
    if ROBODOJO_ROOT in sys.path:
        sys.path.remove(ROBODOJO_ROOT)
    sys.path.insert(0, ROBODOJO_ROOT)
    # EvalEnv uses this ID for resume state and temporary video cleanup. Distinct
    # workers must not reuse another world's native output directory.
    os.environ.setdefault("ROBODOJO_RUN_ID", f"roboenv-{secrets.token_hex(8)}")

    from roboenv_runtime.adapters.gpu import launch_isaaclab
    from env.camera_manager.capture.render_sync import zero_delay_kit_args

    # AppLauncher parses sys.argv too; hide the caller's own flags from it.
    saved_argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        # Keep the native extensions: the optional conveyor graph reverses
        # the asset's authored surface velocity.
        _APP = launch_isaaclab(
            "robodojo", headless=headless, enable_cameras=enable_cameras,
            kit_args=(
                "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
                + zero_delay_kit_args()
            ),
        )
    finally:
        sys.argv = saved_argv
    return _APP


class _NoopModelClient:
    """Stands in for XPolicyLab's WsModelClient so no policy server is needed."""

    def call(self, *a, **k):
        return None

    def close(self, *a, **k):
        return None

    def __getattr__(self, _name):
        return lambda *a, **k: None


# --------------------------------------------------------------------------
# Env
# --------------------------------------------------------------------------


class RoboDojoEnv:
    """Native construction shared by the typed device implementation."""

    def __init__(
        self, task_name="pack_objects_into_box", seed=0, env_cfg_type="arx_x5", headless=True
    ):
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.seed = seed
        self._app = _boot_isaac(headless=headless)
        self._env = None
        self._build()

    def _assemble_config(self, seed: int):
        """Replicates src/eval_client/main.py:249-341."""
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from omegaconf import OmegaConf
        from task.RoboDojo import task_registry
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, self.env_cfg_type + ".yml"))
        eval_cfg["task_name"] = self.task_name
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = 0
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = "roboenv"
        eval_cfg["additional_info"] = ""
        eval_cfg["seed"] = seed
        eval_cfg["physx_monitor_enabled"] = False

        deploy_cfg = {
            "policy_name": "roboenv",
            "port": 0,
            "host": "localhost",
            "protocol": "ws",
            "policy_server_url": "ws://localhost:0",
            "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
            "trial_id": f"{self.task_name}-{os.environ['ROBODOJO_RUN_ID']}",
            "action_case_id": f"{self.task_name}_case",
            "repeat_index": None,
        }

        cfg = OmegaConf.create(
            {
                "sim": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")
                ),
                "scene": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")
                ),
                "camera": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")
                ),
                "robot": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")
                ),
                "task_env": load_yaml(
                    task_registry.task_config_path(
                        os.path.join(benchmark_path, "config"), self.task_name
                    )
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )
        OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(cfg, "eval_cfg.num_envs", 1, force_add=True)
        cfg = process_randomization(cfg)
        cfg, eval_num = process_config(cfg, task_name=self.task_name)
        eval_cfg["eval_num"] = eval_num
        OmegaConf.update(
            cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        cfg.sim.seed = [0]
        return cfg

    def _build(self) -> None:
        import src.eval_client.eval_env as ee_mod

        cfg = self._assemble_config(self.seed)
        # The simulator-only path has no dependency on an optional policy package.
        self._env = ee_mod.create_eval_env(cfg, self._app, model_client=_NoopModelClient())

    def _layout_ids(self) -> list[int]:
        """Return the native selector order used for this environment."""
        sm = self._env.seed_manager
        return list(getattr(sm, "seed_list", None) or sorted(getattr(sm, "seed_info", {}).keys()))

    def _layout_id_for(self, seed: int) -> int:
        """Map a trial seed onto one of the task's saved native layout ids."""
        ids = self._layout_ids()
        if not ids:
            return int(seed)
        return int(ids[int(seed) % len(ids)])
