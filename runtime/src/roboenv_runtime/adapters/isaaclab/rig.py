"""Go2 hardware selection; importing this module never starts Isaac Sim."""

import json
from pathlib import Path

PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
FEET = tuple(f"{leg}_foot" for leg in LEGS)
CAMERA_POSITION = (0.35, 0.0, 0.10)
# Optical +Z points along body +X; optical +X points along body -Y.
CAMERA_ROTATION = (-0.5, 0.5, -0.5, 0.5)
CAMERA_SIZE = 256
FOCAL_LENGTH = 24.0
APERTURE = 27.712812921102035  # 60 degree field of view


def facts():
    return json.loads(Path(__file__).with_name("go2.json").read_text())


def scene_config(asset):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg, ImuCfg
    from isaaclab_assets import UNITREE_GO2_CFG

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, lazy_sensor_update=True)
    cfg.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.robot.spawn.usd_path = str(asset)
    cfg.ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim.GroundPlaneCfg())
    cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=2000.0))
    cfg.camera_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/CameraFront",
        height=CAMERA_SIZE,
        width=CAMERA_SIZE,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            horizontal_aperture=APERTURE,
            vertical_aperture=APERTURE,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=CAMERA_POSITION, rot=CAMERA_ROTATION, convention="ros"),
    )
    cfg.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=ImuCfg.OffsetCfg(pos=tuple(facts()["imu_position"])),
    )
    for foot in FEET:
        setattr(cfg, foot, ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}"))
    return cfg
