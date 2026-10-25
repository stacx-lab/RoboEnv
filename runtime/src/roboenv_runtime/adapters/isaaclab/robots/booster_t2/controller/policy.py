"""The loco-manipulation policy: its config (YAML in `configs/`), the exported ONNX actor and the
metadata stored in it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from roboenv_runtime.adapters.isaaclab.robots.booster_t2.controller.robot import CONFIG_DIR
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.interface import Pose

from roboenv_runtime.adapters.isaaclab.robots.booster_t2.assets import directory


class PolicyConfig:
    """How to run the policy: files, rates, joints, observation constants, command limits.

    Args:
        path: YAML file; a bare file name is looked up in `CONFIG_DIR`.
    """

    def __init__(self, path: str | Path = "locomanip_policy.yaml", *, asset_directory=None) -> None:
        path = Path(path)
        self.path = CONFIG_DIR / path if len(path.parts) == 1 else path
        cfg = yaml.safe_load(self.path.read_text())
        self.onnx: Path = Path(asset_directory or directory()) / cfg["onnx"]
        """Exported actor."""
        self.robot: Path = CONFIG_DIR / cfg["robot"]
        """Robot config the policy was trained for."""
        self.decimation: int = cfg["decimation"]
        """Physics steps per policy step."""
        self.joints: tuple[str, ...] = tuple(cfg["joints"])
        """Observed and driven joints, policy order."""
        self.palm_frames: dict[str, str] = dict(cfg["palm_frames"])
        """URDF link of the palm per side ("left", "right")."""
        self.gait_period: float = cfg["gait_period"]
        """Period of the phase observation, s."""
        self.standing_threshold: float = cfg["standing_threshold"]
        """Velocity command norm below which the phase observation is 0."""
        r = cfg["command_range"]
        self.command_range = np.array([r["lin_vel_x"], r["lin_vel_y"], r["ang_vel_z"]], dtype=float)
        """Trained velocity command range: rows vx, vy (m/s), wz (rad/s); columns min, max."""
        self.rest_palm_targets: dict[str, Pose] = {
            side: Pose.of(t["pos"], t["quat"]) for side, t in cfg["rest_palm_targets"].items()
        }
        """Default palm target per side, command frame (trunk origin, trunk heading, z up)."""
        self.rest_arm_pos: dict[str, np.ndarray] = {
            side: np.array(q, dtype=float) for side, q in cfg["rest_arm_pos"].items()
        }
        """Arm joint positions per side whose palm FK is `rest_palm_targets`, the robot config's
        `joint_groups` order. Shape (7,), rad."""
        self.palm_workspace: dict[str, np.ndarray] = {
            side: np.array([box["min"], box["max"]], dtype=float)
            for side, box in cfg["palm_workspace"].items()
        }
        """Box of the trained palm targets per side, command frame: rows min, max. Shape (2, 3),
        m."""


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split(",")])


@dataclass(frozen=True)
class PolicyMetadata:
    """Fields the training runner wrote into the ONNX file."""

    joint_names: tuple[str, ...]
    """All robot joints, model order."""
    default_joint_pos: np.ndarray
    """Default pose, `joint_names` order. Shape (num_joints,), rad."""
    stiffness: np.ndarray
    """PD kp, `joint_names` order. Shape (num_joints,)."""
    damping: np.ndarray
    """PD kd, `joint_names` order. Shape (num_joints,)."""
    action_joint_names: tuple[str, ...]
    """Joints driven by the action, action order."""
    action_scale: np.ndarray
    """Joint target = default + scale · action. Shape (num_actions,), rad."""
    observation_names: tuple[str, ...]
    """Actor observation terms, concatenation order."""
    history_lengths: tuple[int, ...]
    """Frames per observation term; 0 means the current frame only."""

    @classmethod
    def from_session(cls, session: ort.InferenceSession) -> PolicyMetadata:
        props = session.get_modelmeta().custom_metadata_map
        return cls(
            joint_names=tuple(props["joint_names"].split(",")),
            default_joint_pos=_floats(props["default_joint_pos"]),
            stiffness=_floats(props["joint_stiffness"]),
            damping=_floats(props["joint_damping"]),
            action_joint_names=tuple(props["action_joint_names"].split(",")),
            action_scale=_floats(props["action_scale"]),
            observation_names=tuple(props["observation_names"].split(",")),
            history_lengths=tuple(
                int(float(v)) for v in props["observation_terms_history_length"].split(",")
            ),
        )


class OnnxPolicy:
    """The exported actor, observation normalization included."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.metadata = PolicyMetadata.from_session(self._session)
        self._input = self._session.get_inputs()[0].name
        self.obs_dim: int = int(self._session.get_inputs()[0].shape[1])
        """Length of the flat observation vector."""
        self.action_dim: int = int(self._session.get_outputs()[0].shape[1])
        """Length of the action vector."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Action for one observation. `obs` shape (obs_dim,); returns shape (action_dim,)."""
        out = self._session.run(None, {self._input: obs.astype(np.float32)[None]})[0]
        return out[0].astype(np.float64)
