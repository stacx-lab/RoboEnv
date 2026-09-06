"""RoboCasa365's openpi config, ported from ``robocasa-benchmark/openpi`` for inference.

The public checkpoint ``robocasa/robocasa365_checkpoints/pi05_pretrain_human300``
was trained with ``TrainConfig(name="pi05_pretrain_human300", model=Pi0Config(pi05=True,
max_token_len=200), data=LeRobotRobocasaDataConfig(...))`` whose data transforms are
``RobocasaInputs``/``RobocasaOutputs`` below (copied verbatim from the fork's
``src/openpi/policies/robocasa_policy.py``).  The fork loads normalisation statistics
from its GR00T-format datasets; the checkpoint ships them as ``assets/norm_stats.json``,
which ``serve_openpi.py`` passes explicitly.

Observation contract (the fork's ``examples/robocasa/main.py``):
    observation/image        robot0_agentview_left, resize_with_pad 224
    observation/wrist_image  robot0_eye_in_hand
    observation/right_image  robot0_agentview_right
    observation/state        base_to_eef_pos(3) + base_to_eef_quat(4) + base_pos(3)
                             + base_quat(4) + gripper_qpos(2)
    prompt                   the episode's language instruction
Action (12): ee pos delta(3), ee rot delta(3), gripper_close(1), base_motion(4: vx vy wz torso),
control_mode(1) — ``robocasa.utils.env_utils.convert_action`` + the gym wrapper's ``unmap_action``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """Model inputs from the RoboCasa observation dict (fork: robocasa_policy.py)."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        """Pack images/state/prompt into the model's input dict."""
        mask_padding = self.model_type == _model.ModelType.PI0
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        if self.model_type == _model.ModelType.PI05:
            if "observation/right_image" not in data:
                raise ValueError("observation/right_image is required for pi05")
            right_image = _parse_image(data["observation/right_image"])
            right_mask = np.True_
        else:
            right_image = np.zeros_like(base_image)
            right_mask = np.False_ if mask_padding else np.True_
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """The first 12 action dims are RoboCasa's; the rest is padding."""

    def __call__(self, data: dict) -> dict:
        """Strip the padding from the predicted action chunk."""
        return {"actions": np.asarray(data["actions"][:, :12])}


def make_robocasa_config(name: str = "pi05_pretrain_human300"):
    """The fork's ``pi05_pretrain_human300`` TrainConfig (inference parts only)."""
    import openpi.models.pi0_config as pi0_config
    from openpi.training import config as _config
    from typing_extensions import override

    @dataclasses.dataclass(frozen=True)
    class RobocasaDataConfig(_config.DataConfigFactory):
        repo_id: str | None = None  # type: ignore[assignment]

        @override
        def create(
            self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
        ) -> _config.DataConfig:
            """Inference transforms; norm stats are injected by ``serve_openpi.py``."""
            data_transforms = transforms.Group(
                inputs=[
                    RobocasaInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type
                    )
                ],
                outputs=[RobocasaOutputs()],
            )
            model_transforms = _config.ModelTransformFactory()(model_config)
            base = self.create_base_config(assets_dirs, model_config)
            return dataclasses.replace(
                base,
                repack_transforms=transforms.Group(),
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                # the fork's DataConfig keeps openpi's older default (mean/std normalisation);
                # the vendored openpi would switch pi05 checkpoints to quantile norm
                use_quantile_norm=False,
            )

    return _config.TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
        data=RobocasaDataConfig(),
    )
