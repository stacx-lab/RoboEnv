"""Extra RoboTwin embodiments: the same tasks under a different robot.

RoboTwin builds every task around whatever pair of arms it is given, so a task is
not tied to one robot.  ``roboenv_runtime.adapters.robotwin.backend`` registers the three the
harness earned its certificates on (dual Piper with mplib or curobo, and YAM); this
module adds the ones we want for *pretrained policies*, whose checkpoints are trained
on one specific embodiment and are useless on any other.

``aloha_agilex`` is the RoboTwin 2.0 leaderboard robot — one dual-arm ALOHA-AgileX
(``assets/embodiments/aloha-agilex``, base at ``[0, -0.65, 0]``, curobo planner), and
what every public RoboTwin checkpoint we know of was fine-tuned on.  Measured
2026-08-28: `InternVLA-A1.5-RoboTwin` scores 0/8 on dual Piper because the same joint
angles put its hands 7 cm *below* the table, out of every camera.

    make_adapter("robotwin", task_name="click_bell", embodiment="aloha_agilex")

The Piper registrations are untouched, so every existing certificate keeps its
meaning; a task chooses the robot through ``adapter_kwargs``.
"""

from __future__ import annotations

import functools

__all__ = ["EXTRA_EMBODIMENTS", "register_extra_envs"]

#: roboenv embodiment id -> the RoboTwin ``embodiment`` argument (``[name]`` is one
#: dual-arm robot, ``[left, right, distance]`` is two single arms).
EXTRA_EMBODIMENTS: dict[str, list] = {
    "aloha_agilex": ["aloha-agilex"],
}


def register_extra_envs() -> None:
    """Register ``robotwin_<task>_<embodiment>_low_level`` for :data:`EXTRA_EMBODIMENTS`.

    Idempotent, and cheap: no simulator is imported until an env is instantiated.
    """
    from capx.envs.base import register_env

    from roboenv_runtime.adapters.robotwin.backend import RoboTwinEnv, list_robotwin_tasks

    for task in list_robotwin_tasks():
        for name, embodiment in EXTRA_EMBODIMENTS.items():
            register_env(
                f"robotwin_{task}_{name}_low_level",
                functools.partial(
                    RoboTwinEnv,
                    task_name=task,
                    embodiment=list(embodiment),
                    # the embodiment's own default is curobo, whose prebuilt kernels do not
                    # run on this host ("no kernel image is available"); mplib is CPU-side
                    # and is what every Piper certificate was earned with anyway
                    planner="mplib_RRT",
                ),
            )
