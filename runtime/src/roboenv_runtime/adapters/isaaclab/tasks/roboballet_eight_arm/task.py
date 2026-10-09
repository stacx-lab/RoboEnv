"""Registered factory for the original eight-arm RoboBallet reaching scene."""


class Task:
    name = "roboballet-eight-arm"
    embodiment = "roboballet_eight_panda"

    @classmethod
    def create_environment(cls, **kwargs):
        from ...robots.roboballet_eight_panda.environment import RoboBalletEnv

        return RoboBalletEnv(**kwargs)
