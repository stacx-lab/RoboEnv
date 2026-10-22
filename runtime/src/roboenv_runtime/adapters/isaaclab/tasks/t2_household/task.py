"""Configuration-driven T2 scenes and goals, independent of benchmark selection."""

import copy

from .scoring import validate


class Task:
    name = "t2-household"
    embodiment = "booster_t2"

    @staticmethod
    def definition(task_spec):
        validate(task_spec)
        return copy.deepcopy(task_spec)

    @classmethod
    def create_environment(cls, **kwargs):
        from ...robots.booster_t2.environment import T2Env

        return T2Env(task_name=cls.name, **kwargs)
