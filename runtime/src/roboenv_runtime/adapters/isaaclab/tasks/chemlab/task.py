"""Instrument-only laboratory tasks on the existing Isaac Lab adapter."""


class Task:
    name = "chemlab-instruments"
    embodiment = "chemlab_instruments"

    @classmethod
    def create_environment(cls, **kwargs):
        from .environment import ChemLabEnvironment

        return ChemLabEnvironment(**kwargs)
