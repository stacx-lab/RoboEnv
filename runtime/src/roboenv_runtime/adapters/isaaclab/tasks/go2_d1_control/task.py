"""Flat-ground control practice without a success predicate."""

from ...robots.go2_d1.task import Go2Scene, Go2Task, Go2TaskState


class Scene(Go2Scene):
    def sample(self, fallen):
        return {"fallen": fallen}


class Task(Go2Task):
    name = "go2-d1-control"
    state_type = Go2TaskState
    scene_type = Scene
