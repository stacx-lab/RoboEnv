"""Physical T2 scene construction from an explicit task definition."""

import copy
from dataclasses import dataclass
from pathlib import Path

from .scenes import load_scene


@dataclass(frozen=True)
class Item:
    name: str
    model: str
    xy: tuple
    support: float
    yaw: float = 0.0
    fixed: bool = False


@dataclass(frozen=True)
class Articulated:
    name: str
    model: str
    xy: tuple
    support: float
    yaw: float
    joints: dict


class SceneTask:
    def __init__(self, name, spec, asset_directory):
        from ...tasks.t2_household.release import State, validate

        validate(spec)
        self.asset_directory = Path(asset_directory)
        self.name, self.spec = name, copy.deepcopy(spec)
        self.scene = load_scene(spec["scene"])
        self.scene.asset_directory = Path(asset_directory) / "scenes"
        for obj in spec.get("articulated", {}).values():
            self.scene.excluded_prims += tuple(self.scene.replacements.get(obj["model"], ()))
        self.start = tuple(spec["start"])
        self.items = [Item(name, item["model"], tuple(item["pos"][:2]), item["pos"][2],
                           item["yaw"], item["fixed"]) for name, item in spec["items"].items()]
        self.articulated = [Articulated(name, obj["model"], tuple(obj["pos"][:2]), obj["pos"][2],
                                        obj["yaw"], dict(obj["joints"])) for name, obj in spec["articulated"].items()]
        self.state = State(spec)

    def reset(self):
        self.state.reset()

    def evaluate(self, world):
        self.state.update(world)
