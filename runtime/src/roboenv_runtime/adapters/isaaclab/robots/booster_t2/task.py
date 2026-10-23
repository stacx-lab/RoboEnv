"""Physical T2 scene construction from an explicit task definition."""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scenes import load_scene

HAND_LINKS = tuple(f"{side}_{part}" for side in ("left", "right") for part in (
    "palm", "thumb_touch_link", "index_touch_link", "middle_touch_link",
    "ring_touch_link", "pinky_touch_link",
))


@dataclass(frozen=True)
class Box:
    name: str
    size: tuple
    pos: tuple
    color: tuple = (0.6, 0.6, 0.6)

    def distance(self, point):
        return float(np.linalg.norm(np.maximum(np.abs(np.asarray(point) - self.pos)
                                               - np.asarray(self.size) / 2, 0)))


@dataclass(frozen=True)
class Item:
    name: str
    model: int
    xy: tuple
    support: float
    yaw: float = 0.0
    fixed: bool = False


@dataclass(frozen=True)
class Articulated:
    name: str
    model: int
    xy: tuple
    support: float
    yaw: float
    joints: dict


class SceneTask:
    def __init__(self, name, spec, asset_directory):
        from ...tasks.t2_household.scoring import State, validate

        if spec.get("version") in (3, 4):
            from ...tasks.t2_household.release import State, validate
        validate(spec)
        self.asset_directory = Path(asset_directory)
        self.name, self.spec = name, copy.deepcopy(spec)
        self.scene = load_scene(spec["scene"]) if spec["scene"] else None
        if self.scene:
            self.scene.asset_directory = Path(asset_directory) / "scenes"
        self.start = tuple(spec["start"])
        self.boxes = [Box(b["name"], tuple(b["size"]), tuple(b["pos"]), tuple(b["color"]))
                      for b in spec["boxes"]]
        if spec["version"] in (3, 4):
            self.items = [Item(name, item["model"], tuple(item["pos"][:2]), item["pos"][2],
                               item["yaw"], item["fixed"]) for name, item in spec["items"].items()]
            self.articulated = [Articulated(name, obj["model"], tuple(obj["pos"][:2]), obj["pos"][2],
                                            obj["yaw"], dict(obj["joints"])) for name, obj in spec["articulated"].items()]
            self.state = State(spec)
            return
        self.items = [Item(name, item["model"], tuple(item["xy"]),
                           self.scene.surfaces[item["surface"]].height, item["yaw"])
                      for name, item in spec["items"].items()]
        self.articulated = [Articulated(name, obj["model"], tuple(obj["xy"]),
                                       obj["support"], obj["yaw"], dict(obj["joints"]))
                            for name, obj in spec["articulated"].items()]
        self.state = State(spec)

    def reset(self):
        self.state.reset()

    def evaluate(self, world):
        self.state.update(world)
