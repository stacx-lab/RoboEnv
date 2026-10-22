"""Measured-state task predicates and sustained, ordered goal completion."""

import copy
import math

import numpy as np

from ...robots.booster_t2.task import HAND_LINKS


def _numbers(value, count, field):
    if not isinstance(value, list) or len(value) != count or any(
        type(v) not in (float, int) or not math.isfinite(v) for v in value
    ):
        raise ValueError(f"{field} requires {count} finite numbers")


def validate(spec):
    fields = {"version", "scene", "start", "items", "boxes", "stages", "hold_s"}
    if not isinstance(spec, dict) or set(spec) != fields or spec["version"] != 1:
        raise ValueError("Invalid T2 task definition")
    if spec["scene"] not in (None, "kitchen"):
        raise ValueError("Unknown T2 scene")
    _numbers(spec["start"], 3, "start")
    if type(spec["hold_s"]) not in (float, int) or not 0 < spec["hold_s"] <= 10:
        raise ValueError("hold_s must be positive and at most ten seconds")
    if not isinstance(spec["items"], dict) or not isinstance(spec["boxes"], list):
        raise ValueError("Invalid T2 items or boxes")
    for name, item in spec["items"].items():
        if (not isinstance(name, str) or not name.isidentifier() or not spec["scene"]
                or not isinstance(item, dict) or set(item) != {"model", "xy", "surface", "yaw"}
                or type(item["model"]) is not int or item["model"] not in (0, 2, 6, 12)
                or item["surface"] not in ("dining_table", "kitchen_island")
                or type(item["yaw"]) not in (float, int) or not math.isfinite(item["yaw"])):
            raise ValueError("Invalid T2 item")
        _numbers(item["xy"], 2, "item position")
    names = set()
    for box in spec["boxes"]:
        if (not isinstance(box, dict) or set(box) != {"name", "pos", "size", "color"}
                or not isinstance(box["name"], str) or not box["name"].isidentifier()
                or box["name"] in names or box["name"] in spec["items"]):
            raise ValueError("Invalid or duplicate T2 box")
        names.add(box["name"])
        for field in ("pos", "size", "color"):
            _numbers(box[field], 3, field)
        if min(box["size"]) <= 0 or any(not 0 <= v <= 1 for v in box["color"]):
            raise ValueError("Invalid box extent or color")
    if not isinstance(spec["stages"], list):
        raise ValueError("stages must be a list")
    fields = {
        "touch": {"kind", "box", "side"},
        "lift": {"kind", "item", "height", "side"},
        "place": {"kind", "item", "surface", "xy", "radius", "upright"},
        "contain": {"kind", "item", "container"},
    }
    for stage in spec["stages"]:
        if not isinstance(stage, list) or not stage:
            raise ValueError("Each stage needs at least one goal")
        for goal in stage:
            if not isinstance(goal, dict):
                raise ValueError("Each goal must be a mapping")
            kind = goal.get("kind")
            if kind not in fields or set(goal) != fields[kind]:
                raise ValueError("Invalid T2 goal fields")
            if kind == "touch":
                if goal["box"] not in names:
                    raise ValueError("Unknown touch box")
            elif goal["item"] not in spec["items"]:
                raise ValueError("Unknown goal item")
            if kind in ("touch", "lift") and goal["side"] not in ("either", "left", "right"):
                raise ValueError("Unknown hand side")
            if kind == "lift" and (type(goal["height"]) not in (float, int)
                                   or not 0.05 <= goal["height"] <= 0.3):
                raise ValueError("Invalid lift height")
            if kind == "contain" and (goal["container"] not in spec["items"]
                                       or spec["items"][goal["container"]]["model"] != 6
                                       or goal["item"] == goal["container"]):
                raise ValueError("Containment requires a separate bowl")
            if kind == "place":
                _numbers(goal["xy"], 2, "placement target")
                if (goal["surface"] not in ("dining_table", "kitchen_island")
                        or type(goal["upright"]) is not bool
                        or type(goal["radius"]) not in (float, int)
                        or not 0.04 <= goal["radius"] <= 0.4):
                    raise ValueError("Invalid placement goal")


class State:
    """Advance goals from physics observations only; never edit the world."""

    def __init__(self, spec):
        validate(spec)
        self.spec = copy.deepcopy(spec)
        self.reset()

    def reset(self):
        self.stage = 0
        self.since = None
        self.completed = False
        self.initial = {}
        self.max_lift = {}
        self.lift_since = {}
        self.lifted = {}
        self.metrics = {}

    def initialize(self, world):
        self.initial = {name: np.asarray(world.item_pose(name)[0]).copy()
                        for name in self.spec["items"]}
        self.max_lift = dict.fromkeys(self.initial, 0.0)
        self.lift_since = dict.fromkeys(self.initial, None)
        self.lifted = dict.fromkeys(self.initial, False)

    def _goal(self, goal, world, hands):
        kind = goal["kind"]
        side = goal.get("side", "either")
        selected = hands if side == "either" else hands[:6] if side == "left" else hands[6:]
        if kind == "touch":
            box = next(b for b in world.task.boxes if b.name == goal["box"])
            return min(box.distance(p) for p in selected) <= 0.01
        name = goal["item"]
        pos, quat = world.item_pose(name)
        if kind == "lift":
            return (pos[2] - self.initial[name][2] >= goal["height"]
                    and world.item_gap(name, selected) <= 0.04)
        # A slide, knock or initial placement cannot count as a pick-and-place.
        if (not self.lifted[name] or world.item_speed(name) > 0.05
                or np.linalg.norm(world.item_velocity(name)[3:]) > 0.3
                or world.item_gap(name, hands) < 0.025):
            return False
        corners = world.item_corners(name)
        if kind == "place":
            surface = world.task.scene.surfaces[goal["surface"]]
            w, x, y, z = quat
            return (abs(corners[:, 2].min() - surface.height) <= 0.015
                    and all(surface.contains(p[:2], margin=0.01) for p in corners)
                    and np.linalg.norm(np.asarray(pos[:2]) - goal["xy"]) <= goal["radius"]
                    and (not goal["upright"] or 1 - 2 * (x*x + y*y) >= math.cos(0.2)))
        # The can may protrude above the rim; its bottom must be inside the upright bowl.
        bowl = goal["container"]
        target, rotation = world.item_pose(bowl)
        bounds = world.item_corners(bowl)
        w, x, y, z = rotation
        return (1 - 2 * (x*x + y*y) >= math.cos(0.2)
                and np.linalg.norm(np.asarray(target[:2]) - self.initial[bowl][:2]) <= 0.10
                and np.max(np.linalg.norm(corners[:, :2] - target[:2], axis=1)) <= 0.065
                and bounds[:, 2].min() - 0.005 <= corners[:, 2].min()
                <= bounds[:, 2].max() - 0.01
                and world.item_speed(bowl) <= 0.05)

    def update(self, world):
        if self.completed or not self.spec["stages"]:
            return
        if not self.initial and self.spec["items"]:
            self.initialize(world)
        for name, start in self.initial.items():
            self.max_lift[name] = max(self.max_lift[name], float(world.item_pose(name)[0][2] - start[2]))
        hands = world.link_poses(HAND_LINKS)[0]
        for name, start in self.initial.items():
            held = (world.item_pose(name)[0][2] - start[2] >= 0.08
                    and world.item_gap(name, hands) <= 0.04)
            if not held:
                self.lift_since[name] = None
            elif self.lift_since[name] is None:
                self.lift_since[name] = world.time
            elif world.time - self.lift_since[name] >= 0.15 - 1e-9:
                self.lifted[name] = True
        goals = self.spec["stages"][self.stage]
        ok = all(self._goal(goal, world, hands) for goal in goals)
        if not ok:
            self.since = None
        elif self.since is None:
            self.since = world.time
        elif world.time - self.since >= self.spec["hold_s"] - 1e-9:
            self.stage += 1
            self.since = None
            self.completed = self.stage == len(self.spec["stages"])
            if not self.completed:
                self.initialize(world)
        self.metrics = {"completed_stages": self.stage, "total_stages": len(self.spec["stages"])}

    def record(self):
        return {"stage": self.stage, "since": self.since, "completed": self.completed,
                "initial": {n: p.tolist() for n, p in self.initial.items()},
                "max_lift": dict(self.max_lift), "lift_since": dict(self.lift_since),
                "lifted": dict(self.lifted), "metrics": dict(self.metrics)}
