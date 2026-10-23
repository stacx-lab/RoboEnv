"""Version 2 measured-state predicates and sustained, ordered goal completion."""

import copy
import math

import numpy as np

from ...robots.booster_t2.task import HAND_LINKS

GOAL_FIELDS = {
    "touch": {"box", "side"},
    "lift": {"item", "height", "side"},
    "place": {"item", "surface", "zone", "upright"},
    "contain": {"item", "container"},
    "handover": {"item", "from", "to", "height"},
    "slide": {"item", "surface", "zone", "distance"},
    "rotate": {"item", "surface", "yaw", "tolerance", "drift"},
    "stack": {"item", "support", "tolerance"},
    "carry": {"item", "surface", "zone", "tilt", "contents"},
    "joint": {"object", "joints", "min", "max", "absolute"},
    "push": {"object", "direction", "distance", "lateral"},
}
ARTICULATED_JOINTS = {
    0: {f"articulate_joint_{i}" for i in (*range(16), 17)},
    2: {f"articulate_joint_{i}" for i in range(5)},
    3: {"articulate_joint_1", "articulate_joint_2"},
    4: {"articulate_joint_0", "articulate_joint_1"},
    5: {"articulate_joint_0", "articulate_joint_2"},
}


def _numbers(value, count, field):
    if not isinstance(value, list) or len(value) != count or any(
        type(v) not in (float, int) or not math.isfinite(v) for v in value
    ):
        raise ValueError(f"{field} requires {count} finite numbers")


def _scalar(value, lower, upper, field):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Invalid {field}")


def validate(spec):
    fields = {"version", "scene", "start", "items", "boxes", "articulated", "stages", "hold_s"}
    if not isinstance(spec, dict) or set(spec) != fields or type(spec["version"]) is not int or spec["version"] != 2:
        raise ValueError("Expected a version 2 T2 task definition")
    if spec["scene"] not in (None, "kitchen"):
        raise ValueError("Unknown T2 scene")
    _numbers(spec["start"], 3, "start")
    _scalar(spec["hold_s"], 0.1, 10, "hold_s")
    if (not isinstance(spec["items"], dict) or not isinstance(spec["boxes"], list)
            or not isinstance(spec["articulated"], dict)):
        raise ValueError("Invalid T2 objects")
    names = set()

    def name_check(name):
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError("Invalid or duplicate T2 object name")
        names.add(name)

    for name, item in spec["items"].items():
        name_check(name)
        if (not spec["scene"] or not isinstance(item, dict)
                or set(item) != {"model", "xy", "surface", "yaw"}
                or type(item["model"]) is not int or item["model"] not in (0, 2, 6, 12)
                or item["surface"] not in ("dining_table", "kitchen_island")):
            raise ValueError("Invalid T2 item")
        _numbers(item["xy"], 2, "item position")
        _scalar(item["yaw"], -math.pi, math.pi, "item yaw")
    for name, obj in spec["articulated"].items():
        name_check(name)
        if (not isinstance(obj, dict) or set(obj) != {"model", "xy", "support", "yaw", "joints"}
                or type(obj["model"]) is not int or obj["model"] not in ARTICULATED_JOINTS
                or not isinstance(obj["joints"], dict)
                or set(obj["joints"]) - ARTICULATED_JOINTS[obj["model"]]):
            raise ValueError("Invalid articulated T2 object")
        _numbers(obj["xy"], 2, "articulated position")
        _scalar(obj["support"], 0, 1.5, "articulated support")
        _scalar(obj["yaw"], -math.pi, math.pi, "articulated yaw")
        for value in obj["joints"].values():
            _scalar(value, -math.pi, math.pi, "articulated initial joint")
    boxes = {}
    for box in spec["boxes"]:
        if not isinstance(box, dict) or set(box) != {"name", "pos", "size", "color"}:
            raise ValueError("Invalid T2 box")
        name_check(box["name"])
        boxes[box["name"]] = box
        for field in ("pos", "size", "color"):
            _numbers(box[field], 3, field)
        if min(box["size"]) <= 0 or any(not 0 <= v <= 1 for v in box["color"]):
            raise ValueError("Invalid box extent or color")
    if not isinstance(spec["stages"], list) or not spec["stages"]:
        raise ValueError("stages must be a nonempty list")
    for stage in spec["stages"]:
        if not isinstance(stage, list) or not stage:
            raise ValueError("Each stage needs at least one goal")
        for goal in stage:
            if not isinstance(goal, dict) or goal.get("kind") not in GOAL_FIELDS:
                raise ValueError("Unknown T2 goal")
            kind = goal["kind"]
            if set(goal) != GOAL_FIELDS[kind] | {"kind"}:
                raise ValueError("Invalid T2 goal fields")
            if "item" in goal and goal["item"] not in spec["items"]:
                raise ValueError("Unknown goal item")
            if kind == "touch" and goal["box"] not in boxes:
                raise ValueError("Unknown touch box")
            if kind in ("touch", "lift") and goal["side"] not in ("either", "left", "right"):
                raise ValueError("Unknown hand side")
            if kind in ("lift", "handover"):
                _scalar(goal["height"], .05, .3, "lift height")
            if kind == "handover" and {goal["from"], goal["to"]} != {"left", "right"}:
                raise ValueError("Handover needs two different hands")
            if "surface" in goal and goal["surface"] not in ("dining_table", "kitchen_island"):
                raise ValueError("Unknown target surface")
            if "zone" in goal:
                from ...robots.booster_t2.scenes import load_scene

                if goal["zone"] not in boxes or not spec["scene"]:
                    raise ValueError("Unknown placement zone")
                zone = boxes[goal["zone"]]
                surface = load_scene(spec["scene"]).surfaces[goal["surface"]]
                if (zone["size"][2] > .005 or abs(zone["pos"][2] - surface.height) > .005
                        or any(not surface.contains([zone["pos"][0] + x * zone["size"][0] / 2,
                                                      zone["pos"][1] + y * zone["size"][1] / 2])
                               for x in (-1, 1) for y in (-1, 1))):
                    raise ValueError("Placement zone must lie on the target surface")
            if kind == "place" and type(goal["upright"]) is not bool:
                raise ValueError("upright must be boolean")
            if kind == "contain" and (goal["container"] not in spec["items"]
                    or spec["items"][goal["container"]]["model"] != 6 or goal["item"] == goal["container"]):
                raise ValueError("Containment requires a separate bowl")
            if kind == "slide":
                _scalar(goal["distance"], .1, 2, "slide distance")
            if kind == "rotate":
                _scalar(goal["yaw"], -math.pi, math.pi, "target yaw")
                _scalar(goal["tolerance"], .05, .3, "yaw tolerance")
                _scalar(goal["drift"], .02, .15, "rotation drift")
            if kind == "stack":
                if goal["support"] not in spec["items"] or goal["support"] == goal["item"]:
                    raise ValueError("Stacking needs a separate support item")
                _scalar(goal["tolerance"], .005, .03, "stack alignment")
            if kind == "carry":
                _scalar(goal["tilt"], .15, .6, "carry tilt")
                if (not isinstance(goal["contents"], list) or len(set(goal["contents"])) != len(goal["contents"])
                        or any(n not in spec["items"] or n == goal["item"] for n in goal["contents"])
                        or goal["contents"] and spec["items"][goal["item"]]["model"] != 6):
                    raise ValueError("Carry contents require a bowl and distinct items")
            if kind in ("joint", "push"):
                if goal["object"] not in spec["articulated"]:
                    raise ValueError("Unknown articulated goal object")
                model = spec["articulated"][goal["object"]]["model"]
                if kind == "joint":
                    if (not isinstance(goal["joints"], list) or not goal["joints"]
                            or set(goal["joints"]) - ARTICULATED_JOINTS[model]):
                        raise ValueError("Unknown articulated goal joints")
                    _scalar(goal["min"], -math.pi, math.pi, "joint minimum")
                    _scalar(goal["max"], goal["min"], math.pi, "joint maximum")
                    if type(goal["absolute"]) is not bool or goal["absolute"] and goal["min"] < 0:
                        raise ValueError("Invalid absolute joint interval")
                else:
                    if model != 0:
                        raise ValueError("Push requires the free-base chair")
                    _numbers(goal["direction"], 2, "push direction")
                    if not math.isclose(np.linalg.norm(goal["direction"]), 1, abs_tol=1e-6):
                        raise ValueError("Push direction must have unit length")
                    _scalar(goal["distance"], .2, 2, "push distance")
                    _scalar(goal["lateral"], .05, .5, "push lateral tolerance")


def upright(quat, tolerance=.2):
    _, x, y, _ = quat
    return 1 - 2 * (x*x + y*y) >= math.cos(tolerance)


def yaw(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def angle_error(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def contained(world, item, bowl, tilt=.2):
    target, rotation = world.item_pose(bowl)
    corners, bounds = world.item_corners(item), world.item_corners(bowl)
    return (upright(rotation, tilt) and np.max(np.linalg.norm(corners[:, :2] - target[:2], axis=1)) <= .065
            and bounds[:, 2].min() - .005 <= corners[:, 2].min() <= bounds[:, 2].max() - .01)


class State:
    """Observe physics; stages require sustained goals and new touch events."""

    def __init__(self, spec):
        validate(spec)
        self.spec = copy.deepcopy(spec)
        self.reset()

    def reset(self):
        self.stage, self.since, self.completed = 0, None, False
        self.initial, self.initial_rotation, self.art_initial = {}, {}, {}
        self.max_lift, self.lift_since, self.lifted = {}, {}, {}
        self.touched, self.handover, self.violated, self.armed = {}, {}, {}, {}
        self.metrics = {}
        self.initialized = False

    def initialize(self, world):
        self.initial = {n: np.asarray(world.item_pose(n)[0]).copy() for n in self.spec["items"]}
        self.initial_rotation = {n: np.asarray(world.item_pose(n)[1]).copy() for n in self.initial}
        self.art_initial = {n: world.articulated_state(n) for n in self.spec["articulated"]}
        self.max_lift = dict.fromkeys(self.initial, 0.0)
        self.lift_since = dict.fromkeys(self.initial, None)
        self.lifted = dict.fromkeys(self.initial, False)
        self.touched = dict.fromkeys(self.initial, False)
        self.handover, self.violated, self.armed = {}, {}, {}
        self.initialized = True

    def _resting(self, world, name, hands):
        return (world.item_speed(name) <= .05 and np.linalg.norm(world.item_velocity(name)[3:]) <= .3
                and world.item_gap(name, hands) >= .025)

    def _on_surface(self, world, name, surface):
        corners = world.item_corners(name)
        support = world.task.scene.surfaces[surface]
        return (abs(corners[:, 2].min() - support.height) <= .015
                and all(support.contains(p[:2], margin=.01) for p in corners))

    def _in_zone(self, goal, world):
        box = next(b for b in world.task.boxes if b.name == goal["zone"])
        # All corners must lie in the visible rectangular footprint, including its corners.
        corners = world.item_corners(goal["item"])
        return bool(np.all(np.abs(corners[:, :2] - box.pos[:2]) <= np.asarray(box.size[:2]) / 2 + 1e-9))

    def _goal(self, goal, world, hands, key):
        kind = goal["kind"]
        side = goal.get("side", "either")
        selected = hands if side == "either" else hands[:6] if side == "left" else hands[6:]
        if kind == "touch":
            box = next(b for b in world.task.boxes if b.name == goal["box"])
            touching = min(box.distance(p) for p in selected) <= .01
            if not touching:
                self.armed[key] = True
            return touching and self.armed.get(key, False)
        if kind in ("joint", "push"):
            name = goal["object"]
            current = world.articulated_state(name)
            initial = self.art_initial[name]
            if kind == "joint":
                return any(goal["min"] <= (abs(current["joints"][j]) if goal["absolute"]
                                           else current["joints"][j]) <= goal["max"]
                           and abs(current["joints"][j] - initial["joints"][j]) >= .2
                           and abs(current["joint_velocities"][j]) <= .25 for j in goal["joints"])
            delta = np.asarray(current["pose"][:3]) - initial["pose"][:3]
            along = float(np.dot(delta[:2], goal["direction"]))
            lateral = np.linalg.norm(delta[:2] - along * np.asarray(goal["direction"]))
            return (along >= goal["distance"] and lateral <= goal["lateral"] and abs(delta[2]) <= .08
                    and upright(current["pose"][3:], .35) and np.linalg.norm(current["velocity"][:3]) <= .05
                    and np.linalg.norm(current["velocity"][3:]) <= .3)
        name = goal["item"]
        pos, quat = world.item_pose(name)
        height = pos[2] - self.initial[name][2]
        if kind == "lift":
            return height >= goal["height"] and world.item_gap(name, selected) <= .04
        if kind == "handover":
            source = hands[:6] if goal["from"] == "left" else hands[6:]
            target = hands[:6] if goal["to"] == "left" else hands[6:]
            a, b = world.item_gap(name, source), world.item_gap(name, target)
            progress = self.handover.setdefault(key, {"since": None, "ready": False})
            if height < goal["height"] or min(a, b) > .04:
                progress.update(since=None, ready=False)
            elif a <= .04 and b >= .08:
                if progress["since"] is None:
                    progress["since"] = world.time
                elif world.time - progress["since"] >= .15 - 1e-9:
                    progress["ready"] = True
            elif not progress["ready"]:
                progress["since"] = None
            return (progress["ready"]
                    and height >= goal["height"] and b <= .04 and a >= .08)
        if kind == "carry" and self.max_lift[name] >= .02:
            if not upright(quat, goal["tilt"]) or any(not contained(world, n, name, goal["tilt"]) for n in goal["contents"]):
                self.violated[key] = True
        if kind in ("slide", "rotate"):
            if not self._on_surface(world, name, goal["surface"]):
                self.violated[key] = True
            if not self.touched[name] or self.violated.get(key) or not self._resting(world, name, hands):
                return False
            if kind == "slide":
                return (np.linalg.norm(np.asarray(pos[:2]) - self.initial[name][:2]) >= goal["distance"]
                        and self._in_zone(goal, world))
            return (angle_error(yaw(quat), goal["yaw"]) <= goal["tolerance"]
                    and angle_error(yaw(quat), yaw(self.initial_rotation[name])) >= .5
                    and np.linalg.norm(np.asarray(pos[:2]) - self.initial[name][:2]) <= goal["drift"]
                    and upright(quat))
        if not self.lifted[name] or not self._resting(world, name, hands):
            return False
        if kind in ("place", "carry"):
            return (self._on_surface(world, name, goal["surface"]) and self._in_zone(goal, world)
                    and (not goal.get("upright", False) or upright(quat))
                    and (kind != "carry" or (not self.violated.get(key) and upright(quat, goal["tilt"])
                         and all(contained(world, n, name, goal["tilt"]) for n in goal["contents"]))))
        if kind == "stack":
            bottom = goal["support"]
            a, b = world.item_corners(name), world.item_corners(bottom)
            target, target_q = world.item_pose(bottom)
            return (abs(a[:, 2].min() - b[:, 2].max()) <= .015
                    and np.linalg.norm(np.asarray(pos[:2]) - target[:2]) <= goal["tolerance"]
                    and np.linalg.norm(np.asarray(target) - self.initial[bottom]) <= .04
                    and self._on_surface(world, bottom, self.spec["items"][bottom]["surface"])
                    and upright(quat) and upright(target_q) and self._resting(world, bottom, hands))
        bowl = goal["container"]
        return (contained(world, name, bowl)
                and np.linalg.norm(np.asarray(world.item_pose(bowl)[0][:2]) - self.initial[bowl][:2]) <= .10
                and self._resting(world, bowl, hands))

    def update(self, world):
        if self.completed:
            return
        if not self.initialized:
            self.initialize(world)
        hands = world.link_poses(HAND_LINKS)[0]
        for name, start in self.initial.items():
            height = float(world.item_pose(name)[0][2] - start[2])
            self.max_lift[name] = max(self.max_lift[name], height)
            near = world.item_gap(name, hands) <= .04
            self.touched[name] = self.touched[name] or near
            if not (height >= .08 and near):
                self.lift_since[name] = None
            elif self.lift_since[name] is None:
                self.lift_since[name] = world.time
            elif world.time - self.lift_since[name] >= .15 - 1e-9:
                self.lifted[name] = True
        # Evaluate every conjunct so history and invariant checks cannot short-circuit.
        results = [bool(self._goal(g, world, hands, str(i)))
                   for i, g in enumerate(self.spec["stages"][self.stage])]
        if not all(results):
            self.since = None
        elif self.since is None:
            self.since = world.time
        elif world.time - self.since >= self.spec["hold_s"] - 1e-9:
            self.stage += 1
            self.since = None
            self.completed = self.stage == len(self.spec["stages"])
            if not self.completed:
                self.initialize(world)
        self.metrics = {"completed_stages": self.stage, "total_stages": len(self.spec["stages"]),
                        "goal_conditions": results, "invariant_violations": dict(self.violated)}

    def record(self):
        return {"stage": self.stage, "since": self.since, "completed": self.completed,
                "initial": {n: p.tolist() for n, p in self.initial.items()},
                "initial_rotation": {n: p.tolist() for n, p in self.initial_rotation.items()},
                "art_initial": copy.deepcopy(self.art_initial), "initialized": self.initialized,
                "max_lift": dict(self.max_lift), "lift_since": dict(self.lift_since),
                "lifted": dict(self.lifted), "touched": dict(self.touched), "armed": dict(self.armed),
                "handover": dict(self.handover), "violated": dict(self.violated), "metrics": dict(self.metrics)}
