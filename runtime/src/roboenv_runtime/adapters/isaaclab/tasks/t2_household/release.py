"""Version 3 household contracts, evaluated from native physical evidence.

All event transitions run at the physics rate, including brief hand contacts
and tool impacts. Tolerances implement visible relationships rather than
unannounced travel, lift-height or hand-origin requirements.
"""

import copy
import math

import numpy as np
from roboenv_runtime.adapters.urdf_geometry import matrix

from ...robots.booster_t2.household_models import (
    CART_BAY_INTERIOR,
    MODEL_NAMES,
    PORTAL_X,
    SURFACES,
    model,
)


GOALS = {
    "insert": {"cylinder", "cube", "sorter"},
    "ordered_touch": {"blue", "red"},
    "balance": {"scale", "weights"},
    "contain": {"item", "bowl", "table"},
    "hang": {"mug", "rack"},
    "scrape": {"blocks", "tool", "pan", "stand"},
    "carry_level": {"item", "source", "target"},
    "pull_cart": {"cart"},
    "deliver": {"items", "targets", "source"},
    "slide": {"item", "target", "surface"},
    "illuminate": {"lamp", "book"},
    "turn": {"item", "mat", "plate"},
    "stack": {"item", "support", "table"},
    "open_jar": {"jar", "lid", "dish", "table"},
    "open_lid": {"bin"},
    "pound": {"bench", "tool", "stand"},
    "push_chair": {"chair", "desk"},
    "fetch": {"item", "source", "target"},
    "store": {"remote", "drawer"},
    "parcel": {"parcel", "door", "target", "source"},
}


def vector(value, n, label):
    if (
        not isinstance(value, list)
        or len(value) != n
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"{label} needs {n} finite numbers")


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != {
        "version",
        "scene",
        "start",
        "items",
        "boxes",
        "articulated",
        "stages",
        "hold_s",
    }:
        raise ValueError("Invalid household task fields")
    if type(spec["version"]) is not int or spec["version"] != 3 or spec["scene"] != "household":
        raise ValueError("Expected a version 3 household scene")
    vector(spec["start"], 3, "start")
    if spec["boxes"] != []:
        raise ValueError("Household tasks use physical receivers, not marker boxes")
    if type(spec["hold_s"]) not in (float, int) or not 0.1 <= spec["hold_s"] <= 1.0:
        raise ValueError("Invalid household settling interval")
    names = set()
    for collection, articulated in ((spec["items"], False), (spec["articulated"], True)):
        if not isinstance(collection, dict):
            raise ValueError("Household objects must be a mapping")
        for name, d in collection.items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name in names
                or name in SURFACES
            ):
                raise ValueError("Invalid or duplicate household object")
            names.add(name)
            expected = {"model", "pos", "yaw", "joints" if articulated else "fixed"}
            if not isinstance(d, dict) or set(d) != expected or d["model"] not in MODEL_NAMES:
                raise ValueError("Unknown household asset or object fields")
            vector(d["pos"], 3, "object position")
            if type(d["yaw"]) not in (float, int) or not math.isfinite(d["yaw"]):
                raise ValueError("Invalid object yaw")
            m = model(d["model"])
            if bool(m.joints) != articulated:
                raise ValueError("Asset must use its actual rigid or articulated representation")
            if articulated:
                joints = {j.name: j for j in m.joints}
                if not isinstance(d["joints"], dict) or set(d["joints"]) != set(joints):
                    raise ValueError("Every native joint needs an explicit initial value")
                for key, value in d["joints"].items():
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not joints[key].limits[0] <= value <= joints[key].limits[1]
                    ):
                        raise ValueError("Joint value outside its physical limits")
            elif type(d["fixed"]) is not bool:
                raise ValueError("Fixed flag must be boolean")
    stages = spec["stages"]
    if (
        not isinstance(stages, list)
        or len(stages) != 1
        or not isinstance(stages[0], list)
        or len(stages[0]) != 1
    ):
        raise ValueError("Household tasks declare one complete semantic goal")
    goal = stages[0][0]
    if (
        not isinstance(goal, dict)
        or goal.get("kind") not in GOALS
        or set(goal) != {"kind"} | GOALS[goal["kind"]]
    ):
        raise ValueError("Unknown household goal or binding fields")
    for key, value in goal.items():
        if key == "kind":
            continue
        plural = key in {"weights", "blocks", "items", "targets"}
        if (plural and not isinstance(value, list)) or (not plural and not isinstance(value, str)):
            raise ValueError(f"Invalid household binding type: {key}")
        values = value if isinstance(value, list) else [value]
        if not values or any(
            not isinstance(n, str) or n not in names | set(SURFACES) for n in values
        ):
            raise ValueError(f"Unknown household goal binding: {key}")
        if len(values) != len(set(values)):
            raise ValueError("Repeated goal object")
        if key in {"table", "source", "surface", "desk"}:
            if any(n not in SURFACES for n in values):
                raise ValueError(f"{key} requires a room surface")
        elif any(n not in names for n in values):
            raise ValueError(f"{key} requires a task object")
    if goal["kind"] == "deliver" and (len(goal["items"]) != 2 or len(goal["targets"]) != 2):
        raise ValueError("Delivery requires two distinct object/receiver pairs")
    if goal["kind"] == "scrape" and len(goal["blocks"]) != 2:
        raise ValueError("Scraping requires two blocks")
    # Reject malformed bindings before the simulator starts, including an item
    # being used as its own target and the wrong mechanism masquerading as a goal.
    all_defs = {**spec["items"], **spec["articulated"]}
    types = {
        "insert": {"sorter": "sorter", "cylinder": "cylinder_block", "cube": "cube_block"},
        "balance": {"scale": "balance_scale"},
        "hang": {"mug": "mug", "rack": "mug_rack"},
        "scrape": {"tool": "scraper", "pan": "collection_pan", "stand": "tool_stand"},
        "pull_cart": {"cart": "cart"},
        "illuminate": {"lamp": "lamp", "book": "open_book"},
        "open_jar": {"jar": "jar", "lid": "jar_lid", "dish": "dish"},
        "open_lid": {"bin": "trash_bin"},
        "pound": {"bench": "pounding_bench", "tool": "mallet", "stand": "tool_stand"},
        "push_chair": {"chair": "chair"},
        "store": {"drawer": "drawer_cabinet", "remote": "remote"},
        "parcel": {"door": "door", "parcel": "parcel"},
    }
    for role, asset in types.get(goal["kind"], {}).items():
        if all_defs.get(goal[role], {}).get("model") != asset:
            raise ValueError(f"{role} requires {asset}")
    for a, b in (("item", "target"), ("item", "bowl"), ("item", "support")):
        if a in goal and b in goal and goal[a] == goal[b]:
            raise ValueError("An object cannot be its own receiver")


class State:
    """Native observations only; no task-solving actions or scripted outcomes."""

    def __init__(self, spec):
        validate(spec)
        self.spec = copy.deepcopy(spec)
        self.goal = self.spec["stages"][0][0]
        self.reset()

    def reset(self):
        self.completed = False
        self.initialized = False
        self.since = None
        self.stage = 0
        self.events = {}
        self.previous = {}
        self.initial = {}
        self.metrics = {}

    def initialize(self, world):
        self.reset()
        self.initialized = True
        m = world.measurements
        self.initial = {n: m.pose(n)[0].copy() for n in m.models}
        self.previous = {n: p.copy() for n, p in self.initial.items()}
        self.events = {
            "picked": [],
            "touch_stage": 0,
            "qualified": {},
            "last_touch": {},
            "distance": 0.0,
            "pulled": 0.0,
        }
        self.initial_joints = {
            n: dict(world.articulated_joints(n)) for n in self.spec["articulated"]
        }
        self.previous_joints = copy.deepcopy(self.initial_joints)

    def _pickup_events(self, m):
        for n, d in self.spec["items"].items():
            if d["fixed"] or n in self.events["picked"]:
                continue
            supported = any(
                m.supported(n, t) for t in (*SURFACES, *m.models, "floor", "ground") if t != n
            )
            if m.held(n) and not supported:
                self.events["picked"].append(n)

    def _carry(self, m, item, source, target, *, both=False, level=False):
        e = self.events
        held = m.held(item)
        if held and m.supported(item, source):
            e["carry_armed"] = True
        if held and not m.supported(item, source) and e.get("carry_armed"):
            e.update(carry_armed=False, carry_active=True, carry_invalid=False, tilt_since=None)
        if e.get("carry_active"):
            if (
                both
                and not (m.hand_support(item, "left") and m.hand_support(item, "right"))
                and not m.supported(item, target)
            ):
                e.setdefault("one_hand_since", m.world.time)
                if m.world.time - e["one_hand_since"] > 0.10:
                    e["carry_invalid"] = True
            else:
                e.pop("one_hand_since", None)
            if level and held and not m.upright(item, 0.52):
                if e.get("tilt_since") is None:
                    e["tilt_since"] = m.world.time
                if m.world.time - e["tilt_since"] > 0.10:
                    e["carry_invalid"] = True
            else:
                e["tilt_since"] = None
            if not held and not m.supported(item, target) and not m.supported(item, source):
                e.setdefault("unsupported_since", m.world.time)
                if m.world.time - e["unsupported_since"] > 0.10:
                    e["carry_invalid"] = True
            else:
                e.pop("unsupported_since", None)
        return bool(
            e.get("carry_active") and not e.get("carry_invalid") and m.on(item, target, True)
        )

    def _evaluate(self, m):
        g = self.goal
        k = g["kind"]
        e = self.events
        c = m.contacts
        picked = lambda n: n in e["picked"]
        result = {}
        if k == "ordered_touch":
            blue = m.touch(g["blue"], "left")
            red = m.touch(g["red"], "right")
            stage = e["touch_stage"]
            if stage == 0 and blue:
                e["touch_stage"] = 1
            elif stage == 1 and not blue:
                e["touch_stage"] = 2
                if red and not e["last_touch"].get("red", False):
                    e["touch_stage"] = 3
            elif stage == 2 and red and not e["last_touch"].get("red", False):
                e["touch_stage"] = 3
            elif stage == 3 and not red:
                e["touch_stage"] = 4
            e["last_touch"]["red"] = red
            return {"ordered_contacts_and_withdrawals": e["touch_stage"] == 4}
        if k == "insert":
            for role, region in (("cylinder", "round"), ("cube", "square")):
                n = g[role]
                target = g["sorter"]
                link, center, half = m.models[target].regions[region]
                local = m.local(m.pose(n)[0], target, link) - center
                old = m.local(self.previous[n], target, link) - center
                if old[2] < 0 <= local[2]:
                    e["qualified"].pop(n, None)
                if (
                    old[2] >= 0 > local[2]
                    and picked(n)
                ):
                    e["qualified"][n] = bool(np.all(np.abs(local[:2]) < np.asarray(half[:2])))
                result[role + "_inserted"] = bool(e["qualified"].get(n) and m.inside(n, target))
            return result
        if k == "balance":
            scale = g["scale"]
            valid = [n for n in g["weights"] if m.inside(n, scale, "pan")]
            q = m.joint(scale, "balance")
            # The base legitimately touches the table; only moving links must
            # be unsupported. Examine those links rather than the whole actor.
            external = False
            for x, y, _ in c.pairs.values():
                if y[0] == scale:
                    x, y = y, x
                if x[0] == scale and x[1] in ("beam", "pan", "reference") and y[0] != scale:
                    # A weight on the beam or reference side is an external
                    # counterweight, not a legitimate load in the empty pan.
                    if not (x[1] == "pan" and y[0] in valid):
                        external = True
            return {
                "weights_in_pan": bool(valid),
                "beam_level": abs(q) < 0.045,
                "beam_settled": abs(m.joint_speed(scale, "balance")) < 0.025,
                "unassisted": not external,
                "weights_released": all(m.released(n) for n in g["weights"]),
            }
        if k == "contain":
            return {
                "picked_up": picked(g["item"]),
                "inside": m.inside(g["item"], g["bowl"]),
                "bowl_on_table": m.on(g["bowl"], g["table"]) and m.upright(g["bowl"]),
            }
        if k == "hang":
            mug, rack = g["mug"], g["rack"]
            engaged = any(
                x[2].startswith("handle") for x, _, _ in c.between(mug, rack, part="hook")
            )
            return {
                "picked_up": picked(mug),
                "handle_engaged": engaged,
                "hook_support": c.upward_force(mug, rack, part="hook")
                > 0.5 * m.models[mug].links[0].mass * 9.81,
                "released": m.released(mug),
                "settled": m.quiet(mug),
            }
        if k == "scrape":
            result["tool_picked"] = picked(g["tool"])
            for n in g["blocks"]:
                delta = np.linalg.norm(m.pose(n)[0][:2] - self.previous[n][:2])
                blade = c.touching(n, g["tool"], part="blade") and m.held(g["tool"])
                if (
                    blade
                    and delta > 0.00002
                    and (m.supported(n, "activity_table") or m.supported(n, g["pan"]))
                ):
                    e["qualified"][n] = True
                inside = m.inside(n, g["pan"], settled=False)
                # Preserve causality while the block coasts into the pan;
                # settling latency must not erase a valid tool sweep.
                if any(m.touch(n, side) for side in ("left", "right")):
                    e["qualified"].pop(n, None)
                    e.pop(n + "_swept", None)
                if inside and e["qualified"].get(n):
                    e[n + "_swept"] = True
                if m.supported(n, "activity_table") and not inside:
                    e.pop(n + "_swept", None)
                    if m.quiet(n) and not blade:
                        e["qualified"].pop(n, None)
                result[n + "_swept_inside"] = bool(e.get(n + "_swept") and m.inside(n, g["pan"]))
            result["tool_returned"] = m.on(g["tool"], g["stand"])
            return result
        if k == "carry_level":
            return {
                "level_carry_and_placement": self._carry(
                    m, g["item"], g["source"], g["target"], level=True
                )
            }
        if k == "pull_cart":
            cart = g["cart"]
            pos, q = m.pose(cart)
            delta = pos[:2] - self.previous[cart][:2]
            distance = float(np.linalg.norm(delta))
            if distance > 0.00003:
                e["distance"] += distance
                to_robot = m.world.root_pose()[0][:2] - pos[:2]
                handle = any(
                    c.touching(s + "_hand", cart, part="handle") for s in ("left", "right")
                )
                grounded = m.supported(cart, "floor") or m.supported(cart, "ground")
                if handle and grounded and np.dot(delta, to_robot) > 0:
                    e["pulled"] += distance
            points = m.points(cart)
            parked = bool(
                np.all(
                    (points[:, :2] > CART_BAY_INTERIOR[0]) & (points[:, :2] < CART_BAY_INTERIOR[1])
                )
            )
            facing = (
                float(
                    np.dot(
                        matrix([0, 0, 0, *q[1:], q[0]])[:3, :3] @ np.array([0, -1, 0]), [0, 1, 0]
                    )
                )
                > 0.8
            )
            return {
                "pulled_transport": e["pulled"] > 0.65 * e["distance"],
                "in_bay": parked,
                "handle_outward": facing,
                "upright": m.upright(cart),
                "released_and_still": m.released(cart) and m.quiet(cart),
            }
        if k == "deliver":
            return {
                n + "_delivered": picked(n) and m.on(n, target, True)
                for n, target in zip(g["items"], g["targets"], strict=True)
            }
        if k == "slide":
            n = g["item"]
            support = m.supported(n, g["surface"]) or m.supported(n, g["target"])
            # A retry starts outside the visible mat, not at an arbitrary
            # off-center point after carrying the can onto the mat.
            points = m.local(m.points(n), g["target"])
            outside = np.linalg.norm(points[:, :2], axis=1).min() > m.models[g["target"]].radius
            if support and outside:
                if not e.get("slide_active"):
                    e.update(slide_active=True, slide_invalid=False, slide_touched=False)
            if e.get("slide_active"):
                e["slide_touched"] = e["slide_touched"] or any(
                    m.touch(n, s) for s in ("left", "right")
                )
                if not support:
                    e.setdefault("unsupported_since", m.world.time)
                    if m.world.time - e["unsupported_since"] > 0.05:
                        e.update(slide_active=False, slide_invalid=True)
                else:
                    e.pop("unsupported_since", None)
            return {
                "supported_slide": bool(
                    e.get("slide_active") and e.get("slide_touched") and not e.get("slide_invalid")
                ),
                "on_mat_center": m.on(n, g["target"], True),
            }
        if k == "illuminate":
            lamp = g["lamp"]
            head_touch = any(c.touching(s + "_hand", lamp, link="head") for s in ("left", "right"))
            switch_touch = any(c.touching(s + "_hand", lamp, link="switch") for s in ("left", "right"))
            if head_touch and abs(m.joint(lamp, "aim") - self.initial_joints[lamp]["aim"]) > 0.01:
                e["head_adjusted"] = True
            if switch_touch and m.joint(lamp, "power") > 0.18:
                e["switched_on"] = True
            return {
                "switch_actuated": bool(e.get("switched_on")),
                "power_on": m.joint(lamp, "power") > 0.18,
                "head_adjusted": bool(e.get("head_adjusted")),
                "pages_lit": m.lit_pages(lamp, g["book"]),
                "released": m.released(lamp),
                "head_still": abs(m.joint_speed(lamp, "aim")) < 0.03,
            }
        if k == "turn":
            n = g["item"]
            direction = m.pose(g["plate"])[0] - m.pose(n)[0]
            direction[2] = 0
            q = m.pose(n)[1]
            front = matrix([0, 0, 0, *q[1:], q[0]])[:3, :3] @ np.array([0, -1, 0])
            facing = np.dot(front, direction) / max(np.linalg.norm(direction), 1e-9) > math.cos(0.35)
            if m.supported(n, g["mat"]) and m.upright(n):
                e.pop("turn_unsupported_since", None)
                if not e.get("turn_valid") and not facing:
                    e.update(
                        turn_valid=True,
                        turn_touched=False,
                        turn_rotation=0.0,
                        turn_quat=m.pose(n)[1].tolist(),
                    )
                if e.get("turn_valid") and any(m.touch(n, s) for s in ("left", "right")):
                    e["turn_touched"] = True
                old = np.asarray(e.get("turn_quat", q))
                now = m.pose(n)[1]
                cosine = abs(np.dot(old, now)) / (np.linalg.norm(old) * np.linalg.norm(now))
                e["turn_rotation"] = max(
                    e.get("turn_rotation", 0.0), 2 * math.acos(float(np.clip(cosine, 0, 1)))
                )
            else:
                e.setdefault("turn_unsupported_since", m.world.time)
                if m.world.time - e["turn_unsupported_since"] > 0.05:
                    e["turn_valid"] = False
            return {
                "supported_turn": bool(
                    e.get("turn_valid")
                    and e.get("turn_touched")
                    and e.get("turn_rotation", 0) > 0.02
                ),
                "facing_plate": bool(facing),
                "released_on_mat": m.on(n, g["mat"]) and m.upright(n),
            }
        if k == "stack":
            return {
                "upper_picked": picked(g["item"]),
                "upper_supported": m.on(g["item"], g["support"]),
                "lower_on_table": m.on(g["support"], g["table"]),
                "both_upright": m.upright(g["item"]) and m.upright(g["support"]),
            }
        if k == "open_jar":
            jar, lid = g["jar"], g["lid"]
            roles = any(
                m.hand_support(jar, a) and m.hand_support(lid, b)
                for a, b in (("left", "right"), ("right", "left"))
            )
            lid_points = m.local(m.points(lid), jar)
            separated = not c.touching(jar, lid) and (
                lid_points[:, 2].min() > m.models[jar].height + 0.015
                or np.linalg.norm(m.local(m.pose(lid)[0], jar)[:2])
                > m.models[jar].radius + m.models[lid].radius
            )
            if roles and not m.supported(jar, g["table"]) and separated:
                e["opened_in_hands"] = True
            return {
                "bimanual_opening": bool(e.get("opened_in_hands")),
                "jar_returned": m.on(jar, g["table"]) and m.upright(jar),
                "lid_on_dish": m.on(lid, g["dish"]),
            }
        if k == "open_lid":
            angle = abs(m.joint(g["bin"], "hinge"))
            # Front-edge clearance and exposed opening derive from the lid's
            # real span; no zero-angle assumption about an imported asset.
            return {
                "mouth_accessible": 0.30 * math.sin(angle) > 0.22 and abs(math.cos(angle)) < 0.65
            }
        if k == "pound":
            tool, bench = g["tool"], g["bench"]
            result["tool_picked"] = picked(tool)
            head = next(p for p in m.models[tool].links[0].parts if p.name == "head")
            q = m.pose(tool)[1]
            lever = matrix([0, 0, 0, *q[1:], q[0]])[:3, :3] @ np.asarray(head.pos)
            velocity = m.speed(tool)
            head_vz = float((velocity[:3] + np.cross(velocity[3:], lever))[2])
            for peg in ("peg_a", "peg_b"):
                touch = c.touching(bench, tool, part="head") and any(
                    x[1] == peg for x, _, _ in c.between(bench, tool, part="head")
                )
                key = peg + "_touch"
                if (
                    touch
                    and not e["last_touch"].get(key, False)
                    and m.held(tool)
                    and min(head_vz, e.get("previous_head_vz", 0.0)) < -0.005
                ):
                    e[peg + "_tap"] = m.world.time
                current = m.joint(bench, peg)
                descent = max(0.0, self.previous_joints[bench][peg] - current)
                if m.world.time - e.get(peg + "_tap", -100) < 0.15:
                    e[peg + "_driven"] = e.get(peg + "_driven", 0.0) + descent
                e["last_touch"][key] = touch
                result[peg + "_tapped_flush"] = (
                    current < 0.004
                    and e.get(peg + "_driven", 0.0) >= self.initial_joints[bench][peg] - 0.006
                )
            result["tool_returned"] = m.on(tool, g["stand"])
            e["previous_head_vz"] = head_vz
            return result
        if k == "push_chair":
            n = g["chair"]
            pos, _ = m.pose(n)
            delta = pos[:2] - self.previous[n][:2]
            if (
                any(m.touch(n, s) for s in ("left", "right"))
                and np.dot(delta, pos[:2] - m.world.root_pose()[0][:2]) > 0
            ):
                e["pushed"] = True
            center, size, _ = SURFACES[g["desk"]]
            seat = next(p for p in m.models[n].links[0].parts if p.name == "seat")
            q = m.pose(n)[1]
            seat_center = pos + matrix([0, 0, 0, *q[1:], q[0]])[:3, :3] @ np.asarray(seat.pos)
            tucked = (
                abs(seat_center[0] - center[0]) < (size[0] - seat.size[0]) / 2
                and abs(seat_center[1] - center[1]) < size[1] / 2 - 0.05
            )
            return {
                "pushed": bool(e.get("pushed")),
                "under_desk": tucked,
                "upright": m.upright(n),
                "floor_supported": m.supported(n, "floor") or m.supported(n, "ground"),
                "still": m.quiet(n),
            }
        if k == "fetch":
            return {
                "book_picked": picked(g["item"]),
                "book_on_tray": m.on(g["item"], g["target"], True),
            }
        if k == "store":
            drawer = g["drawer"]
            if m.joint(drawer, "slide") < -0.02:
                e["drawer_opened"] = True
            return {
                "opened": bool(e.get("drawer_opened")),
                "remote_inside": m.inside(g["remote"], drawer),
                "closed": abs(m.joint(drawer, "slide")) < 0.012,
                "released": m.released(drawer),
                "settled": abs(m.joint_speed(drawer, "slide")) < 0.02,
            }
        if k == "parcel":
            if abs(m.joint(g["door"], "hinge")) > 0.3 and not picked(g["parcel"]):
                e["door_opened_first"] = True
            delivered = self._carry(m, g["parcel"], g["source"], g["target"], both=True)
            if e.get("carry_active") and not e.get("carry_invalid"):
                pos = m.pose(g["parcel"])[0]
                root = m.world.root_pose()[0]
                if pos[0] > PORTAL_X + 0.15 and root[0] > PORTAL_X + 0.15:
                    e["passed_doorway"] = True
            return {
                "door_opened_first": bool(e.get("door_opened_first")),
                "bilateral_delivery": delivered,
                "actual_passage": bool(e.get("passed_doorway")),
            }
        raise ValueError(f"Unsupported household goal: {k}")

    def update(self, world):
        if not self.initialized or self.completed:
            return
        m = world.measurements
        self._pickup_events(m)
        conditions = {k: bool(v) for k, v in self._evaluate(m).items()}
        if all(conditions.values()):
            if self.since is None:
                self.since = world.time
            if world.time - self.since >= self.spec["hold_s"] - 1e-9:
                self.completed = True
                self.stage = 1
        else:
            self.since = None
        self.metrics = {
            "completed_stages": self.stage,
            "total_stages": 1,
            "conditions": conditions,
            "settled_for_s": 0.0 if self.since is None else world.time - self.since,
        }
        self.previous = {n: m.pose(n)[0].copy() for n in m.models}
        self.previous_joints = {
            n: dict(world.articulated_joints(n)) for n in self.spec["articulated"]
        }
        m.contacts.events.clear()

    def record(self):
        return {
            "version": 3,
            "stage": self.stage,
            "completed": self.completed,
            "since": self.since,
            "events": copy.deepcopy(self.events),
            "metrics": copy.deepcopy(self.metrics),
        }
