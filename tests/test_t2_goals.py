"""Boundary and adversarial traces for T2's measured-state task predicates."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.scenes import load_scene
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.task import Box
from roboenv_runtime.adapters.isaaclab.tasks.t2_household.scoring import State, validate


class ObservedWorld:
    """Synthetic observations test predicates, not physical solvability."""

    def __init__(self, goal, *, stages=None):
        names = {
            g[k]
            for stage in (stages or [[goal]])
            for g in stage
            for k in ("item", "container", "support")
            if k in g
        }
        names.update(
            n for stage in (stages or [[goal]]) for g in stage for n in g.get("contents", [])
        )
        items = {
            n: {
                "model": 6 if n == "bowl" else 2,
                "xy": [0.0, 1.2],
                "surface": "dining_table",
                "yaw": 0.0,
            }
            for n in names
        }
        boxes = [
            dict(
                name="zone", pos=[0.4, 1.25, 0.761], size=[0.36, 0.36, 0.002], color=[0.0, 1.0, 0.0]
            ),
            dict(name="blue", pos=[0.0, 1.2, 1.0], size=[0.1, 0.1, 0.1], color=[0.0, 0.0, 1.0]),
            dict(name="red", pos=[0.5, 1.2, 1.0], size=[0.1, 0.1, 0.1], color=[1.0, 0.0, 0.0]),
        ]
        self.spec = dict(
            version=2,
            scene="kitchen",
            start=[0.0, 0.0, 0.0],
            items=items,
            boxes=boxes,
            articulated={},
            stages=stages or [[goal]],
            hold_s=0.5,
        )
        self.task = SimpleNamespace(scene=load_scene("kitchen"), boxes=[Box(**b) for b in boxes])
        self.pos = {n: np.array([0.0, 1.2, 0.81 if n != "bowl" else 0.79]) for n in names}
        self.quat = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in names}
        self.extent = {
            n: np.array([0.03, 0.03, 0.05]) if n != "bowl" else np.array([0.08, 0.08, 0.03])
            for n in names
        }
        self.gaps = {n: [1.0, 1.0] for n in names}
        self.velocity = {n: np.zeros(6) for n in names}
        self.hands = np.full((12, 3), 10.0)
        self.time = 0.0
        self.state = State(self.spec)
        self.state.initialize(self)

    def item_pose(self, name):
        return self.pos[name].copy(), self.quat[name].copy()

    def item_corners(self, name):
        e = self.extent[name]
        return self.pos[name] + np.array(
            [[x, y, z] for x in (-e[0], e[0]) for y in (-e[1], e[1]) for z in (-e[2], e[2])]
        )

    def item_gap(self, name, hands):
        if len(hands) == 12:
            return min(self.gaps[name])
        return self.gaps[name][0 if np.shares_memory(hands, self.hands[:6]) else 1]

    def item_velocity(self, name):
        return self.velocity[name]

    def item_speed(self, name):
        return np.linalg.norm(self.velocity[name][:3])

    def link_poses(self, names):
        return self.hands, None

    def advance(self, duration=0.6):
        for _ in range(round(duration / 0.02)):
            self.time += 0.02
            self.state.update(self)
            json.dumps(self.state.record(), allow_nan=False)

    def lift(self, name="can"):
        self.pos[name][2] += 0.12
        self.gaps[name] = [0.0, 1.0]
        self.advance(0.2)
        assert self.state.lifted[name]
        self.gaps[name] = [1.0, 1.0]


def placement():
    return dict(kind="place", item="can", surface="dining_table", zone="zone", upright=False)


@pytest.mark.parametrize("offset,expected", [([0.14, 0.14], True), ([0.17, 0.0], False)])
def test_square_uses_all_corners_including_diagonal(offset, expected):
    world = ObservedWorld(placement())
    world.lift()
    world.pos["can"] = np.array([0.4 + offset[0], 1.25 + offset[1], 0.81])
    world.advance()
    assert world.state.completed is expected


def test_ordered_touch_requires_a_fresh_second_contact():
    stages = [
        [dict(kind="touch", box="blue", side="left")],
        [dict(kind="touch", box="red", side="right")],
    ]
    world = ObservedWorld(None, stages=stages)
    world.advance(0.02)
    world.hands[:6] = [0.0, 1.2, 1.0]
    world.hands[6:] = [0.5, 1.2, 1.0]
    world.advance(2.0)
    assert world.state.stage == 1 and not world.state.completed
    world.hands[6:] = 10.0
    world.advance(0.04)
    world.hands[6:] = [0.5, 1.2, 1.0]
    world.advance()
    assert world.state.completed


@pytest.mark.parametrize("shortcut", ["none", "no_source", "instant_source", "drop", "both_hands"])
def test_handover_requires_sustained_source_then_exclusive_target_without_drop(shortcut):
    world = ObservedWorld(
        dict(kind="handover", item="can", **{"from": "right", "to": "left", "height": 0.1})
    )
    world.pos["can"][2] += 0.12
    if shortcut != "no_source":
        world.gaps["can"] = [1.0, 0.0]
        world.advance(0.02 if shortcut == "instant_source" else 0.2)
    if shortcut == "drop":
        world.pos["can"][2] -= 0.12
        world.advance(0.02)
        world.pos["can"][2] += 0.12
    world.gaps["can"] = [0.0, 0.0 if shortcut == "both_hands" else 1.0]
    world.advance()
    assert world.state.completed is (shortcut == "none")


@pytest.mark.parametrize("kind", ["slide", "rotate"])
@pytest.mark.parametrize("shortcut", ["none", "no_contact", "lift"])
def test_surface_manipulation_rejects_teleport_and_lift(kind, shortcut):
    goal = (
        dict(kind=kind, item="can", surface="dining_table", zone="zone", distance=0.3)
        if kind == "slide"
        else dict(
            kind=kind,
            item="can",
            surface="dining_table",
            yaw=np.pi / 2,
            tolerance=0.175,
            drift=0.08,
        )
    )
    world = ObservedWorld(goal)
    if shortcut != "no_contact":
        world.gaps["can"] = [0.0, 1.0]
        world.advance(0.04)
    if shortcut == "lift":
        world.pos["can"][2] += 0.03
        world.advance(0.02)
        world.pos["can"][2] -= 0.03
    world.gaps["can"] = [1.0, 1.0]
    if kind == "slide":
        world.pos["can"][:2] = [0.4, 1.25]
    else:
        world.quat["can"] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    world.advance()
    assert world.state.completed is (shortcut == "none")


@pytest.mark.parametrize("tilt", [False, True])
def test_carry_remembers_transient_tilt_after_pickup(tilt):
    world = ObservedWorld(
        dict(kind="carry", item="bowl", surface="dining_table", zone="zone", tilt=0.35, contents=[])
    )
    world.lift("bowl")
    if tilt:
        # Lowering below the initial height must not suspend the carry invariant.
        world.pos["bowl"][2] = .75
        world.quat["bowl"] = [0.8, 0.6, 0.0, 0.0]
        world.advance(0.02)
        world.quat["bowl"] = [1.0, 0.0, 0.0, 0.0]
    world.pos["bowl"] = np.array([0.4, 1.25, 0.79])
    world.advance()
    assert world.state.completed is (not tilt)


@pytest.mark.parametrize("shortcut", ["none", "support_moved", "support_floating", "unreleased"])
def test_stack_requires_lift_alignment_and_stable_support(shortcut):
    world = ObservedWorld(dict(kind="stack", item="can", support="lower", tolerance=0.015))
    world.lift()
    world.pos["can"] = world.pos["lower"] + [0.0, 0.0, 0.1]
    if shortcut == "support_moved":
        world.pos["can"][0] += 0.05
        world.pos["lower"][0] += 0.05
    if shortcut == "support_floating":
        world.pos["can"][2] += 0.03
        world.pos["lower"][2] += 0.03
    if shortcut == "unreleased":
        world.gaps["can"] = [0.0, 1.0]
    world.advance()
    assert world.state.completed is (shortcut == "none")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(version=1),
        lambda s: s.update(stages=[]),
        lambda s: s["stages"][0][0].update(zone="missing"),
        lambda s: s["boxes"][0]["pos"].__setitem__(0, 100.0),
    ],
)
def test_invalid_geometry_and_old_protocol_are_rejected(mutate):
    spec = copy.deepcopy(ObservedWorld(placement()).spec)
    mutate(spec)
    with pytest.raises(ValueError):
        validate(spec)


@pytest.mark.parametrize(
    "angle,velocity,absolute,expected",
    [
        (0.0, 0.0, False, False),
        (1.0, 0.0, False, True),
        (1.0, 1.0, False, False),
        (-1.0, 0.0, False, False),
        (-1.0, 0.0, True, True),
        (1.7, 0.0, True, False),
    ],
)
def test_joint_goal_needs_motion_in_range_and_sustained_low_velocity(
    angle, velocity, absolute, expected
):
    spec = dict(
        version=2,
        scene=None,
        start=[0.0, 0.0, 0.0],
        items={},
        boxes=[],
        hold_s=0.5,
        articulated={"fixture": dict(model=4, xy=[1.0, 0.0], support=0.75, yaw=0.0, joints={})},
        stages=[
            [
                dict(
                    kind="joint",
                    object="fixture",
                    joints=["articulate_joint_0"],
                    min=0.7,
                    max=1.57,
                    absolute=absolute,
                )
            ]
        ],
    )
    native = {
        "pose": [1.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
        "velocity": [0.0] * 6,
        "joints": {"articulate_joint_0": 0.0},
        "joint_velocities": {"articulate_joint_0": 0.0},
    }
    world = SimpleNamespace(
        time=0.0,
        articulated_state=lambda name: copy.deepcopy(native),
        link_poses=lambda names: (np.zeros((12, 3)), None),
    )
    state = State(spec)
    state.initialize(world)
    native["joints"]["articulate_joint_0"] = angle
    native["joint_velocities"]["articulate_joint_0"] = velocity
    for world.time in (0.0, 0.1, 0.6):
        state.update(world)
    assert state.completed is expected


@pytest.mark.parametrize(
    "delta,quat,expected",
    [
        ([0.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], True),
        ([0.6, 0.3, 0.0], [1.0, 0.0, 0.0, 0.0], False),
        ([0.6, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0], False),
        ([0.6, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], False),
        ([0.0, 0.6, 0.0], [1.0, 0.0, 0.0, 0.0], False),
    ],
)
def test_chair_push_rejects_sideways_lift_and_tip(delta, quat, expected):
    spec = dict(
        version=2,
        scene=None,
        start=[0.0, 0.0, 0.0],
        items={},
        boxes=[],
        hold_s=0.5,
        articulated={"chair": dict(model=0, xy=[1.0, 0.0], support=0.0, yaw=0.0, joints={})},
        stages=[
            [dict(kind="push", object="chair", direction=[1.0, 0.0], distance=0.5, lateral=0.25)]
        ],
    )
    native = {
        "pose": [1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "velocity": [0.0] * 6,
        "joints": {},
        "joint_velocities": {},
    }
    world = SimpleNamespace(
        time=0.0,
        articulated_state=lambda name: copy.deepcopy(native),
        link_poses=lambda names: (np.zeros((12, 3)), None),
    )
    state = State(spec)
    state.initialize(world)
    native["pose"] = (np.array([1.0, 0.0, 0.4]) + delta).tolist() + quat
    for world.time in (0.0, 0.1, 0.6):
        state.update(world)
    assert state.completed is expected


@pytest.mark.parametrize("spill", [False, True])
def test_loaded_carry_retains_contents_throughout_motion(spill):
    world = ObservedWorld(
        dict(
            kind="carry",
            item="bowl",
            surface="dining_table",
            zone="zone",
            tilt=0.35,
            contents=["can"],
        )
    )
    world.gaps["bowl"] = [0.0, 1.0]
    world.pos["bowl"][2] += 0.12
    world.pos["can"][2] += 0.12
    world.advance(0.2)
    if spill:
        world.pos["can"][0] += 0.3
        world.advance(0.02)
        world.pos["can"][0] -= 0.3
    world.pos["can"] = np.array([0.4, 1.25, 0.81])
    world.pos["bowl"] = np.array([0.4, 1.25, 0.79])
    world.gaps["bowl"] = [1.0, 1.0]
    world.advance()
    assert world.state.completed is (not spill)
