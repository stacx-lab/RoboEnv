"""Semantic counterexamples for native household contacts and goal events."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.household_models import MODEL_NAMES, model
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.household_physics import (
    ContactPairs,
    collider_identity,
)
from roboenv_runtime.adapters.isaaclab.robots.booster_t2.scenes import load_scene
from roboenv_runtime.adapters.isaaclab.tasks.t2_household.release import State, validate


def touch_spec():
    return {
        "version": 3,
        "scene": "household",
        "start": [-2.0, 0.7, 1.57],
        "items": {
            n: {"model": n + "_block", "pos": [x, 1.76, 0.82], "yaw": 0.0, "fixed": True}
            for n, x in (("blue", -2.25), ("red", -1.75))
        },
        "boxes": [],
        "articulated": {},
        "hold_s": 0.1,
        "stages": [[{"kind": "ordered_touch", "blue": "blue", "red": "red"}]],
    }


class TouchTrace:
    """Labeled contact traces, without hidden hand-origin coordinates."""

    def __init__(self):
        self.contacts = ContactPairs()
        self.models = {n: model(n + "_block") for n in ("blue", "red")}
        self.world = SimpleNamespace(measurements=self, time=0.0)
        self.state = State(touch_spec())
        self.state.initialize(self.world)

    def pose(self, name):
        return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])

    def touch(self, name, side):
        return self.contacts.touching(name, side + "_hand")

    def step(self, blue=False, red=False, dt=0.005):
        for name, side, active in (("blue", "left", blue), ("red", "right", red)):
            self.contacts.feed(
                name,
                (name, "body", "shape"),
                (side + "_hand", "hand", "palm_surface"),
                [],
                lost=not active,
            )
        self.world.time += dt
        self.state.update(self.world)


def test_brief_real_contacts_pass_without_one_second_or_link_origin_hold():
    trace = TouchTrace()
    trace.step(blue=True)
    trace.step()
    trace.step(red=True)
    trace.step()
    trace.step(dt=0.11)
    assert trace.state.completed


def test_simultaneously_held_red_contact_does_not_count_as_a_fresh_second_touch():
    trace = TouchTrace()
    trace.step(blue=True, red=True)
    trace.step(red=True)
    trace.step(red=True, dt=2.0)
    assert not trace.state.completed
    trace.step()
    trace.step(red=True)
    trace.step()
    trace.step(dt=0.11)
    assert trace.state.completed


def test_first_withdrawal_and_fresh_second_contact_can_share_a_physics_step():
    trace = TouchTrace()
    trace.step(blue=True)
    trace.step(red=True)
    trace.step()
    trace.step(dt=0.11)
    assert trace.state.completed


def test_hovering_does_not_create_touch_and_wrong_order_can_be_retried():
    trace = TouchTrace()
    trace.step(dt=2.0)
    assert trace.state.events["touch_stage"] == 0
    trace.step(red=True)
    trace.step()
    trace.step(blue=True)
    trace.step()
    trace.step(red=True)
    trace.step()
    trace.step(dt=0.11)
    assert trace.state.completed


def test_contact_orientation_support_and_lost_events():
    contacts = ContactPairs()
    can = ("can", "body", "bottom")
    table = ("dining_table", "body", "top")
    samples = [(np.array([0.0, 0.0, 0.82]), np.array([0.0, 0.0, 1.0]), 1.2)]
    contacts.feed("pair", can, table, samples)
    assert contacts.upward_force("can", "dining_table") == pytest.approx(1.2)
    assert contacts.upward_force("dining_table", "can") == 0
    assert contacts.touching("can", "dining_table")
    # A sleeping body has no new event; the extant support is retained.
    assert contacts.touching("can", "dining_table")
    contacts.feed("pair", can, table, [], lost=True)
    assert not contacts.touching("can", "dining_table")


def test_collision_identity_keeps_tool_part_and_joint_link():
    assert collider_identity("/World/Task/bench/peg_a/peg") == ("bench", "peg_a", "peg")
    assert collider_identity("/World/Task/mallet/body/head") == ("mallet", "body", "head")
    assert collider_identity("/World/Household/dining_table/top")[0] == "dining_table"
    assert (
        collider_identity("/World/Robot/left_index_proximal_link/collisions/mesh")[0] == "left_hand"
    )
    assert collider_identity("/World/Robot/left_base_link")[0] == "left_hand"
    assert (
        collider_identity("/World/Robot/left_index_link/collisions/mesh")[2]
        != collider_identity("/World/Robot/left_thumb_link/collisions/mesh")[2]
    )
    assert collider_identity("/World/Robot/left_elbow_link/collisions/mesh")[0] == "robot"


@pytest.mark.parametrize("change", ["version", "marker", "unknown_object", "nonfinite", "model"])
def test_invalid_release_definition_is_rejected_before_simulator_creation(change):
    spec = copy.deepcopy(touch_spec())
    if change == "version":
        spec["version"] = True
    elif change == "marker":
        spec["boxes"] = [{"name": "green_square"}]
    elif change == "unknown_object":
        spec["stages"][0][0]["red"] = "absent"
    elif change == "nonfinite":
        spec["items"]["red"]["pos"][0] = float("nan")
    else:
        spec["items"]["red"]["model"] = "imaginary_asset"
    with pytest.raises(ValueError):
        validate(spec)


def test_all_authored_assets_have_valid_unique_physical_parts_and_joint_frames():
    for name in MODEL_NAMES:
        definition = model(name)
        links = {link.name for link in definition.links}
        assert len(links) == len(definition.links)
        for link in definition.links:
            assert link.mass > 0
            assert len({p.name for p in link.parts}) == len(link.parts)
            assert all(p.name.isidentifier() and min(p.size) > 0 for p in link.parts)
        assert all(
            j.parent in links
            and j.child in links
            and j.parent != j.child
            and j.limits[0] < j.limits[1]
            for j in definition.joints
        )


def test_peg_flush_endpoint_matches_the_visible_bench_surface():
    bench = model("pounding_bench")
    top = next(p for p in bench.links[0].parts if p.name == "top_front")
    surface = top.pos[2] + top.size[2] / 2
    for link in bench.links[1:]:
        peg = link.parts[0]
        assert link.pos[2] + peg.size[1] / 2 == pytest.approx(surface)


class PhysicalFacts:
    """Labeled support/contact facts for semantic boundary tests, not dynamics."""

    def __init__(self, assets):
        from roboenv_runtime.adapters.isaaclab.robots.booster_t2.household_physics import (
            Measurements,
        )

        self.measurements = Measurements.__new__(Measurements)
        self.measurements.scene = load_scene("household")
        self.measurements.models = {n: model(asset) for n, asset in assets.items()}
        self.measurements.definitions = {n: {"model": asset} for n, asset in assets.items()}
        self.measurements.contacts = ContactPairs()
        self.measurements.world = SimpleNamespace(time=0.0)
        self.poses = {n: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) for n in assets}
        self.velocities = {n: np.zeros(6) for n in assets}
        self.boundary = {}
        self.measurements.pose = lambda n, link=None: self.poses[n]
        self.measurements.speed = lambda n: self.velocities[n]
        self.measurements.points = lambda n: self.boundary[n]

    def support(self, item, receiver, *, receiver_part="floor", receiver_link="body"):
        self.measurements.contacts.feed(
            (item, receiver),
            (item, "body", "body"),
            (receiver, receiver_link, receiver_part),
            [(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0)],
        )


def test_can_containment_allows_protrusion_but_rejects_rim_balance_and_overlap():
    facts = PhysicalFacts({"can": "can_tomato", "bowl": "bowl"})
    # A can resting on the cavity floor extends above the bowl's rim.
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    facts.boundary["can"] = np.array(
        [[0.033 * np.cos(a), 0.033 * np.sin(a), z] for a in angles for z in (0.012, 0.15)]
    )
    facts.support("can", "bowl", receiver_part="bottom")
    m = facts.measurements
    assert m.inside("can", "bowl")
    facts.support("can", "bowl", receiver_part="rim_0")
    assert not m.inside("can", "bowl")
    facts.support("can", "bowl", receiver_part="bottom")
    facts.boundary["can"][:, 0] += 0.15
    assert not m.inside("can", "bowl")
    facts.boundary["can"][:, 0] -= 0.15
    m.contacts.reset()
    assert not m.inside("can", "bowl")


def test_pan_entry_is_detectable_before_settling_but_final_placement_must_settle():
    facts = PhysicalFacts({"block": "small_block", "pan": "collection_pan"})
    facts.boundary["block"] = np.array([[-0.02, -0.02, 0.004], [0.02, 0.02, 0.044]])
    facts.support("block", "pan")
    facts.velocities["block"][0] = 0.2
    assert facts.measurements.inside("block", "pan", settled=False)
    assert not facts.measurements.inside("block", "pan")
    facts.velocities["block"][:] = 0
    assert facts.measurements.inside("block", "pan")


def test_release_distinguishes_unloaded_touch_from_a_loaded_grasp():
    facts = PhysicalFacts({"can": "can_tomato", "plate": "plate"})
    facts.support("can", "plate")
    m = facts.measurements
    m.contacts.feed(
        "touch",
        ("can", "body", "wall"),
        ("left_hand", "hand", "palm"),
        [(np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0)],
    )
    assert m.touch("can", "left") and m.released("can")
    assert m.on("can", "plate", True)
    for name, sign in (("index", 1), ("thumb", -1)):
        m.contacts.feed(
            name,
            ("can", "body", "wall"),
            ("left_hand", "hand", name),
            [(np.zeros(3), np.array([sign, 0.0, 0.0]), 0.2)],
        )
    assert m.held("can") and not m.released("can")
    assert not m.on("can", "plate", True)


def test_receiver_center_follows_moved_receiver_without_hidden_world_anchor():
    facts = PhysicalFacts({"can": "can_tomato", "plate": "plate"})
    facts.support("can", "plate")
    for n in facts.poses:
        facts.poses[n] = (np.array([2.0, -3.0, 0.8]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert facts.measurements.on("can", "plate", True)
    facts.poses["can"][0][0] += 0.14
    assert not facts.measurements.on("can", "plate", True)


@pytest.mark.parametrize("value", [["red"], "activity_table", 4])
def test_scalar_object_bindings_reject_lists_room_surfaces_and_numbers(value):
    spec = touch_spec()
    spec["stages"][0][0]["red"] = value
    with pytest.raises(ValueError):
        validate(spec)


class CarryFacts:
    def __init__(self):
        self.world = SimpleNamespace(time=0.0)
        self.grasp = True
        self.sources = {"table"}
        self.level = True
        self.both = True
        self.placed = False

    def held(self, name):
        return self.grasp

    def supported(self, name, receiver):
        return receiver in self.sources

    def hand_support(self, name, side):
        return self.grasp and (side == "left" or self.both)

    def upright(self, name, tolerance):
        return self.level

    def on(self, name, receiver, center):
        return self.placed


def carry_state():
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.events = {}
    return state


def test_level_requirement_only_applies_during_carry_and_source_return_allows_retry():
    state, facts = carry_state(), CarryFacts()
    facts.level = False  # Tilting while taking hold on the source is recoverable.
    state._carry(facts, "bowl", "table", "plate", level=True)
    facts.level = True
    facts.sources.clear()
    state._carry(facts, "bowl", "table", "plate", level=True)
    facts.level = False
    facts.world.time = 0.5
    state._carry(facts, "bowl", "table", "plate", level=True)
    facts.world.time = 0.7
    state._carry(facts, "bowl", "table", "plate", level=True)
    facts.placed = True
    facts.sources = {"plate"}
    assert not state._carry(facts, "bowl", "table", "plate", level=True)
    facts.sources = {"table"}
    facts.placed = False
    state._carry(facts, "bowl", "table", "plate", level=True)
    facts.sources.clear()
    facts.level = True
    state._carry(facts, "bowl", "table", "plate", level=True)
    facts.sources = {"plate"}
    facts.placed = True
    facts.grasp = False
    assert state._carry(facts, "bowl", "table", "plate", level=True)


def test_bilateral_carry_tolerates_one_contact_step_but_rejects_sustained_one_hand():
    state, facts = carry_state(), CarryFacts()
    state._carry(facts, "parcel", "table", "tray", both=True)
    facts.sources.clear()
    facts.both = False
    state._carry(facts, "parcel", "table", "tray", both=True)
    facts.world.time = 0.005
    facts.both = True
    state._carry(facts, "parcel", "table", "tray", both=True)
    assert not state.events.get("carry_invalid")
    facts.both = False
    facts.world.time = 0.1
    state._carry(facts, "parcel", "table", "tray", both=True)
    facts.world.time = 0.3
    state._carry(facts, "parcel", "table", "tray", both=True)
    facts.placed = True
    facts.sources = {"tray"}
    assert not state._carry(facts, "parcel", "table", "tray", both=True)


def test_turn_requires_real_rotation_without_accumulating_quaternion_roundoff():
    facts = PhysicalFacts({"box": "cracker_box", "mat": "rectangular_mat", "plate": "plate"})
    facts.support("box", "mat")
    facts.poses["plate"][0][0] = 0.3
    m = facts.measurements
    m.contacts.feed("touch", ("box", "body", "carton"), ("left_hand", "hand", "palm"), [])
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.goal = {"kind": "turn", "item": "box", "mat": "mat", "plate": "plate"}
    state.events = {}
    for _ in range(100):
        facts.poses["box"] = (np.zeros(3), np.array([0.9999999, 0, 0, 0]))
        assert not state._evaluate(m)["supported_turn"]
    facts.poses["box"] = (np.zeros(3), np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
    assert all(state._evaluate(m).values())


def test_carried_placement_cannot_be_reclassified_as_a_short_slide_on_the_mat():
    facts = PhysicalFacts({'can': 'can_tomato', 'mat': 'round_mat'})
    m = facts.measurements
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.goal = {'kind': 'slide', 'item': 'can', 'target': 'mat', 'surface': 'reading_desk'}
    state.events = {}
    facts.support('can', 'mat')
    facts.boundary['can'] = np.array([[-0.03, -0.03, 0], [0.03, 0.03, 0.1]])
    m.contacts.feed('touch', ('can', 'body', 'wall'), ('left_hand', 'hand', 'palm'), [])
    assert not state._evaluate(m)['supported_slide']
    # Returning wholly outside the visible mat permits a complete supported retry.
    facts.boundary['can'][:, 0] += 0.4
    facts.support('can', 'reading_desk')
    state._evaluate(m)
    facts.boundary['can'][:, 0] -= 0.4
    assert all(state._evaluate(m).values())


def test_lamp_requires_contacts_with_the_operated_parts_not_just_its_base():
    facts = PhysicalFacts({'lamp': 'lamp', 'book': 'open_book'})
    m = facts.measurements
    m.joint = lambda n, j: {'aim': 0.65, 'power': 0.3}[j]
    m.joint_speed = lambda n, j: 0.0
    m.lit_pages = lambda lamp, book: True
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.goal = {'kind': 'illuminate', 'lamp': 'lamp', 'book': 'book'}
    state.initial_joints = {'lamp': {'aim': 0.1, 'power': -0.3}}
    state.events = {}
    for link in ('base', 'head', 'switch'):
        m.contacts.feed(link, ('left_hand', 'hand', 'palm'), ('lamp', link, link), [])
        result = state._evaluate(m)
        if link == 'base':
            assert not result['switch_actuated'] and not result['head_adjusted']
        elif link == 'head':
            assert result['head_adjusted'] and not result['switch_actuated']
        else:
            assert all(result.values())


def test_scale_rejects_a_weight_used_to_prop_or_counterbalance_the_beam():
    facts = PhysicalFacts({'scale': 'balance_scale', 'weight': 'weight_1'})
    m = facts.measurements
    m.inside = lambda *args, **kwargs: True
    m.joint = lambda *args: 0.0
    m.joint_speed = lambda *args: 0.0
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.goal = {'kind': 'balance', 'scale': 'scale', 'weights': ['weight']}
    state.events = {}
    facts.support('weight', 'scale', receiver_link='pan')
    assert all(state._evaluate(m).values())
    for link in ('beam', 'reference'):
        facts.support('weight', 'scale', receiver_link=link)
        assert not state._evaluate(m)['unassisted']


def test_reinsertion_must_use_the_matching_aperture_again():
    facts = PhysicalFacts({'sorter': 'sorter', 'cylinder': 'cylinder_block', 'cube': 'cube_block'})
    m = facts.measurements
    m.inside = lambda *args: True
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.goal = {'kind': 'insert', 'sorter': 'sorter', 'cylinder': 'cylinder', 'cube': 'cube'}
    state.events = {'picked': ['cylinder', 'cube'], 'qualified': {}}
    state.previous = {'cylinder': np.array([-0.104, 0, 0.20]), 'cube': np.array([0.105, 0, 0.20])}
    for n, x in (('cylinder', -0.104), ('cube', 0.105)):
        facts.poses[n] = (np.array([x, 0, 0.05]), np.array([1.0, 0, 0, 0]))
    assert all(state._evaluate(m).values())
    state.previous['cylinder'] = facts.poses['cylinder'][0].copy()
    facts.poses['cylinder'][0][2] = 0.20
    assert not state._evaluate(m)['cylinder_inserted']
    state.previous['cylinder'] = facts.poses['cylinder'][0].copy()
    facts.poses['cylinder'][0][:] = [0.105, 0, 0.05]
    assert not state._evaluate(m)['cylinder_inserted']


def test_airborne_cart_motion_does_not_count_as_pulling_on_its_wheels():
    facts = PhysicalFacts({'cart': 'cart'})
    m = facts.measurements
    p = np.array([3.6, -2.18, 0.0])
    facts.poses['cart'] = (p, np.array([0.0, 0, 0, 1.0]))
    facts.boundary['cart'] = p + np.array([[-0.275, -0.25, 0.1], [0.275, 0.25, 0.9]])
    m.world.root_pose = lambda: (p + [0, 0.6, 1.0], np.array([1.0, 0, 0, 0]))
    m.contacts.feed('handle', ('left_hand', 'hand', 'palm'), ('cart', 'base', 'handle'), [])
    state = State.__new__(State)
    state.spec = {"version": 3}
    state.goal = {'kind': 'pull_cart', 'cart': 'cart'}
    state.previous = {'cart': p - [0, 0.02, 0]}
    state.events = {'distance': 0.0, 'pulled': 0.0}
    result = state._evaluate(m)
    assert result['in_bay'] and not result['pulled_transport']
    facts.support('cart', 'floor')
    state.events = {'distance': 0.0, 'pulled': 0.0}
    assert all(state._evaluate(m).values())


@pytest.mark.parametrize(
    "version,scene", [(3, "household"), (4, "simple_household"), (5, "hssd_household")]
)
def test_asset_version_selects_its_scene_and_preserves_goal_contract(version, scene):
    spec = touch_spec()
    spec.update(version=version, scene=scene)
    validate(spec)
    assert State(spec).record()["version"] == version
    spec["scene"] = "household" if version == 4 else "simple_household"
    with pytest.raises(ValueError, match="scene"):
        validate(spec)


def test_imported_furniture_contacts_keep_named_support_not_ground():
    path = "/World/Household/dining_table/furniture/Geometry/Mesh/mesh/mesh_0"
    assert collider_identity(path) == ("dining_table", "body", "mesh_0")
    contacts = ContactPairs()
    contacts.feed("support", ("can", "body", "bottom"), collider_identity(path),
                  [(np.zeros(3), np.array([0., 0., 1.]), 1.2)])
    assert contacts.upward_force("can", "dining_table") == pytest.approx(1.2)
    assert contacts.upward_force("can", "ground") == 0


def test_scanned_bowl_has_open_sloped_shell_and_load_bearing_bottom():
    from roboenv_runtime.adapters.isaaclab.robots.booster_t2.simple_household import (
        model as scan_model,
    )

    original = model("bowl")
    bowl = scan_model("bowl")
    assert bowl.links[0].mass == original.links[0].mass
    floor, *walls = bowl.links[0].parts
    assert floor.name == "floor" and floor.shape == "cylinder"
    assert floor.pos[2] + floor.size[1] / 2 < .01
    assert all(p.shape == "hull" for p in walls)
    vertices = np.concatenate([np.asarray(p.size) for p in walls])
    radial = np.linalg.norm(vertices[:, :2], axis=1)
    # A centred 69 mm can fits all heights. No collision wedge crosses the
    # cavity, and the visible scan's 260 mm rim is the physical outer diameter.
    assert radial.min() > .035
    assert radial.max() == pytest.approx(.13)
    assert vertices[:, 2].max() == pytest.approx(.09)
    assert radial[vertices[:, 2] > .085].min() > radial[vertices[:, 2] < .005].max()
    assert {p.shape for p in model("bowl").links[0].parts} == {"cylinder", "box"}


@pytest.mark.parametrize("relative,entity", [
    ("furniture/eeaf34edd2065a3fa2af3fc021cd343ca029f696/mesh", "dining_table"),
    ("walls/geometry_rs_3785/mesh", "floor"),
    ("furniture/unrelated/mesh", "scene:furniture/unrelated"),
    ("furniture/eeaf34edd2065a3fa2af3fc021cd343ca029f696_extra/mesh",
     "scene:furniture/eeaf34edd2065a3fa2af3fc021cd343ca029f696_extra"),
])
def test_full_house_contacts_distinguish_floor_furniture_and_exact_bindings(relative, entity):
    scene = load_scene("hssd_household")
    assert collider_identity("/World/Scene/" + relative, scene)[0] == entity


def test_full_house_alias_uses_real_support_and_rotated_desk_uses_local_axes():
    facts = PhysicalFacts({"can": "can_tomato"})
    facts.measurements.scene = scene = load_scene("hssd_household")
    facts.support("can", "dining_table")
    assert facts.measurements.supported("can", "activity_table")
    assert not facts.measurements.supported("can", "reading_desk")
    desk = scene.surfaces["reading_desk"]
    assert desk.contains(np.asarray(desk.center) + [0, .4])
    assert not desk.contains(np.asarray(desk.center) + [.4, 0])


def test_v5_contact_with_unbound_scene_furniture_prevents_false_pickup():
    spec = touch_spec()
    spec.update(version=5, scene="hssd_household")
    spec["items"]["can"] = {"model": "can_tomato", "pos": [0, 0, .8], "yaw": 0., "fixed": False}
    state = State(spec)
    state.events = {"picked": []}
    facts = PhysicalFacts({"can": "can_tomato"})
    facts.measurements.scene = load_scene("hssd_household")
    facts.measurements.held = lambda name: True
    facts.support("can", "scene:furniture/unbound_shelf")
    state._pickup_events(facts.measurements)
    assert state.events["picked"] == []
    facts.measurements.contacts.reset()
    state._pickup_events(facts.measurements)
    assert state.events["picked"] == ["can"]
