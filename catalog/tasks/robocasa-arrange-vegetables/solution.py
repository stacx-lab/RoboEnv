"""RoboCasa family program - the goal shape "an object ends up somewhere, hand clear".

One program for the pick-and-place shape, atomic and composite alike.  Every one of their
``_check_success`` bodies is the same conjunction:

    <containment or contact test on the object>  AND  OU.gripper_obj_far(<object>)

and the whole difficulty is **which** object and **which** receptacle, because RoboCasa
never says so in a machine-readable form and the referent rules a program can write from
outside do not survive the composite suite:

* ``objects[0]`` picked the *meat on the board* in ``PrepForTenderizing``, where the
  rolling pin is what moves;
* "the receptacle is ``<obj>_container``" is right for the atomic tasks and exactly wrong
  for a composite, where ``marshmallow_container`` is the plate the marshmallow STARTED
  on - releasing it there is why 22 composites scored 0.0 after a clean grasp and a clean
  placement;
* "the destination is the fixture the sentence names last" cannot see that
  ``AddMarshmallow``'s destination is an *object* (a mug) standing on a fixture the
  sentence never names.

``get_placement_goal()`` ends the guessing: it runs the episode's own ``_check_success``
with the benchmark's placement predicates wrapped and reports the arguments each one was
called with and the value it really returns.  The pairs to act on are the placements that
are FALSE right now - a conjunct already true at reset is staging, not work
(``PrepForTenderizing`` scores the meat on its own board *and* the rolling pin onto it;
only the pin moves), and a conjunct the scorer NEGATES is satisfied by being false.  It
costs no simulation steps.

There can be more than one, and a coded transport across ONE counter is 250-350 frames, not
the 900 a policy transport across a kitchen costs: ``DessertUpgrade`` moves two desserts
onto a tray and the first cost 349 frames of the 1000.  So this walks the pending list
while the horizon can still pay for another carry plus the finish.

The rest is the shape-D machinery the atomic round measured:

* **the grasp** - ``pick_object``, which aims BELOW the object's own bottom (the descent is
  floor-seeking: it bottoms out 2-3 cm above the support whatever it was asked for) and
  judges the grasp by whether the object rose WITH THE HAND;
* **the standing pose** - ``stand_to_pick``, which prices every candidate pose by IK on a
  kinematic copy for zero frames and usually declines to move;
* **the finish** - ``gripper_obj_far`` wants 0.25 m, so ``open_gripper`` + ``retreat_from``
  + ``tuck_arm``, never ``place_gently`` (it retreats 6 cm).  Reserve ~170 frames.

Where to stand comes from the world too: ``support`` in a placement record is the fixture
the destination object is resting on *now*, read from MuJoCo's own contacts, which is a
better answer than any fixture the sentence names.
"""
import numpy as np

RECEPTACLE_WORDS = ("container", "plate", "receptacle", "bowl", "tray", "pan", "cup", "pot",
                    "mug", "jug", "glass", "basket", "kettle", "pitcher")
FINISH = 170          # open_gripper + retreat + tuck_arm
CARRY = 260           # what one more pick-carry-place needs before it is worth starting
REACH = 0.72          # the arm's envelope around the base; farther than this, drive first
NARROW_RIM = 0.14     # an opening narrower than the open gripper cannot be lowered into


def budget():
    b = get_step_budget()
    return int(b.get("max", 1000)) - int(b.get("used", 0))


def solved():
    try:
        return bool(get_goal()["satisfied"])
    except Exception:  # noqa: BLE001
        return False


def P(n):
    return np.asarray(get_object_pose(n)[0], float)


def reach(point):
    b = np.asarray(get_base_pose(), float)
    return float(np.linalg.norm(np.asarray(point, float)[:2] - b[:2])) <= REACH


def go(fixture, ref=None):
    """Drive to a fixture's standing pose over the BFS route when the straight line is
    blocked.  A composite's destination is often across the kitchen (AddMarshmallow's mug
    stands on the DINING counter) and drive_to is a straight line: it reported `reached`
    False with 2.75 m of error after 83 frames, twice, and the carried marshmallow was
    scraped out of the jaws in the process.  plan_base_route costs no frames, so ask it
    first and only fall back to the straight line when it finds nothing.
    """
    try:
        x, y, yaw = get_base_placement(fixture, ref)
        plan = plan_base_route(float(x), float(y), float(yaw))
        print("[route-plan]", fixture, {k: plan.get(k) for k in ("ok", "length_m", "reason")},
              "waypoints", len(plan.get("waypoints") or []))
        if plan.get("ok"):
            r = drive_route(float(x), float(y), float(yaw))
            r["target"] = [float(x), float(y), float(yaw)]
            return r
    except Exception as exc:  # noqa: BLE001
        print("[route-plan]", exc)
    return drive_to_fixture(fixture, ref)


goal = get_goal()
print("[goal]", goal["task"], "decoded", goal["decoded"], "satisfied", goal["satisfied"])
print("[instruction]", goal["instruction"])
fixtures = {k: v for k, v in (goal.get("fixtures") or get_task_fixtures() or {}).items()
            if isinstance(v, str)}
objects = list_objects()
print("[fixtures]", fixtures, "[objects]", objects)

text = (goal.get("instruction") or "").lower()

# ------------------------------------------------------ what the SCORER says has to move
PG = {}
try:
    PG = get_placement_goal() or {}
except Exception as exc:  # noqa: BLE001
    print("[placement-goal]", exc)
pending = [m for m in (PG.get("pending") or []) if m.get("object") in objects]
print("[placements]", PG.get("placements"))
print("[pending]", pending, "single", PG.get("single"), "opaque", PG.get("opaque"))


def named_in_sentence(name):
    """Where the sentence last mentions any word of an object's name (-1: never)."""
    return max((text.rfind(w) for w in name.split("_") if len(w) > 3 and w in text), default=-1)


def is_recep(o):
    return any(w in o for w in RECEPTACLE_WORDS)


if not pending:
    # the scorer named no pending placement (it threw, or its conjunction is one this
    # cannot read): fall back to the sentence, which is what the atomic round used
    print("[referent] the scorer named no pending placement; falling back to the sentence")
    if "obj" in objects:
        obj = "obj"
    else:
        movers = [o for o in objects if not is_recep(o)] or objects
        by_text = sorted(movers, key=lambda o: -named_in_sentence(o))
        obj = by_text[0] if named_in_sentence(by_text[0]) >= 0 else movers[0]
    cands = [o for o in objects if o != obj and is_recep(o)]
    own = [o for o in cands if o.startswith(obj + "_") or o.startswith("obj_")]
    rest = [o for o in cands if o not in own]
    rest.sort(key=lambda o: (o.endswith("_container"), -named_in_sentence(o)))
    guess = next(iter(rest or own), None)
    pending = [{"object": obj, "destination": guess, "destination_is_fixture": False,
                "support": None, "kind": "guessed"}]

# ------------------------------------------- which fixture the sentence sends things to
# Only a fallback now: with the receptacle known, the destination fixture is the one it is
# STANDING ON, read from the simulator's contacts, not the one the sentence names last.
ranked = []
for role, fx in fixtures.items():
    words = [role.replace("_", " "), role.split("_")[0]]
    hit = max((text.rfind(w) for w in words if w and text.rfind(w) >= 0), default=-1)
    ranked.append((hit, role, fx))
ranked.sort()
SENTENCE_FX = ranked[-1][2] if ranked and ranked[-1][0] >= 0 else None

TORSO = [0.0]


def torso_for(z):
    """The torso is the whole answer above z ~ 1.30, and must come back down below ~1.10."""
    want = 0.34 if z > 1.28 else 0.0
    if abs(want - TORSO[0]) > 0.05:
        try:
            set_torso_height(want)
            TORSO[0] = want
            print("[torso]", want)
        except Exception as exc:  # noqa: BLE001
            print("[torso]", exc)


def aim_point(recep, dest_fx):
    """Where the OBJECT's centre has to end up, and the surface it lands on (or None)."""
    if recep is not None:
        rp, rext = P(recep), get_object_extent(recep)
        return np.array([rp[0], rp[1], float(rext["top_z"])]), float(rext["top_z"]), "receptacle"
    regions = {}
    try:
        regions = get_fixture_regions(dest_fx) or {}
    except Exception:  # noqa: BLE001
        regions = {}
    if regions:
        b = np.asarray(get_base_pose(), float)
        pref = [k for k in regions if any(w in k for w in ("int", "basin", "shelf", "rack"))]
        keys = pref or sorted(regions)
        k = min(keys, key=lambda n: float(np.linalg.norm(
            np.asarray(regions[n]["center"], float)[:2] - b[:2])))
        r = regions[k]
        c = np.asarray(r["center"], float)
        size = np.asarray(r.get("size", [0.4, 0.4, 0.2]), float)
        floor = float(c[2] - abs(size[2]) / 2.0)
        # an opening narrower than the open gripper cannot be lowered into: release above
        # its rim instead of trying to reach its floor
        narrow = float(min(abs(size[0]), abs(size[1]))) < NARROW_RIM
        if narrow:
            rim = float(c[2] + abs(size[2]) / 2.0)
            return np.array([c[0], c[1], rim + 0.06]), None, f"rim:{k}"
        return np.array([c[0], c[1], floor]), floor, f"region:{k}"
    fe = get_fixture_extent(dest_fx)
    top = max(float(np.asarray(p, float)[2]) for p in fe["bbox"])
    return np.array([float(fe["pos"][0]), float(fe["pos"][1]), top]), top, "fixture-top"


def transport(move):
    """One pick-carry-place, for the pair the scorer named."""
    obj = move["object"]
    recep = None if move.get("destination_is_fixture") else move.get("destination")
    if recep is not None and recep not in objects:
        recep = None
    dest_fx = (move.get("destination") if move.get("destination_is_fixture")
               else move.get("support")) or SENTENCE_FX
    p0 = P(obj)
    print("[move]", obj, np.round(p0, 3).tolist(), get_grasp_axis(obj), "-> recep", recep,
          "dest_fx", dest_fx)

    src_fx = None
    try:
        src_fx = get_supporting_fixture(obj)
    except Exception as exc:  # noqa: BLE001
        print("[support]", exc)
    if not reach(p0) and src_fx:
        r = go(src_fx, obj)
        print("[drive->src]", src_fx, {k: r.get(k) for k in ("reached", "error_m", "steps")})
        p0 = P(obj)
    if not reach(p0):
        r = drive_to(float(p0[0]), float(p0[1]))
        print("[drive->obj]", {k: r.get(k) for k in ("reached", "error_m")})

    # `site_above_bottom_m` separates a grasp from a miss and is legible BEFORE the descent:
    # stand_to_pick prices every footprint-clear pose by IK on a kinematic copy -- zero
    # frames -- and drives only when a pose is predicted to reach and standing still is not.
    sp = {}
    try:
        sp = stand_to_pick(obj)
        print("[stand]", {k: sp.get(k) for k in ("moved", "here_site_above_bottom_m",
                                                 "predicted_site_above_bottom_m", "steps",
                                                 "reason")})
    except Exception as exc:  # noqa: BLE001
        print("[stand]", exc)
    if not sp.get("moved"):
        torso_for(float(P(obj)[2]))
    # a pose the search PREDICTED cannot reach the object's lower half fails 7 times in 10
    # (and the three that grasp anyway do it on the first attempt): pay for one descent
    # there, not two
    predicted = sp.get("predicted_site_above_bottom_m")
    doubtful = predicted is not None and float(predicted) >= 0.07
    pick = pick_object(obj, attempts=1 if doubtful or budget() < 600 else 2)
    print("[pick]", pick)
    if not pick.get("grasped"):
        # nothing is in the jaws, so there is nothing to carry.  Carrying anyway is what
        # spent 490-900 frames on an empty-handed `place_object` in eleven of the twenty-two
        # composites measured on 2026-08-28, and every one of them was already lost.
        print("[pick] FAILED - the object is not in the jaws; no carry")
        return {"placed": False, "held": False, "reason": "nothing grasped"}

    aim, surface_z, how = aim_point(recep, dest_fx)
    print("[dest]", how, np.round(aim, 3).tolist(), "surface_z", surface_z, "budget", budget())
    if not reach(aim) and budget() > FINISH + 250:
        r = go(dest_fx, recep) if dest_fx else {"reached": False}
        print("[drive->dest]", dest_fx, {k: r.get(k) for k in ("reached", "error_m", "steps")})
        aim, surface_z, how = aim_point(recep, dest_fx)
        if not reach(aim) and budget() > FINISH + 200:
            # the fixture's own standing pose is not the receptacle's: BreadSelection drove
            # to counter_3_right, reported `reached`, and the cutting board was still 0.50 m
            # outside the arm.  Stand at the POINT the object has to reach.
            r = stand_at([float(aim[0]), float(aim[1]), float(aim[2])])
            print("[stand-at-dest]", {k: r.get(k) for k in ("reached", "shoulder_dist",
                                                            "steps", "reason")})
            aim, surface_z, how = aim_point(recep, dest_fx)
        print("[dest*]", how, np.round(aim, 3).tolist(), "budget", budget())
    if not reach(aim):
        # a release aimed 0.5-3.2 m outside the arm is a drop, and the servo spends 300-900
        # frames failing to get there first: keep hold of it and go to the finish
        print("[place] SKIPPED - the destination is", round(float(np.linalg.norm(
            aim[:2] - np.asarray(get_base_pose(), float)[:2])), 2), "m away, out of reach")
        return {"placed": False, "held": True, "reason": "destination out of reach"}

    drop_only = how.startswith("rim")
    torso_for(float(aim[2]))
    rep = place_object(obj, x=float(aim[0]), y=float(aim[1]), surface_z=surface_z,
                       clearance_m=0.10 if drop_only else 0.04, gentle=not drop_only)
    print("[place]", rep)
    if rep.get("held") and not rep.get("placed") \
            and rep.get("carry_error_m", 1.0) <= 0.06 and rep.get("descend_error_m", 0.0) > 0.12:
        # the xy is right and only the DESCENT is refused (an oven rack 0.35 m below what
        # the arm reaches from any clear standing spot).  Driving cannot fix a vertical
        # envelope and costs ~90 frames: release here, straight down onto the surface.
        rep = place_object(obj, x=float(aim[0]), y=float(aim[1]), surface_z=surface_z,
                           clearance_m=0.04, gentle=False, force=True)
        print("[place-drop]", rep)
    elif rep.get("held") and not rep.get("placed"):
        torso_for(max(float(aim[2]), 1.30) if float(aim[2]) > 1.15 else float(aim[2]))
        if dest_fx and budget() > FINISH + 200:
            r = go(dest_fx, recep)
            print("[drive->dest2]", {k: r.get(k) for k in ("reached", "error_m", "steps")})
            aim, surface_z, how = aim_point(recep, dest_fx)
            drop_only = how.startswith("rim")
        rep = place_object(obj, x=float(aim[0]), y=float(aim[1]), surface_z=surface_z,
                           clearance_m=0.10 if drop_only else 0.04, gentle=not drop_only,
                           force=budget() < FINISH + 200)
        print("[place2]", rep)
    if rep.get("held") and not rep.get("placed"):
        rep = place_object(obj, x=float(aim[0]), y=float(aim[1]), surface_z=surface_z,
                           clearance_m=0.10 if drop_only else 0.04, gentle=not drop_only,
                           force=True)
        print("[place3]", rep)
    return rep


# ------------------------------------------------------ every placement the scorer wants
done = []
for i, m in enumerate(pending):
    if solved():
        break
    if i and budget() < FINISH + CARRY:
        print("[stop] no room for another carry:", budget(), "frames left")
        break
    print(f"--- transport {i + 1}/{len(pending)} budget {budget()}")
    done.append(transport(m))

# ------------------------------------------------------------------- the cheap finish
# `gripper_obj_far` is scored on whatever object the SCORER named - often the receptacle
# (AddMarshmallow: `gripper_obj_far(self, "mug")`), not the thing that moved.  Retreat from
# every one of them, and never with `place_gently`, which backs off only 6 cm.
print("[before finish] satisfied", solved(), "budget", budget())
torso_for(0.0)
open_gripper()
far = [n for n in (PG.get("gripper_far") or []) if n in objects]
moved = [m["object"] for m in pending if m.get("object") in objects]
for name in (far + [n for n in moved if n not in far]):
    if solved() or budget() < 60:
        break
    print("[retreat]", name, retreat_from(name, 0.28))
if not solved() and budget() > 90:
    try:
        r = tuck_arm()
        print("[tuck]", {k: r.get(k) for k in ("reached", "steps")})
    except Exception as exc:  # noqa: BLE001
        print("[tuck]", exc)
print("[final] satisfied", get_goal()["satisfied"], "budget", get_step_budget(),
      "moved", {m["object"]: np.round(P(m["object"]), 3).tolist() for m in pending
                if m.get("object") in objects})
