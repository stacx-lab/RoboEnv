"""libero.90_8 — open the top drawer of the cabinet and put the bowl in it.

Break the posture trade-off by moving the bowl OUT first.

`(Open wooden_cabinet_1_top_region)` needs `qpos < -0.14`; `(In akita_black_bowl_1 ...)`
needs the bowl's origin in the drawer's box.  Measured: the pull reaches the -0.16 stop at
`stroke=0.10` (969 steps), but at that depth `rim_pick` on the bowl fails on all four
walls and all four radial offsets; at the shallow -0.1432 the pick works and costs the
drawer 21.5 mm against 3.2 mm of margin.  Neither depth affords both.

The way out is to change WHERE the bowl is when the deep pick happens: pick it up while
the drawer is still shut (nothing to disturb), set it down clear of the cabinet, pull the
drawer to its stop, and pick it again from there.

Measured 2026-08-28, all three numbers from this program's own log: the staging pick costs
the drawer **nothing** (it is shut), the pull reaches **-0.15992** in 2 strokes and 915
steps, and the second pick and the carry leave the joint at **-0.15992** — the give-back
is zero, against 21.5 mm when the bowl is picked from where it starts, 0.28 m from the
cabinet.  2706 sim steps of 4000.

The stroke matters as much as the staging.  Swept at `target_fraction=0.936`: stroke 0.05
reaches -0.15992 in 4 strokes and 1218 steps, **0.10 reaches -0.15997 in 2 strokes and
969**, 0.14 (the family program's) stops at **-0.14322**, 0.20 at -0.15773.  LIBERO scores
this drawer with `WoodenCabinet.is_open` = `qpos < -0.14`, so 0.14 leaves 3.2 mm of margin
and any carry spends more than that.

Nothing is read from the sentence: the object, the region and the joint all come out of
`get_goal_predicates()`, and the staging point is derived from the fixture's own position.
"""
import numpy as np

goal = get_goal_predicates()
PREDS = [[str(x) for x in p] for p in goal["predicates"]]
print("goal:", goal.get("language"), PREDS)
REG = next(p[2] for p in PREDS if p[0].lower() == "in")
_obj = next(p[1] for p in PREDS if p[0].lower() == "in")
POSES = get_all_object_poses()
OBJ = _obj if _obj in POSES else _obj.rsplit("_", 1)[0]
JI = get_joint_info()
_stem = REG[: -len("_region")] if REG.endswith("_region") else REG
J = next(j for j in (_stem + "_level", _stem, _stem + "_joint") if j in JI)
# stage the object on the far side of the robot from the fixture the drawer belongs to,
# and away from anything else standing on the table.
_fix = np.asarray(body_position(get_region_boxes()[REG]["parent"]), float)
_here = np.asarray(body_position(OBJ), float)
STAGE = [float(_here[0]) - 0.11, float(_fix[1]) + 0.43]
print("joint", J, "object", OBJ, "staging at", [round(v, 3) for v in STAGE])
LO, HI = (float(v) for v in JI[J]["range"])
CLOSED, OPENED = (HI, LO) if abs(HI) < abs(LO) else (LO, HI)
SPAN = abs(HI - LO)
TARGET = CLOSED + (-0.883 * SPAN if OPENED < CLOSED else 0.883 * SPAN)


def q():
    return float(get_joint_states()[J])


def left():
    return get_step_budget()["remaining"]


def in_ok():
    b = get_region_boxes()[REG]
    lo = np.asarray(b["lo"], float).copy()
    lo[2] -= 0.01                       # LIBERO's own floor relaxation
    hi = np.asarray(b["hi"], float)
    p = np.asarray(body_position(OBJ), float)
    return bool(np.all(p > lo) and np.all(p < hi)), np.round(p, 4).tolist()


_HP = get_hand_pose()
HOME_POS = np.asarray(_HP.get("position", _HP.get("tcp")), float)
HOME_QUAT = np.asarray(_HP["quat_wxyz"], float)


def go_home():
    move_by_displacement(0.0, 0.0, 0.12)
    goto_pose_closed(HOME_POS.tolist(), HOME_QUAT.tolist())
    return np.round(tcp(), 3).tolist()


print("target qpos <", round(TARGET, 4))

# 1. stage the bowl clear of the cabinet, with the drawer still shut
r = rim_pick(OBJ, depth_m=0.02, max_steps=800)
print("stage pick", {k: r.get(k) for k in ("grasped", "side", "rise_m", "reason")},
      "q", round(q(), 5), "left", left())
if r.get("grasped"):
    carry_to(STAGE[0], STAGE[1], height_m=0.28, object_name=OBJ)
    open_gripper()
    wait(20)
    print("staged at", np.round(body_position(OBJ), 4).tolist(), "left", left())
    park(height_m=0.30)

# 2. pull to the stop
reach = abs(OPENED - q()) / SPAN
rep = open_articulated(J, target_fraction=min(0.936, max(reach - 0.005, 0.05)),
                       stroke=0.10, max_strokes=6, pitches=(0.0,))
print("pull", {k: rep.get(k) for k in ("opened", "qpos", "strokes", "regrasps")},
      "q", round(q(), 5), "left", left())
print("home", go_home(), "q", round(q(), 5), "left", left())

# 3. pick it again from the staging spot and carry it in
ok, where = in_ok()
if not ok and left() > 900:
    r = rim_pick(OBJ, depth_m=0.02, max_steps=max(500, left() - 450))
    print("pick", {k: r.get(k) for k in ("grasped", "side", "rise_m", "reason")},
          "q", round(q(), 5), "left", left())
    if r.get("grasped"):
        b = get_region_boxes()[REG]
        c = np.asarray(b["center"], float)
        hang = float(np.asarray(tcp(), float)[2]) - float(get_object_extents(OBJ)["bottom_z"])
        travel = max(float(b["hi"][2]), float(b["top_z"])) + 0.10 + hang
        carry_to(float(c[0]), float(c[1]),
                 height_m=max(travel, float(np.asarray(tcp(), float)[2])), object_name=OBJ)
        print("pre-carry q", round(q(), 5), "left", left())
        rp = place_in_region(OBJ, REG, mode="drop", margin_m=0.08)
        print("place", {k: rp.get(k) for k in ("inside", "released")},
              "q", round(q(), 5), "left", left())
    ok, where = in_ok()

print("VERDICT open", round(q(), 5), "<", round(TARGET, 4), "->", q() < TARGET,
      "| in", ok, where, get_step_budget())
assert q() < TARGET and ok, (round(q(), 5), ok, where)
