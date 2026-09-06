"""libero.10_9 — KITCHEN_SCENE6: put the yellow-and-white mug in the microwave and close it.

Goal: ``(In white_yellow_mug_1 microwave_1_heating_region)`` AND ``(Close microwave_1)``
(the hinge's ``qpos > -0.005`` on a ``[-2.094, 0]`` range; the door starts at -1.643).

Why a plain top-down place cannot work, measured on this scene: the cavity is
x 0.5137..0.7207, y 0.2607..0.4292, floor z 0.032, ROOF z 0.176 — 14.4 cm of headroom.
The Panda hand needs ~9-10 cm above the fingertips and spans ~21 cm across the closing
axis, which is the cavity's own width; a hand holding the mug by its rim or body has to
be INSIDE the cavity and hits the roof.  What makes the task expressible is that the mug's
handle stands 6.4 cm clear of the point the predicate tests:

* the handle's top bar (``..._g3``, a bar running in y at the -y side of the cup) is
  pinched top-down closing across it in x.  Measured: the mug hangs essentially upright
  (AABB 0.1061 x 0.1403 x 0.1041 held against 0.1061 x 0.1397 x 0.1032 at rest) and the
  origin sits ``(-0.0004, +0.0635, -0.0819)`` from the fingertips.
* so putting the mug's ORIGIN 4 cm inside the cavity's front edge leaves the whole hand
  at y ~0.237 — 2.4 cm IN FRONT of the opening plane — and the wrist rises in free air.
* the traverse is at mug-bottom z = floor + 1.3 cm (mug top 0.148 against the 0.176 roof),
  then one 1.1 cm descent and release.

Everything is read from the microwave's own collision geoms
(``get_object_parts``: g3/g4 side walls, g5 back, g6 roof, g7 floor) and from the region
box, so no coordinate is hard-coded.

Closing the door afterwards also pushes the mug the rest of the way in (origin y
0.280 -> 0.331 measured), because the handle is what protrudes through the opening.
``close_articulated(target_qpos=-0.003)`` leaves 2 mm inside the ``Close`` predicate.
Cost: ~2950 of 4000 steps.
"""

import numpy as np

OBJ = "white_yellow_mug_1"
REGION = "microwave_1_heating_region"
FIXTURE = "microwave_1"
JOINT = "microwave_1_microjoint"

QD = tool_down_quat_wxyz([1.0, 0.0, 0.0])  # tool down, fingers closing across x


def r4(v):
    return np.round(np.asarray(v, float), 4).tolist()


def tcp_now():
    return np.asarray(get_hand_pose()["tcp"], float)


def go(p, tag, tries=2):
    """goto_pose_closed with the wrist pinned, repeated while it has not arrived."""
    r = None
    for _ in range(int(tries)):
        r = goto_pose_closed(np.asarray(p, float), QD)
        if r["reached"]:
            break
    print(" ", tag, r["position_error_m"], r["orientation_dev_deg"],
          "origin", r4(body_position(OBJ)), "left", get_step_budget()["remaining"])
    return r


# --- the cavity, from the microwave's own collision geoms --------------------
parts = {p["geom"]: p for p in get_object_parts(FIXTURE)}
# the cavity floor and roof are the two FLAT boxes; the floor plate's own footprint is
# the cavity footprint (the side walls sit outside it) and its front edge the opening.
boxes = [p for p in parts.values() if p["type"] == "box"]
flat = sorted((p for p in boxes if float(p["hi"][2]) - float(p["lo"][2]) < 0.05),
              key=lambda p: float(p["center"][2]))
floor_g, roof_g = flat[0], flat[-1]
CAV_X = (float(floor_g["lo"][0]), float(floor_g["hi"][0]))
CAV_FLOOR = float(floor_g["hi"][2])
CAV_ROOF = float(roof_g["lo"][2])
CAV_FRONT = float(floor_g["lo"][1])
reg = get_region_boxes()[REGION]
print("cavity x", r4(CAV_X), "floor", round(CAV_FLOOR, 4), "roof", round(CAV_ROOF, 4),
      "front", round(CAV_FRONT, 4))
print("region lo", r4(reg["lo"]), "hi", r4(reg["hi"]))

# --- pinch the outer end of the handle's top bar -----------------------------
mug_parts = get_object_parts(OBJ)
o0 = np.asarray(body_position(OBJ), float)
# the handle is the material on the far side of the cup from its centre: take the bar
# whose centre is farthest from the object's origin in the horizontal plane and which
# still lies in the upper half of the mug
top = float(get_object_extents(OBJ)["top_z"])
cands = [p for p in mug_parts if float(p["center"][2]) > o0[2] + 0.25 * (top - o0[2])]
bar = max(cands, key=lambda p: float(np.linalg.norm(np.asarray(p["center"][:2]) - o0[:2])))
print("handle bar", bar["geom"], "lo", r4(bar["lo"]), "hi", r4(bar["hi"]))
GRIP = [float(o0[0]), float(bar["lo"][1]) + 0.004, float(bar["hi"][2]) - 0.005]
print("grip", r4(GRIP))

g = grasp_at(GRIP, QD, object_name=OBJ, lift_m=0.06)
print("grasp", {k: g.get(k) for k in ("opening_m", "held", "rise_m", "grasp_error_m",
                                      "descent_reason")},
      "left", get_step_budget()["remaining"])
assert g["held"], g

off = np.asarray(body_position(OBJ), float) - tcp_now()
print("origin - fingertips", r4(off))

TX = 0.5 * (CAV_X[0] + CAV_X[1])
TY = max(float(reg["lo"][1]) + 0.025, CAV_FRONT + 0.04)
Z_IN = CAV_FLOOR + 0.013      # mug bottom while it travels through the opening
Z_DOWN = CAV_FLOOR + 0.002
print("target origin", round(TX, 4), round(TY, 4), Z_IN, Z_DOWN)

go([TX - off[0], TY - off[1] - 0.10, Z_IN - off[2]], "stage")
go([TX - off[0], TY - off[1], Z_IN - off[2]], "insert")
off2 = np.asarray(body_position(OBJ), float) - tcp_now()
print("origin - fingertips after the insert", r4(off2))
go([TX - off2[0], TY - off2[1], Z_DOWN - off2[2]], "lower", tries=1)
print("open", open_gripper())
print("released", r4(body_position(OBJ)), "left", get_step_budget()["remaining"])

# lift the pads clear of the mug, then back out of the opening
t = tcp_now()
go([t[0], t[1], 0.24], "clear", tries=1)
t = tcp_now()
go([t[0], t[1] - 0.12, t[2]], "retreat", tries=1)

lo = np.asarray(reg["lo"], float) - np.array([0.0, 0.0, 0.01])
hi = np.asarray(reg["hi"], float)
pos = np.asarray(body_position(OBJ), float)
print("in region before the close?", bool(np.all(pos > lo) and np.all(pos < hi)), r4(pos))

print("hinge before", round(get_joint_states()[JOINT], 4), "left", get_step_budget()["remaining"])
c = close_articulated(JOINT, target_qpos=-0.003)
print("close", {k: c.get(k) for k in ("closed", "qpos", "strokes", "reason",
                                      "budget_exhausted")})

q = get_joint_states()[JOINT]
pos = np.asarray(body_position(OBJ), float)
inside = bool(np.all(pos > lo) and np.all(pos < hi))
print("FINAL hinge", round(q, 5), "Close?", q > -0.005, "In?", inside, r4(pos),
      "left", get_step_budget()["remaining"])
assert inside, (pos.tolist(), lo.tolist(), hi.tolist())
assert q > -0.005, q
