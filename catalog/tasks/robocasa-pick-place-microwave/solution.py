import numpy as np

def grasp_yaw(extent):
    cands = []
    for i in range(3):
        ax = np.asarray(extent["axes_world"][i], float)
        if abs(ax[2]) < 0.7:
            cands.append((float(extent["size"][i]), float(np.arctan2(ax[1], ax[0]))))
    size, yaw = min(cands)
    return yaw, size

def P(name):
    return np.asarray(get_object_pose(name)[0], float)

def tip():
    return np.asarray(get_gripper_pose()[:3], float)

fx = get_task_fixtures()
print("fixtures", fx, "objects", list_objects())
for o in list_objects():
    e = get_object_extent(o)
    print(" ", o, np.round(P(o), 3).tolist(), "size", np.round(e["size"], 3).tolist(),
          "bot", round(e["bottom_z"], 3), "top", round(e["top_z"], 3))
MW = fx["microwave"]
reg = get_fixture_regions(MW)
print("regions", reg)
b = np.asarray(get_base_pose(), float)
print("base", np.round(b, 3).tolist(), "tip", np.round(tip(), 3).tolist())

# ---- pick obj off the counter
e = get_object_extent("obj")
p0 = P("obj")
yaw, w = grasp_yaw(e)
print("[pick] yaw", round(yaw, 3), "width", round(w, 3))
quat = top_down_grasp_quat(yaw)
open_gripper()
gz = e["bottom_z"] + 0.5 * (e["top_z"] - e["bottom_z"])
r1 = goto_pose([p0[0], p0[1], gz + 0.12], quat, 0.015)
print("[hover]", {k: r1.get(k) for k in ("reached", "error_m", "steps")})
r2 = goto_pose([p0[0], p0[1], gz], quat, 0.008)
print("[descend]", {k: r2.get(k) for k in ("reached", "error_m", "steps")}, "tip", np.round(tip(), 3).tolist())
op = close_gripper()
move_by_displacement(z=0.15)
p1 = P("obj")
print(f"[grasp] opening {op:.3f} rose {p1[2]-p0[2]:.3f} budget {get_step_budget()}")
if p1[2] - p0[2] < 0.05:
    raise RuntimeError("no lift")

# ---- torso up, carry into the microwave
t = set_torso_height(0.34)
print("[torso]", {k: t.get(k) for k in ("height", "shoulder_z", "eef_z", "reached", "steps")})
print("  obj now", np.round(P("obj"), 3).tolist(), "tip", np.round(tip(), 3).tolist())

c = P("container")
ce = get_object_extent("container")
off = tip()[:2] - P("obj")[:2]
oh = get_object_extent("obj")
half = 0.5 * (oh["top_z"] - oh["bottom_z"])
zt = ce["top_z"] + half + 0.045
tgt = np.array([c[0] + off[0], c[1] + off[1], zt])
print("[target]", np.round(tgt, 3).tolist(), "container", np.round(c, 3).tolist(),
      "top", round(ce["top_z"], 3), "off", np.round(off, 3).tolist())

# in front of the mouth at carry height, then straight in
mouth_x = max(reg[list(reg)[0]]["center"][0] + 0.5 * reg[list(reg)[0]]["size_xy"][0], tgt[0]) + 0.22
for q in ([mouth_x, tgt[1], zt], tgt.tolist()):
    try:
        r = goto_pose(q, None, 0.02)
        print("[carry]", np.round(q, 3).tolist(), {k: r.get(k) for k in ("reached", "error_m", "steps")},
              "tip", np.round(tip(), 3).tolist())
    except Exception as exc:
        print("[carry] FAIL", np.round(q, 3).tolist(), exc, "tip", np.round(tip(), 3).tolist())
print("budget", get_step_budget())
pg = place_gently("obj")
print("[place]", pg, "obj", np.round(P("obj"), 3).tolist())

# ---- retreat straight out of the mouth, then far
for dx in (0.15, 0.15, 0.15):
    p = P("obj")
    if np.linalg.norm(tip() - p) > 0.28:
        break
    try:
        r = move_by_displacement(x=dx)
        print("[out]", {k: r.get(k) for k in ("reached", "error_m")}, "tip", np.round(tip(), 3).tolist())
    except Exception as exc:
        print("[out] FAIL", exc)
        break
p = P("obj")
c2 = P("container")
print(f"[final] obj {np.round(p,3).tolist()} container {np.round(c2,3).tolist()} "
      f"d_obj_cont {np.linalg.norm(p-c2):.3f} grip-obj {np.linalg.norm(tip()-p):.3f} budget {get_step_budget()}")
