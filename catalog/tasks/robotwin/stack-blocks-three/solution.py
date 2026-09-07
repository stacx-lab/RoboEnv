import numpy as np


def pose(name):
    return np.asarray(get_object_pose(name)[0], dtype=float)


def stacked(upper, lower):
    return (
        np.linalg.norm(upper[:2] - lower[:2]) <= 0.025
        and abs(upper[2] - lower[2] - 0.05) <= 0.012
    )


def try_open(arm):
    try:
        globals()[f"open_gripper_{arm}"]()
        return True
    except Exception:
        return False


def reset(arm):
    try_open(arm)
    try:
        globals()[f"back_to_origin_{arm}"]()
    except Exception:
        pass


def move_held(name, arm, target):
    obj = pose(name)
    gripper = np.asarray(globals()[f"get_gripper_pose_{arm}"](), dtype=float)
    globals()[f"goto_pose_{arm}"](
        gripper[:3] + np.asarray(target) - obj, gripper[3:]
    )


def place_block2():
    for _ in range(2):
        reset("right")
        try:
            grasp_object(
                "block2", "right",
                pre_grasp_dis=0.08,
                grasp_dis=0.0,
                contact_point_ids=1,
            )
            g = np.asarray(get_gripper_pose_right(), dtype=float)
            goto_pose_right(g[:3] + [0.04, -0.03, 0.10], g[3:])

            p1 = pose("block1")
            move_held("block2", "right", p1 + [0.0, 0.0, 0.13])
            move_held("block2", "right", p1 + [0.0, 0.0, 0.053])
        except Exception:
            try_open("right")
            continue

        if try_open("right") and stacked(pose("block2"), pose("block1")):
            return True
    return False


def handoff_block3():
    reset("left")
    grasp_object("block3", "left", pre_grasp_dis=0.10, grasp_dis=0.0)

    g = np.asarray(get_gripper_pose_left(), dtype=float)
    goto_pose_left(g[:3] + [-0.04, -0.03, 0.11], g[3:])

    p1 = pose("block1")
    target = p1 + [-0.3183, 0.0058, 0.1070]
    move_held("block3", "left", target)

    released = try_open("left")
    try:
        back_to_origin_left()
    except Exception:
        pass

    p3 = pose("block3")
    return (
        released
        and np.linalg.norm(p3[:2] - target[:2]) <= 0.025
        and abs(p3[2] - p1[2]) <= 0.012
    )


def place_block3():
    reset("right")
    grasp_object("block3", "right", pre_grasp_dis=0.10, grasp_dis=0.0)

    g = np.asarray(get_gripper_pose_right(), dtype=float)
    goto_pose_right(g[:3] + [0.04, -0.03, 0.11], g[3:])

    for offset in (
        [-0.1400, -0.0111, 0.1050],
        [-0.0700, -0.0061, 0.0600],
        [0.0, 0.0, 0.060],
        [0.0, 0.0, 0.052],
    ):
        move_held("block3", "right", pose("block2") + offset)

    released = try_open("right")
    p1, p2, p3 = pose("block1"), pose("block2"), pose("block3")
    return released and stacked(p2, p1) and stacked(p3, p2)


stage1_ok = place_block2()
stage2_ok = stage1_ok and handoff_block3()
stage3_ok = stage2_ok and place_block3()

left_open = try_open("left")
right_open = try_open("right")
p1, p2, p3 = pose("block1"), pose("block2"), pose("block3")

solved = (
    stage1_ok
    and stage2_ok
    and stage3_ok
    and stacked(p2, p1)
    and stacked(p3, p2)
    and left_open
    and right_open
)
print("SOLVED" if solved else "NOT_SOLVED")
