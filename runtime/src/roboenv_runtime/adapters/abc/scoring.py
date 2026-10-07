"""RoboEnv-defined scoring for ABC's three marker tasks and block tower."""

import itertools

import mujoco
import numpy as np


DRAWER_INSTRUCTION = (
    "Put all five markers in the {level} drawer, release them, and close the drawer."
)
TOWER_INSTRUCTION = (
    "Build a self-supporting tower of at least five blocks and release it for one second."
)


def geom_points(model, data, geom):
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start : start + count]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        points = np.asarray(list(itertools.product((-1, 1), repeat=3))) * model.geom_size[geom]
    else:
        raise ValueError("Supplemental evaluator requires mesh or box object collision geometry")
    return points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def descendant(model, body, root):
    while body and body != root:
        body = int(model.body_parentid[body])
    return body == root


def collisions(model, root):
    return [
        g
        for g in range(model.ngeom)
        if (model.geom_contype[g] or model.geom_conaffinity[g])
        and descendant(model, int(model.geom_bodyid[g]), root)
    ]


class StableEvaluator:
    def __init__(self, model):
        self.model = model
        self.since = None
        self.last_time = None
        self.robot_bodies = {
            b
            for b in range(model.nbody)
            if any(descendant(model, b, model.body(side + "_arm").id) for side in ("left", "right"))
        }

    def evaluate(self, data):
        good, metrics = self.condition(data)
        now = float(data.time)
        if not good:
            self.since = None
        elif self.since is None:
            self.since = now
        self.last_time = now
        stable = 0.0 if self.since is None else now - self.since
        success = good and stable >= 1.0 - 1e-9
        return {
            "reward": float(success),
            "success": bool(success),
            "scoring": "roboenv-defined",
            "rule": self.rule,
            "stable_s": stable,
            **metrics,
        }


class MarkersInDrawer(StableEvaluator):
    rule = "abc/markers-contained-and-closed/1"

    def __init__(self, model, level):
        super().__init__(model)
        self.instruction = DRAWER_INSTRUCTION.format(level=level)
        self.drawer = model.body(f"link_{level}_drawer_low").id
        self.joint = model.joint(f"joint_{level}_drawer_low")
        self.markers = [model.body(f"marker_{i}").id for i in range(1, 6)]
        self.marker_geoms = [collisions(model, b) for b in self.markers]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        boxes = []
        for g in collisions(model, self.drawer):
            points = (geom_points(model, data, g) - data.xpos[self.drawer]) @ data.xmat[
                self.drawer
            ].reshape(3, 3)
            boxes.append((points.min(0), points.max(0)))
        # Identify floor and walls by their thin axis, excluding tiny handle meshes.
        walls = [(lo, hi) for lo, hi in boxes if np.sort(hi - lo)[-2] > 0.04]
        bounds = []
        for axis in range(3):
            panels = [(lo, hi) for lo, hi in walls if np.argmin(hi - lo) == axis]
            if axis < 2:
                if len(panels) != 2:
                    raise ValueError("Expected two opposing native drawer walls")
                panels.sort(key=lambda pair: pair[0][axis])
                bounds.append((panels[0][1][axis], panels[1][0][axis]))
            else:
                if len(panels) != 1:
                    raise ValueError("Expected one native drawer floor")
                bounds.append(
                    (panels[0][1][axis], min(hi[2] for lo, hi in walls if np.argmin(hi - lo) != 2))
                )
        self.lower, self.upper = np.asarray(bounds).T

    def condition(self, data):
        contained = []
        for geoms in self.marker_geoms:
            points = np.concatenate([geom_points(self.model, data, g) for g in geoms])
            local = (points - data.xpos[self.drawer]) @ data.xmat[self.drawer].reshape(3, 3)
            contained.append(
                bool(np.all(local >= self.lower - 0.002) and np.all(local <= self.upper + 0.002))
            )
        held = False
        for contact in data.contact:
            a, b = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if (a in self.markers and b in self.robot_bodies) or (
                b in self.markers and a in self.robot_bodies
            ):
                held = True
        closed = (
            abs(float(data.qpos[self.joint.qposadr[0]] - self.model.qpos0[self.joint.qposadr[0]]))
            <= 0.005
        )
        return all(contained) and closed and not held, {
            "markers_contained": sum(contained),
            "drawer_closed": closed,
            "robot_contact": held,
        }


class BlockTower(StableEvaluator):
    rule = "abc/five-level-released-tower/1"
    instruction = TOWER_INSTRUCTION

    def __init__(self, model):
        super().__init__(model)
        self.blocks = {
            int(model.jnt_bodyid[j])
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        }
        if len(self.blocks) != 9:
            raise ValueError("Expected nine native building blocks")
        self.table_geoms = {
            g
            for g in range(model.ngeom)
            if model.geom(g).name == "table_plane"
            or model.body(int(model.geom_bodyid[g])).name == "play_table"
        }

    def condition(self, data):
        support = {b: set() for b in self.blocks}
        grounded, invalid = set(), set()
        for contact_id, contact in enumerate(data.contact):
            ga, gb = int(contact.geom1), int(contact.geom2)
            a, b = int(self.model.geom_bodyid[ga]), int(self.model.geom_bodyid[gb])
            if a not in self.blocks and b not in self.blocks:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, contact_id, force)
            if force[0] <= 1e-6:
                continue
            for body, other, other_geom, direction in ((a, b, gb, -1), (b, a, ga, 1)):
                if body not in self.blocks:
                    continue
                if other not in self.blocks and other_geom not in self.table_geoms:
                    invalid.add(body)
                if direction * contact.frame[2] <= 0.5:
                    continue
                if other_geom in self.table_geoms:
                    grounded.add(body)
                elif other in self.blocks and data.xipos[body, 2] > data.xipos[other, 2] + 0.005:
                    support[body].add(other)
        for body in self.blocks:
            if (
                np.linalg.norm(data.cvel[body, :3]) >= 0.1
                or np.linalg.norm(data.cvel[body, 3:]) >= 0.01
            ):
                invalid.add(body)
        levels = {}
        for body in sorted(self.blocks, key=lambda b: data.xipos[b, 2]):
            levels[body] = (
                0
                if body in invalid
                else max(
                    [1 if body in grounded else 0]
                    + [levels.get(other, 0) + 1 for other in support[body] if levels.get(other, 0)]
                )
            )
        height = max(levels.values(), default=0)
        return height >= 5, {"stable_levels": height}


def supplemental_evaluator(task, model):
    if task.startswith("put_markers_in_"):
        return MarkersInDrawer(model, task.removeprefix("put_markers_in_").removesuffix("_drawer"))
    if task == "build_wood_block_tower":
        return BlockTower(model)
    return None
