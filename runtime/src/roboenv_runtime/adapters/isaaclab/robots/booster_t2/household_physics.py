"""Owner-only contact and functional-state measurements for household tasks."""

import math

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix
from .household_models import SURFACES, model


def quat_to_matrix(q):
    return matrix([0, 0, 0, *q[1:], q[0]])[:3, :3]


def collider_identity(path):
    """Keep entity, articulation link and collision part distinct."""
    pieces = str(path).split("/")
    if "Robot" in pieces:
        text = "/".join(pieces[pieces.index("Robot") + 1 :]).lower()
        for side in ("left", "right"):
            if side + "_" in text and any(
                s in text
                for s in (
                    "palm",
                    "thumb",
                    "index",
                    "middle",
                    "ring",
                    "pinky",
                    "hand",
                    side + "_base_link",
                )
            ):
                # The importer can retain a shared leaf name such as "mesh".
                # Keep the link path so distinct fingers remain distinct.
                return (side + "_hand", "hand", text)
        return ("robot", "body", pieces[-1])
    if "Task" in pieces:
        i = pieces.index("Task")
        return (pieces[i + 1], pieces[i + 2] if len(pieces) > i + 2 else "body", pieces[-1])
    if "Household" in pieces:
        i = pieces.index("Household")
        return (pieces[i + 1], "body", pieces[-1])
    return ("ground", "body", pieces[-1])


class ContactPairs:
    """Persist sleeping contacts until PhysX emits a lost event; clear on reset."""

    def __init__(self):
        self.pairs = {}
        self.events = []

    def reset(self):
        self.pairs.clear()
        self.events.clear()

    def feed(self, key, a, b, samples, lost=False):
        if lost:
            self.pairs.pop(key, None)
            return
        self.pairs[key] = (a, b, samples)
        self.events.append((a, b))

    def between(self, a, b, *, part=None, link=None):
        for x, y, samples in self.pairs.values():
            if x[0] == b and y[0] == a:
                x, y = y, x
                samples = [(p, -n, f) for p, n, f in samples]
            if (
                x[0] == a
                and y[0] == b
                and (part is None or y[2].startswith(part))
                and (link is None or y[1] == link)
            ):
                yield x, y, samples

    def touching(self, a, b, **kwargs):
        return any(True for _ in self.between(a, b, **kwargs))

    def upward_force(self, a, b, **kwargs):
        return sum(
            max(0.0, float(n[2])) * f
            for _, _, samples in self.between(a, b, **kwargs)
            for _, n, f in samples
        )


class Measurements:
    def __init__(self, world):
        import carb
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, UsdPhysics

        self.world = world
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.contacts = ContactPairs()
        self.models = {
            n: model(d["model"])
            for collection in (world.task.spec["items"], world.task.spec["articulated"])
            for n, d in collection.items()
        }
        self.definitions = {**world.task.spec["items"], **world.task.spec["articulated"]}
        stage = world.sim.stage if hasattr(world.sim, "stage") else None
        if stage is None:
            import isaaclab.sim as sim_utils

            stage = sim_utils.get_current_stage()
        self.stage = stage
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.interface = get_physx_simulation_interface()
        self._samples = {}
        self._lamps = {}

    def refresh(self):
        # The native owner steps PhysX directly without pumping Kit events.
        # Read the current-step report immediately after fetch_results instead
        # of relying on a renderer-driven subscription callback.
        report = self.interface.get_contact_report()
        self._report(report[0], report[1])

    def _report(self, headers, data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType

        for header in headers:
            paths = [
                str(PhysicsSchemaTools.intToSdfPath(v))
                for v in (header.collider0, header.collider1)
            ]
            key = tuple(paths)
            points = []
            for i in range(
                header.contact_data_offset, header.contact_data_offset + header.num_contact_data
            ):
                c = data[i]
                # Contact normals point toward actor0; impulses are accumulated
                # over this native physics step, hence division by dt gives N.
                force = float(np.linalg.norm(np.asarray(c.impulse))) / self.world.physics_dt
                points.append(
                    (np.asarray(c.position, dtype=float), np.asarray(c.normal, dtype=float), force)
                )
            self.contacts.feed(
                key,
                *[collider_identity(p) for p in paths],
                points,
                lost=header.type == ContactEventType.CONTACT_LOST,
            )

    def reset(self):
        self.contacts.reset()

    def pose(self, name, link=None):
        if name in SURFACES:
            xy, _, height = SURFACES[name]
            return np.array([*xy, height]), np.array([1.0, 0, 0, 0])
        if name in self.world._items:
            return self.world.item_pose(name)
        art = self.world._articulated[name]
        index = art.body_names.index(link or self.models[name].links[0].name)
        value = art.data.body_link_pose_w.torch[0, index].detach().cpu().numpy()
        return value[:3].astype(float), value[[6, 3, 4, 5]].astype(float)

    def joint(self, name, joint):
        return float(self.world.articulated_joints(name)[joint])

    def joint_speed(self, name, joint):
        return float(self.world.articulated_state(name)["joint_velocities"][joint])

    def local(self, point, name, link=None):
        p, q = self.pose(name, link)
        return (np.asarray(point) - p) @ quat_to_matrix(q)

    def points(self, name):
        """Actual primitive boundary samples, not corners of a cylinder's AABB."""
        from .controller.rotations import rpy_matrix

        if name not in self._samples:
            points = []
            for p in self.models[name].links[0].parts:
                if not p.collision:
                    continue
                if p.shape == "box":
                    local = np.array(
                        [
                            [x * p.size[0] / 2, y * p.size[1] / 2, z * p.size[2] / 2]
                            for x in (-1, 1)
                            for y in (-1, 1)
                            for z in (-1, 1)
                        ]
                    )
                elif p.shape == "cylinder":
                    local = np.array(
                        [
                            [p.size[0] * math.cos(a), p.size[0] * math.sin(a), z * p.size[1] / 2]
                            for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)
                            for z in (-1, 1)
                        ]
                    )
                else:
                    local = np.array(
                        [
                            [p.size[0] * s * (i == axis) * p.size[i + 1] for i in range(3)]
                            for axis in range(3)
                            for s in (-1, 1)
                        ]
                    )
                points.extend(np.asarray(p.pos) + local @ rpy_matrix(p.rpy).T)
            self._samples[name] = np.asarray(points)
        p, q = self.pose(name)
        return p + self._samples[name] @ quat_to_matrix(q).T

    def speed(self, name):
        if name in self.world._items:
            return self.world.item_velocity(name)
        return np.asarray(self.world.articulated_state(name)["velocity"])

    def quiet(self, name):
        v = self.speed(name)
        return bool(np.linalg.norm(v[:3]) < 0.035 and np.linalg.norm(v[3:]) < 0.25)

    def upright(self, name, tolerance=0.30):
        return bool(quat_to_matrix(self.pose(name)[1])[2, 2] > math.cos(tolerance))

    def touch(self, name, side):
        return self.contacts.touching(name, side + "_hand")

    def supported(self, name, target, link=None):
        return self.contacts.upward_force(name, target, link=link) > 0.02

    def hand_support(self, name, side):
        weight = self.models[name].links[0].mass * 9.81
        pairs = list(self.contacts.between(name, side + "_hand"))
        # A palm can support from below; opposing finger surfaces can grasp
        # through friction. Mere proximity and an unloaded single touch cannot.
        return (
            self.contacts.upward_force(name, side + "_hand") > max(0.02, 0.15 * weight)
            or len({y[2] for _, y, s in pairs if sum(f for _, _, f in s) > 0.02}) >= 2
        )

    def held(self, name):
        return self.hand_support(name, "left") or self.hand_support(name, "right")

    def released(self, name):
        return not self.held(name)

    def on(self, name, target, center=False):
        if not self.supported(name, target) or not self.released(name) or not self.quiet(name):
            return False
        if center:
            local = self.local(self.pose(name)[0], target)
            # Receiver-relative central area, generous enough for stable
            # placements while excluding a visibly rim-adjacent placement.
            radius = self.models[target].radius
            return bool(np.linalg.norm(local[:2]) <= 0.40 * radius)
        return True

    def inside(self, name, target, region="inside", *, settled=True):
        link, center, half = self.models[target].regions[region]
        pts = self.local(self.points(name), target, link) - np.asarray(center)
        # Vertical protrusion is allowed. The receiver floor/support and
        # actual horizontal shape distinguish containment from rim balancing.
        m = self.definitions[target]["model"]
        if m in ("bowl", "plate", "dish", "balance_scale"):
            fits = np.max(np.linalg.norm(pts[:, :2], axis=1)) <= half[0] + 0.003
        else:
            fits = np.all(np.abs(pts[:, :2]) <= np.asarray(half[:2]) + 0.003)
        floor = self.contacts.touching(
            name, target, part="floor", link=link
        ) or self.contacts.touching(name, target, part="bottom", link=link)
        return bool(
            fits
            and floor
            and self.supported(name, target, link)
            and (not settled or self.released(name) and self.quiet(name))
        )

    def lamp_light(self, name):
        from pxr import Gf
        from .household_usd import reading_light

        if name not in self._lamps:
            self._lamps[name] = reading_light(self.stage, name)
        light, translation, orientation = self._lamps[name]
        p, q = self.pose(name, "head")
        origin = p + quat_to_matrix(q) @ np.array([0, -0.077, 0])
        translation.Set(Gf.Vec3d(*map(float, origin)))
        # USD emits along -Z; the physical lamp emits along head-local -Y.
        orientation.Set(
            Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))
            * Gf.Quatd(math.sqrt(0.5), Gf.Vec3d(-math.sqrt(0.5), 0, 0))
        )
        light.GetIntensityAttr().Set(2200.0 if self.joint(name, "power") > 0.18 else 0.0)

    def lit_pages(self, lamp, book):
        # Same local optical axis and cone as the authored USD light.
        if self.joint(lamp, "power") <= 0.18:
            return False
        p, q = self.pose(lamp, "head")
        rotation = quat_to_matrix(q)
        origin = p + rotation @ np.array([0, -0.077, 0])
        axis = rotation @ np.array([0, -1, 0])
        # Sample the page interior in the book frame. Rays aimed exactly at
        # collision-box corners can miss because of floating-point rounding.
        body = next(part for part in self.models[book].links[0].parts if part.name == "body")
        bp, bq = self.pose(book)
        br = quat_to_matrix(bq)
        if br[2, 2] < 0.8:
            return False
        top = [
            bp + br @ np.array([x * (body.size[0] / 2 - 0.008),
                                y * (body.size[1] / 2 - 0.008), body.size[2]])
            for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0))
        ]
        from omni.physx import get_physx_scene_query_interface

        for target in top:
            delta = target - origin
            distance = float(np.linalg.norm(delta))
            if distance <= 0 or np.dot(delta / distance, axis) < math.cos(math.radians(38)):
                return False
            hit = get_physx_scene_query_interface().raycast_closest(
                tuple(origin), tuple(delta / distance), distance + 0.005
            )
            if (
                not hit.get("hit")
                or collider_identity(hit.get("collision", hit.get("rigidBody", "")))[0] != book
            ):
                return False
        return True
