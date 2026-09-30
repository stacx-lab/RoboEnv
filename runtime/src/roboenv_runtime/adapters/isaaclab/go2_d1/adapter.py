"""Go2–D1 low-level leg and arm controllers.

Trot generates leg targets from bounded body commands on horizontal ground.
SerialArm and ArmIKController provide arm-local kinematics and bounded
Cartesian-to-joint control. Neither controller owns a simulator or clock.
RobotModel reads whole-robot URDF geometry and COM; SerialArm reads public arm
profiles. Both pose-IK interfaces share one solver. Arm twists use angular-first
ordering in arm-root axes.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(roll, pitch, yaw):
    return rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)


def quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat(r):
    # Largest diagonal branch is stable even at rotations of pi.
    candidates = np.array(
        [
            1 + r[0, 0] - r[1, 1] - r[2, 2],
            1 - r[0, 0] + r[1, 1] - r[2, 2],
            1 - r[0, 0] - r[1, 1] + r[2, 2],
            1 + np.trace(r),
        ]
    )
    i = int(np.argmax(candidates))
    q = np.zeros(4)
    q[i] = math.sqrt(max(candidates[i], 0)) / 2
    d = 4 * q[i]
    if i == 3:
        q[:3] = [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]
        q[:3] /= d
    else:
        j, k = (i + 1) % 3, (i + 2) % 3
        q[j], q[k], q[3] = (r[j, i] + r[i, j]) / d, (r[k, i] + r[i, k]) / d, (r[k, j] - r[j, k]) / d
    return q / np.linalg.norm(q)


def pose(t):
    return {"translation_m": t[:3, 3].tolist(), "rotation_xyzw": matrix_quat(t[:3, :3]).tolist()}


class RobotModel:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.joints = {}
        for elem in self.root.findall("joint"):
            origin = elem.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, :3] = rpy_matrix(*map(float, origin.get("rpy", "0 0 0").split()))
                t[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
            axis = elem.find("axis")
            limit = elem.find("limit")
            self.joints[elem.get("name")] = {
                "parent": elem.find("parent").get("link"),
                "child": elem.find("child").get("link"),
                "type": elem.get("type"),
                "origin": t,
                "axis": np.array(list(map(float, axis.get("xyz").split())))
                if axis is not None
                else np.array([1.0, 0, 0]),
                "limits": None
                if limit is None
                else [float(limit.get("lower")), float(limit.get("upper"))],
            }
        self.arm_names = [f"d1_Joint{i}" for i in range(1, 7)]

    def forward(self, positions):
        frames = {"base": np.eye(4)}
        pending = list(self.joints.items())
        while pending:
            before = len(pending)
            for name, j in pending[:]:
                if j["parent"] not in frames:
                    continue
                t = j["origin"].copy()
                q = positions.get(name, 0.0)
                if j["type"] in ("revolute", "continuous"):
                    t[:3, :3] = t[:3, :3] @ rotation(j["axis"], q)
                elif j["type"] == "prismatic":
                    t[:3, 3] += t[:3, :3] @ (j["axis"] * q)
                frames[j["child"]] = frames[j["parent"]] @ t
                pending.remove((name, j))
            if len(pending) == before:
                raise ValueError("URDF is not a connected tree rooted at base")
        return frames

    def center_of_mass(self, positions):
        """Assembly COM in the base frame, including fixed payload/camera masses."""
        frames = self.forward(positions)
        weighted = np.zeros(3)
        total = 0.0
        for link in self.root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            transform = frames[link.get("name")]
            weighted += mass * (transform[:3, :3] @ xyz + transform[:3, 3])
            total += mass
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Robot assembly must have a positive finite mass")
        return weighted / total

    def arm_fk(self, q):
        frames = self.forward(dict(zip(self.arm_names, q, strict=True)))
        return np.linalg.inv(frames["d1_base_link"]) @ frames["d1_Link6"]

    def arm_ik(self, target, seed, iterations=150):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("IK pose must be finite [x,y,z,qx,qy,qz,qw]")
        if abs(np.linalg.norm(target[3:]) - 1) > 1e-3:
            raise ValueError("IK quaternion must be unit length")
        bounds = np.array([self.joints[n]["limits"] for n in self.arm_names])
        q = np.asarray(seed, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("IK seed must contain six finite joint angles")
        desired = transform({"translation_m": target[:3], "rotation_xyzw": target[3:]})
        return solve_pose_ik(
            desired, q, bounds, self.arm_fk, iterations=iterations,
            failure="Arm IK did not converge within joint limits; no command applied",
        ).tolist()


def solve_pose_ik(target, seed, bounds, fk, *, jacobian=None, iterations=250,
                  failure="Unreachable arm target"):
    """Shared bounded pose solve for profile TCP and URDF flange targets.

    Jacobians are angular-first in the target frame. URDF callers obtain a
    numerical geometric Jacobian from FK; profile callers supply their analytic
    geometric Jacobian. This calculation neither sends commands nor advances time.
    """
    target = rigid_transform(target)
    seed = joint_vector(seed, len(bounds))
    if type(iterations) is not int or iterations < 1:
        raise ValueError("IK iterations must be a positive integer")
    q = np.clip(seed, bounds[:, 0], bounds[:, 1])
    for _ in range(iterations):
        measured = fk(q)
        error = np.r_[
            rotation_error(target[:3, :3], measured[:3, :3]),
            target[:3, 3] - measured[:3, 3],
        ]
        if np.linalg.norm(error[:3]) < 0.01 and np.linalg.norm(error[3:]) < 0.001:
            return q
        if jacobian is None:
            columns = []
            for delta in np.eye(len(q)) * 1e-5:
                shifted = fk(q + delta)
                columns.append(np.r_[
                    rotation_error(shifted[:3, :3], measured[:3, :3]),
                    shifted[:3, 3] - measured[:3, 3],
                ] / 1e-5)
            j = np.column_stack(columns)
        else:
            j = jacobian(q)
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), error)
        q = np.clip(q + np.clip(dq, -0.15, 0.15), bounds[:, 0], bounds[:, 1])
    raise ValueError(failure)


# Target order: front-left, front-right, rear-left, rear-right.
LEGS = ("FL", "FR", "RL", "RR")
# Each leg contributes hip, thigh, calf angles: 12 targets total.
LEG_NAMES = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]
# Vector order shared by validation, gait commands, and device profiles.
COMMAND_FIELDS = ("vx", "vy", "wz", "height", "roll", "pitch")
# Default: stationary, 0.30 m high, zero roll and pitch.
DEFAULT = np.array([0.0, 0.0, 0.0, 0.30, 0.0, 0.0])
# Hip origins in body coordinates, metres, in LEGS order.
HIP = np.array(
    [[0.1934, 0.0465, 0], [0.1934, -0.0465, 0], [-0.1934, 0.0465, 0], [-0.1934, -0.0465, 0]]
)


def validate_command(args):
    # Require a complete structured command with no missing or extra top-level keys.
    if not isinstance(args, dict) or set(args) != {"mode", "velocity", "posture"}:
        raise ValueError("Base requires mode, velocity and posture")
    # Locomotion is the only mode; zero velocity requests standing.
    if args["mode"] != "locomotion":
        raise ValueError("Unsupported base mode")
    # Check nested container types before examining their field names.
    if not isinstance(args["velocity"], dict) or not isinstance(args["posture"], dict):
        raise ValueError("Velocity and posture must be objects")
    # Require exactly three velocity fields and three posture fields.
    if set(args["velocity"]) != {"vx", "vy", "wz"} or set(args["posture"]) != {
        "height",
        "roll",
        "pitch",
    }:
        raise ValueError("Base requires vx/vy/wz and height/roll/pitch")
    # Flatten both field groups for validation and vector packing.
    values = {**args["velocity"], **args["posture"]}
    for key in COMMAND_FIELDS:
        x = values[key]
        if type(x) not in (float, int) or not math.isfinite(x):
            raise ValueError(f"{key} must be a finite number")
    return np.array([values[k] for k in COMMAND_FIELDS])


def leg_ik(foot, side):
    # Foot position relative to the hip origin, in body axes and metres.
    x, y, z = np.asarray(foot, dtype=float)
    # Lateral hip offset h and equal thigh/calf lengths L, in metres.
    h, L = 0.0955, 0.213
    # Remove the lateral offset to obtain squared sagittal-plane height.
    d = y * y + z * z - h * h
    # Reject targets inside or on the lateral-offset singularity.
    if d <= 0:
        raise ValueError("Foot target inside hip offset")
    # Choose the branch with the foot below the hip in the leg sagittal plane.
    z0 = -math.sqrt(d)
    # Rotate the hip plane toward the target; side is +1 for left, -1 for right.
    hip = math.atan2(z, y) - math.atan2(z0, side * h)
    # Wrap the hip angle into [-pi, pi).
    hip = (hip + math.pi) % (2 * math.pi) - math.pi
    # Apply the cosine rule to obtain the cosine of the calf joint angle.
    c = (x * x + z0 * z0 - 2 * L * L) / (2 * L * L)
    # Reject distances that the two links cannot reach.
    if not -1 <= c <= 1:
        raise ValueError("Foot target outside leg reach")
    # Select the negative calf-angle branch used by the standing pose.
    calf = -math.acos(c)
    # Aim toward the foot and subtract the angular contribution of the calf link.
    thigh = math.atan2(-x, -z0) - math.atan2(L * math.sin(calf), L + L * math.cos(calf))
    # Return absolute hip, thigh, calf angles in radians.
    return [hip, thigh, calf]


class Trot:
    """Diagonal gait with body-posture feedback; only outputs leg joints."""

    # Seconds per full stance-and-swing cycle.
    period = 0.32

    def __init__(self, support_center=(0.0, 0.0)):
        # Native initialization supplies the folded assembly's URDF-derived COM
        # projection in body x/y (metres); this support offset stays fixed during control.
        self.support_center = np.asarray(support_center, dtype=float).copy()
        # Start the internal rate-limited command at the default standing posture.
        self.command = DEFAULT.copy()
        # Normalized cycle progress in [0, 1); it advances only while moving.
        self.phase = 0.0
        # Accumulated height/roll/pitch corrections, initially zero.
        self.posture_integral = np.zeros(3)
        # Accumulated vx/vy/wz corrections, initially zero.
        self.velocity_integral = np.zeros(3)
        # Smoothed measured body vx/vy/wz, initially zero.
        self.filtered_velocity = np.zeros(3)
        # Smoothed world vertical velocity and body angular x/y rates.
        self.filtered_posture_rates = np.zeros(3)

    def state(self):
        # Convert array-valued controller memory to lists for serializable snapshots.
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    def targets(
        self, command, dt, measured_posture=None, measured_velocity=None, posture_rates=None
    ):
        # dt is seconds. Measurements are height/roll/pitch, body vx/vy/wz,
        # and world vertical velocity/body angular x/y rates, respectively.
        # Maximum command changes per second: m/s², m/s², rad/s², m/s, rad/s, rad/s.
        rates = np.array([0.5, 0.4, 1.0, 0.04, 0.18, 0.18])
        # Move each internal command toward its request by at most rate * dt.
        self.command += np.clip(command - self.command, -rates * dt, rates * dt)
        # Begin with feedforward values; feedback below adjusts these local values.
        vx, vy, wz, height, roll, pitch = self.command
        # Update the rate estimate only when rate measurements are supplied.
        if posture_rates is not None:
            # First-order smoothing with a 0.05 s time scale; cap the update weight at 1.
            self.filtered_posture_rates += min(dt / 0.05, 1.0) * (
                # Innovation: new measured rates minus the stored rate estimate.
                np.asarray(posture_rates) - self.filtered_posture_rates
            )
        # Apply posture feedback only when measured height/roll/pitch are available.
        if measured_posture is not None:
            # Positive error means the measured component is below its smoothed target.
            error = self.command[3:] - np.asarray(measured_posture)
            # Integrate error with gain 0.6 and cap accumulated corrections to avoid windup.
            self.posture_integral = np.clip(
                # Integral caps: ±0.03 m height and ±0.06 rad roll/pitch.
                self.posture_integral + error * dt * 0.6, [-0.03, -0.06, -0.06], [0.03, 0.06, 0.06]
            )
            # Add bounded feedback to the requested posture; limits below bound the
            # correction, not the final absolute height/roll/pitch.
            height, roll, pitch = self.command[3:] + np.clip(
                # Proportional correction for current posture error, gain 0.3.
                0.3 * error
                # Integral correction compensates for persistent posture bias.
                + self.posture_integral
                # Subtract filtered rate feedback to damp vertical and angular motion.
                - np.array([0.03, 0.08, 0.08]) * self.filtered_posture_rates,
                # Minimum feedback corrections, in metres/radians/radians.
                [-0.035, -0.08, -0.08],
                # Maximum feedback corrections, in metres/radians/radians.
                [0.035, 0.08, 0.08],
            )
        # A small numerical deadband on the smoothed velocity vector selects standing.
        moving = np.linalg.norm(self.command[:3]) > 0.002
        if moving:
            # Advance cycle progress by dt/period and wrap at one complete cycle.
            self.phase = (self.phase + dt / self.period) % 1
        # Correct stride velocity only while moving and when velocity is measured.
        if moving and measured_velocity is not None:
            # Smooth measured body vx/vy/wz with a 0.15 s time scale.
            self.filtered_velocity += min(dt / 0.15, 1.0) * (
                # Innovation: measured velocity minus the stored velocity estimate.
                np.asarray(measured_velocity) - self.filtered_velocity
            )
            # Compare filtered velocity against the rate-limited request.
            error = self.command[:3] - self.filtered_velocity
            # Integrate velocity error with gain 1.2 and bounded per-axis corrections.
            self.velocity_integral = np.clip(
                # Integral caps are ±0.22 m/s, ±0.15 m/s, and ±0.4 rad/s.
                self.velocity_integral + error * dt * 1.2, [-0.22, -0.15, -0.4], [0.22, 0.15, 0.4]
            )
            # Bound the feedback-adjusted velocities used to generate foot trajectories.
            vx, vy, wz = np.clip(
                # Feedforward request plus proportional (gain 0.5) and integral corrections.
                self.command[:3] + 0.5 * error + self.velocity_integral,
                # Internal lower limits in m/s, m/s, rad/s; these differ from public bounds.
                [-1.5, -0.8, -1.2],
                # Internal upper limits allow extra stride motion to correct tracking error.
                [1.5, 0.8, 1.2],
            )
        # When standing, clear accumulated velocity bias gradually.
        elif not moving:
            # Decay the integral by up to 4*dt of its current value without reversing sign.
            self.velocity_integral *= max(0.0, 1.0 - dt * 4.0)
        # Collect three joint angles per leg in LEGS order.
        q = []
        # Desired body tilt relative to level ground; yaw is handled by foot motion.
        body_r = rpy_matrix(roll, pitch, 0)
        # Build and solve one foot target for each of the four legs.
        for i in range(4):
            # Neutral foot center in level axes aligned with the body heading.
            p = HIP[i] + [
                # Shift the support polygon fore/aft toward the initial COM projection.
                self.support_center[0],
                # Add the lateral COM offset and left/right hip-link offset.
                self.support_center[1] + (0.0955 if i in (0, 2) else -0.0955),
                # Foot center is 0.022 m above ground, relative to the requested body height.
                0.022 - height,
            ]
            # Add stance/swing motion only for nonzero smoothed velocity commands.
            if moving:
                # Raibert-style placement: brake overspeed and lean into acceleration.
                # A 0.05 s gain shifts feet toward measured-minus-requested planar velocity.
                p[:2] += 0.05 * (self.filtered_velocity[:2] - self.command[:2])
                # FL/RR share a phase; FR/RL are offset by half a cycle.
                phase = (self.phase + (0 if i in (0, 3) else 0.5)) % 1
                # Velocity at this foot = body translation + yaw angular velocity cross position.
                velocity = np.array([vx - wz * p[1], vy + wz * p[0]])
                # Each leg spends 60% of a cycle in stance and 40% in swing.
                stance = 0.60
                # Stance portion: keep the foot at its nominal ground-contact height.
                if phase < stance:
                    # Move the stance foot backward relative to the body at -velocity,
                    # from +half-stride to -half-stride.
                    p[:2] += velocity * self.period * (stance / 2 - phase)
                else:
                    # Normalize swing progress to [0, 1).
                    u = (phase - stance) / (1 - stance)
                    # Cubic smoothstep returns the foot from -half-stride to +half-stride.
                    p[:2] += velocity * self.period * stance * (u * u * (3 - 2 * u) - 0.5)
                    # Lift from zero to 6.5 cm at mid-swing and back to zero, with smooth endpoints.
                    p[2] += 0.065 * math.sin(math.pi * u) ** 2
            # Undo desired body tilt, then translate from the body origin to this hip.
            p = body_r.T @ p - HIP[i]
            # Keep the private leg solve away from full extension even when
            # commanded posture, feedback correction and stride combine.
            # Compute the hip-to-foot distance in metres.
            length = np.linalg.norm(p)
            # Shorten targets beyond 0.395 m while preserving their direction.
            p *= min(1.0, 0.395 / length)
            # Solve the left/right leg geometry and append hip, thigh, calf angles.
            q.extend(leg_ik(p, 1 if i in (0, 2) else -1))
        # Return 12 absolute joint angles; the native loop applies joint bounds and drives.
        return np.array(q)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def transform(pose):
    """Public translation/xyzw pose to X_AB."""
    p = np.asarray(pose["translation_m"], dtype=float)
    q = np.asarray(pose["rotation_xyzw"], dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
        raise ValueError("Invalid pose")
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("Quaternion must be unit length")
    t = np.eye(4)
    t[:3, :3] = quat_matrix(q)
    t[:3, 3] = p
    return t


def rotation_error(desired, measured):
    """Log(R_des R_meas.T)^vee, stable at pi via the symmetric eigensystem."""
    r = desired @ measured.T
    angle = math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-6:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        _, vectors = np.linalg.eigh((r + r.T) / 2)
        axis = vectors[:, -1]
        if axis @ skew < 0:
            axis = -axis
        return angle * axis
    return angle / (2 * math.sin(angle)) * skew


def limit(vector, maximum):
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))


class SerialArm:
    """FK/IK/Jacobian in the arm root (default) or explicit robot ``base`` frame.

    ``arm_base`` excludes the profile mount transform. TCP is caller calibration.
    """

    def __init__(self, profile, tcp=(0, 0, 0.105), *, frame="arm_base"):
        if frame not in {"base", "arm_base"}:
            raise ValueError("Expected frame 'base' or 'arm_base'")
        self.frame = frame
        self.model = profile["intrinsics"]["model"]
        self.base_mount = transform(profile["extrinsics"]["mount"])
        self.mount = self.base_mount.copy() if frame == "base" else np.eye(4)
        self.offset = np.eye(4)
        self.offset[:3, 3] = tcp
        self.bounds = np.array(
            [[j["limits"]["min"], j["limits"]["max"]] for j in self.model["joints"]]
        )
        self.origins = [
            transform({"translation_m": j["origin"][:3], "rotation_xyzw": j["origin"][3:]})
            for j in self.model["chain"]
        ]

    def fk(self, q, tcp=True):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.bounds),) or not np.isfinite(q).all():
            raise ValueError("Invalid joint vector")
        t = self.mount.copy()
        for origin, joint in zip(self.origins, self.model["chain"], strict=True):
            t = t @ origin
            t[:3, :3] = t[:3, :3] @ rotation(joint["axis"], q[joint["joint"]] - joint["reference"])
        return t @ self.offset if tcp else t

    def jacobian(self, q):
        q = joint_vector(q, len(self.bounds))
        t = self.mount.copy()
        axes, centers = [], []
        for origin, joint in zip(self.origins, self.model["chain"], strict=True):
            t = t @ origin
            axes.append(t[:3, :3] @ joint["axis"])
            centers.append(t[:3, 3].copy())
            t[:3, :3] = t[:3, :3] @ rotation(joint["axis"], q[joint["joint"]] - joint["reference"])
        p = (t @ self.offset)[:3, 3]
        return np.array(
            [np.r_[a, np.cross(a, p - c)] for a, c in zip(axes, centers, strict=True)]
        ).T

    def ik(self, target, seed):
        return solve_pose_ik(
            target, seed, self.bounds, self.fk, jacobian=self.jacobian, iterations=250,
        )



def joint_vector(value, count):
    value = np.asarray(value, dtype=float)
    if value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError("Invalid measured joints")
    return value


def rigid_transform(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1.0)
    ):
        raise ValueError("Invalid rigid transform")
    return value


class ArmIKController:
    """Bounded differential IK; inputs and Jacobian are strictly arm-root local.

    ``step`` returns absolute joint positions. It does not write devices or advance
    time. One instance owns one command stream; call ``hold`` on target loss/reset.
    """

    def __init__(self, arm):
        if arm.frame != "arm_base":
            raise ValueError("ArmIKController requires arm-base kinematics")
        self.arm = arm
        self.qref = None
        self.qspeed = np.zeros(len(arm.bounds))
        self.saturated = False

    def hold(self, q):
        q = joint_vector(q, len(self.arm.bounds))
        self.qref = q.copy()
        self.qspeed = np.zeros_like(q)
        self.saturated = False
        return q.copy()

    def step(self, q, twist, dt):
        q = joint_vector(q, len(self.arm.bounds))
        twist = np.asarray(twist, dtype=float)
        if twist.shape != (6,) or not np.isfinite(twist).all():
            raise ValueError("Invalid arm-frame TCP twist")
        if not np.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Invalid servo interval")
        qref = q.copy() if self.qref is None else self.qref
        bounded = np.r_[limit(twist[:3], 0.5), limit(twist[3:], 0.15)]
        weight = np.array([0.15, 0.15, 0.15, 1.0, 1.0, 1.0])
        j = weight[:, None] * self.arm.jacobian(q)
        sigma = np.linalg.svd(j, compute_uv=False)[-1]
        damping = 0.01 + max(0.0, 0.015 - sigma)
        dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), weight * bounded)
        limited = np.clip(np.clip(dq, -1.0, 1.0), self.qspeed - 3 * dt, self.qspeed + 3 * dt)
        lower = np.maximum.reduce([self.arm.bounds[:, 0], q - 0.10, qref - dt])
        upper = np.minimum.reduce([self.arm.bounds[:, 1], q + 0.10, qref + dt])
        if np.any(lower > upper):
            raise ValueError("Reference/measurement limits conflict")
        new = np.clip(qref + limited * dt, lower, upper)
        saturated = bool(
            np.max(abs(bounded - twist)) > 1e-6
            or np.max(abs(limited - dq)) > 1e-6
            or np.max(abs(new - qref - limited * dt)) > 1e-6
        )
        self.qspeed = (new - qref) / dt
        self.qref = new
        self.saturated = saturated
        return new.copy(), {"saturated": saturated, "sigma_min": float(sigma)}
