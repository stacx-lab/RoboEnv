"""Leg-only feedback trot with analytic IK and bounded body commands.

No arm targets or external forces are generated. Ground is assumed horizontal.
"""

import math
import numpy as np

from .kinematics import rpy_matrix

LEGS = ("FL", "FR", "RL", "RR")
LEG_NAMES = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]
BOUNDS = {
    "vx": (-1.0, 1.0),
    "vy": (-0.5, 0.5),
    "wz": (-math.pi / 6, math.pi / 6),
    "height": (0.26, 0.34),
    "roll": (-0.12, 0.12),
    "pitch": (-0.12, 0.12),
}
DEFAULT = np.array([0.0, 0.0, 0.0, 0.30, 0.0, 0.0])
HIP = np.array(
    [[0.1934, 0.0465, 0], [0.1934, -0.0465, 0], [-0.1934, 0.0465, 0], [-0.1934, -0.0465, 0]]
)


def validate_command(args):
    if not isinstance(args, dict) or set(args) != {"mode", "velocity", "posture"}:
        raise ValueError("Base requires mode, velocity and posture")
    if args["mode"] != "locomotion":
        raise ValueError("Unsupported base mode")
    if not isinstance(args["velocity"], dict) or not isinstance(args["posture"], dict):
        raise ValueError("Velocity and posture must be objects")
    if set(args["velocity"]) != {"vx", "vy", "wz"} or set(args["posture"]) != {
        "height",
        "roll",
        "pitch",
    }:
        raise ValueError("Base requires vx/vy/wz and height/roll/pitch")
    values = {**args["velocity"], **args["posture"]}
    for key, (lo, hi) in BOUNDS.items():
        x = values[key]
        if type(x) not in (float, int) or not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f"{key} must be finite and within [{lo}, {hi}]")
    return np.array([values[k] for k in BOUNDS])


def leg_ik(foot, side):
    x, y, z = np.asarray(foot, dtype=float)
    h, L = 0.0955, 0.213
    d = y * y + z * z - h * h
    if d <= 0:
        raise ValueError("Foot target inside hip offset")
    z0 = -math.sqrt(d)
    hip = math.atan2(z, y) - math.atan2(z0, side * h)
    hip = (hip + math.pi) % (2 * math.pi) - math.pi
    c = (x * x + z0 * z0 - 2 * L * L) / (2 * L * L)
    if not -1 <= c <= 1:
        raise ValueError("Foot target outside leg reach")
    calf = -math.acos(c)
    thigh = math.atan2(-x, -z0) - math.atan2(L * math.sin(calf), L + L * math.cos(calf))
    return [hip, thigh, calf]


class Trot:
    """Diagonal gait with body-posture feedback; only outputs leg joints."""

    period = 0.32

    def __init__(self, support_center=(0.0, 0.0)):
        self.support_center = np.asarray(support_center, dtype=float).copy()
        self.command = DEFAULT.copy()
        self.phase = 0.0
        self.posture_integral = np.zeros(3)
        self.velocity_integral = np.zeros(3)
        self.filtered_velocity = np.zeros(3)
        self.filtered_posture_rates = np.zeros(3)

    def state(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    def targets(
        self, command, dt, measured_posture=None, measured_velocity=None, posture_rates=None
    ):
        rates = np.array([0.5, 0.4, 1.0, 0.04, 0.18, 0.18])
        self.command += np.clip(command - self.command, -rates * dt, rates * dt)
        vx, vy, wz, height, roll, pitch = self.command
        if posture_rates is not None:
            self.filtered_posture_rates += min(dt / 0.05, 1.0) * (
                np.asarray(posture_rates) - self.filtered_posture_rates
            )
        if measured_posture is not None:
            error = self.command[3:] - np.asarray(measured_posture)
            self.posture_integral = np.clip(
                self.posture_integral + error * dt * 0.6, [-0.03, -0.06, -0.06], [0.03, 0.06, 0.06]
            )
            height, roll, pitch = self.command[3:] + np.clip(
                0.3 * error
                + self.posture_integral
                - np.array([0.03, 0.08, 0.08]) * self.filtered_posture_rates,
                [-0.035, -0.08, -0.08],
                [0.035, 0.08, 0.08],
            )
        moving = np.linalg.norm(self.command[:3]) > 0.002
        if moving:
            self.phase = (self.phase + dt / self.period) % 1
        if moving and measured_velocity is not None:
            self.filtered_velocity += min(dt / 0.15, 1.0) * (
                np.asarray(measured_velocity) - self.filtered_velocity
            )
            error = self.command[:3] - self.filtered_velocity
            self.velocity_integral = np.clip(
                self.velocity_integral + error * dt * 1.2, [-0.22, -0.15, -0.4], [0.22, 0.15, 0.4]
            )
            vx, vy, wz = np.clip(
                self.command[:3] + 0.5 * error + self.velocity_integral,
                [-1.5, -0.8, -1.2],
                [1.5, 0.8, 1.2],
            )
        elif not moving:
            self.velocity_integral *= max(0.0, 1.0 - dt * 4.0)
        q = []
        body_r = rpy_matrix(roll, pitch, 0)
        for i in range(4):
            p = HIP[i] + [
                self.support_center[0],
                self.support_center[1] + (0.0955 if i in (0, 2) else -0.0955),
                0.022 - height,
            ]
            if moving:
                # Raibert-style placement: brake overspeed and lean into acceleration.
                p[:2] += 0.05 * (self.filtered_velocity[:2] - self.command[:2])
                phase = (self.phase + (0 if i in (0, 3) else 0.5)) % 1
                velocity = np.array([vx - wz * p[1], vy + wz * p[0]])
                stance = 0.60
                if phase < stance:
                    p[:2] += velocity * self.period * (stance / 2 - phase)
                else:
                    u = (phase - stance) / (1 - stance)
                    p[:2] += velocity * self.period * stance * (u * u * (3 - 2 * u) - 0.5)
                    p[2] += 0.065 * math.sin(math.pi * u) ** 2
            p = body_r.T @ p - HIP[i]
            # Keep the private leg solve away from full extension even when
            # commanded posture, feedback correction and stride combine.
            length = np.linalg.norm(p)
            p *= min(1.0, 0.395 / length)
            q.extend(leg_ik(p, 1 if i in (0, 2) else -1))
        return np.array(q)
