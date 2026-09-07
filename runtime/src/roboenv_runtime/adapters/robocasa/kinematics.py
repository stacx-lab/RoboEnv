"""MuJoCo kinematics on scratch data: never changes or steps the live world."""

import numpy as np


def vector(value, size, *, lower=None, upper=None):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite numbers")
    if lower is not None and (result < lower).any():
        raise ValueError("Target below device limits")
    if upper is not None and (result > upper).any():
        raise ValueError("Target above device limits")
    return result


def pose(model, data, body, base=0):
    import mujoco

    rotation = data.xmat[base].reshape(3, 3).T
    position = rotation @ (data.xpos[body] - data.xpos[base])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, (rotation @ data.xmat[body].reshape(3, 3)).ravel())
    return [*position.tolist(), *quaternion[1:].tolist(), float(quaternion[0])]


class Kinematics:
    def __init__(self, model, data, joints, body, base):
        import mujoco

        self.model, self.live = model, data
        self.body, self.base = body, base
        self.joints = np.asarray(joints, dtype=int)
        if not len(joints) or not np.isin(model.jnt_type[self.joints], [2, 3]).all():
            raise ValueError("Kinematics requires scalar hinge/slide joints")
        self.qpos = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]
        limits = model.jnt_range[self.joints]
        limited = model.jnt_limited[self.joints].astype(bool)
        self.lower = np.where(limited, limits[:, 0], -np.inf)
        self.upper = np.where(limited, limits[:, 1], np.inf)
        self.scratch = mujoco.MjData(model)

    def _prepare(self, joints, *, validate_limits=True):
        import mujoco

        q = vector(
            joints, len(self.joints),
            lower=self.lower if validate_limits else None,
            upper=self.upper if validate_limits else None,
        )
        self.scratch.qpos[:] = self.live.qpos
        self.scratch.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        return q

    def fk(self, joints):
        self._prepare(joints)
        return {"pose": pose(self.model, self.scratch, self.body, self.base)}

    def measured_pose(self):
        # MuJoCo joint limits are soft constraints. Report the actual state,
        # including overshoot, without treating it as an invalid command.
        self._prepare(self.live.qpos[self.qpos], validate_limits=False)
        return pose(self.model, self.scratch, self.body, self.base)

    def ik(self, pose, seed=None):
        import mujoco

        target = vector(pose, 7)
        if abs(np.linalg.norm(target[3:]) - 1) > 1e-4:
            raise ValueError("Pose quaternion must have unit length (xyzw)")
        # An implicit seed may come from a softly violated physical limit.
        # Project only the scratch solver seed; explicit user seeds stay strict.
        initial = (np.clip(self.live.qpos[self.qpos], self.lower, self.upper)
                   if seed is None else seed)
        q = self._prepare(initial)
        rotation = self.scratch.xmat[self.base].reshape(3, 3).copy()
        position = self.scratch.xpos[self.base] + rotation @ target[:3]
        local_rotation = np.empty(9)
        mujoco.mju_quat2Mat(local_rotation, target[[6, 3, 4, 5]])
        target_rotation = rotation @ local_rotation.reshape(3, 3)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(200):
            self._prepare(q)
            current = self.scratch.xmat[self.body].reshape(3, 3)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (target_rotation @ current.T).ravel())
            if quat[0] < 0:
                quat *= -1
            angular = np.empty(3)
            mujoco.mju_quat2Vel(angular, quat, 1.0)
            linear = position - self.scratch.xpos[self.body]
            if np.linalg.norm(linear) < 1e-4 and np.linalg.norm(angular) < 1e-3:
                return {"joints": q.tolist()}
            mujoco.mj_jacBody(self.model, self.scratch, jp, jr, self.body)
            jacobian = np.vstack((jp[:, self.dofs], jr[:, self.dofs]))
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(6), np.r_[linear, angular]
            )
            q = np.clip(q + np.clip(delta, -0.15, 0.15), self.lower, self.upper)
        raise ValueError("No IK solution found within limits from the supplied seed")
