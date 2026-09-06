"""BEHAVIOR-1K backend: one OmniGibson process, booted lazily, driven symbolically.

Everything heavy lives here so :mod:`roboenv_runtime.adapters.behavior.adapter` can be
imported (and the adapter constructed) in any interpreter; ``import
omnigibson`` happens on the first :meth:`BehaviorBackend.boot`, inside the
``behavior`` conda env (see docs — Isaac Sim boots in ~95 s warm).

Hard-won specifics this module encodes (all measured 2026-09-02):

* Inherited cache settings can point at unwritable locations. Cache overrides
  are checked before OmniGibson is imported and isolated to this run.
* Action-primitive modules must be imported AFTER ``og.Environment`` exists —
  importing them first initialises CUDA ahead of Isaac and the process
  segfaults.
* ``task.object_scope`` maps BDDL names straight to sim objects (no
  ``BDDLEntity`` wrapper in OmniGibson 3.9.2).
* The symbolic ``PLACE_INSIDE`` sampler cannot find an "inside" pose for
  common containers (the challenge trash can among them) while ``GRASP`` /
  ``PLACE_ON_TOP`` / ``OPEN`` work; :meth:`drop_into` is the placement that
  does work — teleport above the container mouth and let physics settle — and
  it flips the BDDL ``Inside`` predicate honestly.
* Goal state can be read WITHOUT stepping: the task's ``PredicateGoal``
  termination condition exposes its checker (:meth:`goal_report`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["BehaviorBackend", "fix_cache_env", "load_challenge_index"]

_CACHE_VARS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME")

# symbolic primitive name -> (enum attribute, takes an object argument)
PRIMITIVES: dict[str, tuple[str, bool]] = {
    "grasp": ("GRASP", True),
    "place_on_top": ("PLACE_ON_TOP", True),
    "place_inside": ("PLACE_INSIDE", True),
    "open": ("OPEN", True),
    "close": ("CLOSE", True),
    "toggle_on": ("TOGGLE_ON", True),
    "toggle_off": ("TOGGLE_OFF", True),
    "soak_under": ("SOAK_UNDER", True),
    "soak_inside": ("SOAK_INSIDE", True),
    "wipe": ("WIPE", True),
    "cut": ("CUT", True),
    "release": ("RELEASE", False),
}

# binary/unary object states an agent may probe (lowercase -> omnigibson class name)
STATES: dict[str, tuple[str, bool]] = {
    "inside": ("Inside", True),
    "ontop": ("OnTop", True),
    "under": ("Under", True),
    "nextto": ("NextTo", True),
    "touching": ("Touching", True),
    "covered": ("Covered", True),
    "open": ("Open", False),
    "toggled_on": ("ToggledOn", False),
    "filled": ("Filled", True),
    "folded": ("Folded", False),
    "cooked": ("Cooked", False),
    "frozen": ("Frozen", False),
    "heated": ("Heated", False),
}


def fix_cache_env() -> None:
    """Re-point unwritable cache env vars at ``<repo>/.cache`` (see module doc)."""
    from roboenv_runtime.paths import repo_root

    base = repo_root() / ".cache"
    for var in _CACHE_VARS:
        current = os.environ.get(var)
        target = Path(current).expanduser() if current else None
        if target is not None:
            try:
                target.mkdir(parents=True, exist_ok=True)
                probe = target / ".roboenv-probe"
                probe.touch()
                probe.unlink()
                continue  # writable — leave it alone
            except OSError:
                pass
        fallback = base / var.lower()
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(fallback)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")


def load_challenge_index() -> dict[str, dict[str, Any]]:
    """The 2026 challenge task index (id -> instruction/scene/rooms), or ``{}``.

    Source: ``third_party/BEHAVIOR-1K/docs/challenge/task_data.json``.
    """
    import json

    from roboenv_runtime.paths import repo_root

    path = repo_root() / "third_party/BEHAVIOR-1K/docs/challenge/task_data.json"
    if not path.is_file():
        return {}
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    except (OSError, ValueError, KeyError):
        return {}
    return {t["id"]: t for t in tasks if isinstance(t, dict) and "id" in t}


class BehaviorBackend:
    """Owns the OmniGibson environment and the symbolic-primitive controller."""

    def __init__(
        self,
        activity_name: str,
        scene_model: str | None = None,
        activity_instance_id: int = 0,
        max_steps: int = 6000,
    ) -> None:
        self.activity_name = activity_name
        self.activity_instance_id = int(activity_instance_id)
        self.max_steps = int(max_steps)
        self.challenge = load_challenge_index().get(activity_name, {})
        self.scene_model = scene_model or self.challenge.get("scene_model")
        if not self.scene_model:
            raise ValueError(
                f"activity {activity_name!r} is not in the challenge index and no "
                "scene_model was given (adapter_kwargs: {scene_model: ...})"
            )
        self.env: Any = None
        self.robot: Any = None
        self._controller: Any = None
        self.steps_used = 0
        self._frames: list[Any] = []
        self._cam_key: tuple[str, str] | None = None
        # cinematic record camera (ROBOENV_BEHAVIOR_RECORD=<out.mp4>): a follow-cam
        # external sensor streamed straight to ffmpeg — the robot's head camera
        # films the ceiling during symbolic motion, so records need this.
        self._rec_path = os.environ.get("ROBOENV_BEHAVIOR_RECORD") or None
        size = os.environ.get("ROBOENV_BEHAVIOR_RECORD_SIZE", "1280x720")
        self._rec_w, self._rec_h = (int(v) for v in size.lower().split("x"))
        self._rec_every = max(1, int(os.environ.get("ROBOENV_BEHAVIOR_RECORD_EVERY", "2")))
        self._rec_sensor: Any = None
        self._rec_proc: Any = None
        self._last_rec_frame: Any = None
        self._cam_pos: Any = None
        self._cam_tgt: Any = None
        self._cam_u: Any = None  # last robot->focus direction, kept while they coincide
        self._cam_choice: int | None = None  # candidate spot in use (continuity)
        self._focus_obj: Any = None  # what the camera frames besides the robot

    # ------------------------------------------------------------ config

    def build_config(self) -> dict[str, Any]:
        """The roboenv-owned R1Pro symbolic config, specialised to this activity."""
        import yaml

        cfg = yaml.safe_load((Path(__file__).parent / "r1pro_symbolic.yaml").read_text())
        cfg["task"]["activity_name"] = self.activity_name
        cfg["task"]["activity_instance_id"] = self.activity_instance_id
        cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        cfg["scene"]["scene_model"] = self.scene_model
        # loading only the task's rooms keeps boot time sane; whole scene when unknown
        cfg["scene"]["load_room_types"] = self.challenge.get("rooms") or None
        if self._rec_path:
            cfg["env"]["external_sensors"] = [
                {
                    "sensor_type": "VisionSensor",
                    "name": "record_cam",
                    "relative_prim_path": "/record_cam",
                    "modalities": ["rgb"],
                    "sensor_kwargs": {"image_height": self._rec_h, "image_width": self._rec_w},
                    "position": [0.0, 0.0, 2.0],
                    "orientation": [0.707, 0.0, 0.0, 0.707],
                    "pose_frame": "parent",
                }
            ]
        return cfg

    # ------------------------------------------------------------ lifecycle

    def boot(self) -> None:
        """Import omnigibson and create the environment (idempotent)."""
        if self.env is not None:
            return
        fix_cache_env()
        import omnigibson as og

        self.env = og.Environment(configs=self.build_config())
        self.robot = self.env.robots[0]
        if self._rec_path:
            self._rec_sensor = (getattr(self.env, "_external_sensors", None) or {}).get(
                "record_cam"
            )
            if self._rec_sensor is None:
                log.warning("record camera requested but not found; recording disabled")
        log.info("BEHAVIOR env up: %s on %s", self.activity_name, self.scene_model)

    def reset(self, seed: int = 0) -> None:
        self.boot()
        self.env.reset()
        # the primitives' pose samplers draw from the global RNGs; unseeded, two
        # processes running the same program diverge (2026-09-02: a 4574-step win
        # replayed to 0.0). Seed after env.reset() so its own sampling is untouched.
        import random

        import torch as th

        random.seed(seed)
        np.random.seed(int(seed) % 2**32)
        th.manual_seed(int(seed))
        self.steps_used = 0
        self._frames = []
        self._controller = None  # its cached state belongs to the previous episode

    def close(self) -> None:
        self._finish_recording()
        if self.env is None:
            return
        import atexit

        import omnigibson as og

        # og.shutdown() ends the PROCESS (exit 0), not just the simulator.  Callers that still
        # have work after close() — roboenv_runtime.evolve.distill's replay() writes its verdict after the
        # adapter is closed — lost that work silently (2026-09-03: every BEHAVIOR fresh-process
        # certification came back "exited without a verdict").  Defer the shutdown to interpreter
        # exit; the CLI's force-exit path never reaches it, which is fine (os._exit).
        atexit.register(og.shutdown)
        self.env = None
        self.robot = None
        self._controller = None

    # ------------------------------------------------------------ acting

    def controller(self) -> Any:
        """The symbolic-primitive controller (created after boot — see module doc)."""
        if self._controller is None:
            from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
                SymbolicSemanticActionPrimitives,
            )

            self._controller = SymbolicSemanticActionPrimitives(self.env, self.robot)
        return self._controller

    def _step(self, action: Any) -> dict[str, Any]:
        # aim sparsely: per-step USD pose writes are needless scene-graph churn
        if self._rec_sensor is not None and self.steps_used % 3 == 0:
            self._aim_record_cam()
        obs, _r, _term, _trunc, info = self.env.step(action)
        self.steps_used += 1
        if self._rec_sensor is not None:
            self._record(obs)
        if self.steps_used % 15 == 0:
            self.capture_frame()
        return info

    # ------------------------------------------------------------ record camera

    def _focus_point(self, pos: np.ndarray) -> np.ndarray | None:
        """What the robot is working on: the held object, else the nearest task
        object within 4 m — its aabb point nearest the robot, since a car's centre
        is nowhere near its trunk — else None."""
        self._focus_obj = None
        try:
            held = self.obj_in_hand()
            if held is not None:
                self._focus_obj = held
                return held.get_position_orientation()[0].cpu().numpy()
        except Exception:
            pass
        best, best_d = None, 4.0
        scope = getattr(getattr(self.env, "task", None), "object_scope", None) or {}
        for name, obj in scope.items():
            if obj is None or name.startswith("agent") or not hasattr(obj, "aabb"):
                continue
            try:
                lo, hi = (t.cpu().numpy() for t in obj.aabb)
            except Exception:
                continue
            p = np.clip(pos, lo, hi)
            d = float(np.linalg.norm((p - pos)[:2]))
            if d < best_d:
                best, best_d, self._focus_obj = p, d, obj
        return best

    def _cam_clear(self, tgt: np.ndarray, cam: np.ndarray) -> bool:
        """No wall or furniture between the look-at point and a camera spot (nor
        0.3 m behind it, so the near plane is not inside a surface). The robot
        and the focus object are allowed in the way — they are the subject."""
        import torch as th

        from omnigibson.utils.sampling_utils import raytest

        ignore = list(getattr(self.robot, "link_prim_paths", []) or [])
        ignore += list(getattr(self._focus_obj, "link_prim_paths", []) or [])
        v = cam - tgt
        end = cam + v / (np.linalg.norm(v) or 1.0) * 0.3
        try:
            hit = raytest(
                th.tensor(tgt, dtype=th.float32), th.tensor(end, dtype=th.float32),
                only_closest=True, ignore_bodies=ignore or None,
            )
        except Exception:
            return True
        return not bool(hit.get("hit", False))

    def _aim_record_cam(self) -> None:
        """Follow-cam framing the robot AND what it works on (the first take of
        loading_the_car never showed the car): behind the robot relative to the
        focus point, offset to one side so the body does not occlude the object,
        pulled back and raised with the robot-object spread."""
        import torch as th

        import omnigibson.utils.transform_utils as T

        pos, orn = (t.cpu().numpy() for t in self.robot.get_position_orientation())
        focus = self._focus_point(pos)
        if focus is None:  # nothing near: look the way the robot faces
            try:
                fwd = T.quat2mat(th.tensor(orn, dtype=th.float32)).numpy()[:, 0]
            except Exception:
                fwd = np.array([1.0, 0.0, 0.0])
            focus = pos + np.array([fwd[0], fwd[1], 0.0])
        u = (focus - pos) * np.array([1.0, 1.0, 0.0])
        d = float(np.linalg.norm(u))
        if d > 0.3:
            self._cam_u = u / d
        u = self._cam_u if self._cam_u is not None else np.array([1.0, 0.0, 0.0])
        side = np.array([-u[1], u[0], 0.0])
        mid = (pos + focus) / 2.0
        tgt = np.array([mid[0], mid[1], 0.75])
        # candidate spots: both sides at three pull-back scales (rooms are small —
        # the first desk take filmed the inside of a wall), then a steep overhead
        # fallback; only spots with a clear line of sight to the subject are used
        back, lateral, height = 1.6 + 0.5 * d, 1.4 + 0.3 * d, 1.6 + 0.35 * d
        cands = [
            tgt - u * back * k + s * side * lateral * k + np.array([0.0, 0.0, (height - 0.75) * k + 0.3])
            for k in (1.0, 0.7, 0.5)
            for s in (1.0, -1.0)
        ]
        cands.append(tgt - u * 0.5 + np.array([0.0, 0.0, 1.6]))
        clear = [i for i, c in enumerate(cands) if self._cam_clear(tgt, c)] or [len(cands) - 1]
        if self._cam_choice in clear:
            i = self._cam_choice  # keep the shot while it still works
        elif self._cam_pos is not None:
            i = min(clear, key=lambda k: float(np.linalg.norm(cands[k] - self._cam_pos)))
        else:
            i = clear[0]
        self._cam_choice = i
        cam = cands[i]
        a = 0.2  # smoothing: teleport-style motion must not snap the camera
        self._cam_tgt = tgt if self._cam_tgt is None else (1 - a) * self._cam_tgt + a * tgt
        self._cam_pos = cam if self._cam_pos is None else (1 - a) * self._cam_pos + a * cam
        z = self._cam_pos - self._cam_tgt
        z = z / (np.linalg.norm(z) or 1.0)
        x = np.cross([0.0, 0.0, 1.0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        rot = th.tensor(np.stack([x, y, z], axis=1), dtype=th.float32)
        self._rec_sensor.set_position_orientation(
            position=th.tensor(self._cam_pos, dtype=th.float32), orientation=T.mat2quat(rot)
        )

    def _record(self, obs: dict[str, Any]) -> None:
        try:
            rgb = obs["external"]["record_cam"]["rgb"]
        except (KeyError, TypeError):
            return
        arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        self._last_rec_frame = arr[..., :3].astype(np.uint8)
        if self.steps_used % self._rec_every:
            return
        if self._rec_proc is None:
            import subprocess

            Path(self._rec_path).parent.mkdir(parents=True, exist_ok=True)
            h, w = self._last_rec_frame.shape[:2]
            self._rec_proc = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", "30", "-i", "-",
                 "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", self._rec_path],
                stdin=subprocess.PIPE,
            )
        try:
            self._rec_proc.stdin.write(self._last_rec_frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            log.warning("record stream died: %s", exc)
            self._rec_proc = None
            self._rec_sensor = None

    def _finish_recording(self) -> None:
        if self._rec_proc is not None:
            try:
                self._rec_proc.stdin.close()
                self._rec_proc.wait(timeout=120)
                log.info("record camera video written: %s", self._rec_path)
            except Exception as exc:
                log.warning("record finalise failed: %s", exc)
            self._rec_proc = None

    def settle(self, steps: int = 30) -> None:
        zero = np.zeros(self.robot.action_dim, dtype=np.float32)
        for _ in range(int(steps)):
            self._step(zero)

    # PhysX occasionally diverges under a teleport-style placement (can_meat,
    # 2026-09-02: place_on_top of a jar just lifted out of a cabinet left the
    # robot base and the jar with NaN orientations, and every state query
    # after that asserted).  Every acting call therefore runs against a scene
    # snapshot and rolls the world back when the simulator throws — the tool
    # contract is "never raises, reports honestly", and a rolled-back world is
    # reported as such so the agent changes approach instead of retrying.
    _NAN_MARKS = ("nan", "unit quaternion", "must not be scaled")

    def _snapshot(self) -> Any:
        import omnigibson as og

        return og.sim.dump_state(serialized=False)

    def _restore(self, state: Any, what: str) -> bool:
        """Roll the world back to ``state``; True when the robot reads sane again."""
        import omnigibson as og

        try:
            og.sim.load_state(state, serialized=False)
            self.settle(3)
            self.robot.get_position_orientation()  # asserts a unit quaternion
            return True
        except Exception as exc:  # the run is in real trouble; say so loudly
            log.error("rollback after %s failed: %s", what, exc)
            return False

    def _guarded(self, what: str, run: Any) -> dict[str, Any]:
        """Run an acting closure under the never-raises contract (see above)."""
        from omnigibson.action_primitives.action_primitive_set_base import (
            ActionPrimitiveError,
            ActionPrimitiveErrorGroup,
        )

        snapshot = self._snapshot()
        start = self.steps_used
        try:
            run()
        except (ActionPrimitiveError, ActionPrimitiveErrorGroup) as exc:
            reasons = [str(e) for e in getattr(exc, "exceptions", None) or [exc]]
            self.capture_frame()
            return {"ok": False, "steps": self.steps_used - start, "errors": reasons[:3]}
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"[:300]
            diverged = any(m in text.lower() for m in self._NAN_MARKS)
            restored = self._restore(snapshot, what)
            held = self.obj_in_hand()
            why = (
                f"PHYSICS_NAN: the simulator diverged during {what}"
                if diverged
                else f"simulator error during {what} — {text}"
            )
            state = (
                "world rolled back to just before the call"
                + (f", still holding {held.name}" if held is not None else "")
                if restored
                else "rollback FAILED, the world may be unusable"
            )
            return {
                "ok": False,
                "steps": self.steps_used - start,
                "restored": restored,
                "errors": [f"{why}; {state}. Repeating the same call will diverge again — "
                           "change the approach (release() and drop_into(), another target)."],
            }
        self.capture_frame()
        return {"ok": True, "steps": self.steps_used - start}

    def run_primitive(self, name: str, obj_name: str | None = None) -> dict[str, Any]:
        """Execute one symbolic primitive; never raises, returns an honest report."""
        from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
            SymbolicSemanticActionPrimitiveSet,
        )

        enum_name, takes_obj = PRIMITIVES[name]
        primitive = getattr(SymbolicSemanticActionPrimitiveSet, enum_name)
        ctrl = self.controller()
        args = (self.resolve(obj_name),) if takes_obj else ()

        def run() -> None:
            for action in ctrl.apply_ref(primitive, *args):
                self._step(action)

        return self._guarded(f"{name}({obj_name or ''})", run)

    def settle_report(self, steps: int) -> dict[str, Any]:
        """``settle`` under the never-raises contract (the tool surface)."""
        return self._guarded(f"settle({steps})", lambda: self.settle(steps))

    def drop_into(self, obj_name: str, container_name: str) -> dict[str, Any]:
        """Teleport ``obj`` above ``container``'s mouth and settle 40 steps.

        The placement that works where the symbolic PLACE_INSIDE sampler fails.
        Refuses while the object is grasped — release first, honesty over magic.
        """
        import torch as th

        obj = self.resolve(obj_name)
        container = self.resolve(container_name)
        held = self.obj_in_hand()
        if held is not None and held.name == obj.name:
            return {"ok": False, "errors": [f"{obj_name} is grasped; call release() first"]}
        lo, hi = (t.cpu().numpy() for t in container.aabb)
        target = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(hi[2] + 0.15)]

        def run() -> None:
            obj.set_position_orientation(
                position=th.tensor(target, dtype=th.float32),
                orientation=th.tensor([0.0, 0.0, 0.0, 1.0]),
            )
            self.settle(40)

        report = self._guarded(f"drop_into({obj_name}, {container_name})", run)
        if not report["ok"]:
            return report
        inside = self.check_state("inside", obj.name, container.name)
        report.update(ok=bool(inside), inside=bool(inside))
        return report

    def obj_in_hand(self) -> Any | None:
        try:
            return self.controller()._get_obj_in_hand()
        except Exception:
            return None

    # ------------------------------------------------------------ observing

    def resolve(self, name: str) -> Any:
        """A sim object from a sim name, a BDDL scope name, or a unique substring."""
        if not name:
            raise ValueError("empty object name")
        scope = self.env.task.object_scope
        if name in scope and scope[name] is not None:
            return scope[name]
        obj = self.env.scene.object_registry("name", name)
        if obj is not None:
            return obj
        matches = {o.name: o for o in self.env.scene.objects if name.lower() in o.name.lower()}
        matches.update(
            {k: v for k, v in scope.items() if v is not None and name.lower() in k.lower()}
        )
        if len(matches) == 1:
            return next(iter(matches.values()))
        hint = sorted(matches)[:8]
        raise KeyError(
            f"object {name!r}: {'ambiguous, matches ' + str(hint) if matches else 'not found'}"
        )

    def task_objects(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for bddl_name, obj in self.env.task.object_scope.items():
            if obj is None or not hasattr(obj, "get_position_orientation"):
                out[bddl_name] = None
                continue
            pos = obj.get_position_orientation()[0].cpu().numpy()
            lo, hi = (t.cpu().numpy() for t in obj.aabb)
            out[bddl_name] = {
                "sim_name": obj.name,
                "category": getattr(obj, "category", ""),
                "position": [round(float(v), 3) for v in pos],
                "aabb": [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]],
            }
        return out

    def check_state(self, kind: str, name: str, other: str | None = None) -> bool | None:
        cls_name, binary = STATES[kind]
        import omnigibson.object_states as object_states

        state_cls = getattr(object_states, cls_name)
        obj = self.resolve(name)
        if state_cls not in obj.states:
            return None
        if binary:
            if other is None:
                raise ValueError(f"state {kind!r} needs a second object")
            return bool(obj.states[state_cls].get_value(self.resolve(other)))
        return bool(obj.states[state_cls].get_value())

    def goal_report(self) -> dict[str, Any]:
        """Fresh goal evaluation without stepping the simulator."""
        task = self.env.task
        predicate = task._termination_conditions.get("predicate")
        if predicate is not None and getattr(predicate, "_check_goal_fn", None):
            done, status = predicate._check_goal_fn()
        else:  # fall back to the last stepped status
            done, status = bool(task.success), {"satisfied": [], "unsatisfied": []}
        return {
            "instruction": self.challenge.get("instruction", ""),
            "conditions": list(task.activity_natural_language_goal_conditions),
            "satisfied": list(status.get("satisfied", [])),
            "unsatisfied": list(status.get("unsatisfied", [])),
            "success": bool(done),
        }

    # ------------------------------------------------------------ media

    def capture_frame(self) -> Any | None:
        """One RGB frame for the harness record — the follow-cam when recording,
        else the robot's head (zed) camera. HxWx3 uint8."""
        if self._last_rec_frame is not None:
            self._frames.append(self._last_rec_frame.copy())
            return self._frames[-1]
        try:
            obs = self.env.get_obs()[0]
            if self._cam_key is None:
                for group, sub in obs.items():
                    if not isinstance(sub, dict):
                        continue
                    for cam in sub:
                        if "zed" in cam and isinstance(sub[cam], dict) and "rgb" in sub[cam]:
                            self._cam_key = (group, cam)
            if self._cam_key is None:
                return None
            group, cam = self._cam_key
            rgb = obs[group][cam]["rgb"]
            arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
            frame = arr[..., :3].astype(np.uint8)
            self._frames.append(frame)
            return frame
        except Exception as exc:  # a dropped frame must never kill an episode
            log.debug("frame capture failed: %s", exc)
            return None

    @property
    def frames(self) -> list[Any]:
        return list(self._frames)
