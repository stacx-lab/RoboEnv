"""Observe actual native evaluations without replaying history-dependent checks."""
import numpy as np


class PredicateDiagnostics:
    def __init__(self, env, task_name):
        if task_name not in {"ScrubCuttingBoard", "MultistepSteaming"}:
            raise ValueError("Predicate diagnostics support the two audited RoboCasa tasks")
        self.env, self.task_name = env, task_name
        self.evaluations, self.updates, self.errors = [], [], []
        self.originals = {}
        self._wrap("_check_success", self.evaluations, self.conditions)
        if task_name == "ScrubCuttingBoard":
            self._wrap("update_state", self.updates, self.scrub_contact)

    def _wrap(self, name, records, measure):
        original = getattr(self.env, name)
        self.originals[name] = self.env.__dict__.get(name)

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            try:
                value = {"native_time_s": float(self.env.sim.data.time), **measure()}
                if name == "_check_success": value["native_result"] = bool(result)
                records.append(value)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message not in self.errors: self.errors.append(message)
            return result

        setattr(self.env, name, observed)

    def detach(self):
        for name, value in self.originals.items():
            if value is None: delattr(self.env, name)
            else: setattr(self.env, name, value)

    def conditions(self):
        from robocasa.utils import object_utils as OU

        env = self.env
        if self.task_name == "ScrubCuttingBoard":
            points = np.asarray(env.board_contact_positions, dtype=float).reshape(-1, 2)
            sweep = 0.0 if not len(points) else float(np.linalg.norm(np.ptp(points, axis=0)))
            return {"board_contact_timer": int(env.board_contact_timer),
                    "board_contact_positions_xy_m": points.tolist(), "sweep_range_m": sweep,
                    "gripper_obj_far": bool(OU.gripper_obj_far(env, "sponge", th=0.15)),
                    "conditions": {"contact_count_ge_5": env.board_contact_timer >= 5,
                                   "sweep_range_ge_0_1m": sweep >= 0.1}}
        handle = env.sink.get_handle_state(env=env)
        pot_location = env._check_obj_location_on_stove(obj_name="pot")
        return {"sink_handle": {k: v.item() if hasattr(v, "item") else v for k, v in handle.items()},
                "specified_burner": str(env.knob), "pot_burner": pot_location,
                "vegetable_in_sink": bool(OU.obj_inside_of(env, "vegetable1", env.sink)),
                "conditions": {
                    "water_was_turned_on": bool(env.water_was_turned_on),
                    "vegetable_was_in_sink_while_water_on": bool(env.vegetable_was_in_sink),
                    "water_off": not bool(handle["water_on"]),
                    "pot_on_specified_burner": pot_location == env.knob,
                    "vegetable_in_pot": bool(OU.check_obj_in_receptacle(env, "vegetable1", "pot"))}}

    def scrub_contact(self):
        from robocasa.utils import object_utils as OU

        env = self.env
        def geoms(obj):
            body = env.obj_body_id[obj]
            return [env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom)
                    if env.sim.model.geom_bodyid[i] == body]
        return {"sponge_grasped": bool(OU.check_obj_grasped(env, "sponge")),
                "sponge_board_contact": bool(env.check_contact(geoms("sponge"), geoms("cutting_board"))),
                "board_contact_timer": int(env.board_contact_timer),
                "sponge_position_m": np.asarray(env.sim.data.body_xpos[env.obj_body_id["sponge"]]).tolist()}

    def snapshot(self):
        return {"complete": not self.errors, "errors": self.errors, "native_task": self.task_name,
                "scope": "Actual native success calls and post-update read-only measurements; no success reevaluation.",
                "native_evaluations": self.evaluations, "native_updates": self.updates,
                "terminal_native_evaluation": self.evaluations[-1] if self.evaluations else None}
