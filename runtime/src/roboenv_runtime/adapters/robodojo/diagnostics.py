"""Owner-only native predicate traces and read-only terminal geometry checks."""

import copy
import math
from types import SimpleNamespace


# Only these native functions have been audited for read-only evaluation. History
# predicates (including is_not_moved) must use their actual execution records.
PURE_PREDICATES = frozenset({
    "is_axis_up", "is_axis_aligned", "is_A_up_B", "is_A_in_B_support_circle",
    "is_AB_xy_distance_within_threshold", "is_A_depth_in_B",
    "is_all_gripper_open", "all_robot_back_to_origin",
})
HISTORY_PREDICATES = frozenset({"is_moved", "is_not_moved"})


def json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite native diagnostic value")
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if callable(value):
        return {"callable": getattr(value, "__qualname__", type(value).__name__)}
    raise TypeError(f"Unsupported native diagnostic type: {type(value).__name__}")


def passed(value, rule):
    # Mirror native check_once's distinct scalar and list-member comparisons.
    if rule == "or":
        return value is not None and value > 0
    if rule == "and":
        return value is None or value >= 1
    return value >= 1


class PredicateDiagnostics:
    def __init__(self, env):
        self.env = env
        self.reward = env.reward_manager
        self.parser = self.reward.func_parser
        self.leaves = {}
        self.references = {}
        self.errors = []
        self.calls = 0
        self.progress = []
        self.hooks_available = hasattr(self.parser, "diagnostic_measurements")
        self.goal = self._groups(self.reward.check_list[0], "goal")
        self.final_goal = self._groups(self.reward.final_check_list[0], "final_goal")
        self.scores = {}
        for prefix in ("score", "final_score"):
            meta = copy.deepcopy(getattr(self.reward, prefix + "_meta")[0])
            entries = getattr(self.reward, prefix + "_list")[0]
            self.scores[prefix] = {
                "meta": meta,
                "stages": [self._conjunction(
                    entry[0] if meta["mode"] == "paired" else entry,
                    f"{prefix}/{index}",
                ) for index, entry in enumerate(entries)],
            }
        self.unsupported_groups = {
            name: json_value(getattr(self.reward, name)[0])
            for name in ("query_list", "trigger_check_list", "trigger_query_list")
            if getattr(self.reward, name)[0]
        }
        if self.reward.score_trigger_meta[0] is not None:
            self.unsupported_groups["score_trigger_meta"] = json_value(self.reward.score_trigger_meta[0])
        self.reward.diagnostic_observer = self.record
        self.checkpoint()

    def _groups(self, groups, path):
        return [self._conjunction(checks, f"{path}/{i}") for i, checks in enumerate(groups)]

    def _conjunction(self, checks, path):
        return {"id": path, "op": "and", "children": [
            self._node(check, f"{path}/{i}") for i, check in enumerate(checks)
        ]}

    def _node(self, check, path, op="or", rule="scalar"):
        if isinstance(check, tuple):
            name, args = check
            node = {"id": path, "predicate": name,
                    "arguments": json_value({k: v for k, v in args.items() if k != "env_idx"}),
                    "pass_rule": rule, "last_native_evaluation": None}
            self.leaves[path] = node
            self.references.setdefault(id(check), (check, []))[1].append(path)
            return node
        if not isinstance(check, list):
            raise TypeError("Unexpected native condition tree")
        return {"id": path, "op": op, "children": [
            self._node(child, f"{path}/{i}", "and" if op == "or" else "or", rule=op)
            for i, child in enumerate(check)
        ]}

    def _error(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        if message not in self.errors and len(self.errors) < 20:
            self.errors.append(message)

    def record(self, predicate, env_idx, result, measurements):
        """Observe the existing call; diagnostic errors never change native results."""
        try:
            if env_idx != 0:
                raise ValueError("Expected one native environment")
            self.calls += 1
            record = {
                "frame": int(self.env.take_action_cnt[0]), "call": self.calls,
                "value": json_value(result), "measurements": json_value(measurements),
            }
            for path in self.references.get(id(predicate), (None, []))[1]:
                leaf = self.leaves[path]
                leaf["last_native_evaluation"] = {
                    **record, "passed": passed(result, leaf["pass_rule"]),
                }
        except Exception as exc:
            self._error(exc)

    def checkpoint(self):
        """Retain only changes to native progression, not a per-frame scene dump."""
        try:
            state = {
                "remaining_goal_groups": len(self.reward.check_list[0]),
                "remaining_final_goal_groups": len(self.reward.final_check_list[0]),
                **{name: json_value(getattr(self.reward, name)[0]) for name in (
                    "score_completed_count", "score_achieved",
                    "final_score_completed_count", "final_score_achieved",
                )},
            }
            if not self.progress or self.progress[-1]["state"] != state:
                self.progress.append({"frame": int(self.env.take_action_cnt[0]), "state": state})
        except Exception as exc:
            self._error(exc)

    def _terminal_evaluation(self, leaf):
        name, args = leaf["predicate"], leaf["arguments"]
        if name in HISTORY_PREDICATES:
            return {"status": "not_evaluated", "reason": "requires_native_history"}
        if name not in PURE_PREDICATES or any(
            key.endswith("_args") and value is not None for key, value in args.items()
        ):
            return {"status": "not_evaluated", "reason": "unsupported_terminal_check"}
        parser = copy.copy(self.parser)
        try:
            parser.diagnostic_measurements = []
            if name in {"is_all_gripper_open", "all_robot_back_to_origin"}:
                # Native aggregate checks short-circuit within the robot loop.
                # Evaluate each arm with the same native function on a read-only
                # view so the report includes both arms even when the first fails.
                results = []
                manager = self.parser.robot_manager
                for robot in manager.robot_list:
                    if robot.type == "target":
                        # Keep getters bound to the original manager: it indexes
                        # robot_key by its full robot_list, including support arms.
                        parser.robot_manager = SimpleNamespace(
                            robot_list=[robot], get_real_endpose=manager.get_real_endpose,
                            get_end_effector_real_val=manager.get_end_effector_real_val,
                        )
                        results.append(getattr(parser, name)({**copy.deepcopy(args), "env_idx": 0}))
                result = float(all(value >= 1 for value in results))
            else:
                result = getattr(parser, name)({**copy.deepcopy(args), "env_idx": 0})
            return {"status": "evaluated", "source": "terminal_read_only",
                    "frame": int(self.env.take_action_cnt[0]), "value": json_value(result),
                    "passed": passed(result, leaf["pass_rule"]),
                    "measurements": json_value(parser.diagnostic_measurements)}
        except Exception as exc:
            self._error(exc)
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def _summarize_tree(self, node):
        if "predicate" in node:
            return node["terminal_evaluation"].get("passed")
        values = [self._summarize_tree(child) for child in node["children"]]
        if node["op"] == "and":
            result = False if False in values else None if None in values else True
        else:
            result = True if True in values else None if None in values else False
        node["terminal_passed"] = result
        return result

    def snapshot(self):
        self.checkpoint()
        # Copy the registered tree and trace before adding terminal-only results.
        report = copy.deepcopy({"goal": self.goal, "final_goal": self.final_goal,
                                "scores": self.scores})
        counts = {"evaluated": 0, "not_evaluated": 0, "error": 0}
        unsupported_checks = []
        objects = {}

        def visit(node):
            if "predicate" not in node:
                for child in node["children"]:
                    visit(child)
                return
            result = self._terminal_evaluation(node)
            node["terminal_evaluation"] = result
            counts[result["status"]] += 1
            if result.get("reason") == "unsupported_terminal_check":
                unsupported_checks.append(node["id"])
            for key in ("label", "label_A", "label_B"):
                label = node["arguments"].get(key)
                if not isinstance(label, str) or label in objects:
                    continue
                try:
                    layout = self.parser.layout_manager
                    name = layout.get_instance_name(label=label, env_idx=0)
                    objects[label] = {"instance": name, "pose": json_value(
                        layout.get_instance_pose(inst_name=name, env_idx=0)),
                        "metadata": json_value(layout.get_instance_metadata(inst_name=name, env_idx=0))}
                except Exception as exc:
                    self._error(exc)

        roots = report["goal"] + report["final_goal"]
        roots += [stage for score in report["scores"].values() for stage in score["stages"]]
        for root in roots:
            visit(root)
            self._summarize_tree(root)
        unsupported = sorted({leaf["predicate"] for leaf in self.leaves.values()}
                             - PURE_PREDICATES - HISTORY_PREDICATES)
        report.update(
            complete=self.hooks_available and not self.errors
            and not self.unsupported_groups and not unsupported and not unsupported_checks,
            native_hooks_available=self.hooks_available, terminal_coverage=counts,
            frame=int(self.env.take_action_cnt[0]), objects=objects,
            coordinate_convention={"position": "native environment relative metres",
                                   "quaternion": "wxyz",
                                   "environment_origin_world_m": json_value(
                                       self.env.scene_manager.env_origins[0])},
            progression=copy.deepcopy(self.progress), errors=list(self.errors),
            unsupported_groups=copy.deepcopy(self.unsupported_groups),
            unsupported_predicates=unsupported,
            unsupported_terminal_checks=unsupported_checks,
            semantics="Terminal checks do not replace native verdict or historical stage completion.",
        )
        return report
