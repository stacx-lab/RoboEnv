"""Execute native reward code with CPU geometry collaborators, without Isaac."""

import ast
import copy
import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboenv_runtime.adapters.robodojo.diagnostics import PredicateDiagnostics

SOURCE = Path(__file__).resolve().parents[1] / "third_party/RoboDojo"


def definitions(path, namespace, names=None):
    if not path.is_file():
        pytest.skip("Native reward checks require the selected RoboDojo source")
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and (names is None or node.name in names)]
    module = ast.parse("from __future__ import annotations\n")
    module.body.extend(nodes)
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture
def native():
    ns = {"np": np, "copy": copy, "deepcopy": copy.deepcopy, "List": list, "Tuple": tuple,
          "combinations": combinations}
    definitions(SOURCE / "utils/transformer.py", ns)
    # Footprints are unused by the tested depth predicate; retain native bbox
    # transformation and extrema calculations while replacing only the XY hull.
    ns["MultiPoint"] = lambda points: SimpleNamespace(
        convex_hull=SimpleNamespace(buffer=lambda margin: None))
    ns["t3d"] = SimpleNamespace(quaternions=SimpleNamespace(quat2mat=ns["quat_to_mat"]))
    ns["cal_quat_dis"] = lambda a, b: 0.0
    definitions(SOURCE / "env/reward_manager/func_parser.py", ns)
    definitions(SOURCE / "env/reward_manager/reward_manager.py", ns)
    poses = {name: (np.zeros(3), np.array([1., 0, 0, 0]))
             for name in [*(f"block{i}" for i in range(8)),
                          *(f"nut{i}" for i in range(3)), *(f"bolt{i}" for i in range(3))]}
    boxes = {name: np.array([[-.01, -.01, -.005], [.01, .01, .005]]) for name in poses}
    layout = SimpleNamespace(
        get_instance_name=lambda label, env_idx: label,
        get_instance_pose=lambda inst_name, env_idx: poses[inst_name],
        get_instance_bbox_vertices=lambda inst_name, env_idx: boxes[inst_name],
        get_instance_metadata=lambda inst_name, env_idx: {"name": inst_name},
        get_scene_object=lambda inst_name, env_idx: object(),
        get_support_points=lambda **kwargs: ([np.zeros(3)], [.02]),
    )
    robots = [SimpleNamespace(type="target", arm_name=side, gripper_scale=[0., 1.])
              for side in ("left_arm", "right_arm")]
    endposes = {robot.arm_name: [0., 0, 0, 1, 0, 0, 0] for robot in robots}
    openings = {robot.arm_name: 1. for robot in robots}

    class RobotManager:
        # Match the native index relationship between robot_list and robot_key.
        robot_list = robots
        robot_key = [robot.arm_name for robot in robots]

        def get_real_endpose(self, robot):
            return [endposes[self.robot_key[self.robot_list.index(robot)]]]

        def get_end_effector_real_val(self, robot, env_idx_list):
            return [openings[self.robot_key[self.robot_list.index(robot)]]]

    manager = ns["RewardManager"](1)
    env = SimpleNamespace(reward_manager=manager, take_action_cnt=[0], success=[True],
                          scene_manager=SimpleNamespace(layout_manager=layout,
                                                        env_origins=[np.zeros(3)]),
                          robot_manager=RobotManager())
    manager.initialize(env)
    manager.func_parser.robot_origin_endpose = [copy.deepcopy(endposes)]
    manager.func_parser.pre_state = [{name: {"pose": np.r_[pos, quat]}
                                     for name, (pos, quat) in poses.items()}]
    return SimpleNamespace(env=env, rm=manager, poses=poses, boxes=boxes,
                           openings=openings, endposes=endposes, namespace=ns)


def leaves(node):
    if "predicate" in node:
        return [node]
    return [leaf for child in node["children"] for leaf in leaves(child)]


def test_short_circuit_trace_and_read_only_snapshot_preserve_history(native):
    rm = native.rm
    native.poses["block0"] = (np.zeros(3), np.array([0., 1, 0, 0]))
    rm.check([rm.is_axis_up(label="block0", axis=[0, 0, 1], threshold=5),
              rm.is_A_depth_in_B(label_A="nut0", label_B="bolt0", z_threshold=.009)])
    rm.score([[rm.is_not_moved(label="block7", dis_threshold=.002, update=True)]],
             [100], score_mode="transition")
    native.poses["block7"][0][0] = .01
    trace = PredicateDiagnostics(native.env)
    native.env.take_action_cnt[0] = 1
    rm.step([0])
    before = copy.deepcopy(rm.func_parser.pre_state)
    state = copy.deepcopy((rm.score_completed_count, rm.check_list))
    report = trace.snapshot()
    first, skipped = leaves(report["goal"][0])
    assert first["last_native_evaluation"]["passed"] is False
    assert first["terminal_evaluation"]["measurements"][0]["value"] == 180
    assert skipped["last_native_evaluation"] is None
    assert skipped["terminal_evaluation"]["passed"] is True
    history = leaves(report["scores"]["score"]["stages"][0])[0]
    assert history["terminal_evaluation"]["status"] == "not_evaluated"
    assert history["last_native_evaluation"]["passed"] is False
    assert history["last_native_evaluation"]["frame"] == 1
    assert history["last_native_evaluation"]["measurements"][0]["op"] == "<="
    assert (rm.score_completed_count, rm.check_list) == state
    for label in before[0]:
        np.testing.assert_array_equal(
            before[0][label]["pose"], rm.func_parser.pre_state[0][label]["pose"])
    assert native.env.take_action_cnt == [1]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("depth,expected", [(.008, False), (.009, False), (.010, True)])
def test_depth_records_native_extrema_and_strict_boundary(native, depth, expected):
    native.boxes["nut0"][:] = 0
    native.boxes["bolt0"][:] = 0
    native.poses["bolt0"][0][2] = depth
    native.rm.check([native.rm.is_A_depth_in_B(
        label_A="nut0", label_B="bolt0", z_threshold=.009)])
    report = PredicateDiagnostics(native.env).snapshot()
    result = leaves(report["goal"][0])[0]["terminal_evaluation"]
    assert result["passed"] is expected
    measured = result["measurements"][0]
    assert measured["value"] == depth and measured["op"] == ">"
    assert measured["inputs"] == {"A_z_min": 0., "B_z_max": depth}


def test_native_or_alternatives_are_not_flattened_into_failures(native):
    native.poses["block0"] = (np.zeros(3), np.array([0., 1, 0, 0]))
    rm = native.rm
    rm.check([[rm.is_axis_up(label=label, axis=[0, 0, 1], threshold=5)
               for label in ("block0", "block1")]])
    trace = PredicateDiagnostics(native.env)
    rm.step([0])
    assert rm.check_list == [[]]
    report = trace.snapshot()
    assert report["goal"][0]["terminal_passed"] is True
    assert report["goal"][0]["children"][0]["op"] == "or"
    assert len(leaves(report["goal"][0])) == 2  # Retained after native consumption.


def test_aggregate_terminal_checks_report_both_arms(native):
    native.openings.update(left_arm=.5, right_arm=.7)
    native.endposes["left_arm"][0] = .1
    native.endposes["right_arm"][1] = .2
    native.rm.check([native.rm.is_all_gripper_open(open_threshold=.9),
                     native.rm.all_robot_back_to_origin()])
    report = PredicateDiagnostics(native.env).snapshot()
    gripper, origin = leaves(report["goal"][0])
    for leaf in (gripper, origin):
        result = leaf["terminal_evaluation"]
        assert result["passed"] is False
        assert {row["inputs"]["arm"] for row in result["measurements"]} == {
            "left_arm", "right_arm"}
    assert len(native.env.robot_manager.robot_list) == 2
    errors = [row["value"] for row in origin["terminal_evaluation"]["measurements"]
              if row["name"] == "origin_position_error"]
    assert errors == [[.1, 0., 0.], [0., .2, 0.]]


@pytest.mark.parametrize("task", ["build_tower", "fasten_screws"])
def test_task_predicate_coverage_and_native_progression_are_unchanged(native, task):
    if task == "build_tower":
        for index, height in enumerate([.1, .05, .05, .23, .25, .15, .15, .2]):
            native.poses[f"block{index}"][0][2] = height
        native.poses["block4"] = (native.poses["block4"][0],
                                  np.array([2 ** -.5, 2 ** -.5, 0., 0.]))
    ns = native.namespace
    definitions(SOURCE / f"task/RoboDojo/tasks/{task}.py", ns,
                {"BuildTowerCommon", "FastenScrewsCommon"})
    task_class = ns["BuildTowerCommon" if task == "build_tower" else "FastenScrewsCommon"]
    task_obj = task_class.__new__(task_class)
    task_obj.reward_manager = native.rm

    def run(enabled):
        native.env.take_action_cnt[0] = 0
        native.rm.diagnostic_observer = None
        native.rm.reset()
        native.rm.func_parser.robot_origin_endpose = [copy.deepcopy(native.endposes)]
        native.rm.func_parser.pre_state = [{name: {"pose": np.r_[pos, quat]}
                                           for name, (pos, quat) in native.poses.items()}]
        task_obj.run_reward()
        task_obj.get_score()
        trace = PredicateDiagnostics(native.env) if enabled else None
        results = []
        for frame in range(1, 4):
            native.env.take_action_cnt[0] = frame
            native.rm.step([0])
            results.append((native.rm.get_reward(), copy.deepcopy(native.rm.score_completed_count)))
            if trace:
                trace.checkpoint()
        return results, trace

    baseline, _ = run(False)
    observed, trace = run(True)
    assert baseline == observed
    assert native.rm.score_completed_count == [3]
    report = trace.snapshot()
    assert report["complete"], report["errors"]
    assert report["unsupported_predicates"] == []
    assert report["terminal_coverage"]["error"] == 0
    assert report["terminal_coverage"]["not_evaluated"] == (2 if task == "build_tower" else 0)
    assert report["goal"][0]["terminal_passed"] is True


def test_unsupported_predicates_and_diagnostic_errors_remain_explicit(native):
    rm = native.rm
    rm.check([rm.is_lift(label="block0", z_threshold=.1)])
    trace = PredicateDiagnostics(native.env)
    rm.step([0])
    report = trace.snapshot()
    assert report["complete"] is False
    assert report["unsupported_predicates"] == ["is_lift"]
    assert leaves(report["goal"][0])[0]["terminal_evaluation"]["status"] == "not_evaluated"
    # A serialization failure in the observer cannot replace the native result.
    rm.func_parser.is_lift = lambda args: object()
    result = rm.call_func_parser(rm.check_list[0][0][0], 0)
    assert type(result) is object
    assert trace.errors
