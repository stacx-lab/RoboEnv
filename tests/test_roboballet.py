"""Native coupled motion and completion rules need no simulator installation."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
from roboenv_runtime.adapters.isaaclab.tasks.roboballet_eight_arm.feasibility import (
    wrist_distance_lower_bound,
)
from roboenv_runtime.adapters.isaaclab.tasks.roboballet_eight_arm.reaching import (
    ReachingTask,
    achieved,
)
from scipy.spatial.transform import Rotation


class World:
    def __init__(self):
        self.q = np.zeros((8, 7))

    def GetJointValues(self, name):
        return self.q[int(name)].copy()

    def SetJointValues(self, name, q):
        self.q[int(name)] = q

    def GetRobotTcpTransform(self, name):
        pose = np.eye(4)
        pose[0, 3] = self.q[int(name), 0]
        return pose

    def GetRobotBaseTransform(self, name):
        return np.eye(4)

    def RobotsInCollision(self):
        return {str(i) for i in range(8) if self.q[i, 0] > .1}


def task():
    value = ReachingTask.__new__(ReachingTask)
    value.world, value.probe = World(), World()
    value.robots = list(map(str, range(8)))
    value.home = np.zeros((8, 7))
    value.lower, value.upper = np.full((8, 7), -3.), np.full((8, 7), 3.)
    value.velocity, value.dwell = np.zeros((8, 7)), np.zeros(8, dtype=int)
    value.targets = [{"position": [2., 0., 0.], "rotation": np.eye(3).tolist()}] * 40
    value.solved_by = np.full(40, -1, dtype=int)
    value.rng, value.events = random.Random(0), []
    value.frame, value.collision_robot_steps, value.last_blocked = 0, 0, []
    value.motion_histogram, value.work_histogram = [0] * 9, [0] * 9
    value.unsafe_proposal_steps, value.unsafe_proposal_robot_steps = 0, 0
    value.measured_velocity, value.telemetry = np.zeros((8, 7)), []
    value.collision_pairs = lambda world: []
    return value


def test_reaching_requires_position_and_full_orientation():
    target = {"position": [0., 0., 0.], "rotation": np.eye(3).tolist()}
    pose = np.eye(4)
    pose[0, 3] = .0249
    assert achieved(pose, target)
    pose[0, 3] = .0251
    assert not achieved(pose, target)
    pose[0, 3] = 0.
    pose[:3, :3] = Rotation.from_euler("z", 14.9, degrees=True).as_matrix()
    assert achieved(pose, target)
    pose[:3, :3] = Rotation.from_euler("z", 15.1, degrees=True).as_matrix()
    assert not achieved(pose, target)


def test_all_eight_arms_advance_together_and_only_colliding_arms_roll_back():
    value = task()
    references = np.full((8, 7), -.5)
    references[0, 0] = .5
    value.step(references)
    assert np.allclose(value.positions()[0, 0], .08)
    assert value.motion_histogram[8] == 1
    value.step(references)
    assert value.last_blocked == [0]
    assert value.positions()[0, 0] == pytest.approx(.08)
    assert np.allclose(value.positions()[1:, 0], -.24)
    assert value.collision_robot_steps == 1


def test_dwell_excludes_other_targets_and_home_return_is_required():
    value = task()
    value.targets = [{"position": [0., 0., 0.], "rotation": np.eye(3).tolist()}] * 40
    value.step(value.home)
    assert (value.solved_by >= 0).sum() == 8
    assert value.dwell.tolist() == [5] * 8
    for _ in range(4):
        value.step(value.home)
    assert (value.solved_by >= 0).sum() == 8
    value.step(value.home)
    assert (value.solved_by >= 0).sum() == 16
    value.solved_by[:] = 0
    assert not value.completed
    value.dwell[:] = 0
    value.world.q[:] = -.0011
    assert not value.completed
    value.step(value.home)
    assert value.completed
    value.world.q[:] = -.0009
    assert value.completed


def test_safety_telemetry_distinguishes_unsafe_proposals_from_rollbacks():
    class PairWorld(World):
        def RobotsInCollision(self):
            return {"0", "1"} if self.q[0, 0] + self.q[1, 0] > .4 else set()

    value = task()
    value.world = PairWorld()
    reference = value.home.copy()
    reference[:2, 0] = .5
    value.step(reference)
    value.step(reference)
    assert value.unsafe_proposal_steps == 1
    assert value.unsafe_proposal_robot_steps == 2
    assert value.collision_robot_steps == 1
    sample = value.telemetry[-1]
    assert sample["unsafe_proposal_arms"] == [1, 2]
    assert len(sample["rollback_arms"]) == 1
    assert len(sample["moving_arms"]) == 1
    assert sample["targets_completed"] == 0
    assert len(sample["native_state_sha256"]) == 64
    assert value.telemetry[0]["native_state_sha256"] != sample["native_state_sha256"]


def test_query_validation_cannot_move_or_advance_live_world():
    value = task()
    for configurations in ([], [[[0.]]], np.full((1, 8, 7), np.nan)):
        with pytest.raises(ValueError):
            value.check(configurations)
    result = value.check(np.full((1, 8, 7), 9.))
    assert result["configurations"][0]["reason"] == "joint_limits"
    assert value.frame == 0 and np.array_equal(value.positions(), value.home)


def test_frozen_scene_retains_all_original_target_sets_and_obstacles():
    root = Path(__file__).resolve().parents[1]
    scene = root / "catalog/tasks/isaaclab/roboballet-eight-arm/scene.json"
    spec = json.loads(scene.read_text())
    assert len(spec["obstacles"]) == 30
    assert len(spec["target_sets"]) == 10
    assert all(len(targets) == 40 for targets in spec["target_sets"])
    assert spec["source_doi"] == "10.5281/zenodo.16546403"


def test_reach_bound_accounts_for_both_pose_tolerances_and_sampling_error():
    vectors = np.array([[0., 0., .25]])
    arguments = {"position_tolerance": .025, "angle_tolerance": np.pi / 12,
                 "sampling_error": .001}
    lower = wrist_distance_lower_bound(
        [0., 0., 1.], np.eye(3), np.zeros(3), vectors, **arguments)
    assert lower == pytest.approx(.724)
    tilted = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    lower = wrist_distance_lower_bound([0., 0., 1.], tilted, np.zeros(3), vectors, **arguments)
    expected = np.sqrt(1. + .25**2 - .5 * np.cos(np.pi / 12)) - .026
    assert lower == pytest.approx(expected)


def test_reach_bound_never_excludes_an_observed_accepted_pose():
    rng = np.random.default_rng(7309)
    vectors = np.array([[.08, -.02, .25]])
    for _ in range(100):
        actual = Rotation.random(random_state=rng)
        wrist = rng.normal(size=3) * .2
        position = wrist + actual.apply(vectors[0])
        offset = rng.normal(size=3)
        position += offset / np.linalg.norm(offset) * .024
        target_rotation = actual * Rotation.from_rotvec([.1, .1, .1])
        lower = wrist_distance_lower_bound(
            position, target_rotation.as_matrix(), np.zeros(3), vectors,
            position_tolerance=.025, angle_tolerance=np.pi / 12, sampling_error=0.)
        assert lower <= np.linalg.norm(wrist) + 1e-12


def test_unreachable_task_is_rejected_as_configuration_error(monkeypatch):
    from roboenv_runtime.adapters.isaaclab.tasks.roboballet_eight_arm import feasibility

    value = task()
    value.target_set_index = 0
    monkeypatch.setattr(feasibility, "unreachable_targets",
                        lambda *args, **kwargs: [{"target": 32, "gap_lower_m": .011}])
    with pytest.raises(ValueError, match="target set 0: targets 32.*configuration error"):
        value.validate_reachability()
    assert value.frame == 0


def test_registered_environment_preserves_clock_budget_and_owner_evidence(monkeypatch, tmp_path):
    from roboenv.tasks.resolve import resolve_task
    from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
    from roboenv_runtime.adapters.isaaclab.robots.roboballet_eight_panda import environment, native
    from roboenv_runtime.bindings.backends.isaaclab.roboballet_eight_panda import bind
    from roboenv_runtime.runtime.world import WorldSession

    value = task()
    value.scene_hash = "frozen-scene"
    value.validate_reachability = lambda: None
    captures, closes = [], []
    monkeypatch.setattr(environment, "verify_assets", lambda path: {"sha256": "verified-assets"})
    monkeypatch.setattr(environment, "ReachingTask", lambda *a, **kw: value)
    monkeypatch.setattr(environment, "version", lambda name: "test-version")

    def capture(name, directory):
        captures.append((name, directory))
        return {"file": "camera.png", "depth_file": "camera.pgm", "timestamp": value.frame * .1}

    monkeypatch.setattr(native, "ReachingScene", lambda *a: SimpleNamespace(
        sync=lambda: None, capture=capture, close=lambda: closes.append(True)))
    resolved = resolve_task("isaaclab/roboballet-eight-arm")
    assert resolved.task.execution_mode == "stepped" and resolved.task.stop_on_success
    assert resolved.task.backend_kwargs["target_set_index"] == 1
    adapter = IsaacLabAdapter(**resolved.task.backend_kwargs, stop_on_success=True)
    adapter.run_dir = tmp_path
    clock = WorldSession(adapter, "stepped")
    clock.reset(7)
    world = adapter.environment
    context = SimpleNamespace(backend="isaaclab", embodiment=adapter.embodiment, native=adapter,
                              devices={s.name: s.name for s in resolved.bindings.devices})
    bound = {s.name: bind(s, resolved.robot, context) for s in resolved.bindings.devices}
    assert "check" in bound["workcell"].actions and "write" not in bound["workcell"].actions
    assert "write" not in bound["camera_side"].actions
    assert set(bound["world"].actions) == {"read", "wait", "load", "unload", "status"}
    before = world.owner_state()
    for name in ("arm1", "arm8"):
        world.set_target(name, [-.5] * 7)
    world.read_camera("camera_side", tmp_path)
    world.read_camera("camera_side", tmp_path)
    assert len(captures) == 1 and world.owner_state() == before
    bad = {"actions": [[0.] * 7, [float("nan")] * 7], "duration_s": .2}
    with pytest.raises(ValueError):
        bound["arm1"].actions["write"].validate(bad)
    assert world.targets["arm1"] == [-.5] * 7
    clock.advance()
    assert value.frame == 0
    waiting = clock.wait(.1)
    clock.advance()
    assert waiting.result()["time_s"] == pytest.approx(.1)
    assert value.frame == 1 and value.motion_histogram[2] == 1
    assert world.read_device("arm1")["timestamp"] == pytest.approx(.1)
    world.stop("arm1")
    assert world.targets["arm1"] == world.read_device("arm1")["positions"]
    assert world.targets["arm8"] == [-.5] * 7
    record = world.record()
    assert record["seed"] == 7 and record["max_steps"] == 420
    assert record["initial_state"] == before
    record["state"]["state"]["trajectory_telemetry"].clear()
    assert value.telemetry
    value.frame = 420
    assert adapter.budget().exhausted
    with pytest.raises(Exception, match="frame limit") as error:
        world.tick()
    assert error.value.code == "episode_finished" and value.frame == 420
    adapter.configure_execution_mode("realtime")
    assert adapter.budget().max is None
    world.tick()
    assert value.frame == 421
    adapter.close()
    assert closes == [True]


def test_prepared_assets_detect_changes_without_overwriting(tmp_path):
    import hashlib

    from roboenv.provisioning.roboballet_assets import MODULES, verify_assets

    selected = tmp_path / "selected"
    names = [*(f"roboballet/environments/{name}" for name in MODULES),
             "roboballet/data/mujoco_world/8_pandas_world.xml"]
    manifest = {}
    for name in names:
        file = selected / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name)
        manifest[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path / "selected-files.json").write_text(json.dumps(manifest))
    assert verify_assets(selected)["files"] == len(names)
    changed = selected / names[0]
    changed.write_text("changed")
    with pytest.raises(ValueError, match="differs from its prepared manifest"):
        verify_assets(selected)
    assert changed.read_text() == "changed"
