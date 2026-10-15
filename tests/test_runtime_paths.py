"""The runtime's own defaults find the checkout's catalog without launcher variables."""

from pathlib import Path

import pytest
from roboenv.tasks.model import task_path
from roboenv_runtime.paths import tasks_root
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig


def test_worker_resolves_task_ids_without_roboenv_tasks(monkeypatch):
    # roboenv-worker and roboenv-supervise resolve a task ID below tasks_root(); launch sets
    # ROBOENV_TASKS, a direct run did not and looked in directories that no longer exist.
    monkeypatch.delenv("ROBOENV_TASKS", raising=False)
    assert task_path("libero/libero-spatial-01", tasks_root()).name == "task.yaml"


def test_explicit_task_root_does_not_fall_back_to_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOENV_TASKS", str(tmp_path))
    assert tasks_root() == tmp_path
    with pytest.raises(ValueError, match="found 0"):
        task_path("libero/libero-spatial-01", tasks_root())


def test_worker_runtime_data_stays_in_disposable_cache(tmp_path):
    config = SupervisorConfig(run_dir=tmp_path, task="fakeshell/toy", seed=0,
                              budget_mode="official", session="test")
    environment = Supervisor(config)._worker_environment()
    for name in ("XDG_DATA_HOME", "OV_DATA_ROOT"):
        path = Path(environment[name])
        assert path.is_relative_to(tmp_path / "cache")
        assert path.is_dir()
    assert Path(environment["ROBOENV_RUN_DIR"]) == tmp_path
    assert not (tmp_path / "data").exists()
