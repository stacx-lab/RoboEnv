"""The runtime's own defaults find the checkout's catalog without launcher variables."""

import pytest
from roboenv.tasks.model import task_path
from roboenv_runtime.paths import tasks_root


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
