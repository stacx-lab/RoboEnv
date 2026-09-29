"""The runtime's own defaults find the checkout's catalog without launcher variables."""

from roboenv.tasks.model import task_path
from roboenv_runtime.paths import tasks_root


def test_worker_resolves_task_ids_without_roboenv_tasks(monkeypatch, tmp_path):
    # roboenv-worker and roboenv-supervise resolve a task ID below tasks_root(); launch sets
    # ROBOENV_TASKS, a direct run did not and looked in directories that no longer exist.
    monkeypatch.delenv("ROBOENV_TASKS", raising=False)
    assert task_path("libero/bowl-onto-plate", tasks_root()).name == "task.yaml"
    monkeypatch.setenv("ROBOENV_TASKS", str(tmp_path))
    assert tasks_root() == tmp_path
