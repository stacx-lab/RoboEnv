"""Optional stage reporting without exposing subprocess output or private inputs."""

import threading
import time
from contextlib import contextmanager


@contextmanager
def stage(progress, label, *, interval=15):
    """Report stage boundaries and elapsed time while a blocking operation runs."""
    if progress is None:
        yield
        return
    started = time.monotonic()
    finished = threading.Event()
    progress(f"{label}...")

    def heartbeat():
        while not finished.wait(interval):
            progress(f"{label}: still running ({time.monotonic() - started:.0f}s elapsed)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    outcome = "done"
    try:
        yield
    except BaseException:
        outcome = "failed or interrupted"
        raise
    finally:
        finished.set()
        thread.join()
        progress(f"{label}: {outcome} ({time.monotonic() - started:.0f}s)")
