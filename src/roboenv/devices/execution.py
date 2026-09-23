"""Deferred device results, independent of simulator and transport ownership."""

from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar

request_identity = ContextVar("device_request_identity", default=None)

_scheduler = ContextVar("device_motion_scheduler", default=None)


@contextmanager
def scheduling(schedule):
    token = _scheduler.set(schedule)
    try:
        yield
    finally:
        _scheduler.reset(token)


def motion_scheduler():
    return _scheduler.get()


def map_result(value, success, failure=None):
    """Transform immediate or deferred results without blocking a native thread."""
    if not isinstance(value, Future):
        return success(value)
    result = Future()

    def complete(source):
        try:
            try:
                output = success(source.result())
            except Exception as error:
                if failure is None:
                    raise
                output = failure(error)
            if isinstance(output, Future):
                output.add_done_callback(complete_result)
            else:
                result.set_result(output)
        except Exception as error:
            result.set_exception(error)

    def complete_result(source):
        try:
            result.set_result(source.result())
        except Exception as error:
            result.set_exception(error)

    value.add_done_callback(complete)
    return result
