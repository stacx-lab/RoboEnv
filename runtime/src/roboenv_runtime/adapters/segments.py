"""Bounded segments yield to the world scheduler instead of occupying its thread."""

import math
import time

from roboenv.devices.execution import segment_scheduler


def require_scheduler(realtime, for_s):
    if realtime and for_s and segment_scheduler() is None:
        raise RuntimeError("bounded realtime writes require the world segment scheduler")


def finish_segment(*, started, frame, interval, for_s, realtime, arrived,
                   exhausted, result):
    if not realtime or not for_s:
        return result((frame() - started) * interval)
    limit = started + max(1, math.ceil(for_s / interval))
    deadline = time.monotonic() + max(1.0, 4 * (limit - started) * interval)
    def complete():
        return (frame() >= limit or exhausted() or bool(arrived and arrived())
                or time.monotonic() >= deadline)
    def snapshot():
        return result((frame() - started) * interval)
    return segment_scheduler()(complete, snapshot)
