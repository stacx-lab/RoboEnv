"""Fallback for standalone controllers; a WorldSession injects its clock wait."""

import time


def wait_for_clock(timeout):
    time.sleep(timeout)
    return True
