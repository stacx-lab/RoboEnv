"""Reject catalog claims that differ from the native device contract."""

# These fields describe client-side calibration or derived/informational data.
# They do not change what the native device produces or how it is commanded.
CALIBRATION = {"source", "method", "calibrated_at", "needs_calibration",
               "bias", "range", "resolution", "fov", "reliable_range_m"}


def validate(actual, expected, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) - CALIBRATION != set(expected) - CALIBRATION:
            raise ValueError(f"{path} fields differ from the native device profile")
        for key in expected.keys() - CALIBRATION:
            validate(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} length differs from the native device profile")
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            validate(value, reference, f"{path}[{index}]")
        return
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ValueError(f"{path} differs from the native device profile")
