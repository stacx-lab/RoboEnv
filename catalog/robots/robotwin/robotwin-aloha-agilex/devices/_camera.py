"""Camera output conversion; the server's artifact protocol stays unchanged."""

import base64
import io
import math
import struct
from pathlib import Path


def output_mode(value="file"):
    if value not in ("file", "array", "base64"):
        raise ValueError("camera output must be file, array or base64")
    return value


def camera_read(data, profile):
    value = data["timestamp"]
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Invalid camera timestamp")
    content = "color" in data
    color = profile["intrinsics"]["color"]
    if content:
        header = base64.b64decode(data["color"], validate=True)[:26]
    else:
        with Path(data["file"]).open("rb") as handle:
            header = handle.read(26)
    if (
        len(header) != 26
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
        or header[24:26] != b"\x08\x02"
        or struct.unpack(">II", header[16:24]) != (color["width"], color["height"])
    ):
        raise ValueError("Remote color must be an RGB8 PNG at the profile resolution")
    key = "color" if content else "file"
    result = {key: data[key], "timestamp": value}
    if profile["type"] == "rgb":
        return result
    depth = profile["intrinsics"]["depth"]
    if depth["registration"] != "color" or depth["quantity"] != "z" or depth["invalid_value"] != 0:
        raise ValueError("RoboCasa serves color-registered optical Z with zero invalid")
    width, height = depth["width"], depth["height"]
    expected = f"P5\n{width} {height}\n65535\n".encode()
    key = "depth" if content else "depth_file"
    if content:
        raw = base64.b64decode(data[key], validate=True)
    else:
        with Path(data[key]).open("rb") as handle:
            raw = handle.read(len(expected) + width * height * 2 + 1)
    if not raw.startswith(expected) or len(raw) != len(expected) + width * height * 2:
        raise ValueError("Remote depth must be PGM16 at the profile resolution")
    limits = depth["valid_range_m"]
    if any(
        pixel != depth["invalid_value"]
        and not limits["min"] <= pixel * depth["scale_m_per_unit"] <= limits["max"]
        for (pixel,) in struct.iter_unpack(">H", raw[len(expected) :])
    ):
        raise ValueError("Remote depth exceeds the declared valid range")
    return {**result, key: data[key]}


def arrays(data, profile):
    import numpy as np
    from PIL import Image

    result = dict(data)
    for key in ("color", "depth") if profile["type"] == "rgbd" else ("color",):
        raw = base64.b64decode(data[key], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            result[key] = np.array(
                image, dtype=np.uint8 if key == "color" else np.uint16, copy=True
            )
    return result
