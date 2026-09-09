"""RoboEnv camera: profile-selected remote device and calibrated file output."""

import math
import struct
import sys
from pathlib import Path

# Importing the support file must not modify the copied catalog.
sys.dont_write_bytecode = True
from _roboenv import Transport, WireError, main  # noqa: E402

transport = Transport(__file__)
settings = transport.settings
remote = transport.remote


def camera_read(data, profile):
    stamps = ["sample_time_s", "received_time_s"]
    if profile["type"] == "rgbd":
        stamps.append("depth_sample_time_s")
    for key in stamps:
        value = data[key]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"Invalid camera timestamp: {key}")
    color = profile["intrinsics"]["color"]
    if (data["width"], data["height"]) != (color["width"], color["height"]):
        raise ValueError("Remote color dimensions differ from the profile")
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
    result = {
        "file": data["file"],
        "width": color["width"],
        "height": color["height"],
        "sample_time_s": data["sample_time_s"],
        "received_time_s": data["received_time_s"],
    }
    if profile["type"] == "rgb":
        return result
    depth = profile["intrinsics"]["depth"]
    if depth["registration"] != "color" or depth["quantity"] != "z" or depth["invalid_value"] != 0:
        raise ValueError("RoboCasa serves color-registered optical Z with zero invalid")
    width, height = depth["width"], depth["height"]
    if (data["depth_width"], data["depth_height"]) != (width, height):
        raise ValueError("Remote depth dimensions differ from the profile")
    expected = f"P5\n{width} {height}\n65535\n".encode()
    with Path(data["depth_file"]).open("rb") as handle:
        if handle.read(len(expected)) != expected:
            raise ValueError("Remote depth must be PGM16 at the profile resolution")
        raster = handle.read(width * height * 2 + 1)
    if len(raster) != width * height * 2:
        raise ValueError("Remote depth payload differs from the declared dimensions")
    pixels = (value[0] for value in struct.iter_unpack(">H", raster))
    limits = depth["valid_range_m"]
    if any(
        raw != depth["invalid_value"]
        and not limits["min"] <= raw * depth["scale_m_per_unit"] <= limits["max"]
        for raw in pixels
    ):
        raise ValueError("Remote depth exceeds the declared valid range")
    return {
        **result,
        "depth_file": data["depth_file"],
        "depth_width": width,
        "depth_height": height,
        "depth_sample_time_s": data["depth_sample_time_s"],
    }


def handle(action, args):
    if action not in {"read", "load", "unload", "status"}:
        return {"ok": False, "error": f"Unsupported camera action: {action}"}
    try:
        if args:
            raise ValueError(f"{action} takes no arguments")
        data = remote(action, args)
        if action == "read":
            data = camera_read(data, settings())
        return {**data, "ok": True}
    except WireError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    main(handle)
