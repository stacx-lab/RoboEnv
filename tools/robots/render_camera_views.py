"""Render nominal pinhole RGB previews from the model's physical optical frames.

Uses the static scene and world frames written by render_go2_d1.py. This is not
RealSense sensor emulation: distortion, stereo reconstruction and noise are absent.
"""

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPECS = {
    "wrist_d405": {
        "frame": "wrist_d405_color_optical_frame",
        "fov_deg": [87, 58],
        "source": "https://www.realsenseai.com/product-family/d405-series/",
    },
    "head_d435": {
        "frame": "head_d435_color_optical_frame",
        "fov_deg": [69, 42],
        "source": "https://www.realsenseai.com/products/stereo-depth-camera-d435/",
    },
}


def render(directory, out):
    out.mkdir(parents=True, exist_ok=False)
    frames = json.loads((directory / "preview_frames.json").read_text())
    width, height = 960, 540
    images, report = [], {}
    for name, spec in SPECS.items():
        scene = ET.parse(directory / "preview_scene.xml").getroot()
        world = scene.find("worldbody")
        # The CAD housings include opaque lens covers. Omit only the current
        # sensor's own housing to approximate rays leaving its physical lens.
        removed = []
        for geom in list(world.findall("geom")):
            if geom.get("name", "").startswith(name + "_link_visual_"):
                removed.append(geom.get("name"))
                world.remove(geom)
        assert len(removed) == 1
        transform = np.array(frames[spec["frame"]])
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.linalg.det(rotation) > 0.999999
        # ROS optical +X right,+Y down,+Z forward -> renderer +X right,+Y up,-Z forward.
        axes = np.concatenate([rotation[:, 0], -rotation[:, 1]])
        fx = width / (2 * math.tan(math.radians(spec["fov_deg"][0]) / 2))
        fy = height / (2 * math.tan(math.radians(spec["fov_deg"][1]) / 2))
        ET.SubElement(scene.find("visual"), "map", znear="0.001", zfar="100")
        ET.SubElement(
            world,
            "camera",
            name="sensor",
            pos=" ".join(map(str, transform[:3, 3])),
            xyaxes=" ".join(map(str, axes)),
            sensorsize="1 1",
            resolution=f"{width} {height}",
            focalpixel=f"{fx} {fy}",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Verify the renderer camera really uses the saved URDF frame.
        camera = model.camera("sensor").id
        np.testing.assert_allclose(data.cam_xpos[camera], transform[:3, 3], atol=1e-8)
        np.testing.assert_allclose(
            -data.cam_xmat[camera].reshape(3, 3)[:, 2], rotation[:, 2], atol=1e-8
        )
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera="sensor")
            rgb = renderer.render().copy()
        image = Image.fromarray(rgb)
        image.save(out / f"{name}_rgb.png")
        images.append(image)
        report[name] = {
            **spec,
            "resolution": [width, height],
            "fx_fy_pixels": [fx, fy],
            "world_from_optical": transform.tolist(),
            "self_housing_excluded": removed,
            "pose_and_optical_axis_checks": "passed",
        }
    board = Image.new("RGB", (1920, 640), "#eef0f4")
    draw = ImageDraw.Draw(board)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title, text = ImageFont.truetype(font, 24), ImageFont.truetype(font, 18)
    for i, (name, label) in enumerate(
        [("wrist_d405", "D405 | wrist camera"), ("head_d435", "D435 | head camera")]
    ):
        board.paste(images[i], (i * 960, 65))
        draw.text((i * 960 + 22, 12), label, fill="#172333", font=title)
        draw.text(
            (i * 960 + 22, 40),
            f"Nominal RGB FOV: {SPECS[name]['fov_deg']} degrees",
            fill="#43546a",
            font=text,
        )
    draw.text(
        (22, 613),
        "Current URDF pose and mounts | empty ground-plane scene | "
        "static pinhole render, not live sensor output",
        fill="#43546a",
        font=text,
    )
    board.save(out / "camera_views.png")
    (out / "capture.json").write_text(
        json.dumps(
            {
                "kind": "static geometric RGB preview",
                "limitations": "Nominal product FOV, no per-unit calibration or sensor noise. "
                "Own housing excluded.",
                "cameras": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(out / "camera_views.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.directory, args.out)
