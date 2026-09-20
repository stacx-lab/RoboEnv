"""Render a static URDF geometry preview using FK and MuJoCo's renderer.

Requires numpy, trimesh, pycollada, mujoco, pillow. No physics is stepped.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from trimesh.transformations import euler_matrix, rotation_matrix


def origin(elem):
    t = np.eye(4)
    if elem is not None:
        t = euler_matrix(*map(float, elem.get("rpy", "0 0 0").split()))
        t[:3, 3] = list(map(float, elem.get("xyz", "0 0 0").split()))
    return t


def render(directory):
    robot = ET.parse(directory / "go2_d1.urdf").getroot()
    pose = json.loads((directory / "preview_pose.json").read_text())
    transforms = {"base": np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        count = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            transform = origin(joint.find("origin"))
            q = pose.get(joint.get("name"), 0)
            axis = joint.find("axis")
            axis = (
                np.array(list(map(float, axis.get("xyz", "1 0 0").split())))
                if axis is not None
                else np.array([1, 0, 0])
            )
            if joint.get("type") in ("revolute", "continuous"):
                transform = transform @ rotation_matrix(q, axis)
            elif joint.get("type") == "prismatic":
                translation = np.eye(4)
                translation[:3, 3] = axis * q
                transform = transform @ translation
            transforms[joint.find("child").get("link")] = transforms[parent] @ transform
            pending.remove(joint)
        assert len(pending) < count
    meshes = []
    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            mesh = geom.find("mesh")
            if mesh is not None:
                loaded = trimesh.load(directory / mesh.get("filename"), force="scene")
                model = loaded.to_geometry()
                model.apply_scale(list(map(float, mesh.get("scale", "1 1 1").split())))
            elif geom.find("sphere") is not None:
                model = trimesh.creation.icosphere(
                    subdivisions=2, radius=float(geom.find("sphere").get("radius"))
                )
            elif geom.find("box") is not None:
                model = trimesh.creation.box(
                    extents=list(map(float, geom.find("box").get("size").split()))
                )
            elif geom.find("cylinder") is not None:
                cylinder = geom.find("cylinder")
                model = trimesh.creation.cylinder(
                    radius=float(cylinder.get("radius")), height=float(cylinder.get("length"))
                )
            else:
                raise ValueError("Unsupported visual geometry")
            model.apply_transform(transforms[link.get("name")] @ origin(visual.find("origin")))
            meshes.append((link.get("name"), model))
    floor = min(m.bounds[0, 2] for _, m in meshes)
    world_frames = {}
    for name, transform in transforms.items():
        value = transform.copy()
        value[:3, 3] += [0, 0, -floor + 0.004]
        world_frames[name] = value.tolist()
    (directory / "preview_frames.json").write_text(json.dumps(world_frames, indent=2) + "\n")
    scene = ET.Element("mujoco", model="go2_d1_static_preview")
    ET.SubElement(scene, "compiler", angle="radian")
    ET.SubElement(scene, "statistic", extent="1.2", center="0 0 0.4")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1100")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(
        visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.65 0.65 0.65", specular="0.2 0.2 0.2"
    )
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.72 0.78 0.86",
        rgb2="0.94 0.95 0.97",
        width="512",
        height="3072",
    )
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        world, "light", pos="1 -2 3", dir="-0.3 0.4 -1", diffuse="0.8 0.8 0.8", directional="true"
    )
    ET.SubElement(world, "geom", type="plane", size="200 200 0.1", rgba="0.65 0.68 0.73 1")
    render_dir = directory / "preview_meshes"
    render_dir.mkdir(exist_ok=True)
    for i, (name, mesh) in enumerate(meshes):
        mesh.apply_translation([0, 0, -floor + 0.004])
        suffix = "obj" if len(mesh.faces) > 200_000 else "stl"
        filename = render_dir / f"{i}.{suffix}"
        mesh.export(filename)
        ET.SubElement(asset, "mesh", name=f"m{i}", file=str(filename.resolve()))
        # Neutral presentation colors; geometry is from the URDF source meshes.
        color = "0.72 0.75 0.79 1" if name.startswith("d1_") else "0.17 0.20 0.24 1"
        if "foot" in name or "Link7" in name:
            color = "0.08 0.09 0.11 1"
        if name == "wrist_d405_link":
            color = "0.85 0.38 0.08 1"
        elif name == "head_d435_link":
            color = "0.08 0.45 0.75 1"
        ET.SubElement(
            world,
            "geom",
            name=f"{name}_visual_{i}",
            type="mesh",
            mesh=f"m{i}",
            rgba=color,
            contype="0",
            conaffinity="0",
        )
    ET.ElementTree(scene).write(directory / "preview_scene.xml")
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames = []
    cameras_added = "wrist_d405_link" in transforms
    views = [
        ("Perspective", 135, -17, [0.04, 0, 0.43], 1.8),
        ("Side", 90, -8, [0.04, 0, 0.43], 1.8),
    ]
    if cameras_added:
        for label, link, distance in [
            ("Wrist", "wrist_d405_link", 0.38),
            ("Head", "head_d435_link", 0.48),
        ]:
            target = transforms[link][:3, 3] + [0, 0, -floor + 0.004]
            views.append((label, 145, -20, target, distance))
    with mujoco.Renderer(model, height=1100, width=1600) as renderer:
        for label, azimuth, elevation, target, distance in views:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = target
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera)
            image = Image.fromarray(renderer.render())
            image.save(directory / f"preview_{label.lower()}.png")
            frames.append(image.resize((960, 660)))
    board = Image.new("RGB", (1920, 1455 if cameras_added else 755), "#eef0f4")
    board.paste(frames[0], (0, 75))
    board.paste(frames[1], (960, 75))
    if cameras_added:
        board.paste(frames[2], (0, 775))
        board.paste(frames[3], (960, 775))
    draw = ImageDraw.Draw(board)
    from PIL import ImageFont

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title = ImageFont.truetype(font_path, 28)
    body = ImageFont.truetype(font_path, 18)
    heading = "Go2 + Unitree D1-550 | " + (
        "RealSense D405 wrist + D435 head" if cameras_added else "Robot geometry preview"
    )
    draw.text((30, 12), heading, fill="#172333", font=title)
    draw.text(
        (30, 48),
        "Official camera meshes | orange: D405 | blue: D435 | provisional mounting brackets"
        if cameras_added
        else "Inherited mount: x = 85 mm, z = 94 mm | illustrative joint pose | static mesh render",
        fill="#43546a",
        font=body,
    )
    draw.text((30, 727), "Perspective", fill="#172333", font=body)
    draw.text((990, 727), "Side view | +X (front) points right", fill="#172333", font=body)
    if cameras_added:
        draw.text(
            (30, 1427),
            "D405 | attached to final wrist; looks toward grasp",
            fill="#172333",
            font=body,
        )
        draw.text(
            (990, 1427), "D435 | mounted above head; looks forward", fill="#172333", font=body
        )
    board.save(directory / "go2_d1_preview.png")
    print(directory / "go2_d1_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
