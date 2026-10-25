"""Local Revo2 grasp-center FK, including the URDF's coupled distal joints."""

import math
from functools import cache
from pathlib import Path
from xml.etree import ElementTree

from kinex.geometry import compose_rt, invert_rt, xyzq_to_rt
from kinex.urdf import numbers, rpy_quaternion


@cache
def joints(path):
    root = ElementTree.parse(path).getroot()
    return {j.find("child").get("link"): j for j in root.findall("joint")}


def grasp_center(profile, directory, positions):
    side = profile["name"].split("_", 1)[0]
    values = dict(zip((j["name"] for j in profile["intrinsics"]["model"]["joints"]),
                      positions, strict=True))
    tree = joints(str(Path(directory) / "model.urdf"))

    def transform(link):
        if link not in tree:
            return xyzq_to_rt([0, 0, 0, 0, 0, 0, 1])
        joint = tree[link]
        origin = joint.find("origin")
        pose = numbers(origin, "xyz", "0 0 0") + rpy_quaternion(*numbers(origin, "rpy", "0 0 0"))
        result = compose_rt(transform(joint.find("parent").get("link")), xyzq_to_rt(pose))
        if joint.get("type") == "fixed":
            return result
        mimic = joint.find("mimic")
        angle = (values.get(joint.get("name"), 0.) if mimic is None else
                 values.get(mimic.get("joint"), 0.) * float(mimic.get("multiplier", "1"))
                 + float(mimic.get("offset", "0")))
        axis = numbers(joint.find("axis"), "xyz", "1 0 0")
        factor = math.sin(angle / 2) / math.hypot(*axis)
        rotation = [v * factor for v in axis] + [math.cos(angle / 2)]
        return compose_rt(result, xyzq_to_rt([0, 0, 0, *rotation]))

    palm_inverse = invert_rt(transform(side + "_palm"))
    points = [(weight, compose_rt(palm_inverse, transform(f"{side}_{finger}_touch_link"))[1])
              for finger, weight in (("thumb", 0.5), ("index", 0.25), ("middle", 0.25))]
    return [sum(w * p[i] for w, p in points) for i in range(3)] + [0, 0, 0, 1]
