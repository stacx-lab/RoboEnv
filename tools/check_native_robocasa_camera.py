"""Rendered RGB-D calibration regression; run in the isolated RoboCasa interpreter."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml
from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv
from roboenv_runtime.adapters.robocasa.depth import write_depth
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def check(out):
    from roboenv.provisioning.gpu import configure_gpu

    configure_gpu("robocasa")
    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

    out.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ("camera_left", "camera_right", "camera_wrist"):
        path = ROOT / "catalog/robots/robocasa/robocasa-panda-omron/devices" / name
        k = yaml.safe_load((path / "profile.yaml").read_text())["intrinsics"]["color"]["projection"]
        fovy = np.rad2deg(2 * np.arctan(128 / k["fy"]))
        for distance in (0.3, 1.5):
            for axis in (0, 1):
                normal = np.array([0.0, 0.0, -1.0])
                normal[axis] = 0.4
                normal /= np.linalg.norm(normal)
                q = Rotation.align_vectors(normal[None, :], [[0, 0, 1]])[0].as_quat()
                quat = " ".join(map(str, [q[3], *q[:3]]))
                xml = f'''<mujoco>
                  <visual><quality offsamples="4"/><global offwidth="256" offheight="256"/>
                  <map znear="0.01" zfar="10"/></visual><statistic extent="1"/>
                  <worldbody><camera name="audit" quat="0 1 0 0" fovy="{fovy}"/>
                  <geom type="plane" pos="0 0 {distance}" quat="{quat}"
                    size="3 3 .01" rgba=".8 .4 .2 1"/></worldbody></mujoco>'''
                sim = MjSim.from_xml_string(xml)
                try:
                    MjRenderContextOffscreen(sim, device_id=0)
                    sim.forward()
                    world = RoboCasaEnv.__new__(RoboCasaEnv)
                    world.env = SimpleNamespace(sim=sim)
                    world._render, world._camera_names = True, ("audit",)
                    world._camera_size = (256, 256)
                    before = sim.get_state().flatten().copy()
                    _, z = world.render_rgbd("audit")
                    np.testing.assert_array_equal(sim.get_state().flatten(), before)
                    assert sim.model._model.vis.quality.offsamples == 0
                    v, u = np.indices(z.shape)
                    mask = (u > 8) & (u < 247) & (v > 8) & (v < 247) & (z > 0)
                    expected = normal[2] * distance / (
                        normal[0] * (u - k["cx"]) / k["fx"]
                        + normal[1] * (v - k["cy"]) / k["fy"] + normal[2]
                    )
                    file = out / f"{name}-{distance}-{axis}.pgm"
                    write_depth(file, z)
                    raster = file.read_bytes().split(b"65535\n", 1)[1]
                    encoded = np.frombuffer(raster, dtype=">u2").reshape(z.shape) * 0.001
                    error = float(np.max(np.abs(z[mask] - expected[mask])))
                    encoded_error = float(np.max(np.abs(encoded[mask] - expected[mask])))
                    assert error < 3e-5, error
                    assert encoded_error < 0.00055, encoded_error
                    results.append({
                        "camera": name, "distance_m": distance, "axis": axis,
                        "depth_error_m": error, "encoded_error_m": encoded_error,
                    })
                finally:
                    sim.free()
    report = {"passed": True, "cases": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    check(parser.parse_args().out)
