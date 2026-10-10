"""Owner-only BEHAVIOR first-frame delivery probe; no model or task-success claim."""

import argparse
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from check_native_behavior import CalibrationWorld
from roboenv_runtime.adapters.behavior.devices import pose
from roboenv_runtime.adapters.urdf_geometry import matrix_pose
from roboenv_runtime.runtime.worker import write_png
from roboenv_runtime.spectator import ViewSpec
from roboenv_runtime.spectator.isaac import Engine


def centroid(rgb):
    red, green, blue = rgb.astype(float).transpose(2, 0, 1)
    y, x = np.where((red > 80) & (blue > 80) & (green < .6 * np.minimum(red, blue)))
    return np.array([x.mean(), y.mean()]) if len(x) > 20 else None


def check(world, out, report):
    import carb.settings
    import omnigibson as og
    import torch

    report.update(initialization=world.initialization, implementation=world.implementation_identity)
    report["settings"] = {key: carb.settings.get_settings().get(key) for key in (
        "/app/renderer/waitIdle", "/app/hydraEngine/waitIdle",
        "/app/updateOrder/checkForHydraRenderComplete", "/app/asyncRendering",
    )}
    engine = Engine(SimpleNamespace(_env=world, name="behavior"))
    views = [ViewSpec(role, source=role, width=640, height=480) for role in engine.sources]
    engine.open(views)
    try:
        # Warm new render products only; never warm after a measured scene change.
        for _ in range(10):
            engine.capture(views)
        for name, sensor_name in world.cameras.items():
            sensor = world.robot.sensors[sensor_name]
            role = next(role for role, source in engine.sources.items() if source.native is sensor)
            view = next(view for view in views if view.name == role)
            marker = world.fixture(name)
            transform = pose(sensor) @ np.diag([1, -1, -1, 1])
            for kind in ("agent", "spectator"):
                positions = (-.035, .035, -.035) if kind == "agent" else (.035, -.035, .035)
                previous_reference = None
                for index, x in enumerate(positions):
                    xyz = np.array([x, 0, .18])
                    marker.set_position_orientation(
                        position=torch.tensor((transform @ np.r_[xyz, 1])[:3]),
                        orientation=torch.tensor(matrix_pose(transform)[3:]),
                    )
                    before = world.owner_state()["sha256"]
                    started = time.monotonic()
                    rgb = (world.device_image(name)[0] if kind == "agent"
                           else engine.capture([view])[role])
                    seconds = time.monotonic() - started
                    first = centroid(rgb)
                    for _ in range(8):
                        og.sim.render()
                    reference = (world.device_image(name)[0] if kind == "agent"
                                 else engine.capture([view])[role])
                    expected = centroid(reference)
                    invariant = world.owner_state()["sha256"] == before
                    difference = (None if first is None or expected is None
                                  else float(np.max(np.abs(first - expected))))
                    # A static background must not masquerade as our moving fixture.
                    moved = (expected is not None and (previous_reference is None
                             or abs(expected[0] - previous_reference[0]) > 50))
                    sample = {"camera": name, "kind": kind, "x_m": x,
                              "first_centroid": None if first is None else first.tolist(),
                              "reference_centroid": None if expected is None else expected.tolist(),
                              "error_px": difference, "read_s": seconds,
                              "physics_unchanged": invariant,
                              "passed": bool(invariant and moved and difference is not None
                                             and difference < 2)}
                    previous_reference = expected
                    report["samples"].append(sample)
                    prefix = f"{name}-{kind}-{index}"
                    write_png(out / f"{prefix}-first.png", rgb)
                    write_png(out / f"{prefix}-reference.png", reference)
                    print(json.dumps(sample), flush=True)
                    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            world.park(name)
            for _ in range(8):
                og.sim.render()
        report["passed"] = len(report["samples"]) == 18 and all(
            sample["passed"] for sample in report["samples"]
        )
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "scope": "native camera delivery; no task success or replay claim",
              "samples": []}
    world = None
    try:
        world = CalibrationWorld("turning_on_radio", "house_double_floor_lower",
                                 args.instance, max_steps=100)
        world.reset(seed=args.seed)
        check(world, out, report)
        if not report["passed"]:
            raise RuntimeError("BEHAVIOR first-frame camera delivery failed")
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if world is not None:
            world.close()
        import omnigibson as og

        og.shutdown()


if __name__ == "__main__":
    main()
