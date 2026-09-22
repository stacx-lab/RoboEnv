"""Compare real LIBERO camera pixels over HTTP with owner-thread rendering."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import roboenv
from PIL import Image
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.runtime.worker import write_png

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {"camera": "agentview", "camera_wrist": "robot0_eye_in_hand"}


def check(out, realtime):
    out.mkdir(parents=True, exist_ok=True)
    adapter = LiberoAdapter("libero_90", 8, realtime=False)
    try:
        adapter.reset(0)
        reference = {name: adapter._env.render(camera) for name, camera in CAMERAS.items()}
        for name, pixels in reference.items():
            write_png(out / f"reference-{name}.png", pixels)
    finally:
        adapter.close()

    errors = []
    with roboenv.launch(
        "libero/drawer-and-bowl", project=ROOT, interpreter=sys.executable,
        realtime=realtime, port=0, run_dir=out / "runs",
    ) as run, roboenv.connect(run.endpoint) as env:
        def read(name, index):
            result = env.device(name).read()
            with Image.open(result["file"]) as image:
                pixels = np.asarray(image).copy()
            assert pixels.shape == reference[name].shape == (256, 256, 3)
            difference = np.abs(pixels.astype(float) - reference[name])
            error = float(difference.mean())
            if realtime:
                # Physics may settle between boots; random framebuffer bytes
                # cannot resemble the scene rendered directly on its owner.
                assert error < 10, (name, error)
            else:
                np.testing.assert_array_equal(pixels, reference[name])
            write_png(out / f"rpc-{index}-{name}.png", pixels)
            return error

        # Warm the client download directory before sharing it across requests.
        errors.extend(read(name, "initial") for name in CAMERAS)
        with ThreadPoolExecutor(max_workers=5) as pool:
            writer = None
            if realtime:
                target = env.device("arm").read()["positions"]
                writer = pool.submit(
                    run.client.device("arm").write, target, for_s=2,
                    request_id="camera-overlap",
                )
                deadline = time.monotonic() + 5
                while run.client.operation("camera-overlap")["state"] != "running":
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
            reads = [pool.submit(read, name, index) for index in range(4) for name in CAMERAS]
            errors.extend(future.result(timeout=10) for future in reads)
            if writer is not None:
                assert not writer.done(), "Camera reads waited for the bounded write"
                assert writer.result(timeout=10)["data"]["elapsed_s"] >= 2
        report = {
            "passed": True, "realtime": realtime, "camera_reads": len(errors),
            "max_mean_absolute_pixel_error": max(errors),
            "model_calls": 0, "task_success_claimed": False,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    check(args.out, args.realtime)
