"""The same seed builds the same RoboCasa kitchen, within and across processes. No GPU.

Needs the prepared RoboCasa interpreter, this checkout's synced RoboCasa source and its
kitchen asset bundles; skipped otherwise.
"""

import json
import os
import subprocess

import pytest
from helpers import ROOT, native_interpreter
from roboenv.provisioning.robocasa_assets import ARCHIVES

SOURCE = ROOT / "third_party/robocasa"
ASSETS = SOURCE / "robocasa/models/assets"
PROCESSES, BUILDS = 2, 3


def prepared_kitchen():
    interpreter = native_interpreter()
    if not (SOURCE / "robocasa/__init__.py").is_file():
        pytest.skip("Requires the synced RoboCasa source (roboenv sources sync)")
    missing = [
        relative
        for relative in ARCHIVES.values()
        if not (ASSETS / relative).is_dir() or not any((ASSETS / relative).iterdir())
    ]
    if missing:
        pytest.skip(f"Requires the prepared RoboCasa kitchen assets: {missing}")
    return interpreter


# Counter-top regions left and right of the sink used to come out of a set of XML
# elements, i.e. in memory-address order: this kitchen, seed 0 changed layout of its
# objects (and the burner in its instruction) from one build or launch to the next.
@pytest.mark.parametrize("task, seed", [("MultistepSteaming", 0)])
def test_same_seed_builds_the_same_kitchen(task, seed):
    interpreter = prepared_kitchen()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            str(path) for path in (ROOT / "src", ROOT / "runtime/src", SOURCE)
        ),
    }
    command = [interpreter, str(ROOT / "tests/robocasa_kitchen_cpu.py"), task, str(seed)]
    runs = [
        subprocess.Popen(
            [*command, str(BUILDS)],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(PROCESSES)
    ]
    scenes = []
    try:
        for run in runs:
            stdout, stderr = run.communicate(timeout=600)
            assert run.returncode == 0, stdout + stderr
            report = json.loads(stdout.splitlines()[-1])
            assert report["robocasa"] == str((SOURCE / "robocasa").resolve())
            scenes += report["scenes"]
    finally:
        for run in runs:
            run.kill()
            run.wait()
    assert len(scenes) == PROCESSES * BUILDS
    assert scenes[0]["poses"]
    differing = [i for i, other in enumerate(scenes) if other != scenes[0]]
    assert not differing, (
        f"builds {differing} of {len(scenes)} differ from build 0 at seed {seed}: "
        + json.dumps([scenes[0]] + [scenes[i] for i in differing])
    )
