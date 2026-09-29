"""Verify bundled official LIBERO assets and configure the registered interpreter."""

import hashlib
import json
import subprocess

from roboenv.provisioning.interpreter import resolve_interpreter
from roboenv.provisioning.progress import stage
from roboenv.tasks.libero import entries


def prepare_assets(root, progress=None):
    interpreter = resolve_interpreter(root, "libero")
    base = root / "third_party/LIBERO/libero/libero"
    with stage(progress, "Checking official BDDL and fixed-state assets"):
        tasks = list(entries(base))
        if len(tasks) != 130:
            raise ValueError("Official LIBERO must contain 130 standard task entries")
        if not any((base / "assets").rglob("*.xml")):
            raise ValueError("LIBERO model assets are missing")
        config = root / ".cache/roboenv/libero-config"
        subprocess.run(
            [
                str(interpreter),
                "-I",
                "-c",
                "from roboenv_runtime.adapters.libero import ensure_libero_config; "
                "import sys; ensure_libero_config(sys.argv[1])",
                str(config),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        receipt = {
            "schema": "roboenv/libero-assets/1",
            "tasks": len(tasks),
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for t in tasks
                for p in (t["bddl"], t["states"])
            },
        }
        output = root / ".cache/roboenv/libero-assets.json"
        output.write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        "environment_assets_verified": True,
        "asset_receipt": str(output.relative_to(root)),
        "config_directory": str(config.relative_to(root)),
        "gpu_replay_verified": False,
    }
