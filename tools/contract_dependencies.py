"""Stage pinned device-service dependencies without changing a prepared interpreter."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "runtime/locks/device-service.txt"


def prepare_contract_dependencies(interpreter, destination):
    """Return a portable receipt after installing and probing a new target directory.

    The supplied interpreter determines the wheel ABI. It is never an installation
    destination. Do not reuse this target with a different Python minor version.
    The caller owns image staging and must add the target to the runtime import path.
    """
    interpreter = Path(interpreter).absolute()
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("contract dependency destination must be a new directory")
    if not interpreter.is_file() or not shutil.which("uv"):
        raise ValueError("an existing Python interpreter and uv are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    environment.update(PYTHONNOUSERSITE="1", UV_CACHE_DIR=str(ROOT / ".cache/uv"))
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(interpreter),
        "--target",
        str(destination),
        "--no-deps",
        "--no-compile-bytecode",
        "-r",
        str(LOCK),
    ]
    # uv output can contain interpreter/cache locations; do not publish it.
    try:
        subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
        probe = subprocess.run(
            [
                str(interpreter),
                "-I",
                "-c",
                (
                    "import sys, json, pathlib, importlib.metadata as metadata; "
                    "target=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(target)); "
                    "import jsonschema, jsonschema_specifications, referencing, rpds; "
                    "from jsonschema import Draft202012Validator; "
                    "schema={'type':'object','properties':{'value':{'type':'integer'}},"
                    "'required':['value'],'additionalProperties':False}; "
                    "Draft202012Validator.check_schema(schema); "
                    "assert Draft202012Validator(schema).is_valid({'value':1}); "
                    "assert not Draft202012Validator(schema).is_valid({'value':'invalid'}); "
                    "assert pathlib.Path(jsonschema.__file__).resolve().is_relative_to(target); "
                    "assert pathlib.Path(rpds.__file__).resolve().is_relative_to(target); "
                    "names=['attrs','jsonschema','jsonschema-specifications','referencing',"
                    "'rpds-py','typing-extensions']; "
                    "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:2])),"
                    "'packages':{name:metadata.version(name) for name in names}}))"
                ),
                str(destination),
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "contract dependency staging/probe failed; target is not certified"
        ) from error
    result = {
        "schema": "roboenv/contract-dependencies/1",
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        **json.loads(probe.stdout),
        "schema_validation_verified": True,
    }
    (destination / "_roboenv_contract_dependencies.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
