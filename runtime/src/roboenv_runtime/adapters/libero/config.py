"""Noninteractive paths for the installed official LIBERO source and assets."""

import importlib.util
import os
from pathlib import Path

import yaml


def ensure_libero_config(config_dir=None):
    # Do not import libero.libero: its module body prompts before paths exist.
    namespace = importlib.util.find_spec("libero")
    if namespace is None or not namespace.submodule_search_locations:
        raise RuntimeError("Official LIBERO is not installed in the selected interpreter")
    roots = [Path(p) / "libero" for p in namespace.submodule_search_locations]
    roots = [p for p in roots if (p / "benchmark/libero_suite_task_map.py").is_file()]
    if len(roots) != 1:
        raise RuntimeError("Expected one installed official LIBERO package")
    root = roots[0].resolve()
    directory = (
        Path(
            config_dir
            or os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".cache/roboenv/libero")
        )
        .expanduser()
        .resolve()
    )
    wanted = {
        k: str(root / rel)
        for k, rel in {
            "benchmark_root": ".",
            "bddl_files": "bddl_files",
            "init_states": "init_files",
            "assets": "assets",
        }.items()
    }
    if any(not Path(p).is_dir() for p in wanted.values()):
        raise RuntimeError("Official LIBERO source is missing its BDDL, states, or assets")
    directory.mkdir(parents=True, exist_ok=True)
    wanted["datasets"] = str(directory / "datasets")
    Path(wanted["datasets"]).mkdir(exist_ok=True)
    file = directory / "config.yaml"
    encoded = yaml.safe_dump(wanted, sort_keys=True)
    if not file.exists() or file.read_text() != encoded:
        # A per-process temporary name also permits concurrent world startup.
        temporary = directory / f"config.{os.getpid()}.tmp"
        temporary.write_text(encoded)
        temporary.replace(file)
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    return directory
