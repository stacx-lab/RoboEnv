"""Locations of independently prepared, checksum-verified T2 assets."""

import os
from pathlib import Path


def directory():
    from roboenv.provisioning.booster_t2_assets import asset_root
    from roboenv_runtime.paths import repo_root

    return asset_root(repo_root())


def generated_directory():
    import tempfile

    # Each worker owns its generated USDs; concurrent worlds never edit each other's files.
    return Path(tempfile.gettempdir()) / "roboenv-t2" / str(os.getpid())
