"""Pinned HSSD scene layout, furniture supports and semantic collider bindings.

Each scene is a folder here with a `scene.yaml`: which of SIMPLE's HSSD houses it is, where it
sits in the world, the robot's default start and the furniture tops tasks put objects on. The
houses and objects are prepared by `tools/prepare_booster_t2.py --download`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
import numpy as np

from roboenv_runtime.adapters.isaaclab.robots.booster_t2 import assets

SCENE_DIR = Path(__file__).parent / "scenes"
"""Folder holding one subfolder per scene."""

DEFAULT_CACHE_DIR = assets.directory() / "scenes"
"""Where SIMPLE's downloads live: `hssd/<id>/` per house, `graspnet/` for the objects."""


@dataclass(frozen=True)
class Surface:
    """A furniture top with a semantic contact identity, in the world frame."""

    center: tuple[float, float]
    """Center of the top, x y, m."""
    size: tuple[float, float]
    """Extent along x and y, m."""
    height: float
    """Height of the top above the floor, m."""
    yaw: float = 0.0
    entity: str = ""

    def local(self, xy):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return (np.asarray(xy) - self.center) @ np.array([[c, -s], [s, c]])

    def contains(self, xy, margin: float = 0.0) -> bool:
        """Whether world point `xy` (m) lies over the top, at least `margin` (m) inside its
        edges."""
        return bool(np.all(np.abs(self.local(xy)) <= np.asarray(self.size) / 2 - margin))


class Scene:
    """One house.

    Args:
        name: The scene's folder name.
        spec: The contents of its `scene.yaml`.
    """

    def __init__(self, name: str, spec: dict) -> None:
        self.asset_directory = DEFAULT_CACHE_DIR
        self.name = name
        """The scene's folder name."""
        if not isinstance(spec["hssd"], str):
            raise TypeError(f"scene {name}: quote the hssd id, or YAML reads it as a number")
        self.hssd_id: str = spec["hssd"]
        """Id of the HSSD house, as in SIMPLE's `scenes_hssd_<id>.zip`."""
        self.translation: tuple[float, float, float] = tuple(spec["translation"])
        """World position of the house's origin once turned Z up, m."""
        self.start: tuple[float, float, float] = tuple(spec["start"])
        """Default robot start: x m, y m, yaw rad, world frame."""
        self.surfaces: dict[str, Surface] = {
            k: Surface(tuple(v["center"]), tuple(v["size"]), v["height"],
                       v.get("yaw", 0.0), v.get("entity", k))
            for k, v in spec.get("surfaces", {}).items()
        }
        """Named furniture tops tasks put objects on."""
        self.collider_bindings = dict(spec.get("collider_bindings", {}))
        self.cart_bay = spec.get("cart_bay")
        self.portal = spec.get("portal")
        self.excluded_prims = tuple(spec.get("excluded_prims", ()))
        self.replacements = spec.get("replacements", {})

    def collider_entity(self, path):
        """Resolve a full-house collider without confusing furniture with floor."""
        relative = str(path).removeprefix("/World/Scene/")
        for prefix, entity in self.collider_bindings.items():
            if relative == prefix or relative.startswith(prefix + "/"):
                return entity
        return "scene:" + "/".join(relative.split("/")[:2])

    @property
    def directory(self) -> Path:
        """Folder of the downloaded house."""
        return self.asset_directory / "hssd" / self.hssd_id

    @property
    def source(self) -> Path:
        """The house's USD as downloaded (Y up).

        Raises:
            FileNotFoundError: The house has not been downloaded.
        """
        path = self.directory / f"{self.hssd_id}.usd"
        if not path.is_file():
            raise FileNotFoundError(
                f"scene {self.name}: {path} is missing; run tools/prepare_booster_t2.py --download"
            )
        return path


def scene_names() -> list[str]:
    """Names of the scenes in `SCENE_DIR`, sorted."""
    return sorted(p.name for p in SCENE_DIR.iterdir() if (p / "scene.yaml").is_file())


def load_scene(name: str) -> Scene:
    """The scene in folder `name`, built from its `scene.yaml`.

    Raises:
        ValueError: No such scene.
    """
    if name not in scene_names():
        raise ValueError(f"no scene {name!r}; scenes: {', '.join(scene_names())}")
    return Scene(name, yaml.safe_load((SCENE_DIR / name / "scene.yaml").read_text()))


def house_usd(scene: Scene) -> Path:
    """The house's USD with its material and texture paths pointing into its own folder, matching
    file names regardless of case; written to the process-local conversion directory."""
    import hashlib
    from pxr import Sdf, UsdUtils
    from .assets import generated_directory

    folder = generated_directory() / hashlib.sha256(str(scene.directory).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "scene.usd"
    if path.is_file():
        return path
    files = {
        f.relative_to(scene.directory).as_posix().lower(): f for f in scene.directory.rglob("*")
    }

    def repair(asset: str) -> str:
        for folder in ("props/", "textures/"):
            if folder in asset:
                local = asset[asset.index(folder) :]
                match = files.get(local.lower())
                return str(match or (scene.directory / local))
        return asset

    layer = Sdf.Layer.OpenAsAnonymous(str(scene.source))
    UsdUtils.ModifyAssetPaths(layer, repair)
    layer.Export(str(path))
    return path
