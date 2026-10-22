"""Scenes: the house a task takes place in, shared by many tasks, and SIMPLE's objects that tasks
put in it.

Each scene is a folder here with a `scene.yaml`: which of SIMPLE's HSSD houses it is, where it
sits in the world, the robot's default start and the furniture tops tasks put objects on. The
houses and the objects are downloaded into `CACHE_DIR` by `tools/prepare_booster_t2.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from roboenv_runtime.adapters.isaaclab.robots.booster_t2 import assets

SCENE_DIR = Path(__file__).parent / "scenes"
"""Folder holding one subfolder per scene."""

DEFAULT_CACHE_DIR = assets.directory() / "scenes"
"""Where SIMPLE's downloads live: `hssd/<id>/` per house, `graspnet/` for the objects."""


@dataclass(frozen=True)
class Surface:
    """A furniture top, axis aligned, world frame."""

    center: tuple[float, float]
    """Center of the top, x y, m."""
    size: tuple[float, float]
    """Extent along x and y, m."""
    height: float
    """Height of the top above the floor, m."""

    def contains(self, xy, margin: float = 0.0) -> bool:
        """Whether world point `xy` (m) lies over the top, at least `margin` (m) inside its
        edges."""
        return all(abs(p - c) <= s / 2 - margin for p, c, s in zip(xy, self.center, self.size))


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
            k: Surface(tuple(v["center"]), tuple(v["size"]), v["height"])
            for k, v in spec.get("surfaces", {}).items()
        }
        """Named furniture tops tasks put objects on."""

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
                f"scene {self.name}: {path} is missing; run tools/prepare_booster_t2.py {self.name}"
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


def graspnet_usd(model: int, asset_directory: Path) -> Path:
    """SIMPLE's USD of GraspNet-1B object `model`, as downloaded.

    Raises:
        FileNotFoundError: It has not been downloaded.
    """
    path = asset_directory / "scenes" / "graspnet" / f"{model:03d}_ruled.usd"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run tools/prepare_booster_t2.py")
    return path
