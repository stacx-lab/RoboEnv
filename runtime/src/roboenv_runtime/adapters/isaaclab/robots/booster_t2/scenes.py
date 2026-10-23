"""Scenes: the house a task takes place in, shared by many tasks, and SIMPLE's objects that tasks
put in it.

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
                f"scene {self.name}: {path} is missing; run tools/prepare_booster_t2.py --download"
            )
        return path


def scene_names() -> list[str]:
    """Names of the scenes in `SCENE_DIR`, sorted."""
    return sorted(["household", "simple_household", *(p.name for p in SCENE_DIR.iterdir() if (p / "scene.yaml").is_file())])


def load_scene(name: str) -> Scene:
    """The scene in folder `name`, built from its `scene.yaml`.

    Raises:
        ValueError: No such scene.
    """
    if name in ("household", "simple_household"):
        from .household_models import SURFACES

        return Scene(name, {"hssd": "102344022" if name == "simple_household" else "authored-household-v1", "translation": [0., 0., 0.],
            "start": [-2., .7, float(np.pi / 2)], "surfaces": {
                k: {"center": list(xy), "size": list(size), "height": height}
                for k, (xy, size, height) in SURFACES.items()}})
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

@dataclass(frozen=True)
class JointDynamics:
    """Passive dynamics of one joint of an articulated object, from SIMPLE's MJCF."""

    damping: float
    """Viscous damping, N·m·s/rad (hinge) or N·s/m (slide)."""
    friction: float
    """Friction loss, N·m (hinge) or N (slide)."""
    armature: float
    """Armature, kg·m² (hinge) or kg (slide)."""


@dataclass(frozen=True)
class ArticulatedModel:
    """One of SIMPLE's articulated objects (`articulated:<id>`), as downloaded."""

    mjcf: Path
    """The MJCF SIMPLE simulates it with."""
    floating: bool
    """Whether it has a free base; otherwise its base is fixed in the world."""
    joints: dict[str, JointDynamics]
    """Hinge and slide joints by name."""
    bounds: np.ndarray
    """Bounding box of its geometry at rest in its model's frame (the MJCF's world frame): rows
    min, max. Shape (2, 3), m."""


def articulated_model(model: int, asset_directory: Path) -> ArticulatedModel:
    """SIMPLE's articulated object `model`, from the MJCF SIMPLE simulates it with.

    Raises:
        FileNotFoundError: It has not been downloaded.
    """
    import mujoco
    import numpy as np

    mjcf = asset_directory / "scenes" / "articulated" / f"{model:03d}" / "output_mjcf" / f"{model:03d}.xml"
    if not mjcf.is_file():
        raise FileNotFoundError(f"{mjcf} is missing; run tools/prepare_booster_t2.py --download")
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    kinds = [int(k) for k in m.jnt_type]
    joints = {}
    for j, kind in enumerate(kinds):
        if kind in (mujoco.mjtJoint.mjJNT_HINGE.value, mujoco.mjtJoint.mjJNT_SLIDE.value):
            dof = m.jnt_dofadr[j]
            joints[m.joint(j).name] = JointDynamics(
                float(m.dof_damping[dof]),
                float(m.dof_frictionloss[dof]),
                float(m.dof_armature[dof]),
            )
    points = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH.value:
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            points.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    points = np.concatenate(points)
    return ArticulatedModel(
        mjcf,
        mujoco.mjtJoint.mjJNT_FREE.value in kinds,
        joints,
        np.array([points.min(0), points.max(0)]),
    )


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
