"""Generate complete native LIBERO scene rigs without task objects or physics."""

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import yaml
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv.devices.export import export_devices, write_driver
from roboenv.tasks.libero import ROBOTS
from roboenv_runtime.adapters.libero.config import CAMERA_SIZE, ensure_libero_config
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]
META = {"created_at": "2026-09-07T00:00:00Z", "updated_at": "2026-09-08T00:00:00Z"}


def native_model(problem):
    """Extract constructor literals; use native camera and robot assembly methods."""
    import ast

    ensure_libero_config()
    from libero.libero.envs import TASK_MAPPING
    from robosuite.controllers import load_controller_config
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.tasks import ManipulationTask
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.utils.binding_utils import MjSim

    cls = TASK_MAPPING[problem]
    # Never instantiate the problem: its constructor samples task objects.
    tree = ast.parse(inspect.getsource(cls))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr.endswith("full_size"):
                    if isinstance(node.value, ast.Tuple):
                        defaults[target.attr] = ast.literal_eval(node.value)
    family = problem.removeprefix("libero_").removesuffix("_manipulation")
    robot_name = (
        "OnTheGroundPanda" if family in {"floor", "living_room_tabletop"} else "MountedPanda"
    )
    robot = ROBOT_CLASS_MAPPING[robot_name](
        robot_name,
        controller_config=load_controller_config(default_controller="JOINT_POSITION"),
        initialization_noise=None,
        control_freq=20,
    )
    robot.load_model()
    key = {
        "tabletop": "table",
        "floor": "empty",
        "kitchen_tabletop": "kitchen_table",
        "living_room_tabletop": "living_room_table",
        "study_tabletop": "study_table",
    }[family]
    offset = robot.robot_model.base_xpos_offset[key]
    if callable(offset):
        offset = offset(defaults[key + "_full_size"][0])
    robot.robot_model.set_base_xpos(offset)
    arena = EmptyArena()
    arena.set_origin([0, 0, 0])
    cls._setup_camera(None, arena)
    model = ManipulationTask(arena, [robot.robot_model])
    sim = MjSim.from_xml_string(model.get_xml())
    robot.reset_sim(sim)
    robot.setup_references()
    robot.reset(deterministic=True)
    sim.forward()
    return robot, sim


def canonical(value):
    if isinstance(value, float):
        return round(value, 12) or 0.0
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def binding_plan():
    return {
        "schema_version": "roboenv/bindings/1",
        "backend": "libero",
        "embodiment": "panda",
        "devices": {
            name: {
                "binding": "libero.device/1",
                "entity": entity,
                "profile": f"devices/{name}/profile.yaml",
                **extra,
            }
            for name, entity, extra in [
                ("arm", "arm", {}),
                ("gripper", "gripper", {}),
                ("camera", "camera", {"config": {"camera": "agentview"}}),
                ("camera_wrist", "camera", {"config": {"camera": "robot0_eye_in_hand"}}),
                ("world", "world", {}),
            ]
        },
    }


def descriptors(problem):
    robot, sim = native_model(problem)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = binding_plan()
            (root / "bindings.yaml").write_text(yaml.safe_dump(plan))
            for name, entry in plan["devices"].items():
                type_ = {
                    "arm": "joint",
                    "gripper": "gripper",
                    "camera": "rgb",
                    "world": "world",
                }[entry["entity"]]
                path = root / entry["profile"]
                path.parent.mkdir(parents=True)
                path.write_text(yaml.safe_dump({"name": name, "type": type_}))
            native = SimpleNamespace(
                _robot=robot,
                sim=sim,
                _render=True,
                _camera_names=("agentview", "robot0_eye_in_hand"),
                _camera_size=CAMERA_SIZE,
            )
            context = BindingContext("libero", "panda", SimpleNamespace(_env=native))
            parsed = BindingPlan.load(root, backend="libero")
            before = default_registry().assemble(parsed, root, context).inventory()["devices"]
            sim.data.qpos[robot._ref_joint_pos_indexes] += 0.01
            sim.forward()
            after = default_registry().assemble(parsed, root, context).inventory()["devices"]
            for first, second in zip(before, after, strict=True):
                assert canonical(first["intrinsics"]) == canonical(second["intrinsics"])
                assert canonical(first["extrinsics"]) == canonical(second["extrinsics"])
            assert sim.data.time == 0
            return before
    finally:
        sim.free()


def generate(out, provenance=None, *, bindings_out=None):
    for problem, rig in ROBOTS.items():
        facts = {d["name"]: canonical(d) for d in descriptors(problem)}
        destination = out / "robots" / rig
        destination.mkdir(parents=True, exist_ok=True)
        workspace = {
            "identity": {
                "source": "user",
                "id": destination.name,
                "name": destination.name,
                "platform": "sim",
                "kind": "arm",
                "model": "Native LIBERO Panda and PandaGripper",
            },
            "sim": {"source": "user", "simulator": "libero"},
            "devices": {"source": "user"},
            "safety": {
                "source": "user",
                "max_cartesian_velocity": None,
                "max_joint_velocity": None,
                "keep_out_zones": [],
                "notes": "No additional velocity caps. Native collisions and joint limits "
                "apply. Stop holds its device; it does not pause the world. "
                "Use camera observations to plan and verify motions.",
            },
            "meta": META,
        }
        (destination / "workspace.yaml").write_text(yaml.safe_dump(workspace, sort_keys=False))
        with tempfile.TemporaryDirectory() as temporary:
            export_devices(list(facts.values()), temporary, {"endpoint": "http://127.0.0.1:8123"})
            for path in sorted(Path(temporary).glob("*/profile.yaml")):
                profile = canonical(yaml.safe_load(path.read_text()))
                profile["meta"] = META
                encoded = yaml.safe_dump(profile, sort_keys=False)
                if "???" in encoded:
                    raise ValueError("Unresolved native catalog facts")
                device = destination / "devices" / path.parent.name
                device.mkdir(parents=True, exist_ok=True)
                (device / "profile.yaml").write_text(encoded)
                if profile["type"] != "world":
                    write_driver(device, profile["type"])
        if bindings_out is not None:
            binding = bindings_out / f"{rig}.yaml"
            binding.parent.mkdir(parents=True, exist_ok=True)
            binding.write_text(yaml.safe_dump(binding_plan(), sort_keys=False))
    if provenance:
        import robosuite

        native = Path(robosuite.__file__).parent
        source_paths = [
            ROOT / "tools/generate_libero_config.py",
            ROOT / "runtime/src/roboenv_runtime/bindings/backends/libero/devices.py",
            ROOT / "runtime/src/roboenv_runtime/adapters/robocasa/geometry.py",
            ROOT / "src/roboenv/assets/_roboenv.py",
            ROOT / "src/roboenv/assets/joint-client.py",
            ROOT / "src/roboenv/assets/gripper-client.py",
            ROOT / "src/roboenv/assets/camera-client.py",
        ]
        source_paths += sorted((ROOT / "third_party/LIBERO/libero/libero/envs/robots").glob("*.py"))
        source_paths += sorted(
            (ROOT / "third_party/LIBERO/libero/libero/envs/problems").glob("*.py")
        )
        source_paths += [ROOT / "third_party/LIBERO/libero/libero/envs/bddl_base_domain.py"]
        hashes = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        }
        for rel in [
            "models/assets/robots/panda/robot.xml",
            "models/assets/grippers/panda_gripper.xml",
            "models/assets/mounts/rethink_mount.xml",
            "models/grippers/panda_gripper.py",
            "controllers/config/joint_position.json",
            "controllers/joint_pos.py",
        ]:
            hashes["robosuite-1.4.0/" + rel] = hashlib.sha256(
                (native / rel).read_bytes()
            ).hexdigest()
        provenance.write_text(
            json.dumps(
                {
                    "generator": "tools/generate_libero_config.py",
                    "sources": hashes,
                    "method": "CPU native model assembly; no task objects, sampling, "
                    "renderer, or physics",
                    "camera_size": list(CAMERA_SIZE),
                    "control_frequency_hz": 20,
                    "robots": ROBOTS,
                    "robosuite_version": robosuite.__version__,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "catalog")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--bindings-out", type=Path, help="Separate internal binding resource root")
    args = parser.parse_args()
    generate(args.out, args.provenance, bindings_out=args.bindings_out)
