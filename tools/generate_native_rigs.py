"""Deterministically generate complete source-backed device catalogs without tasks."""

import argparse
from pathlib import Path

import yaml
from roboenv.devices.export import driver_access, write_driver

ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP = "2026-09-09T00:00:00Z"


def write_rig(root, robot, backend, embodiment, facts):
    catalog = Path(root)
    robot_path = f"{backend}/{robot}" if backend in {"robotwin", "robodojo"} else robot
    directory = catalog / "robots" / robot_path
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workspace.yaml").write_text(yaml.safe_dump({
        "identity": {"source": "user", "id": robot, "name": robot, "platform": "sim",
                     "kind": "mobile" if backend == "behavior" else "other", "model": robot},
        "sim": {"source": "user", "simulator": backend},
        "devices": {"source": "user"},
        "safety": {"source": "user", "max_cartesian_velocity": None,
                   "max_joint_velocity": None, "keep_out_zones": [],
                   "notes": "Native joint limits and collisions apply. Stop holds its device. "
                            "Coordinated writes may share step: {id: N, devices: [...]}; "
                            "all participants must submit once before that step executes."},
        "meta": {"created_at": TIMESTAMP, "updated_at": TIMESTAMP},
    }, sort_keys=False))
    bindings = {"schema_version": "roboenv/bindings/1", "backend": backend,
                "embodiment": embodiment, "devices": {}}
    for name, native in sorted(facts.items()):
        kind = native["type"]
        device = directory / "devices" / name
        device.mkdir(parents=True, exist_ok=True)
        profile = {"name": name, "type": kind, "intrinsics": native["intrinsics"]}
        if kind != "world":
            profile["extrinsics"] = native["extrinsics"]
            actions = ["load", "unload", "status", "read"]
            if kind not in {"rgb", "rgbd", "imu", "force_torque"}:
                actions += ["write", "stop"]
            if kind == "manipulator":
                actions += ["fk", "ik"]
            profile["access"] = driver_access(kind, name, actions, "http://127.0.0.1:8123")
            write_driver(device, kind)
        profile["meta"] = {"created_at": TIMESTAMP, "updated_at": TIMESTAMP}
        (device / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
        bindings["devices"][name] = {"binding": backend + ".device/1", "entity": name}
    path = catalog / "bindings" / (robot_path + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(bindings, sort_keys=False))


def generate_robotwin(project, catalog, *, assets=None):
    from roboenv_runtime.adapters.robotwin.rig import describe

    for embodiment, robot in (("piper", "robotwin-dual-piper"),
                              ("aloha_agilex", "robotwin-aloha-agilex")):
        facts, _, _ = describe(Path(project) / "third_party/RoboTwin", embodiment, assets=assets)
        write_rig(catalog, robot, "robotwin", embodiment, facts)


def generate_robodojo(project, catalog, *, assets=None):
    from roboenv_runtime.adapters.robodojo.rig import describe

    for embodiment, robot in (("arx_x5", "robodojo-arx-x5"), ("yam", "robodojo-yam")):
        facts, _, _ = describe(Path(project) / "third_party/RoboDojo", embodiment, assets=assets)
        write_rig(catalog, robot, "robodojo", embodiment, facts)


def generate_behavior(project, catalog, *, assets=None):
    from roboenv_runtime.adapters.behavior.devices import compile_rig

    if assets is not None:
        raise ValueError("BEHAVIOR uses its registered OmniGibson asset depot")
    try:
        write_rig(catalog, "behavior-r1pro", "behavior", "r1pro", compile_rig())
    finally:
        # A task-free compile still owns an Isaac application. Release it before
        # interpreter teardown rather than leaving native threads to destructors.
        import omnigibson as og

        og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["robotwin", "robodojo", "behavior"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    {"robotwin": generate_robotwin, "robodojo": generate_robodojo,
     "behavior": generate_behavior}[args.backend](ROOT, args.out, assets=args.assets)
