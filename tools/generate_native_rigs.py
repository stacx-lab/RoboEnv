"""Deterministically generate complete source-backed device catalogs without tasks."""

import argparse
from pathlib import Path

import yaml
from roboenv.devices.export import driver_access, write_driver

ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP = "2026-09-09T00:00:00Z"
NATIVE_LIMIT_NOTES = {
    "behavior": "Native base-joint velocity targets clip to +/-1.5 m/s for translation and +/-pi "
    "rad/s for yaw after conversion from body to native joint coordinates. These are "
    "actuator-coordinate limits, not independent body-frame command bounds. Native "
    "position controllers clip joint targets to model ranges. URDF-only "
    "velocity/effort ratings are unspecified. Gripper profiles reject commands outside [0, 1].",
    "robotwin": "SAPIEN enforces physical joint ranges. Native normalized gripper targets clip "
    "to [0, 1] and retain native incremental targets and drive velocities at each physics step. "
    "Gripper profiles reject commands outside [0, 1]. URDF-only arm velocity and effort "
    "ratings are unspecified; no "
    "additional command speed caps apply.",
    "robodojo": "Only upstream ARX X5 is supported. Native arm actuators enforce 5 rad/s "
    "velocity and 100 N m effort limits. Physical joint constraints and native "
    "gripper clipping and interpolation apply; no added arm target-range rejection. "
    "Gripper profiles reject commands outside [0, 1].",
}


def write_rig(root, robot, backend, embodiment, facts, *, bindings_out=None):
    catalog = Path(root)
    robot_path = f"{backend}/{robot}"
    directory = catalog / "robots" / robot_path
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "identity": {
                    "source": "user",
                    "id": robot,
                    "name": robot,
                    "platform": "sim",
                    "kind": "mobile" if backend == "behavior" else "other",
                    "model": robot,
                },
                "sim": {"source": "user", "simulator": backend},
                "devices": {"source": "user"},
                "safety": {
                    "source": "user",
                    "max_cartesian_velocity": None,
                    "max_joint_velocity": None,
                    "keep_out_zones": [],
                    "notes": NATIVE_LIMIT_NOTES.get(
                        backend, "Limits are handled by the native simulator."
                    )
                    + " Stop affects only its device; the world owns simulation time.",
                },
                "meta": {"created_at": TIMESTAMP, "updated_at": TIMESTAMP},
            },
            sort_keys=False,
        )
    )
    bindings = {
        "schema_version": "roboenv/bindings/1",
        "backend": backend,
        "embodiment": embodiment,
        "devices": {},
    }
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
            if kind == "joint":
                actions += ["fk", "ik"]
            profile["access"] = driver_access(kind, name, actions, "http://127.0.0.1:8123")
            write_driver(device, kind)
        profile["meta"] = {"created_at": TIMESTAMP, "updated_at": TIMESTAMP}
        (device / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
        bindings["devices"][name] = {"binding": backend + ".device/1", "entity": name}
    if bindings_out is not None:
        path = Path(bindings_out) / (robot_path + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(bindings, sort_keys=False))


def generate_robotwin(project, catalog, *, assets=None, bindings_out=None):
    from roboenv_runtime.adapters.robotwin.rig import describe

    facts, _, _ = describe(
        Path(project) / "third_party/RoboTwin", "aloha_agilex", assets=assets
    )
    write_rig(
        catalog, "robotwin-aloha-agilex", "robotwin", "aloha_agilex", facts,
        bindings_out=bindings_out,
    )


def generate_robodojo(project, catalog, *, assets=None, bindings_out=None):
    from roboenv_runtime.adapters.robodojo.rig import describe

    for embodiment, robot in (("arx_x5", "robodojo-arx-x5"),):
        facts, _, _ = describe(Path(project) / "third_party/RoboDojo", embodiment, assets=assets)
        write_rig(catalog, robot, "robodojo", embodiment, facts, bindings_out=bindings_out)


def generate_behavior(project, catalog, *, assets=None, bindings_out=None):
    from roboenv_runtime.adapters.behavior.devices import compile_rig

    if assets is not None:
        raise ValueError("BEHAVIOR uses its registered OmniGibson asset depot")
    try:
        write_rig(
            catalog, "behavior-r1pro", "behavior", "r1pro", compile_rig(), bindings_out=bindings_out
        )
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
    parser.add_argument("--bindings-out", type=Path, help="Separate internal binding resource root")
    args = parser.parse_args()
    {"robotwin": generate_robotwin, "robodojo": generate_robodojo, "behavior": generate_behavior}[
        args.backend
    ](ROOT, args.out, assets=args.assets, bindings_out=args.bindings_out)
