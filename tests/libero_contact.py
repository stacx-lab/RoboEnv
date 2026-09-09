"""Owner-only contact fixture: a fixed block obstructs the native Panda fingers.

Only fixture setup edits model XML/state. All tested movement and stop commands
subsequently use the production binding and native physics; no task success claim.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BindingPlan
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.bindings import default_registry

ROOT = Path(__file__).resolve().parents[1]


def check(out):
    adapter = LiberoAdapter(enable_render=False, realtime=True)
    try:
        adapter.reset(0)
        world = adapter._env
        saved = world.sim.get_state()
        site = world.sim.model.site_name2id(world._robot.gripper.important_sites["grip_site"])
        position = world.sim.data.site_xpos[site].copy()
        xml = ET.fromstring(world.env.model.get_xml())
        ET.SubElement(
            xml.find("worldbody"),
            "geom",
            {
                "name": "owner_contact_probe",
                "type": "sphere",
                "size": "0.018",
                "pos": " ".join(map(str, position)),
                "rgba": "1 0 1 1",
                "contype": "1",
                "conaffinity": "1",
                "friction": "1 0.005 0.0001",
            },
        )
        world.env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
        world.sim.set_state(saved)
        world.sim.forward()
        world.stop_part("arm")
        world.stop_part("gripper")
        root = ROOT / "catalog/robots/libero/libero-panda-tabletop"
        plan = BindingPlan.load(root, backend="libero")
        plan = replace(plan, devices=tuple(s for s in plan.devices if s.entity != "camera"))
        devices = default_registry().assemble(
            plan, root, BindingContext("libero", "panda", adapter)
        )

        def call(device, action, **args):
            return devices.call({"device": device, "action": action, "args": args})["data"]

        def contacts():
            obstacle = world.sim.model.geom_name2id("owner_contact_probe")
            return sum(obstacle in (c.geom1, c.geom2) for c in world.sim.data.contact)

        before = world.sim.get_state().flatten().copy()
        assert call("gripper", "write", target=[1]) == {"accepted": True}
        assert np.array_equal(before, world.sim.get_state().flatten())
        for _ in range(60):
            world.tick()
        closed = call("gripper", "read")
        contact_count = contacts()
        assert contact_count >= 2, (closed, contact_count)
        assert abs(closed["positions"][0] - closed["positions"][1]) > 0.02
        # Stop does not force the measured joints through the obstructing geometry.
        before = world.sim.get_state().flatten().copy()
        call("arm", "stop")
        call("gripper", "stop")
        assert np.array_equal(before, world.sim.get_state().flatten())
        for _ in range(40):
            world.tick()
        stopped = call("gripper", "read")
        drift = float(np.max(np.abs(np.array(stopped["positions"]) - closed["positions"])))
        assert drift < 0.005, drift
        assert np.isfinite(world.sim.get_state().flatten()).all()
        call("gripper", "write", target=[-1])
        for _ in range(40):
            world.tick()
        opened = call("gripper", "read")
        assert abs(opened["positions"][0] - opened["positions"][1]) > 0.06
        report = {
            "passed": True,
            "fixture": "owner-only fixed sphere between native fingers",
            "contacts_while_closing": contact_count,
            "closed": closed,
            "stopped": stopped,
            "opened": opened,
            "stop_drift_m": drift,
            "task_success_claimed": False,
        }
        out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.out)
