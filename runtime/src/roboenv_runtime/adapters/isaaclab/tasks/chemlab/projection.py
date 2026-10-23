"""Synchronous, per-instance geometry transactions for native runtime callbacks."""

import copy
import json
from types import SimpleNamespace

import numpy as np
from roboenv.provisioning.chemlab_assets import resource
from roboenv.validation import json_value

from .geometry import component_links, joint_bindings
from .supplements import supplement


class InstrumentProjection:
    def __init__(self, root, row, instance):
        from yourdfpy import URDF

        self.instance = instance
        self.contract = json.loads(resource(root, row["contract"]).read_text())
        self.model = URDF.load(str(resource(root, row["urdf"])))
        self.supplements, self.groups = supplement(root, row, self.contract, self.model)
        self.bindings = joint_bindings(SimpleNamespace(contract=self.contract), self.model)
        self.values, self.config, self.transforms, self.visible = {}, {}, {}, {}
        self.integrated = {}
        self.commit = None
        self.slots = {}
        if row["id"] == "instrument.agilent_hs8697":
            manifest = resource(root, row["path"] + "/geometry_manifest.json")
            self.slots = json.loads(manifest.read_text())["slots"]

    def __call__(self, instance, values):
        if instance != self.instance or not isinstance(values, dict):
            raise ValueError("Scene callback requires its bound instance and typed snapshot")
        json_value(values)
        if self.contract["asset_type_id"] == "instrument.opentrons_flex_single_5_1000ul":
            values = {f"motion.{key}": value for key, value in values.items()}
        previous = copy.deepcopy(self.values)
        try:
            self.update({**self.values, **values})
            if self.commit is not None:
                self.commit()
        except Exception:
            self.update(previous)
            if self.commit is not None:
                self.commit()
            raise
        return True

    def update(self, values, dt=0.0):
        self.values = dict(values)
        self.config = {}
        self.transforms, self.visible = {}, {}
        self.visible.update(
            {
                key: value["default_visible"]
                for key, value in self.supplements.items()
                if "default_visible" in value
            }
        )
        for binding in self.bindings.values():
            value = values.get(binding["state"])
            if type(value) in (bool, int, float) and np.isfinite(value):
                coordinate = float(value) * binding["scale"] + binding["offset"]
                target = binding["target"]
                if binding["velocity"]:
                    self.integrated[target] = self.integrated.get(target, 0.0) + coordinate * dt
                    coordinate = self.integrated[target]
                self.config[target] = coordinate
        # Reviewed release formulas for scene channels not represented as a
        # single scalar URDF joint. Never evaluate free-form contract prose.
        asset = self.contract["asset_type_id"]
        if asset == "instrument.agilent_hs8697":
            slot = values.get("manipulator.position_slot")
            if slot is not None:
                position = (
                    np.asarray([0.15, 0.14, 0.64])
                    if slot == "home"
                    else np.asarray(self.slots[slot]["position_m"]) + [0, 0, 0.22]
                )
                origin = self.model.joint_map["manipulator_carriage_joint"].origin[:3, 3]
                self._translation("manipulator_carriage", position - origin)
        if asset == "instrument.binder_vd56":
            for kind in ("vacuum", "vent"):
                value = values.get(f"{kind}.valve_open")
                if type(value) is bool:
                    self.config[f"{kind}_valve_joint"] = float(value) * np.pi / 2
        if asset == "instrument.metrohm_914":
            for probe in ("ph", "conductance", "do"):
                value = values.get(f"probe.{probe}.installed")
                if type(value) is bool:
                    self._local_position(f"probe_{probe}", 2, 0.008 if value else 0.048)
        if asset == "instrument.biotage_extrahera_classic":
            for component, slot, axis, offset in (
                ("gantry_x", "gantry.x_m", 0, -0.22),
                ("gantry_y", "gantry.y_m", 1, -0.14),
                ("pipette_z", "gantry.z_m", 2, 0),
                ("pressure_manifold", "manifold.z_m", 2, 0.33),
            ):
                if slot in values:
                    self._local_position(component, axis, values[slot] + offset)
        if asset == "instrument.parr_4591":
            if "vent.open" in values:
                self.config["vent_handle_joint"] = float(values["vent.open"]) * np.pi / 2
            pressure = values.get("pressure.indicated_pressure_pa")
            if type(pressure) in (int, float):
                self.config["gauge_needle_joint"] = (
                    1.5 * np.pi * np.clip((pressure - 101325) / 20700000, 0, 1)
                )
        if asset == "instrument.magritek_spinsolve80" and "sample.insertion_fraction" in values:
            self.config["sample_tube_joint"] = 0.16 * values["sample.insertion_fraction"]
        if asset == "instrument.metrohm_omnis" and "dosing.piston_fraction" in values:
            self.config["piston_proxy_joint"] = 0.080 * values["dosing.piston_fraction"]
        if asset in {
            "instrument.agilent_gc8890b",
            "instrument.nabertherm_l9",
            "instrument.milestone_dma80",
            "instrument.miele_plw6111",
        }:
            if "power.enabled" in values:
                target = (
                    "power_rocker_joint"
                    if asset == "instrument.milestone_dma80"
                    else "power_switch_joint"
                )
                joint = self.model.joint_map[target]
                self.config[target] = (
                    joint.limit.upper if values["power.enabled"] else joint.limit.lower
                )
        if asset == "instrument.brooks_preciseflex_400" and isinstance(
            values.get("scene.joints"), dict
        ):
            for target, key in (
                ("j1_z", "z_m"),
                ("j2_shoulder", "shoulder_rad"),
                ("j3_elbow", "elbow_rad"),
                ("j4_wrist", "wrist_rad"),
            ):
                self.config[target] = values["scene.joints"][key]
        if asset == "instrument.park_nx10":
            for key, value in values.items():
                if key.startswith("knob.") and key.endswith("_turns") and value is not None:
                    name = key.removeprefix("knob.").removesuffix("_turns")
                    self.config[f"knob_{name}_joint"] = float(value) * 2 * np.pi
        if asset == "instrument.bronkhorst_p702cv_5bar_gauge_n2":
            if "scene.valve_position" in values:
                self.config["valve_poppet_joint"] = 0.003 * values["scene.valve_position"]
        if asset == "instrument.carbolite_tf1_12_60_300":
            for side in ("left", "right"):
                value = values.get(f"end_seal.{side}.withdrawal_m")
                if type(value) in (int, float):
                    self.config[f"end_seal_{side}_joint"] = value
                    self.config[f"end_seal_{side}_parking_frame_joint"] = 0.10 * np.clip(
                        (value - 0.06) / 0.04, 0, 1
                    )
        if asset == "instrument.mt_xpr204e" and "pan.installed" in values:
            self._translation("pan", [0, 0, 0 if values["pan.installed"] else 0.08])
        if asset == "instrument.metrohm_899":
            for component in ("cell", "generator", "indicator"):
                value = values.get(f"slot.{component}.installed")
                if type(value) is bool:
                    self._translation(component, [0, 0, 0 if value else 0.08])
        if asset == "instrument.radleys_mya4":
            for n in range(1, 5):
                value = values.get(f"zone_{n}.vessel.fill_volume_m3")
                if type(value) in (int, float):
                    transform = np.eye(4)
                    transform[2, 2] = np.clip(value / 50e-6, 0, 2)
                    self.transforms[f"zone_{n}_liquid"] = transform
                    self.visible[f"zone_{n}_liquid"] = value > 0
        for binding in self.contract.get("model_bindings", {}).values():
            if binding.get("channel") != "visibility":
                continue
            slot, component = binding.get("state_id"), binding.get("component_id")
            if self.groups and binding.get("target") in {"lock_indicator", "imbalance_indicator"}:
                continue
            if slot not in values:
                continue
            value = values[slot]
            if type(value) is bool:
                self.visible[component] = value
            elif slot == "scene.rack_format":
                self.visible[component] = value == int(component.rsplit("_", 1)[1])
            elif slot in {"tray.occupancy", "oven.occupancy"} and component.startswith("vial_"):
                self.visible[component] = value[int(component.rsplit("_", 1)[1]) - 1] is not None
            elif slot.endswith("connection_id"):
                self.visible[component] = value is not None
            elif slot in {"extrudate.length_m", "extrudate.last_cut_length_m"}:
                self.visible[component] = value is not None and value > 0
            elif slot == "accessory.mode":
                self.visible[component] = component == f"accessory_{value}"
            elif slot == "deck.format" and value is not None:
                volume = values.get("deck.cartridge_volume_m3")
                self.visible[component] = (
                    component == f"rack_24_{round(volume * 1e6)}ml"
                    if value == "cartridges24" and volume is not None
                    else component == {"plate96": "plate_96", "plate48": "plate_48"}.get(value)
                )
        unknown = set(self.config) - set(self.model.joint_map)
        if unknown:
            raise ValueError(f"Scene mapping refers to absent joints: {sorted(unknown)}")
        self.model.update_cfg(self.config)

    def _translation(self, component, translation):
        if not set(component_links(self.contract, component)).intersection(
            self.model.scene.graph.nodes
        ):
            raise ValueError(f"Scene component has no released geometry: {component}")
        transform = np.eye(4)
        transform[:3, 3] = translation
        self.transforms[component] = transform

    def binding_keys(self):
        resolved = set()
        for key, binding in self.contract.get("model_bindings", {}).items():
            if binding.get("direction") == "scene_to_process":
                continue
            component = binding.get("component_id")
            links = component_links(self.contract, component)
            channel = binding.get("channel")
            if (
                channel == "visibility"
                and component in self.visible
                and any(
                    self.model.scene.graph.transforms.parents[node] in links
                    for node in self.model.scene.graph.nodes_geometry
                )
            ):
                resolved.add(key)
            elif channel in {"joint", "pose"} and (
                component in self.transforms
                or binding.get("target") in self.transforms
                or (key in self.bindings and self.bindings[key]["target"] in self.config)
                or any(
                    joint.child in links and joint.name in self.config
                    for joint in self.model.robot.joints
                )
            ):
                resolved.add(key)
        if self.groups:
            resolved.update(
                {
                    "rotor_configuration",
                    "sample_visibility",
                    "lock_indicator",
                    "imbalance_indicator",
                }
            )
        return resolved

    def colors(self):
        result = {}
        for key, binding in self.contract.get("model_bindings", {}).items():
            if binding.get("channel") != "material":
                continue
            slot = binding.get("state_id", "")
            value = self.values.get(slot)
            if slot.endswith("_rgb") and isinstance(value, list) and len(value) == 3:
                if all(type(v) in (int, float) and 0 <= v <= 1 for v in value):
                    result[key] = value
            elif slot == "rack.occupied_slots" and isinstance(value, list):
                index = int(binding["component_id"].rsplit("_", 1)[1])
                # Release specifies orange/blue-gray occupancy proxies, without
                # photometric calibration. Keep this palette explicit.
                result[key] = [1.0, 0.4, 0.05] if value[index] else [0.35, 0.45, 0.55]
        if self.contract["asset_type_id"] == "instrument.magritek_spinsolve80":
            # Public Spinsolve80BlenderAdapter: one composite writer, running
            # takes precedence over power. Do not let an inactive flag erase it.
            color = (
                [1.0, 0.48, 0.08]
                if self.values.get("acquisition.running")
                else [0.12, 0.65, 0.28]
                if self.values.get("power.enabled")
                else [0.09, 0.13, 0.10]
            )
            result.update(indicator_power=color, indicator_running=color)
        if self.contract["asset_type_id"] == "instrument.ziath_dp5_mirage_2023":
            enabled = self.values.get("power.enabled")
            if type(enabled) is bool:
                # Released materials.mtl: Indicator_on / Indicator_off diffuse RGB.
                result["indicator_power"] = [0.03, 0.85, 0.27] if enabled else [0.03, 0.065, 0.058]
        return result

    def _local_position(self, component, axis, value):
        joint = next(j for j in self.model.robot.joints if j.child == component)
        translation = [0, 0, 0]
        translation[axis] = value - joint.origin[axis, 3]
        self._translation(component, translation)

    def node_transform(self, node):
        transform = self.model.scene.graph[node][0].copy()
        ancestors = set()
        parent = node
        parents = self.model.scene.graph.transforms.parents
        while parent in parents and parent not in ancestors:
            ancestors.add(parent)
            parent = parents[parent]
        ancestors.add(parent)
        for component, delta in self.transforms.items():
            for link in component_links(self.contract, component):
                if link in ancestors:
                    frame = self.model.scene.graph[link][0]
                    transform = frame @ delta @ np.linalg.inv(frame) @ transform
                    break
        return transform

    def node_visible(self, node):
        source = self.groups.get(node, "")
        if source == "lock_indicator":
            return self.values.get("lid.locked") is True
        if source == "imbalance_indicator":
            return self.values.get("alarm.imbalance") is True
        if source.startswith("rotor_large_"):
            return self.values.get("rotor.configuration") == "large_capacity"
        if source.startswith("rotor_fixed_"):
            return self.values.get("rotor.configuration") == "fixed_angle"
        if source.startswith("sample_carrier_"):
            slot = "slot_" + source.split("_")[2]
            return slot in self.values.get("sample.occupied_slots", [])
        parents = self.model.scene.graph.transforms.parents
        ancestors = {node}
        while node in parents and parents[node] not in ancestors:
            node = parents[node]
            ancestors.add(node)
        return all(
            value
            for component, value in self.visible.items()
            if ancestors.intersection(component_links(self.contract, component))
        )
