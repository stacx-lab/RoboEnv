"""Native panel pixels and explicitly labelled contract readout projections."""

import hashlib
import json
from pathlib import Path

from .geometry import panel_spec


def readout_bindings(instrument, spec):
    if not spec["native"]:
        return set(spec["bindings"])
    return {
        key
        for key, binding in instrument.contract.get("model_bindings", {}).items()
        if binding.get("component_id") == spec["component_id"]
        and binding.get("channel") == "display"
    }


def displays(instrument):
    result = {
        name: {**spec, "native": True}
        for name, spec in panel_spec(instrument).get("displays", {}).items()
    }
    covered = {spec.get("component_id") for spec in result.values()}
    for key, binding in instrument.contract.get("model_bindings", {}).items():
        component, slot = binding.get("component_id"), binding.get("state_id")
        schema = instrument.contract["state_slots"].get(slot, {}).get("value_schema", {})
        if (
            binding.get("channel") not in {"display", "material"}
            or not slot
            or component in covered
            or schema.get("type") in {"boolean", "array"}
            or instrument.contract["state_slots"][slot].get("visibility") != "agent"
            or binding.get("attribute", "").startswith("custom_property")
        ):
            continue
        name = "readout_" + component
        entry = result.setdefault(
            name,
            {
                "component_id": component,
                "native": False,
                "kind": "contract_readout",
                "size": [640, 240],
                "bindings": {},
            },
        )
        entry["bindings"][key] = slot
    # Status booleans on the same digit/readout component belong in the existing
    # labelled readout, not a new screen over each physical indicator lamp.
    for key, binding in instrument.contract.get("model_bindings", {}).items():
        slot = binding.get("state_id")
        entry = result.get("readout_" + str(binding.get("component_id")))
        state = instrument.contract["state_slots"].get(slot, {})
        if (
            entry is not None
            and binding.get("channel") in {"display", "material"}
            and state.get("visibility") == "agent"
            and state.get("value_schema", {}).get("type") == "boolean"
        ):
            entry["bindings"][key] = slot
    return result


def capture(instrument, display, spec):
    if spec["native"]:
        response = instrument.invoke("panel.capture", {"display_id": display}, owner=True)
        if response.get("status") == "succeeded":
            return response["result"]["image"], "native"
        code = response.get("error", {}).get("code")
        if response.get("status") != "rejected" or not isinstance(code, str):
            raise RuntimeError(f"Native panel capture failed: {display}: {code}")
        lines = ["Panel unavailable", str(code)]
        mode = "unavailable:" + code
    else:
        state = instrument.read()
        lines = ["PROCESS READOUT (SI)"]
        for slot in dict.fromkeys(spec["bindings"].values()):
            observation = state.get(slot, {})
            value = observation.get("value")
            rendered = f"{value:.6g}" if type(value) in (int, float) else json.dumps(value)
            lines.append(f"{slot}: {rendered} {observation.get('unit', '')}")
        mode = "contract_readout"
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, max(120, 32 * (len(lines) + 1))), (15, 22, 28))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    for index, line in enumerate(lines):
        draw.text((12, 12 + 30 * index), line, fill=(220, 235, 230), font=font)
    import io

    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path = Path(instrument.capture_directory.name) / f"readout_{digest}.png"
    if not path.exists():
        path.write_bytes(data)
    return {"path": str(path), "sha256": digest, "width": image.width, "height": image.height}, mode
