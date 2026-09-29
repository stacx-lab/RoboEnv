"""Agent-neutral Markdown and YAML from public robot and device facts. No drivers."""

import copy
import json
from pathlib import Path

import yaml
from roboenv.devices.model import DeviceDefinition


def public_profiles(devices):
    result = []
    for device in devices:
        DeviceDefinition(
            device["name"],
            device["type"],
            device.get("intrinsics", {}),
            device.get("extrinsics", {}),
        )
        result.append(
            copy.deepcopy(
                {
                    key: device[key]
                    for key in (
                        "name",
                        "type",
                        "intrinsics",
                        "extrinsics",
                        "actions",
                        "execution",
                    )
                    if key in device
                }
            )
        )
    if len({item["name"] for item in result}) != len(result):
        raise ValueError("Duplicate device name")
    return result


def robot_description(robot, devices):
    identity = robot.get("identity", {})
    model = identity.get("model") or identity.get("name") or "Robot"
    simulator = robot.get("sim", {}).get("simulator", "the simulator")
    lines = [
        f"The robot is **{model}**, running in **{simulator}**.",
        "",
        "| Device | Type |",
        "|---|---|",
    ]
    lines += [f"| `{d['name']}` | {d['type']} |" for d in devices]
    return "\n".join(lines)


def _value(value):
    return "not specified" if value is None else json.dumps(value, ensure_ascii=False)


def _facts(value, label, depth=0):
    """Preserve all public parameter values in readable Markdown, without YAML links."""
    prefix = "  " * depth + "- "
    label = label.replace("_", " ")
    if isinstance(value, dict) and value:
        return [
            f"{prefix}{label}:",
            *[line for key, item in value.items() for line in _facts(item, key, depth + 1)],
        ]
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        return [
            f"{prefix}{label} (in order):",
            *[
                line
                for index, item in enumerate(value)
                for line in _facts(item, f"Item {index + 1}", depth + 1)
            ],
        ]

    def word(item):
        if item is None:
            return "not specified"
        if isinstance(item, bool):
            return "yes" if item else "no"
        return str(item)

    if isinstance(value, list):
        text = ", ".join(word(item) for item in value) if value else "none listed"
    elif isinstance(value, dict):
        text = "no entries"
    else:
        text = word(value)
    return [f"{prefix}{label}: {text}."]


def _schema(schema, prefix="arguments"):
    """Explain argument structure, including arrays and alternative command modes."""
    lines = []
    details = [schema.get("type", "value")]
    for key, label in (
        ("const", "must equal"),
        ("enum", "one of"),
        ("minimum", "minimum"),
        ("maximum", "maximum"),
        ("minItems", "minimum length"),
        ("maxItems", "maximum length"),
        ("description", "meaning"),
    ):
        if key in schema:
            details.append(f"{label}: {_value(schema[key])}")
    lines.append(f"- `{prefix}`: {'; '.join(details)}.")
    for key, value in schema.get("properties", {}).items():
        requirement = "required" if key in schema.get("required", []) else "optional"
        lines += _schema(value, f"{prefix}.{key} ({requirement})")
    if schema.get("required") and not schema.get("properties"):
        lines.append("- Required result fields: " + ", ".join(schema["required"]) + ".")
    for index, value in enumerate(schema.get("prefixItems", [])):
        lines += _schema(value, f"{prefix}[{index}]")
    if isinstance(schema.get("items"), dict):
        lines += _schema(schema["items"], f"{prefix}[i]")
    for kind in ("oneOf", "anyOf", "allOf"):
        for index, value in enumerate(schema.get(kind, [])):
            lines += _schema(value, f"{prefix} / {kind} variant {index + 1}")
    if schema.get("additionalProperties") is False:
        lines.append(f"- `{prefix}` accepts only the named fields.")
    return lines


def _dimensions(device):
    facts = device.get("intrinsics", {})
    control = facts.get("control", {})
    lines = []
    if control.get("limit_handling") == "native":
        lines.append(
            "Command limits are handled by the native simulator. Model joint ranges "
            "describe geometry, not additional write restrictions. Native clipping and "
            "physical constraints still apply; see workspace safety notes."
        )
    joints = facts.get("model", {}).get("joints", [])
    if joints:
        lines += [
            f"Control mode: **{control.get('mode', 'see profile')}**. "
            "Joint vectors use the following zero-based order. In position mode, "
            "targets are absolute joint coordinates, not end-effector pose deltas.",
            "",
            "| Index | Joint | Meaning / units | Minimum | Maximum |",
            "|---|---|---|---|---|",
        ]
        for index, joint in enumerate(joints):
            meaning = {
                "revolute": "rotation about the declared joint axis / radians",
                "continuous": "rotation about the declared joint axis / radians",
                "prismatic": "translation along the declared joint axis / metres",
            }.get(joint["type"], joint["type"] + " / see profile")
            limits = joint.get("limits") or {}
            lines.append(
                f"| {index} | `{joint['name']}` | {meaning} | "
                f"{_value(limits.get('min'))} | {_value(limits.get('max'))} |"
            )
        lines += [
            "",
            "`read()` reports measured positions in this order; velocities, when "
            "provided, are the corresponding units per second. Joint axes and the "
            "serial chain are in the detailed parameters below. Positive rotation follows "
            "the right-hand rule. Null velocity/effort limits are unspecified, not zero.",
        ]
    if device["type"] in {"gripper", "suction"}:
        lines.append(
            "Write a flat `actions: [value, ...]` sequence with `duration_s`; "
            "a single command uses `[value]`."
        )
    if control.get("mode") == "direction":
        lines += [
            f"Each gripper action commands opening/closing direction: "
            f"`{control.get('open_target')}` opens and `{control.get('close_target')}` "
            "closes. It is a controller command, not an aperture in metres.",
            f"Physical aperture range in metres: {_value(facts.get('aperture_m'))}.",
        ]
        if "increment_per_interval" in control:
            lines.append(
                "The direction command changes the internal gripper target by "
                f"{control['increment_per_interval']} per control interval; "
                "zero retains that target. Inspect finger feedback to measure opening."
            )
    meanings = {
        "vx": "linear X component",
        "vy": "linear Y component",
        "vz": "linear Z component",
        "wx": "angular X component",
        "wy": "angular Y component",
        "wz": "angular Z component (yaw)",
    }
    for mode, channels in control.get("modes", {}).items():
        lines += [f"For `write`, set `mode={mode!r}` and supply the named channel dictionaries."]
        for channel in ("velocity", "posture"):
            data = channels.get(channel)
            if not data:
                continue
            lines += [
                f"`{channel}` components use the **{data.get('frame', 'declared')}** frame.",
                "",
                "| Field | Meaning | Units | Bounds |",
                "|---|---|---|---|",
            ]
            for component in data["components"]:
                axis = component["axis"]
                lines.append(
                    f"| `{channel}.{axis}` | {meanings.get(axis, axis)} | "
                    f"{component['units']} | {_value(component.get('bounds'))} |"
                )
            lines += [
                "Normalized values are dimensionless controller targets, not m/s or rad/s. "
                + (
                    "Limits are handled by the native simulator; null bounds are unspecified. "
                    "See workspace safety notes for native clipping and coordinate frames."
                    if control.get("limit_handling") == "native"
                    else "Bounds constrain commands, not measured motion."
                )
            ]
    for index, position in enumerate(facts.get("feedback", {}).get("positions", [])):
        lines.append(
            f"Feedback `positions[{index}]` is `{position['name']}` in {position['units']}."
        )
    feedback = facts.get("feedback", {})
    if feedback.get("pose_frame"):
        lines.append(f"Feedback pose uses the `{feedback['pose_frame']}` frame.")
    if feedback.get("velocity_components"):
        lines.append(
            "Measured velocity components, in order: "
            + ", ".join(f"`{axis}`" for axis in feedback["velocity_components"])
            + f"; frame: `{feedback.get('velocity_frame')}`. "
            "Linear feedback is metres/second; angular feedback is radians/second."
        )
    if device["type"] in {"rgb", "rgbd"}:
        color, depth = facts.get("color", {}), facts.get("depth", {})
        lines += [
            f"Color image: {color.get('width')} × {color.get('height')} pixels, "
            f"{color.get('pixel_format')}, {color.get('image_format')}.",
            "Pixel coordinates are (u, v) = (column, row); arrays are indexed [v, u]. "
            "Pinhole fx/fy are focal lengths in pixels; cx/cy are the principal point. "
            "Camera optical coordinates use +X right, +Y down, +Z forward.",
            f"Projection: {_value(color.get('projection'))}.",
        ]
        if depth:
            lines += [
                f"Depth encoding: {depth.get('image_format')}, {depth.get('pixel_format')}; "
                f"Multiply a valid pixel by {depth.get('scale_m_per_unit')} "
                "to obtain metres. "
                f"Invalid value: {depth.get('invalid_value')}. "
                f"Quantity: {depth.get('quantity')} "
                "(`z` means optical-axis depth, not Euclidean distance).",
                f"Depth registration: {depth.get('registration')}. "
                f"Valid range: {_value(depth.get('valid_range_m'))} metres.",
            ]
        lines += [
            "`read()` downloads image artifacts and returns local "
            "`file` / `depth_file` paths. "
            "Copy any images you need before closing the SDK connection; its temporary "
            "downloads are removed on close."
        ]
    mount = device.get("extrinsics", {}).get("mount")
    if mount:
        lines += [
            f"Mount: parent `{mount.get('parent')}`, frame `{mount.get('parent_frame')}`; "
            f"translation {_value(mount.get('translation_m'))} metres; "
            f"rotation {_value(mount.get('rotation_xyzw'))} in quaternion [x, y, z, w]."
        ]
    if facts.get("geometry", {}).get("description"):
        lines.append(facts["geometry"]["description"])
    if facts.get("stop"):
        lines.append(
            f"Device stop behavior: {_value(facts['stop'].get('behavior'))}; "
            f"affected devices: {_value(facts['stop'].get('affected_devices'))}."
        )
    return lines


def render_documentation(robot, devices, connection):
    """Render the live SDK contract without adding benchmark goals or scoring policy."""
    devices = public_profiles(devices)
    lines = [
        "# Robot and device interface",
        "",
        robot_description(robot, devices),
        "",
        "## Connect",
        "",
        f"The running service endpoint is `{connection['endpoint']}`. "
        "The orchestrator sets `ROBOENV_ENDPOINT`; do not start a second service.",
        "",
        "```python",
        "import os",
        "import roboenv",
        "",
        "env = roboenv.connect(os.environ['ROBOENV_ENDPOINT'])",
        "print(env.devices())  # public device names and supported actions",
        "print(env.status())",
        "print(env.task.instruction)  # public scene goal",
        "```",
        "",
        "Read with `env.device(name).read()`. Joint `write(actions, duration_s=...)` submits "
        'buffered references and returns immediately. Use `env.device("world").wait(seconds)` '
        "to wait in world time and `.read()` for time_s. The device owns resampling. "
        "Goal planning and arrival checks belong to the harness. "
        "Concurrent device writes share the world clock; starts are not atomic. "
        "Other writes use the schemas below.",
        "",
        "CLI: `roboenv device NAME ACTION 'JSON_ARGUMENTS' --endpoint \"$ROBOENV_ENDPOINT\"`. "
        "`env.status()` reports state and budget. `env.stop()` finalizes the service when "
        "permitted; device `stop()` affects only its declared scope.",
        "",
        "## Frames and feedback",
        "",
        "Poses use [x, y, z, qx, qy, qz, qw]: metres and a unit quaternion. "
        "Structured poses use `translation_m` and `rotation_xyzw`. Frames are right-handed; "
        "`arm:flange` names a local frame. Mounts map child coordinates to the parent. "
        "Null local frames come from measured `frame_poses`; `???` means unknown.",
        "",
        "Sequence completion is not arrival. Reads never request physics steps. "
        "After an uncertain write, inspect `roboenv operation ID` before sending another. "
        "Compare timestamps only within their declared clock. Local kinematic geometry "
        "does not imply server-side IK/FK support.",
        "",
        "## Robot specification",
        "",
        *_facts(robot, "Robot"),
    ]
    for device in devices:
        lines += ["", f"## {device['name']} ({device['type']})", "", *_dimensions(device)]
        for name, action in device.get("actions", {}).items():
            if not action.get("supported"):
                lines += [
                    "",
                    f"The `{name}` operation is unavailable: "
                    f"{action.get('reason', 'not supported')}.",
                ]
                continue
            lines += ["", f"### {name}", "", *_schema(action.get("input_schema", {}))]
            if "read_only" in action:
                lines.append(
                    "This operation is read-only."
                    if action["read_only"]
                    else "This operation may change device or simulation state."
                )
            if action.get("output_schema"):
                lines += _schema(action["output_schema"], "result")
        lines += ["", "### Detailed device parameters", ""]
        for field, label in (
            ("intrinsics", "Intrinsic parameters"),
            ("extrinsics", "Mount and frames"),
            ("execution", "Execution behavior"),
        ):
            if field in device:
                lines += _facts(device[field], label)
    return "\n".join(lines) + "\n"


def export_documentation(robot, devices, directory, connection):
    """Export harness.md, public YAMLs and connection metadata; never executable files."""
    devices = public_profiles(devices)
    markdown = render_documentation(robot, devices, connection)
    directory = Path(directory)
    (directory / "devices").mkdir(parents=True, exist_ok=True)
    (directory / "harness.md").write_text(markdown)
    for device in devices:
        (directory / "devices" / f"{device['name']}.yaml").write_text(
            yaml.safe_dump(device, sort_keys=False)
        )
    (directory / "devices.json").write_text(json.dumps(devices) + "\n")
    (directory / "connection.json").write_text(
        json.dumps(
            {
                key: connection[key]
                for key in ("endpoint", "session", "generation")
                if key in connection
            }
        )
        + "\n"
    )
    (directory / "workspace.yaml").write_text(yaml.safe_dump(robot, sort_keys=False))
