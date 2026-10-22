"""Explicit instrument commands on the existing owner/device transport."""

import re

from roboenv.devices.model import Action, BoundDevice

from roboenv_runtime.bindings.motion import control_binding

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def action_name(capability):
    """Kinex driver verbs are identifiers; retain native IDs inside the owner."""
    name = capability.replace(".", "_")
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", name) is None:
        raise ValueError("Unsupported instrument capability identifier")
    return name


@control_binding
def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "chemlab_instruments":
        raise ValueError("ChemLab bindings require an instrument world")
    if spec.config or spec.name != spec.entity or context.devices.get(spec.entity) != spec.name:
        raise ValueError("Instrument bindings require canonical names")
    env = context.native.environment
    definition = spec.definition(root)
    if definition.type == "world":
        if spec.name != "world":
            raise ValueError("ChemLab requires one canonical world device")
        return BoundDevice(definition, {}, implementation="World reference", binding=spec.as_dict())
    if definition.type != "other" or spec.name not in env.instruments:
        raise ValueError("ChemLab requires a declared instrument device")
    instrument = env.instruments[spec.name]
    actions = {
        "read": Action(lambda args, execution: env.read_device(spec.name), EMPTY),
        "status": Action(lambda args, execution: {"loaded": True}, EMPTY),
        "load": Action(lambda args, execution: {"loaded": True}, EMPTY),
        "unload": Action(lambda args, execution: {"unloaded": True}, EMPTY),
    }
    for capability, contract in instrument.capabilities.items():
        name = action_name(capability)
        if name in actions:
            raise ValueError("Instrument capability aliases collide")

        def invoke(args, execution, capability=capability):
            return execution.perform(
                capability, args, lambda: env.invoke(spec.name, capability, args), mutating=True
            )

        supported = contract.get("availability") == "implemented"
        actions[name] = Action(
            invoke if supported else None,
            {**contract["input_schema"], "title": capability},
            reason=None if supported else "Unsupported by upstream runtime",
        )
    operation_schema = {
        "type": "object",
        "properties": {"operation_id": {"type": "string", "minLength": 1}},
        "required": ["operation_id"],
        "additionalProperties": False,
    }
    for method in ("poll", "cancel"):
        if callable(getattr(instrument.agent, method, None)):
            capability = f"operation.{method}"

            def operation(args, execution, capability=capability):
                return execution.perform(
                    capability, args, lambda: env.invoke(spec.name, capability, args), mutating=True
                )

            actions[action_name(capability)] = Action(operation, operation_schema)
    return BoundDevice(
        definition, actions, implementation="ChemLab instrument interface", binding=spec.as_dict()
    )
