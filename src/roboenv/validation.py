"""Strict, finite JSON-compatible data at owner configuration boundaries."""

from __future__ import annotations

import json
import math


def json_value(value):
    """Reject coercion, recursive aliases, and values JSON cannot represent."""
    active = set()
    validated = set()

    def visit(item, depth=0):
        if depth > 128:
            raise ValueError("configuration nesting exceeds the supported depth")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("configuration numbers must be finite")
            return
        if type(item) not in (dict, list):
            raise ValueError("configuration must contain only JSON-compatible values")
        identity = id(item)
        if identity in active:
            raise ValueError("recursive configuration is not supported")
        if identity in validated:
            return
        active.add(identity)
        try:
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError("configuration mappings require string keys")
                children = item.values()
            else:
                children = item
            for child in children:
                visit(child, depth + 1)
        finally:
            active.remove(identity)
        validated.add(identity)

    visit(value)
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("configuration mappings require unique string keys")
        result[key] = value
    return result


def json_loads(raw):
    try:
        return json_value(json.loads(raw, object_pairs_hook=_unique_pairs))
    except RecursionError as exc:
        raise ValueError("configuration nesting exceeds the supported depth") from exc


def _unique_mapping(loader, node, deep=False):
    return _unique_pairs(
        (loader.construct_object(key, deep=deep), loader.construct_object(value, deep=deep))
        for key, value in node.value
    )


def yaml_load(raw):
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        def compose_node(self, parent, index):
            # Owner files are finite JSON-compatible documents, not object graphs.
            if self.check_event(yaml.AliasEvent):
                raise ValueError("configuration YAML aliases are not supported")
            return super().compose_node(parent, index)

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
    try:
        return json_value(yaml.load(raw, Loader=UniqueLoader))
    except (RecursionError, yaml.YAMLError) as exc:
        raise ValueError("invalid or excessively nested YAML configuration") from exc


def action_validator(schema):
    """Compile a standard JSON Schema without permitting network resolution."""
    try:
        from jsonschema import Draft202012Validator, validators
        from jsonschema.exceptions import ValidationError
        from referencing import Registry
        from referencing.exceptions import NoSuchResource
    except ImportError as exc:
        raise RuntimeError(
            "Device serving requires the server extra: install 'roboenv[server]'"
        ) from exc
    json_value(schema)
    if not isinstance(schema, dict):
        raise ValueError("device action input_schema must be a JSON Schema object")

    def inspect(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise ValueError("device schemas permit only local fragment references")
                inspect(value)
        elif isinstance(item, list):
            for value in item:
                inspect(value)

    inspect(schema)
    validator_type = validators.validator_for(schema, default=Draft202012Validator)
    if "$schema" in schema and validators.validator_for(schema, default=None) is None:
        raise ValueError("unsupported device JSON Schema dialect")
    validator_type.check_schema(schema)

    def reject_remote(uri):
        raise NoSuchResource(ref=uri)

    validator = validator_type(schema, registry=Registry(retrieve=reject_remote))

    def validate(value):
        try:
            validator.validate(value)
        except ValidationError as exc:
            raise ValueError("arguments do not match the declared device input schema") from exc
        except Exception as exc:
            raise ValueError("device input schema could not be resolved locally") from exc

    return validate
