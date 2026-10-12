"""Read native language templates without importing or running a simulator."""

import ast


def method(classes, name, method_name):
    """Find a method in a task class or its local base classes."""
    cls = classes.get(name)
    if cls is None:
        return None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    for base in cls.bases:
        if isinstance(base, ast.Name):
            found = method(classes, base.id, method_name)
            if found is not None:
                return found
    return None


def _field(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return _field(node.func.value)
        return _field(node.args[0])
    if isinstance(node, ast.Subscript):
        indices = [n.value for n in ast.walk(node.slice)
                   if isinstance(n, ast.Constant) and isinstance(n.value, (int, str))]
        return _field(node.value) + ("_" + str(indices[-1]) if indices else "")
    raise ValueError("Unsupported native instruction field")


def language_template(node):
    """Keep native wording; dynamic fields are filled by the native episode."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join("{" + _field(part.value) + "}"
                       if isinstance(part, ast.FormattedValue) else language_template(part)
                       for part in node.values)
    raise ValueError("Native instruction must be a string or f-string")


def robodojo_instruction(tree, name):
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    definition = method(classes, name, "gen_instruction")
    if definition is not None:
        for node in ast.walk(definition):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "templates" for target in node.targets
            ) or isinstance(node, ast.Return):
                value = node.value
                if isinstance(value, ast.List) and value.elts:
                    instruction = language_template(value.elts[0])
                    if instruction.strip():
                        return instruction
    raise ValueError(f"RoboDojo task lacks a native instruction: {name}")


def robocasa_instruction(classes, name):
    definition = method(classes, name, "get_ep_meta")
    if definition is not None:
        for node in ast.walk(definition):
            if not isinstance(node, ast.Assign):
                continue
            if (isinstance(node.value, (ast.Constant, ast.JoinedStr))
                    and any(isinstance(target, ast.Subscript)
                            and isinstance(target.slice, ast.Constant)
                            and target.slice.value == "lang" for target in node.targets)):
                return language_template(node.value)
    raise ValueError(f"RoboCasa task lacks a native instruction: {name}")
