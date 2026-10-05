"""Discover SIMPLE's decoupled G1 registrations using syntax, without GPU imports."""

import ast

from roboenv.tasks.model import Task

TASK_NAMES = {
    "g1_wholebody_locomotion_pick_between_tables_teleop": "transfer-between-tables",
}


def _instruction(cls, template, package):
    """Fill native language with the effective default target used by success checks."""
    if "{}" not in template:
        # Articulated tasks name their fixture directly. Their unused target
        # actor is not the object evaluated by check_success.
        return template.strip()
    target = None
    for node in ast.walk(cls):
        if isinstance(node, ast.keyword) and node.arg == "target":
            if isinstance(node.value, ast.Call):
                for keyword in node.value.keywords:
                    if keyword.arg == "asset_id":
                        target = ast.literal_eval(keyword.value)
    constructor = next(n for n in cls.body
                       if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = constructor.args.defaults
    for argument, default in zip(constructor.args.args[-len(defaults):], defaults, strict=True):
        if argument.arg in {"target", "target_object"}:
            override = ast.literal_eval(default)
            if override is not None:
                target = override
    if not isinstance(target, str) or not target.startswith("graspnet1b:"):
        raise ValueError(f"Unsupported SIMPLE instruction target: {target}")
    names_tree = ast.parse((package / "assets/graspnet.py").read_text())
    names = next(ast.literal_eval(node.value) for node in names_tree.body
                 if isinstance(node, ast.Assign) and any(
                     isinstance(name, ast.Name) and name.id == "GraspNet_1B_Object_Names"
                     for name in node.targets))
    return template.format(names[int(target.split(":")[1])]).strip()


def entries(source):
    package = source / "src/simple"
    for node in ast.parse((package / "envs/__init__.py").read_text()).body:
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "register"
        ):
            continue
        values = {item.arg: ast.literal_eval(item.value) for item in node.value.keywords}
        if values.get(
            "entry_point"
        ) != "simple.envs.sonic_loco_manip:SonicLocoManipEnv" or not values["id"].endswith(
            "Teleop-v0"
        ):
            continue
        task = values["kwargs"]["task"]
        path = package / "tasks" / (task + ".py")
        metadata = {}
        description = task.replace("_", " ")
        instructions = []
        instruction = ""
        for cls in ast.parse(path.read_text()).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for field in cls.body:
                if not isinstance(field, ast.AnnAssign) or not isinstance(field.target, ast.Name):
                    continue
                if field.target.id == "metadata":
                    metadata = ast.literal_eval(field.value)
                if field.target.id == "description":
                    description = ast.literal_eval(field.value)
            for call in ast.walk(cls):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "LanguageDRCfg"):
                    continue
                for keyword in call.keywords:
                    if keyword.arg == "instructions":
                        instructions = ast.literal_eval(keyword.value)
                        if instructions:
                            instruction = _instruction(cls, instructions[0], package)
        if not instructions:
            raise ValueError(f"SIMPLE task needs an instruction based on its success rule: {task}")
        if metadata.get("physics_dt") != 0.005 or metadata.get("render_hz") != 50:
            raise ValueError(f"Unsupported SIMPLE task cadence: {task}")
        yield {
            "env_id": values["id"],
            "task": task,
            "metadata": metadata,
            "description": description,
            "instruction": instruction,
            "source": path,
            "name": TASK_NAMES.get(
                task, task.removeprefix("g1_wholebody_").removesuffix("_teleop").replace("_", "-")
            ),
        }


def discover_tasks(root):
    for entry in entries(root / "third_party/SIMPLE"):
        yield (
            Task(
                id="simple/" + entry["name"],
                robot="simple/simple-g1",
                instruction=entry["instruction"],
                budget_mode="practice",
                execution_mode="realtime",
                backend_kwargs={
                    "env_id": entry["env_id"],
                    "sim_mode": "mujoco_isaac",
                    "dr_level": 0,
                    "arm_gravity_compensation": True,
                },
            ),
            entry["source"],
        )
