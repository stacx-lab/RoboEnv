"""Discover SIMPLE's decoupled G1 registrations using syntax, without GPU imports."""

import ast

from roboenv.tasks.model import Task

REPRESENTATIVES = {
    "g1_wholebody_xmove_pick_teleop": "g1-xmove-pick",
    "g1_wholebody_locomotion_pick_between_tables_teleop": "g1-transfer-between-tables",
    "g1_wholebody_bend_pick_teleop": "g1-bend-pick",
}


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
        if metadata.get("physics_dt") != 0.005 or metadata.get("render_hz") != 50:
            raise ValueError(f"Unsupported SIMPLE task cadence: {task}")
        yield {
            "env_id": values["id"],
            "task": task,
            "metadata": metadata,
            "description": description,
            "source": path,
            "name": REPRESENTATIVES.get(
                task, task.removeprefix("g1_wholebody_").removesuffix("_teleop").replace("_", "-")
            ),
        }


def discover_tasks(root):
    for entry in entries(root / "third_party/SIMPLE"):
        yield (
            Task(
                id="simple/" + entry["name"],
                backend="simple",
                backend_task=entry["env_id"].replace("/", ".", 1),
                robot="simple/simple-g1",
                instruction=entry["description"],
                budget_mode="practice",
                action_abstraction="devices",
                backend_kwargs={
                    "env_id": entry["env_id"],
                    "realtime": True,
                    "sim_mode": "mujoco_isaac",
                    "dr_level": 0,
                    "arm_gravity_compensation": True,
                },
            ),
            entry["source"],
        )
