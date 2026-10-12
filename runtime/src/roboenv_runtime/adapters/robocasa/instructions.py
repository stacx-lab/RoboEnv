"""Public names and assignments sampled by native kitchen initialization."""

# Only roles already disclosed by each task's native language belong here.
OBJECT_ROLES = {
    "LoadCondimentsInFridge": ("condiment1", "condiment2"),
    "FilterMicrowavableItem": ("fruit", "meat"),
    "DivideBuffetTrays": ("vegetable1", "vegetable2", "meat1", "meat2"),
    "MultistepSteaming": ("vegetable1",),
    "ScalePortioning": ("meat",),
    "PrepareVeggieDip": ("vegetable",),
    "PrepareVegetableRoasting": ("vegetable",),
}


def task_context(task_name, env):
    names = {role: env.get_obj_lang(role) for role in OBJECT_ROLES.get(task_name, ())}
    context = {"objects": names} if names else {}
    if task_name == "DivideBuffetTrays":
        # Match the native language's grouping, including two items of one kind.
        for kind in ("vegetable", "meat"):
            first, second = names[kind + "1"], names[kind + "2"]
            context[kind + "s"] = first + "s" if first == second else f"{first} and {second}"
    if task_name == "MultistepSteaming":
        context["burner"] = env.knob.replace("_", " ")
    return context
