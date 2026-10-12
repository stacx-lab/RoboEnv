"""Scene language is public, stable, and independent of native goal rendering."""

import itertools
from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter
from roboenv_runtime.adapters.robodojo.adapter import RoboDojoAdapter


@pytest.mark.parametrize("name,names,extra,expected", [
    ("LoadCondimentsInFridge", {"condiment1": "ketchup", "condiment2": "mustard"}, {},
     {"objects": {"condiment1": "ketchup", "condiment2": "mustard"}}),
    ("FilterMicrowavableItem", {"fruit": "apple", "meat": "steak"}, {},
     {"objects": {"fruit": "apple", "meat": "steak"}}),
    ("DivideBuffetTrays", {"vegetable1": "carrot", "vegetable2": "carrot",
                          "meat1": "steak", "meat2": "chicken"}, {},
     {"objects": {"vegetable1": "carrot", "vegetable2": "carrot",
                  "meat1": "steak", "meat2": "chicken"},
      "vegetables": "carrots", "meats": "steak and chicken"}),
    ("MultistepSteaming", {"vegetable1": "broccoli"}, {"knob": "front_left"},
     {"objects": {"vegetable1": "broccoli"}, "burner": "front left"}),
    ("ScalePortioning", {"meat": "chicken"}, {}, {"objects": {"meat": "chicken"}}),
    ("PrepareVeggieDip", {"vegetable": "cucumber"}, {}, {"objects": {"vegetable": "cucumber"}}),
    ("PrepareVegetableRoasting", {"vegetable": "carrot"}, {},
     {"objects": {"vegetable": "carrot"}}),
    ("StoreDumplings", {}, {}, {}),
])
def test_robocasa_context_only_discloses_language_roles(name, names, extra, expected):
    adapter = RoboCasaAdapter.__new__(RoboCasaAdapter)
    native = SimpleNamespace(get_obj_lang=names.__getitem__, private_pose=[1, 2, 3], **extra)
    adapter._env = SimpleNamespace(task_name=name, env=native)
    assert adapter.task_context() == expected
    assert adapter.task_context() == expected
    if name == "MultistepSteaming":
        names["vegetable1"] = "corn"
        native.knob = "back_right"
        assert adapter.task_context() == {"objects": {"vegetable1": "corn"}, "burner": "back right"}


class Layout:
    def __init__(self, metadata):
        self.metadata = metadata

    def get_instance_metadata(self, *, label, env_idx):
        assert env_idx == 0
        return self.metadata[label]

    def get_labels_by_prefix(self, *, prefix, env_idx):
        assert env_idx == 0
        return [label for label in self.metadata if label.startswith(prefix)]

    def get_label_descriptions(self, *, label, env_idx):
        return self.get_instance_metadata(label=label, env_idx=env_idx)["description"]


def dojo(name, metadata):
    adapter = RoboDojoAdapter.__new__(RoboDojoAdapter)
    # No reward/observation/description manager: context reads cannot invoke them.
    adapter._env = SimpleNamespace(task_name=name, _env=SimpleNamespace(
        scene_manager=SimpleNamespace(layout_manager=Layout(metadata)), private_pose=[1, 2, 3]))
    return adapter


def test_robodojo_categories_and_pickup_are_stable_and_follow_new_layout():
    metadata = {"cat0_0": {"model_name": "fruit"}, "cat1_0": {"model_name": "toy"},
                "cat2_0": {"model_name": "cup"}}
    adapter = dojo("classify_objects_by_language", metadata)
    assert adapter.task_context() == {"cat0": "fruit", "cat1": "toy", "cat2": "cup"}
    metadata["cat0_0"]["model_name"] = "bottle"
    assert adapter.task_context()["cat0"] == "bottle"
    del metadata["cat2_0"]
    with pytest.raises(ValueError, match="category: cat2"):
        adapter.task_context()
    metadata = {"target": {"description": ["blue cup", "small blue cup"]}}
    adapter = dojo("general_pickup", metadata)
    assert adapter.task_context() == adapter.task_context() == {"target": "blue cup"}
    metadata["target"]["description"] = ["red bottle"]
    assert adapter.task_context() == {"target": "red bottle"}
    metadata["target"]["description"] = []
    with pytest.raises(ValueError, match="target description"):
        adapter.task_context()
    assert dojo("solve_equation", {}).task_context() == {}


def test_robodojo_color_assignments_follow_layout_roles():
    bottles = [(0, "red"), (1, "turquoise"), (2, "violet")]
    bowls = [(0, "black"), (1, "white"), (2, "brown")]
    for a, b in itertools.product(itertools.permutations(bottles), itertools.permutations(bowls)):
        metadata = {f"{kind}_{i}": {"model_id": model}
                    for kind, entries in (("bottle", a), ("bowl", b))
                    for i, (model, _) in enumerate(entries)}
        expected = {f"bottle_color_{i}": color for i, (_, color) in enumerate(a)}
        expected.update({f"bowl_color_{i + 3}": color for i, (_, color) in enumerate(b)})
        assert dojo("pour_by_language", metadata).task_context() == expected
    colors = [(1, "blue"), (2, "green"), (3, "red"), (7, "yellow"), (8, "orange"), (9, "cyan")]
    for entries in itertools.permutations(colors, 3):
        metadata = {f"block_{i}": {"model_id": model} for i, (model, _) in enumerate(entries)}
        expected = {f"color_{i}": color for i, (_, color) in enumerate(entries)}
        assert dojo("stack_blocks_by_language", metadata).task_context() == expected
