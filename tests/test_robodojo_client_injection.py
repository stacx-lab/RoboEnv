"""Optional policy-client injection without importing Isaac or forging sys.modules.

The runtime wiring and lazy default helper are always tested. With locked sources
present, execute the actual upstream constructor AST with explicit lightweight
world collaborators. Scheduled source CI includes those constructor tests.
"""

import ast
import builtins
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/RoboDojo/src/eval_client/eval_env.py"
PATCH = ROOT / "third_party/patches/robodojo-optional-policy-client.patch"
BACKEND = ROOT / "runtime/src/roboenv_runtime/adapters/robodojo/backend.py"


def compiled(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "isolated-constructor.py", "exec"), namespace)
    return namespace


def helper_source():
    additions = [
        line[1:]
        for line in PATCH.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    first = additions.index("def _default_model_client_factory(**kwargs):")
    last = additions.index("def create_eval_env(")
    return "\n".join(additions[first:last])


def test_policy_import_is_lazy_and_missing_optional_package_fails_only_on_use():
    imports = []

    def blocked(name, *args, **kwargs):
        imports.append(name)
        raise ModuleNotFoundError(name)

    namespace = {"__builtins__": {**vars(builtins), "__import__": blocked}}
    exec(helper_source(), namespace)
    assert imports == []
    with pytest.raises(ModuleNotFoundError, match="client_server"):
        namespace["_default_model_client_factory"](url="ws://localhost:1")
    assert imports == ["client_server.ws.model_client"]
    assert "-from client_server.ws.model_client import WsModelClient" in PATCH.read_text()


def test_default_helper_preserves_client_kwargs_and_proxy_compatibility():
    events = []
    expected = {"url": "ws://localhost:1", "evaluation_id": "fixture"}

    def factory(**kwargs):
        events.append(("construct", kwargs))
        return "client"

    def importer(name, *args, **kwargs):
        assert name == "client_server.ws.model_client"
        events.append(("import", name))
        return SimpleNamespace(WsModelClient=factory)

    namespace = {
        "__builtins__": {**vars(builtins), "__import__": importer},
        "_patch_websockets_proxy_compat": lambda: events.append(("compat", None)),
    }
    exec(helper_source(), namespace)
    assert namespace["_default_model_client_factory"](**expected) == "client"
    assert events == [
        ("import", "client_server.ws.model_client"),
        ("compat", None),
        ("construct", expected),
    ]


def test_runtime_passes_its_client_without_modifying_module_or_replacing_it_later():
    tree = ast.parse(BACKEND.read_text())
    noop = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_NoopModelClient"
    )
    env_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RoboDojoEnv"
    )
    build = next(
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build"
    )
    captured = []
    sentinel = object()

    def create(config, app, *, model_client):
        captured.append((config, app, model_client))
        return SimpleNamespace(model_client=model_client)

    evaluation = SimpleNamespace(create_eval_env=create, WsModelClient=sentinel)
    source_namespace = SimpleNamespace(eval_client=SimpleNamespace(eval_env=evaluation))

    def importer(name, *args, **kwargs):
        assert name == "src.eval_client.eval_env"
        return source_namespace

    namespace = compiled(
        [noop, build], {"__builtins__": {**vars(builtins), "__import__": importer}}
    )
    wrapper = SimpleNamespace(seed=7, _app=object(), _assemble_config=lambda seed: {"seed": seed})
    namespace["_build"](wrapper)
    assert captured[0][:2] == ({"seed": 7}, wrapper._app)
    assert isinstance(captured[0][2], namespace["_NoopModelClient"])
    assert wrapper._env.model_client is captured[0][2]
    assert evaluation.WsModelClient is sentinel
    assert wrapper._env.model_client.call(func_name="reset") is None


@pytest.fixture
def constructor(tmp_path, monkeypatch):
    if not UPSTREAM.is_file():
        pytest.skip("constructor integration requires selectively initialized RoboDojo source")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBODOJO_RUN_ID", "fixture-run")
    events = []

    class Task:
        def __init__(self, config, app, **kwargs):
            self.task_kwargs = kwargs
            self.num_envs, self.dt, self.env_seed_list = 1, 0.01, [0]
            self.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(replay=False))

    class Obs:
        def __init__(self, **kwargs):
            self.config = kwargs

    class Seeds:
        def __init__(self, config):
            pass

        def init_eval(self, **kwargs):
            events.append(("seeds", kwargs))

    def registry(name):
        assert name == "task.RoboDojo.task_registry"
        events.append(("registry", name))
        return SimpleNamespace(load_task_class=lambda task: ("fixture-task", Task))

    def missing_policy(name, *args, **kwargs):
        raise AssertionError("optional policy imported while using explicit injection")

    tree = ast.parse(UPSTREAM.read_text())
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "client_server.ws.model_client"
        for node in tree.body
    )
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"create_eval_env", "_default_model_client_factory"}
    ]
    namespace = compiled(
        selected,
        {
            "__builtins__": {**vars(builtins), "__import__": missing_policy},
            "importlib": SimpleNamespace(import_module=registry),
            "BENCHMARK": "RoboDojo",
            "datetime": datetime,
            "os": os,
            "deepcopy": deepcopy,
            "np": np,
            "ObsManager": Obs,
            "SeedManager": Seeds,
            "get_robot_action_dim_info": lambda **kwargs: {"fixture": 1},
            "_patch_websockets_proxy_compat": lambda: events.append(("compat", None)),
        },
    )
    config = SimpleNamespace(
        eval_cfg={
            "task_name": "fixture-task",
            "policy_name": "fixture",
            "config_name": "fixture",
            "num_envs": 1,
        },
        deploy_cfg={},
    )
    return namespace, config, events


class Client:
    def call(self, **kwargs):
        return None

    def __bool__(self):
        return False  # an explicit falsey object must not select the default


def test_injected_instance_needs_no_policy_package_or_network_port(constructor):
    namespace, config, events = constructor
    client = Client()
    env = namespace["create_eval_env"](config, object(), model_client=client, task_option=3)
    assert env.model_client is client
    assert env.port is None
    assert env.task_kwargs == {"task_option": 3}
    assert not any(event[0] == "compat" for event in events)


def test_factory_gets_original_websocket_options_without_importing_policy(constructor):
    namespace, config, events = constructor
    config.deploy_cfg = {"port": 1234, "repeat_index": 2, "ws_ping_timeout_s": 9.0}
    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return Client()

    env = namespace["create_eval_env"](config, object(), model_client_factory=factory)
    assert isinstance(env.model_client, Client)
    assert captured == [
        {
            "url": "ws://localhost:1234",
            "evaluation_id": "fixture-run",
            "trial_id": "fixture-task-fixture-run",
            "action_case_id": "fixture-task_case",
            "repeat_index": 2,
            "ws_ping_interval_s": 20.0,
            "ws_ping_timeout_s": 9.0,
        }
    ]
    assert not any(event[0] == "compat" for event in events)


def test_default_policy_branch_still_uses_default_factory(constructor):
    namespace, config, _ = constructor
    config.deploy_cfg = {"port": 1234, "policy_server_url": "ws://localhost:5678"}
    captured = []
    namespace["_default_model_client_factory"] = lambda **kwargs: (
        captured.append(kwargs) or Client()
    )
    namespace["create_eval_env"](config, object())
    assert captured[0]["url"] == "ws://localhost:5678"


def test_default_policy_still_requires_port(constructor):
    namespace, config, _ = constructor
    with pytest.raises(ValueError, match="Port must"):
        namespace["create_eval_env"](config, object())


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"model_client": Client(), "model_client_factory": Client}, ValueError),
        ({"model_client": object()}, TypeError),
        ({"model_client_factory": 5}, TypeError),
    ],
)
def test_invalid_injection_fails_before_task_construction(constructor, kwargs, error):
    namespace, config, events = constructor
    with pytest.raises(error):
        namespace["create_eval_env"](config, object(), **kwargs)
    assert events == []


def test_invalid_factory_result_is_not_accepted(constructor):
    namespace, config, _ = constructor
    config.deploy_cfg = {"port": 1234}
    with pytest.raises(TypeError, match="must return"):
        namespace["create_eval_env"](config, object(), model_client_factory=lambda **kwargs: None)
