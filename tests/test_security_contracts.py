"""Negative cases at owner configuration, device dispatch and verification boundaries."""

import builtins
import io
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from roboenv.client import Client, UncertainResult
from roboenv.devices.bindings.spec import BindingSpec
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, DeviceError
from roboenv.devices.service import DeviceService
from roboenv.harbor.deployment import load_deployment, native_service
from roboenv.harbor.taskset import deployment_environment, export_taskset, run_taskset
from roboenv.tasks.artifacts import Bundle
from roboenv.tasks.model import Task
from roboenv.validation import action_validator, json_loads, json_value, yaml_load
from roboenv_runtime.runtime.protocol import ProtocolError, decode_ndjson

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["", ".", "../escape", "/app/escape", "a//b", "bundle.json"])
def test_bundle_rejects_ambiguous_or_reserved_paths(tmp_path, name):
    with pytest.raises(ValueError, match="safe relative"):
        Bundle(tmp_path).add(name, "test")


def test_bundle_replacement_does_not_modify_external_hard_links(tmp_path):
    out = Bundle(tmp_path / "out")
    out.add("instruction.md", "old")
    out.commit({})
    external_content = tmp_path / "retained-content"
    external_manifest = tmp_path / "retained-manifest"
    os.link(out.root / "instruction.md", external_content)
    os.link(out.root / "bundle.json", external_manifest)
    old_manifest = external_manifest.read_bytes()
    updated = Bundle(out.root)
    updated.add("instruction.md", "new")
    updated.commit({"updated": True})
    assert (out.root / "instruction.md").read_text() == "new"
    assert external_content.read_text() == "old"
    assert external_manifest.read_bytes() == old_manifest


def test_bundle_rejects_even_in_tree_symlink_ancestors(tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "linked").symlink_to("actual", target_is_directory=True)
    bundle = Bundle(tmp_path)
    bundle.add("linked/file.md", "test")
    with pytest.raises(ValueError, match="real directories"):
        bundle.commit({})
    assert not (tmp_path / "actual/file.md").exists()


@pytest.mark.parametrize(
    "raw",
    ['{"seed":0,"seed":1}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e9999}'],
)
def test_json_configuration_has_no_ambiguous_or_nonfinite_values(raw):
    with pytest.raises(ValueError):
        json_loads(raw)


def test_json_configuration_rejects_non_string_keys_and_recursive_aliases():
    with pytest.raises(ValueError, match="string keys"):
        json_value({"nested": {0: "value"}})
    recursive = []
    recursive.append(recursive)
    with pytest.raises(ValueError, match="recursive"):
        json_value(recursive)
    nested = 0
    for _ in range(130):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        json_value(nested)


def test_yaml_alias_expansion_is_rejected_before_export():
    with pytest.raises(ValueError, match="aliases"):
        yaml_load("first: &shared [value]\nsecond: [*shared, *shared]\n")


def test_task_and_profile_duplicate_yaml_keys_are_not_silently_accepted(tmp_path):
    source = ROOT / "catalog/tasks/robocasa/arrange-vegetables/task.yaml"
    task = tmp_path / "task.yaml"
    task.write_text(source.read_text() + "\nseed: 0\nseed: 1\n")
    with pytest.raises(ValueError, match="unique string"):
        Task.load(task)
    profile = tmp_path / "devices/arm/profile.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("type: arm\nintrinsics: {}\nintrinsics: {joint_count: 7}\n")
    binding = BindingSpec.parse("arm", {"binding": "test/1", "entity": "arm"})
    with pytest.raises(ValueError, match="unique string"):
        binding.definition(tmp_path)


def deployment(tmp_path, **updates):
    depot = tmp_path / "asset-depot"
    depot.mkdir(exist_ok=True)
    spec = {
        "schema": "roboenv/deployment/1",
        "backend": "libero",
        "legacy_tree_required": False,
        "image": "sha256:" + "a" * 64,
        "mounts": [
            {
                "source": depot.name,
                "target": "/opt/roboenv/runtime/assets",
                "read_only": True,
            }
        ],
        **updates,
    }
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(spec))
    return path, spec


@pytest.mark.parametrize(
    "target",
    [
        "/opt/roboenv/runtime/",
        "/opt/roboenv/runtime/../code",
        "/opt/roboenv/runtime/assets:rw",
        "/opt/roboenv/runtime//assets",
        "/opt/roboenv/runtime/assets\nother",
    ],
)
def test_deployment_rejects_mount_aliases_and_compose_syntax(tmp_path, target):
    path, spec = deployment(tmp_path)
    spec["mounts"][0]["target"] = target
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="runtime data prefix"):
        load_deployment(path, "libero")


def test_run_time_depot_mapping_enforces_readonly_and_redacts_paths(tmp_path):
    path, spec = deployment(tmp_path)
    spec["mounts"][0]["read_only"] = False
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="read-only"):
        deployment_environment({"libero": path})
    spec["mounts"][0]["read_only"] = True
    spec["mounts"][0]["source"] = "missing-assets"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="independent existing") as error:
        deployment_environment({"libero": path})
    assert str(tmp_path) not in str(error.value)


def test_deployment_rejects_overlapping_depots(tmp_path):
    path, spec = deployment(tmp_path)
    spec["mounts"].append({**spec["mounts"][0], "target": "/opt/roboenv/runtime/assets/nested"})
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="overlap"):
        load_deployment(path, "libero")


def test_healthcheck_does_not_execute_an_unverified_interpreter_depot(tmp_path):
    path, _ = deployment(tmp_path)
    task = Task.load(ROOT / "catalog/tasks/libero/bowl-into-bottom-drawer/task.yaml")
    service = native_service(path, task, portable=True)
    assert service["healthcheck"]["test"] == [
        "CMD",
        "/usr/bin/python3",
        "-I",
        "/opt/serve/healthcheck.py",
    ]


def test_task_runner_rejects_changed_deployment_identity_before_launch(tmp_path, monkeypatch):
    path, spec = deployment(tmp_path)
    mapping = tmp_path / "deployments.json"
    mapping.write_text(json.dumps({"libero": path.name}))
    wheel = tmp_path / "roboenv-test.whl"
    wheel.write_bytes(b"test-only input")
    destination = tmp_path / "generated"
    export_taskset(ROOT, ROOT / "catalog/tasks", mapping, destination, wheel, backends=["libero"])
    monkeypatch.setattr("roboenv.harbor.taskset.shutil.which", lambda name: name)
    spec["image"] = "sha256:" + "b" * 64
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="regenerate the taskset"):
        run_taskset(destination, mapping, agent="test-agent")


def device(schema):
    calls = []
    bound = BoundDevice(
        DeviceDefinition("arm", "arm"),
        {"write": Action(lambda args, context: calls.append(args), schema)},
        implementation="test-binding",
    )
    return DeviceService([bound]), calls


@pytest.mark.parametrize(
    "arguments",
    [{}, {"target": True}, {"target": 3}, {"target": 0.5, "ignored": 1}, {"target": float("nan")}],
)
def test_invalid_motion_is_rejected_before_dispatch_without_effects(arguments):
    service, calls = device(
        {
            "type": "object",
            "properties": {"target": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["target"],
            "additionalProperties": False,
        }
    )
    with pytest.raises(DeviceError) as error:
        service.call({"device": "arm", "action": "write", "args": arguments})
    assert error.value.code == "invalid_args"
    assert error.value.details == {"effects": "none"}
    assert calls == []
    assert service.sequence == 0


def test_schema_local_references_work_without_remote_fetch():
    service, calls = device(
        {
            "type": "object",
            "properties": {"target": {"$ref": "#/$defs/position"}},
            "$defs": {"position": {"type": "number", "minimum": 0, "maximum": 1}},
        }
    )
    service.call({"device": "arm", "action": "write", "args": {"target": 0.5}})
    assert calls == [{"target": 0.5}]
    with pytest.raises(ValueError, match="local fragment"):
        device({"$ref": "https://example.invalid/untrusted-schema"})


def test_device_server_missing_extra_fails_with_install_guidance(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("test-only missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match=r"roboenv\[server\]"):
        action_validator({"type": "object"})


@pytest.mark.parametrize("generation", [True, -1, 0.5, "1"])
def test_client_generation_is_not_silently_coerced(generation):
    with pytest.raises(ValueError, match="nonnegative integer"):
        Client(generation=generation)


@pytest.mark.parametrize("request_id", ["", False, "not a request identifier"])
def test_client_rejects_invalid_request_id_before_connecting(request_id):
    with pytest.raises(ValueError, match="request_id"):
        Client().request("health", request_id=request_id)


@pytest.mark.parametrize("defect", ["duplicate", "nonfinite", "boolean-generation"])
def test_untrustworthy_response_is_uncertain_and_never_retried(monkeypatch, defect):
    sent = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def settimeout(self, _):
            pass

        def connect(self, _):
            pass

        def sendall(self, payload):
            sent.append(json.loads(payload))

        def makefile(self, _):
            response = {
                key: sent[-1][key] for key in ("protocol", "session", "generation", "request_id")
            }
            response.update(ok=True, result={})
            if defect == "boolean-generation":
                response["generation"] = False
            raw = json.dumps(response)
            if defect == "duplicate":
                raw = raw[:-1] + ', "ok": false}'
            elif defect == "nonfinite":
                raw = raw[:-1] + ', "untrusted": NaN}'
            return io.BytesIO(raw.encode() + b"\n")

    monkeypatch.setattr("roboenv.client.socket.socket", lambda *_: Connection())
    with pytest.raises(UncertainResult) as error:
        Client().request("health", request_id="check-once")
    assert error.value.request_id == "check-once"
    assert len(sent) == 1


def test_wire_client_import_requires_only_standard_library():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, 'src'); from roboenv.client import Client; "
            "assert 'yaml' not in sys.modules; assert 'jsonschema' not in sys.modules; Client()",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        b'{"op":"observe","op":"call"}\n',
        b'{"args":{"value":NaN}}\n',
        b'{"result":{"value":1e9999}}\n',
    ],
)
def test_runtime_wire_rejects_ambiguous_and_nonfinite_json_before_dispatch(raw):
    with pytest.raises(ProtocolError) as error:
        decode_ndjson(raw)
    assert error.value.code == "invalid_json"


@pytest.mark.parametrize("raw", [b'{"reward":0,"reward":1}', b'{"value":1e9999}'])
def test_verifier_rejects_ambiguous_or_overflow_json(raw):
    verifier = runpy.run_path(str(ROOT / "src/roboenv/assets/verify.py"))
    with pytest.raises(ValueError):
        verifier["strict_json"](raw)


def test_verifier_missing_task_overwrites_stale_success(tmp_path, monkeypatch):
    verifier = runpy.run_path(str(ROOT / "src/roboenv/assets/verify.py"))
    out = tmp_path / "verifier"
    out.mkdir()
    (out / "reward.json").write_text('{"reward": 1.0}')
    (out / "reward.txt").write_text("1.0")
    monkeypatch.setenv("ROBOENV_VERIFIER_OUT", str(out))
    monkeypatch.setenv("ROBOENV_RUN_DIR", str(tmp_path / "absent-run"))
    # The package asset has no adjacent task.json: only generated tasks supply it.
    assert verifier["main"]() == 1
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0.0}
    assert (out / "reward.txt").read_text().strip() == "0.0"
    assert json.loads((out / "submit.json").read_text())["valid"] is False


@pytest.mark.parametrize("operation", ["submit", "reset", "restore", "shutdown", "exec"])
def test_public_client_refuses_trial_control_operations(operation):
    with pytest.raises(ValueError, match="not an agent operation"):
        Client("/unused").request(operation)
