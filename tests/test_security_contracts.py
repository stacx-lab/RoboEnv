"""Negative cases at owner configuration, device dispatch and verification boundaries."""

import builtins
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from roboenv.client import Client, UncertainResult
from roboenv.devices.bindings.spec import BindingSpec
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition, DeviceError
from roboenv.devices.service import DeviceService
from roboenv.tasks.artifacts import Bundle
from roboenv.tasks.model import Task
from roboenv.validation import action_validator, json_loads, json_value, yaml_load
from roboenv.verify import certify, refuse, strict_json
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


def test_bundle_regeneration_cannot_leave_stale_managed_files(tmp_path):
    bundle = Bundle(tmp_path / "task")
    bundle.add("instruction.md", "test")
    bundle.add("solution/solve.sh", "test")
    bundle.commit({})
    replacement = Bundle(bundle.root)
    replacement.add("instruction.md", "test")
    with pytest.raises(ValueError, match="stale managed"):
        replacement.commit({})


def test_bundle_manifest_symlink_is_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    out = tmp_path / "task"
    out.mkdir()
    (out / "bundle.json").symlink_to(outside)
    bundle = Bundle(out)
    bundle.add("instruction.md", "test")
    with pytest.raises(ValueError, match="manifest"):
        bundle.commit({})


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
    with pytest.raises(ValueError):
        strict_json(raw)


def test_verify_module_requires_only_standard_library(tmp_path):
    # Harnesses import roboenv.verify or copy the one file; neither may pull in the package.
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, 'src'); import roboenv.verify; "
            "loaded = sorted(m for m in sys.modules if m.startswith('roboenv')); "
            "assert loaded == ['roboenv', 'roboenv.verify'], loaded; "
            "assert 'yaml' not in sys.modules and 'jsonschema' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    source = (ROOT / "src/roboenv/verify.py").read_text()
    assert not re.search(r"^\s*(from|import) roboenv", source, re.MULTILINE)
    copied = tmp_path / "verify.py"
    copied.write_text(source)
    standalone = subprocess.run(
        [sys.executable, "-I", "-S", str(copied)], cwd=tmp_path, capture_output=True, text=True,
        timeout=10,
    )
    assert standalone.returncode == 0, standalone.stderr


def test_certify_unreachable_run_overwrites_stale_success(tmp_path):
    out = tmp_path / "verifier"
    out.mkdir()
    (out / "reward.json").write_text('{"reward": 1.0}')
    (out / "submit.json").write_text('{"valid": true, "trial_success": true}')
    (out / "actions.jsonl").write_text("stale evidence\n")
    report = certify(
        socket_path=tmp_path / "absent.sock", session="s", task_id="t", seed=0,
        budget_mode="official", journal=tmp_path / "absent-run/journal/actions.jsonl", out=out,
    )
    assert report["valid"] is False and report["trial_success"] is False
    assert report["error"] and "health" not in report and "submit" not in report
    assert "journal_capture_error" in report and "journal_artifact" not in report
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0.0}
    assert json.loads((out / "submit.json").read_text()) == report
    assert not (out / "actions.jsonl").exists() and not (out / "reward.txt").exists()
    # Garbage identity fails before any connection is attempted.
    (out / "reward.json").write_text('{"reward": 1.0}')
    report = certify(
        socket_path=tmp_path / "absent.sock", session="", task_id="t", seed=-1,
        budget_mode="official", journal=tmp_path / "absent.jsonl", out=out,
    )
    assert report["valid"] is False and "identity" in report["error"]
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0.0}
    # A run that cannot be identified at all is refused the same way.
    (out / "reward.json").write_text('{"reward": 1.0}')
    assert refuse(out, "KeyError: 'session'")["valid"] is False
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0.0}
    assert json.loads((out / "submit.json").read_text())["error"] == "KeyError: 'session'"


@pytest.mark.parametrize("operation", ["submit", "reset", "restore", "shutdown", "exec"])
def test_public_client_refuses_trial_control_operations(operation):
    with pytest.raises(ValueError, match="not an agent operation"):
        Client("/unused").request(operation)
