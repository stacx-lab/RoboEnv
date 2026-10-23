"""Diagnostic observers preserve native call counts, state and failures."""
from types import SimpleNamespace

import pytest
from roboenv_runtime.adapters.robocasa.diagnostics import PredicateDiagnostics


@pytest.mark.parametrize("result", [False, True])
def test_native_success_is_called_once_and_diagnostic_failures_do_not_change_it(result):
    class Native:
        calls = 0
        sim = SimpleNamespace(data=SimpleNamespace(time=0.05))

        def _check_success(self):
            self.calls += 1
            return result

    native = Native()
    observer = PredicateDiagnostics.__new__(PredicateDiagnostics)
    observer.env, observer.originals, observer.errors = native, {}, []
    records = []
    observer._wrap("_check_success", records, lambda: {"calls": native.calls})
    assert native._check_success() is result and native.calls == 1
    assert records == [{"native_time_s": 0.05, "calls": 1, "native_result": result}]
    observer.detach()
    assert "_check_success" not in native.__dict__
    observer._wrap("_check_success", records, lambda: 1 / 0)
    assert native._check_success() is result and native.calls == 2
    assert observer.errors == ["ZeroDivisionError: division by zero"]
    observer.detach()


def test_diagnostics_propagate_native_failure_without_recording_an_invented_result():
    def fail():
        raise RuntimeError("native failure")

    native = SimpleNamespace(_check_success=fail)
    observer = PredicateDiagnostics.__new__(PredicateDiagnostics)
    observer.env, observer.originals, observer.errors = native, {}, []
    records = []
    observer._wrap("_check_success", records, lambda: {})
    with pytest.raises(RuntimeError, match="native failure"):
        native._check_success()
    assert records == [] and observer.errors == []
    observer.detach()
    assert native._check_success is fail
