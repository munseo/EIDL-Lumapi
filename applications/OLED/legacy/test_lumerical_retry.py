import numpy as np

from msopt.Lumerical_utill import LumericalOptimizationProblem


class _FakeFdtd:
    def __init__(self):
        self.loads = []
        self.set_calls = []

    def switchtolayout(self):
        return None

    def setnamed(self, name, prop, value):
        self.set_calls.append((name, prop, value))

    def load(self, path):
        self.loads.append(path)


class _FakeSimulator:
    design_monitor_name = "design_monitor"
    _last_run_fsp_path = "/tmp/Adjoint_run.fsp"

    def __init__(self):
        self.fdtd = _FakeFdtd()
        self.run_count = 0

    def run(self, name, save=True):
        assert name == "Adjoint_run"
        assert save is True
        self.run_count += 1

    def _run_log_tail(self, name):
        assert name == "Adjoint_run"
        return "transient license failure"


def test_adjoint_result_retry_recovers_missing_monitor(monkeypatch):
    monkeypatch.setenv("LUMERICAL_ADJOINT_RESULT_RETRIES", "3")
    monkeypatch.setenv("LUMERICAL_ADJOINT_RESULT_RETRY_DELAY", "0")

    problem = LumericalOptimizationProblem.__new__(LumericalOptimizationProblem)
    problem.sim = _FakeSimulator()
    result_attempts = {"count": 0}

    def fake_get_field(monitor_name, H_field=False, check_design_alignment=False):
        assert monitor_name == "design_monitor"
        assert H_field is False
        assert check_design_alignment is True
        result_attempts["count"] += 1
        if result_attempts["count"] < 3:
            raise RuntimeError("monitor result unavailable")
        return np.ones((3, 1, 1, 1, 1), dtype=np.complex128)

    problem.get_field = fake_get_field
    result = problem._run_adjoint_with_result_retry("adjoint_source_0")

    assert result.shape == (3, 1, 1, 1, 1)
    assert problem.sim.run_count == 3
    assert problem.sim.fdtd.loads == [
        "/tmp/Adjoint_run.fsp",
        "/tmp/Adjoint_run.fsp",
    ]
    assert (
        "adjoint_source_0",
        "enabled",
        True,
    ) in problem.sim.fdtd.set_calls

