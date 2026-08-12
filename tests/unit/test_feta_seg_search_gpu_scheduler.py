from __future__ import annotations

import inspect
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

import auto_researcher.tasks.feta_seg_search.gpu_scheduler as scheduler_module
from auto_researcher.tasks.feta_seg_search.gpu_scheduler import (
    GPU_SCHEDULER_VERSION,
    GPUComputeProcess,
    GPUProbeResult,
    GPUSchedulerPolicy,
    NvidiaSmiGPUProbe,
    REGISTERED_FIDELITIES,
    gpu_scheduler_policy,
    scheduler_telemetry_is_valid,
    wait_for_gpu_admission,
)
from auto_researcher.tasks.feta_seg_search.runner import run_search_candidate
from auto_researcher.tasks.models import TaskRuntimeContext


FREE = GPUProbeResult(free_memory_mib=48000, utilization_percent=0)


class SequenceProbe:
    def __init__(self, observations: Sequence[GPUProbeResult]):
        self.observations = list(observations)
        self.calls = 0

    def probe(self, _physical_gpu_index: int) -> GPUProbeResult:
        index = min(self.calls, len(self.observations) - 1)
        self.calls += 1
        return self.observations[index]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _policy(mode="primary", **updates) -> GPUSchedulerPolicy:
    values = {
        "mode": mode,
        "physical_gpu_index": 0 if mode == "primary" else 1,
        "poll_seconds": 20,
        "stable_idle_seconds": 0 if mode == "primary" else 60,
        "minimum_free_memory_mib": 40000,
        "maximum_utilization_percent": 10,
        "allowed_fidelities": [25, 50, 100, 150, 300, 350]
        if mode == "primary"
        else [25, 50, 100],
    }
    values.update(updates)
    return GPUSchedulerPolicy.model_validate(values)


def _admit(policy, observations, *, current_pid=100):
    clock = FakeClock()
    logs: list[str] = []
    probe = SequenceProbe(observations)
    telemetry = wait_for_gpu_admission(
        policy,
        maximum_epochs=25,
        probe=probe,
        sleeper=clock.sleep,
        clock=clock,
        current_pid=current_pid,
        environ={"CUDA_VISIBLE_DEVICES": str(policy.physical_gpu_index)},
        logger=logs.append,
    )
    assert telemetry is not None
    return telemetry, probe, clock, logs


def test_disabled_policy_does_not_gate_existing_runs():
    assert REGISTERED_FIDELITIES == (25, 50, 100, 150, 300, 350)
    policy = gpu_scheduler_policy(TaskRuntimeContext())
    assert policy.mode == "disabled"
    assert (
        wait_for_gpu_admission(
            policy,
            maximum_epochs=300,
            probe=SequenceProbe([]),
            environ={},
        )
        is None
    )


def test_scheduler_configuration_is_strict():
    with pytest.raises(ValueError, match="scheduler_configuration_invalid"):
        gpu_scheduler_policy(
            TaskRuntimeContext(
                task_options={
                    "gpu_scheduler": {
                        "mode": "primary",
                        "physical_gpu_index": 0,
                        "poll_seconds": "20",
                    }
                }
            )
        )
    with pytest.raises(ValueError, match="scheduler_configuration_invalid"):
        gpu_scheduler_policy(
            TaskRuntimeContext(
                task_options={
                    "gpu_scheduler": {
                        "mode": "primary",
                        "physical_gpu_index": 0,
                        "unknown": True,
                    }
                }
            )
        )


def test_enabled_policy_rejects_absent_cuda_visible_devices():
    with pytest.raises(ValueError, match="gpu_binding_mismatch"):
        wait_for_gpu_admission(
            _policy(), maximum_epochs=25, probe=SequenceProbe([FREE]), environ={}
        )


@pytest.mark.parametrize("visible", ["1", "0,1", "gpu-uuid", " 0"])
def test_scheduler_rejects_cuda_binding_mismatch_or_malformed_value(visible):
    with pytest.raises(ValueError, match="gpu_binding_mismatch"):
        wait_for_gpu_admission(
            _policy(),
            maximum_epochs=25,
            probe=SequenceProbe([FREE]),
            environ={"CUDA_VISIBLE_DEVICES": visible},
        )


@pytest.mark.parametrize("mode", ["primary", "opportunistic"])
def test_foreign_compute_pid_blocks_then_automatically_admits(mode):
    policy = _policy(mode, stable_idle_seconds=0)
    occupied = GPUProbeResult(
        free_memory_mib=48000,
        utilization_percent=0,
        compute_processes=(GPUComputeProcess(38122, "luis"),),
    )
    telemetry, probe, clock, logs = _admit(policy, [occupied, FREE])
    assert probe.calls == 2
    assert clock.sleeps == [20]
    assert telemetry.wait_seconds == 20
    assert "reason=foreign_process pid=38122 user=luis" in logs[0]
    assert "state=ADMITTED" in logs[-1]


def test_scheduler_ignores_its_own_compute_pid():
    own_context = GPUProbeResult(
        free_memory_mib=48000,
        utilization_percent=0,
        compute_processes=(GPUComputeProcess(100, "gmorgan"),),
    )
    telemetry, probe, clock, _ = _admit(_policy(), [own_context], current_pid=100)
    assert probe.calls == 1
    assert clock.sleeps == []
    assert telemetry.foreign_process_count == 0


@pytest.mark.parametrize(
    ("blocked", "reason"),
    [
        (GPUProbeResult(39999, 0), "low_free_memory"),
        (GPUProbeResult(48000, 11), "high_utilization"),
    ],
)
def test_resource_thresholds_block_admission(blocked, reason):
    telemetry, probe, clock, logs = _admit(_policy(), [blocked, FREE])
    assert probe.calls == 2
    assert clock.sleeps == [20]
    assert telemetry.wait_seconds == 20
    assert f"reason={reason}" in logs[0]


def test_primary_admits_immediately_when_free():
    telemetry, probe, clock, _ = _admit(_policy(), [FREE])
    assert probe.calls == 1
    assert clock.sleeps == []
    assert telemetry.wait_seconds == 0
    assert telemetry.poll_count == 1


def test_opportunistic_waits_for_full_stable_idle_window_without_real_sleep():
    telemetry, probe, clock, logs = _admit(_policy("opportunistic"), [FREE])
    assert probe.calls == 4
    assert clock.sleeps == [20, 20, 20]
    assert telemetry.wait_seconds == 60
    assert telemetry.stable_idle_seconds == 60
    assert sum("reason=stability_window" in record for record in logs) == 1


def test_opportunistic_stability_timer_resets_when_foreign_process_appears():
    occupied = GPUProbeResult(48000, 0, (GPUComputeProcess(38122, "luis"),))
    telemetry, probe, clock, logs = _admit(
        _policy("opportunistic"), [FREE, FREE, occupied, FREE]
    )
    assert probe.calls == 7
    assert clock.sleeps == [20, 20, 20, 20, 20, 20]
    assert telemetry.wait_seconds == 120
    assert any("reason=foreign_process" in record for record in logs)


def test_opportunistic_admits_after_process_leaves_and_stability_passes():
    occupied = GPUProbeResult(48000, 0, (GPUComputeProcess(200),))
    telemetry, probe, clock, _ = _admit(_policy("opportunistic"), [occupied, FREE])
    assert probe.calls == 5
    assert clock.sleeps == [20, 20, 20, 20]
    assert telemetry.wait_seconds == 80


def test_probe_failure_fails_closed():
    class BrokenProbe:
        def probe(self, _physical_gpu_index):
            raise OSError("nvidia-smi unavailable")

    with pytest.raises(RuntimeError, match="gpu_probe_failed"):
        wait_for_gpu_admission(
            _policy(),
            maximum_epochs=25,
            probe=BrokenProbe(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


def test_keyboard_interrupt_is_not_swallowed():
    class InterruptedProbe:
        def probe(self, _physical_gpu_index):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        wait_for_gpu_admission(
            _policy(),
            maximum_epochs=25,
            probe=InterruptedProbe(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


def test_nvidia_smi_probe_queries_physical_gpu_and_handles_empty_compute_output(
    monkeypatch,
):
    calls: list[tuple[list[str], dict]] = []
    outputs = iter((" 48700, 0 \n", ""))

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout=next(outputs))

    monkeypatch.setattr(scheduler_module.subprocess, "run", fake_run)
    observed = NvidiaSmiGPUProbe().probe(1)
    assert observed == GPUProbeResult(48700, 0)
    assert calls[0][0] == [
        "nvidia-smi",
        "--id=1",
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert calls[1][0] == [
        "nvidia-smi",
        "--id=1",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    assert all("shell" not in kwargs for _, kwargs in calls)


@pytest.mark.parametrize(
    "status",
    ["", "48700", "48700, unknown", "48700, 101", "48700, 0, 1"],
)
def test_nvidia_smi_malformed_gpu_output_fails_closed(monkeypatch, status):
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda _arguments, **_kwargs: SimpleNamespace(returncode=0, stdout=status),
    )
    with pytest.raises(ValueError, match="gpu_probe_parse_failed"):
        NvidiaSmiGPUProbe().probe(1)


def test_nvidia_smi_subprocess_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda _arguments, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(RuntimeError, match="gpu_probe_failed"):
        NvidiaSmiGPUProbe().probe(1)


def test_duplicate_wait_logs_are_rate_limited_but_state_remains_visible():
    blocked = GPUProbeResult(30000, 0)
    telemetry, probe, _, logs = _admit(_policy(), [*[blocked] * 16, FREE])
    waiting = [record for record in logs if "state=WAITING" in record]
    assert probe.calls == 17
    assert telemetry.wait_seconds == 320
    assert len(waiting) == 2
    assert all("reason=low_free_memory" in record for record in waiting)


def test_candidate_admission_is_immediately_before_cuda_setup():
    source = inspect.getsource(run_search_candidate)
    validation_index = source.index("materialise_validation_data")
    admission_index = source.index("wait_for_gpu_admission")
    seed_index = source.index("seed_everything")
    assert validation_index < admission_index < seed_index
    between_admission_and_seed = source[admission_index:seed_index]
    assert "create_model" not in between_admission_and_seed


@pytest.mark.parametrize("fidelity", [150, 300, 350])
def test_opportunistic_rejects_disallowed_long_fidelity(fidelity):
    with pytest.raises(ValueError, match="gpu_fidelity_disallowed"):
        wait_for_gpu_admission(
            _policy("opportunistic"),
            maximum_epochs=fidelity,
            probe=SequenceProbe([FREE]),
            environ={"CUDA_VISIBLE_DEVICES": "1"},
        )


def test_opportunistic_allows_25_epoch_fidelity():
    telemetry, _, _, _ = _admit(_policy("opportunistic", stable_idle_seconds=0), [FREE])
    assert telemetry.mode == "opportunistic"


def test_admission_telemetry_records_wait_and_poll_count_exactly():
    blocked = GPUProbeResult(30000, 0)
    telemetry, _, _, _ = _admit(_policy(), [blocked, blocked, FREE])
    metrics = telemetry.as_metrics()
    assert metrics == {
        "gpu_scheduler_version": GPU_SCHEDULER_VERSION,
        "gpu_scheduler_mode": "primary",
        "physical_gpu_index": 0,
        "gpu_admission_wait_seconds": 40.0,
        "gpu_admission_poll_count": 3,
        "gpu_admission_free_memory_mib": 48000,
        "gpu_admission_utilization_percent": 0,
        "gpu_admission_foreign_process_count": 0,
        "gpu_admission_stable_idle_seconds": 0,
    }
    assert scheduler_telemetry_is_valid(metrics, _policy()) is True


def test_scheduler_telemetry_validation_rejects_missing_or_mismatched_values():
    telemetry, _, _, _ = _admit(_policy(), [FREE])
    metrics = telemetry.as_metrics()
    assert scheduler_telemetry_is_valid({}, _policy()) is False
    assert (
        scheduler_telemetry_is_valid({**metrics, "physical_gpu_index": 1}, _policy())
        is False
    )
