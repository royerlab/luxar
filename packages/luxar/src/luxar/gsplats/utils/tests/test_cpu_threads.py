"""Tests for the CLI's automatic CPU thread count (``cpu_threads.py``)."""

from __future__ import annotations

import pytest
import torch

from luxar.gsplats.utils.cpu_threads import (
    AUTO_CPU_THREADS_CAP,
    cpu_threads_for,
    physical_cpu_count,
    resolve_cpu_threads,
)


def _resolve(env: dict[str, str], current: int = 1, cores: int = 16, **kw) -> object:
    return resolve_cpu_threads(env, current=current, physical_cores=cores, **kw)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},  # the hatch pin
        {"OMP_NUM_THREADS": "1"},
        {"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": " 1 "},
    ],
)
def test_pinned_or_unset_widens_to_capped_physical_count(env: dict[str, str]) -> None:
    assert _resolve(env, cores=16) == AUTO_CPU_THREADS_CAP
    assert _resolve(env, cores=4) == 4


@pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_ineffective_blas_thread_export_still_widens(name: str) -> None:
    assert _resolve({name: "16"}, current=1) == AUTO_CPU_THREADS_CAP
    assert _resolve({name: "2"}, current=4) is None


def test_already_multithreaded_torch_is_left_alone() -> None:
    assert _resolve({}, current=4) is None


def test_worker_is_left_alone() -> None:
    assert _resolve({}, worker=True) is None


def test_luxar_override_takes_precedence() -> None:
    env = {"LUXAR_CPU_THREADS": "3", "OMP_NUM_THREADS": "16"}
    assert _resolve(env, current=16) == 3
    # Exceeds the auto cap and applies to workers: it is an explicit choice.
    assert _resolve({"LUXAR_CPU_THREADS": "12"}, worker=True) == 12
    # The only way to really ask for one thread under the hatch pin.
    assert _resolve({"LUXAR_CPU_THREADS": "1"}) == 1


def test_luxar_override_zero_disables() -> None:
    assert _resolve({"LUXAR_CPU_THREADS": "0"}) is None


@pytest.mark.parametrize("bad", ["", "abc", "-2", "1.5"])
def test_invalid_override_falls_back_to_auto(bad: str) -> None:
    assert _resolve({"LUXAR_CPU_THREADS": bad}, cores=6) == 6


def test_physical_cpu_count_is_positive() -> None:
    assert physical_cpu_count() >= 1


def test_worker_share_sets_torch_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.utils.device import cpu_worker_threads_env, physical_core_count

    monkeypatch.setenv("LUXAR_CPU_THREADS", "0")
    worker_env = cpu_worker_threads_env(4)
    share = max(1, physical_core_count() // 4)
    assert worker_env == {"LUXAR_CPU_THREADS": str(share)}
    assert (
        resolve_cpu_threads(
            worker_env, current=1, physical_cores=1, worker=True
        )
        == share
    )


def test_context_manager_widens_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = torch.get_num_threads()
    target = 2 if previous != 2 else 3
    monkeypatch.setenv("LUXAR_CPU_THREADS", str(target))
    with cpu_threads_for("cpu"):
        assert torch.get_num_threads() == target
    assert torch.get_num_threads() == previous


def test_context_manager_restores_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = torch.get_num_threads()
    target = 2 if previous != 2 else 3
    monkeypatch.setenv("LUXAR_CPU_THREADS", str(target))
    with pytest.raises(RuntimeError, match="boom"), cpu_threads_for("cpu"):
        raise RuntimeError("boom")
    assert torch.get_num_threads() == previous


def test_context_manager_is_noop_off_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = torch.get_num_threads()
    monkeypatch.setenv("LUXAR_CPU_THREADS", str(previous + 1))
    with cpu_threads_for("cuda"):
        assert torch.get_num_threads() == previous


def test_test_suite_pin_is_in_force() -> None:
    """The conftest disables widening so CLI tests keep the hatch pin."""
    import os

    assert os.environ.get("LUXAR_CPU_THREADS") is not None
