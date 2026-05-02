"""Tests for Gaussian-splat PyTorch device-selection helpers."""

from __future__ import annotations

import torch

from luxar.gsplats.utils.device import is_mps_available, resolve_torch_device


class _FakeMPSBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_resolve_explicit_device_overrides_accelerator_flags(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(False), raising=False)

    device = resolve_torch_device("cuda", use_cuda=False, use_metal=False)

    assert device.type == "cuda"


def test_resolve_honors_accelerator_opt_outs(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device(None, use_cuda=False, use_metal=False)

    assert device.type == "cpu"


def test_resolve_prefers_cuda_over_mps_when_both_enabled(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device(None, use_cuda=True, use_metal=True)

    assert device.type == "cuda"


def test_resolve_uses_mps_when_cuda_disabled(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device(None, use_cuda=False, use_metal=True)

    assert device.type == "mps"


def test_is_mps_available_handles_missing_backend(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)

    assert is_mps_available() is False
