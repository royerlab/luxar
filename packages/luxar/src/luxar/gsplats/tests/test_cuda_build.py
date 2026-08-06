"""Unit tests for the shared CUDA build machinery (no GPU/nvcc required).

Exercises the pure arch-selection / gencode logic and confirms the two thin
``build.py`` configs and the shared module import cleanly and expose their
entry points.
"""

from __future__ import annotations

import json

from luxar.gsplats import _cuda_build


def test_get_nvcc_supported_archs_returns_none_or_set() -> None:
    # nvcc is absent on this machine, so the query returns None; on a box with
    # nvcc it returns a non-empty set. Either is acceptable — it must never raise.
    result = _cuda_build._get_nvcc_supported_archs()
    assert result is None or isinstance(result, set)


def test_select_target_archs_env_override() -> None:
    # An explicit CUDA_ARCHS override is honoured verbatim, in order.
    assert _cuda_build._select_target_archs(86, "86;90", None) == [86, 90]


def test_select_target_archs_env_override_ignores_supported() -> None:
    # The override path does not consult the nvcc-supported set.
    assert _cuda_build._select_target_archs(80, "75;80", {75}) == [75, 80]


def test_select_target_archs_default_filters_by_supported() -> None:
    archs = _cuda_build._select_target_archs(86, "", {75, 80, 86})
    assert archs == sorted(archs)
    assert all(a >= 75 for a in archs)
    assert 86 in archs
    # Only the supported archs (>= 75) survive.
    assert set(archs) == {75, 80, 86}


def test_select_target_archs_keeps_current_even_if_unsupported() -> None:
    # The current arch is always retained so an unsupported current surfaces as
    # a clear compiler error rather than an empty list.
    archs = _cuda_build._select_target_archs(90, "", {75})
    assert 90 in archs


def test_select_target_archs_can_be_empty() -> None:
    # A current arch below the sm_75 floor with no other supported archs yields
    # an empty list (the orchestrator turns this into a clear error + exit).
    assert _cuda_build._select_target_archs(70, "", {70}) == []


def test_gencode_flags() -> None:
    flags, highest = _cuda_build._gencode_flags([86, 90])
    assert highest == 90
    assert "-gencode=arch=compute_86,code=sm_86" in flags
    assert "-gencode=arch=compute_90,code=sm_90" in flags
    # PTX for the highest arch is appended last (forward compatibility).
    assert flags[-1] == "-gencode=arch=compute_90,code=compute_90"


def test_shared_module_exposes_build_entry_point() -> None:
    assert callable(_cuda_build.build_cuda_extension)


def test_build_scripts_import_and_expose_build() -> None:
    from luxar.gsplats.models.gsplats.cuda import build as splatting_build
    from luxar.gsplats.preprocessing.cuda import build as nlm_build

    assert callable(splatting_build.build)
    assert callable(nlm_build.build)


def test_loaded_modules_parses_module_list(monkeypatch) -> None:
    class _Result:
        stdout = "Currently Loaded Modules:\n  1) cuda/12.8.0   2) gcc/12.2.0\n"

    monkeypatch.setattr(_cuda_build.subprocess, "run", lambda *a, **k: _Result())
    mods = _cuda_build._loaded_modules()
    assert "cuda/12.8.0" in mods
    assert "gcc/12.2.0" in mods


def test_loaded_modules_handles_failure(monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(_cuda_build.subprocess, "run", _boom)
    assert _cuda_build._loaded_modules() == []


def test_write_build_info_warns_when_modules_loaded(
    tmp_path, monkeypatch, capsys
) -> None:
    # The "load these modules" warning is item #1296's drift fix: it must fire
    # for every extension (it had been dropped from the NLM path) whenever
    # modules were loaded at build time.
    monkeypatch.setattr(_cuda_build, "_loaded_modules", lambda: ["cuda/12.8", "gcc/12"])
    so_path = tmp_path / "nlm_cuda_backend.so"
    _cuda_build._write_build_info(so_path, "nlm_build_info.json")

    info_path = tmp_path / "nlm_build_info.json"
    assert info_path.exists()
    data = json.loads(info_path.read_text())
    assert data["so_file"] == "nlm_cuda_backend.so"
    assert data["loaded_modules"] == ["cuda/12.8", "gcc/12"]

    out = capsys.readouterr().out
    assert "Load these modules before submitting Slurm fit jobs" in out
    assert "module load cuda/12.8 gcc/12" in out


def test_write_build_info_silent_when_no_modules(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(_cuda_build, "_loaded_modules", lambda: [])
    so_path = tmp_path / "cuda_splatting_backend.so"
    _cuda_build._write_build_info(so_path, "cuda_build_info.json")

    assert (tmp_path / "cuda_build_info.json").exists()
    out = capsys.readouterr().out
    assert "Load these modules" not in out
