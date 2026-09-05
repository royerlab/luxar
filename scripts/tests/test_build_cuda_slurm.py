from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "build_cuda_slurm.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_cuda_slurm", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_gcc_module_requires_cxx20_capable_gcc() -> None:
    module = _load()

    assert module.best_gcc_module(["gcc/8.5", "gcc/9.5"]) is None
    assert module.best_gcc_module(["gcc/9.5", "gcc/10.2", "gcc/14.1"]) == "gcc/14.1"


def test_main_warns_when_only_old_gcc_modules_are_available(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    module = _load()
    virtual_env = tmp_path / "env"
    (virtual_env / "bin").mkdir(parents=True)
    (virtual_env / "bin" / "activate").touch()

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--dry-run"])
    monkeypatch.setattr(module, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "detect_pytorch_cuda_version", lambda: "12.8")
    monkeypatch.setattr(
        module, "list_available_cuda_modules", lambda: ["cuda/12.8.0"]
    )
    monkeypatch.setattr(module, "get_virtual_env", lambda: str(virtual_env))
    monkeypatch.setattr(
        module, "list_available_gcc_modules", lambda: ["gcc/8.5", "gcc/9.5"]
    )

    module.main()

    output = capsys.readouterr().out
    assert "gcc/9.5" in output
    assert "cannot compile the shipped C++20 build" in output
    assert "none needed (system GCC >= 10 assumed)" not in output


def test_generated_build_guidance_names_the_cxx20_floor(tmp_path: Path) -> None:
    module = _load()

    script = module.generate_sbatch_script(
        partition="gpu",
        cuda_module="cuda/12.8",
        gcc_module="gcc/10.2",
        virtual_env="/tmp/luxar-env",
        project_root=tmp_path,
        account="",
        qos="",
        time_limit="01:00:00",
        cpus=4,
        mem_gb=16,
    )

    assert "requires GCC >= 10" in script
    assert "C++20 required" in script
    assert "If GCC < 10" in script
    assert "module load gcc/10" in script
