from __future__ import annotations

import importlib.util
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
