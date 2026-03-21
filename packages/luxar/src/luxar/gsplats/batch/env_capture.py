"""Auto-detect and capture the current execution environment for Slurm jobs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Environment variables known to matter for CUDA/PyTorch workloads.
CURATED_ENV_VARS = [
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "TORCH_HOME",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
]


@dataclass
class CapturedEnv:
    """Snapshot of the current execution environment."""

    conda_prefix: Optional[str] = None
    """``$CONDA_PREFIX`` — active conda environment path."""

    virtual_env: Optional[str] = None
    """``$VIRTUAL_ENV`` — active virtualenv/venv path."""

    loaded_modules: List[str] = field(default_factory=list)
    """Currently loaded environment modules (from ``module list``)."""

    env_vars: Dict[str, str] = field(default_factory=dict)
    """Curated subset of environment variables (only those that are set)."""

    luxar_version: str = ""
    """Installed luxar version string."""


def read_cuda_build_info() -> Dict:
    """Return the CUDA build metadata written by build.py, or an empty dict."""
    try:
        import luxar.gsplats.models.gsplats.cuda as cuda_pkg

        info_path = Path(cuda_pkg.__file__).parent / "cuda_build_info.json"
    except Exception:
        return {}
    if info_path.exists():
        try:
            with open(info_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def capture_environment() -> CapturedEnv:
    """Auto-detect the current execution environment.

    Captures:
    1. Active conda environment (``CONDA_PREFIX``)
    2. Active virtualenv (``VIRTUAL_ENV``)
    3. Loaded environment modules (``module list``)
    4. Modules required by the CUDA extension (from cuda_build_info.json),
       merged into loaded_modules so sbatch scripts load them automatically
    5. Curated environment variables (only those that are set)
    6. Luxar version
    """
    env = CapturedEnv()

    # 1. Conda
    env.conda_prefix = os.environ.get("CONDA_PREFIX")

    # 2. Virtualenv
    env.virtual_env = os.environ.get("VIRTUAL_ENV")

    # 3. Environment modules currently loaded
    currently_loaded = _detect_loaded_modules()
    loaded_set = set(currently_loaded)

    # 4. Merge modules required at build time (e.g. gcc/14.2, cuda/12.8.x)
    build_info = read_cuda_build_info()
    build_modules: List[str] = build_info.get("loaded_modules", [])
    missing: List[str] = []
    for mod in build_modules:
        # Only add non-trivial modules (skip slurm/default and similar)
        base = mod.split("/")[0].lower()
        if base in ("slurm",):
            continue
        if mod not in loaded_set:
            currently_loaded.append(mod)
            missing.append(mod)

    if missing:
        print(
            f"\n⚠  The CUDA extension was built with these modules, which are not\n"
            f"   currently loaded.  They will be added to the sbatch preamble:\n"
            + "".join(f"     module load {m}\n" for m in missing)
            + f"   To silence this warning, load them now:\n"
            + f"     module load {' '.join(missing)}\n"
        )

    env.loaded_modules = currently_loaded

    # 5. Curated env vars
    for var in CURATED_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            env.env_vars[var] = val

    # 6. Luxar version
    try:
        from importlib.metadata import version

        env.luxar_version = version("luxar")
    except Exception:
        env.luxar_version = "unknown"

    return env


def generate_env_preamble(env: CapturedEnv) -> str:
    """Generate a shell preamble that reproduces the captured environment.

    Returns a multi-line string suitable for inclusion at the top of an
    sbatch script.
    """
    lines: List[str] = ["# --- Environment (auto-captured by luxar) ---"]

    # Conda activation
    if env.conda_prefix:
        # Find conda.sh relative to CONDA_PREFIX
        # Typical: /opt/conda/envs/myenv -> /opt/conda/etc/profile.d/conda.sh
        conda_base = _find_conda_base(env.conda_prefix)
        if conda_base:
            conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
            lines.append(f"source {shlex.quote(conda_sh)}")
        env_name = os.path.basename(env.conda_prefix)
        lines.append(f"conda activate {shlex.quote(env_name)}")
    elif env.virtual_env:
        activate = os.path.join(env.virtual_env, "bin", "activate")
        lines.append(f"source {shlex.quote(activate)}")

    # Module loads
    for mod in env.loaded_modules:
        lines.append(f"module load {shlex.quote(mod)}")

    # Env vars (shell-escaped to prevent injection)
    for var, val in env.env_vars.items():
        # Skip CONDA_PREFIX/VIRTUAL_ENV — handled by activation
        if var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
            continue
        lines.append(f"export {var}={shlex.quote(val)}")

    lines.append("")
    return "\n".join(lines)


def _detect_loaded_modules() -> List[str]:
    """Detect currently loaded environment modules via ``module list``."""
    try:
        result = subprocess.run(
            ["bash", "-c", "module list 2>&1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.strip()
        if not output or "command not found" in output.lower():
            return []

        # Parse module list output — varies by module system
        # Common formats:
        #   "Currently Loaded Modules:\n  1) cuda/12.1  2) gcc/11.3"
        #   "Currently Loaded Modulefiles:\n  cuda/12.1\n  gcc/11.3"
        modules: List[str] = []
        for line in output.split("\n"):
            line = line.strip()
            if not line or line.startswith("Currently") or line.startswith("No "):
                continue
            # Strip numbering like "1) " or "  1. "
            for prefix_end in (") ", ". "):
                idx = line.find(prefix_end)
                if idx != -1 and idx < 5:
                    line = line[idx + len(prefix_end) :].strip()
                    break
            # Split by whitespace in case multiple modules per line
            for part in line.split():
                part = part.strip()
                if part and "/" in part:
                    modules.append(part)
        return modules
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def _find_conda_base(conda_prefix: str) -> Optional[str]:
    """Find the conda base directory from CONDA_PREFIX.

    Walks up from the prefix looking for ``etc/profile.d/conda.sh``.
    """
    path = conda_prefix
    for _ in range(5):  # Don't walk too far
        conda_sh = os.path.join(path, "etc", "profile.d", "conda.sh")
        if os.path.exists(conda_sh):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None
