"""Auto-detect and capture the current execution environment for Slurm jobs.

Absent optional probes stay quiet; available probes that break emit one
``RuntimeWarning`` and keep their fallback. Warnings are the intended channel
because ``cli/main.py`` installs ``install_arbol_warnings()`` to render them as
arbol lines.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
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

_warned_probe_failures: set[str] = set()


def _warn_probe_failure_once(probe: str, exc: Exception) -> None:
    """Report one unexpected probe failure while preserving its fallback."""
    if probe in _warned_probe_failures:
        return
    _warned_probe_failures.add(probe)
    stacklevel = 2
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        stacklevel += 1
        frame = frame.f_back
    warnings.warn(
        f"{probe} failed with {type(exc).__name__}: {exc}; using fallback.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


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


def get_slurm_scheduler_info() -> Dict:
    """Query Slurm for scheduler type, job limits, and fairshare info.

    Returns a dict with keys:
        scheduler_type: e.g. "sched/backfill"
        preempt_mode: e.g. "REQUEUE"
        max_array_size: int
        max_jobs_per_user: int or None (None = unlimited)
        max_submit_per_user: int or None
        uses_backfill: bool
        uses_fairshare: bool
    """
    info: Dict = {
        "scheduler_type": "unknown",
        "preempt_mode": "OFF",
        "max_array_size": 1000,
        "max_jobs_per_user": None,
        "max_submit_per_user": None,
        "uses_backfill": False,
        "uses_fairshare": False,
    }
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("SchedulerType"):
                info["scheduler_type"] = stripped.split("=", 1)[1].strip()
                info["uses_backfill"] = "backfill" in info["scheduler_type"]
            elif stripped.startswith("PriorityType"):
                info["uses_fairshare"] = "multifactor" in stripped
            elif stripped.startswith("PreemptMode"):
                info["preempt_mode"] = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("MaxArraySize"):
                try:
                    info["max_array_size"] = int(stripped.split("=", 1)[1].strip())
                except ValueError:
                    pass
    except FileNotFoundError:
        pass  # Slurm is optional on local workstations and login environments.
    except subprocess.TimeoutExpired:
        pass  # A busy Slurm controller is an expected transient fallback.
    except Exception as exc:
        _warn_probe_failure_once("get_slurm_scheduler_info", exc)

    # Note: we intentionally do NOT try to parse per-QOS job limits here.
    # Users may have multiple QOS (interactive, normal, etc.) with different
    # limits, and the applicable QOS depends on the target partition.  Parsing
    # this correctly requires knowing which QOS will be used for submission,
    # which we don't know at this point.  max_jobs_per_user=None (unlimited)
    # is the safe default — it just means we prefer shorter jobs for backfill.

    return info


def is_slurm_mps_available() -> bool:
    """Check if Slurm MPS (Multi-Process Service) GRES is available.

    MPS allows the scheduler to pack multiple jobs onto a single GPU
    with isolated memory and compute sharing.  Requires ``GresTypes``
    to include ``mps`` in ``slurm.conf`` (admin-configured).
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith("GresTypes"):
                gres_types = line.split("=", 1)[1].strip().lower()
                return "mps" in [g.strip() for g in gres_types.split(",")]
    except FileNotFoundError:
        pass  # No Slurm installation means MPS is simply unavailable.
    except subprocess.TimeoutExpired:
        pass  # A busy Slurm controller is an expected transient fallback.
    except Exception as exc:
        _warn_probe_failure_once("is_slurm_mps_available", exc)
    return False


def _partition_has_gpu(partition: str) -> bool:
    """Return whether ``sinfo`` reports GPU resources for a partition."""
    try:
        result = subprocess.run(
            ["sinfo", "-p", partition, "--format=%G", "--noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return any("gpu:" in line.lower() for line in result.stdout.splitlines())
    except FileNotFoundError:
        pass  # A partial Slurm client install may omit sinfo.
    except subprocess.TimeoutExpired:
        pass  # A busy Slurm controller is an expected transient fallback.
    except Exception as exc:
        _warn_probe_failure_once("_partition_has_gpu", exc)
    return False


def detect_preemptible_gpu_partition() -> Optional[str]:
    """Find a preemptible partition with GPU resources.

    Looks for partitions whose name suggests preemptibility (e.g.,
    ``preempted``, ``preempt``, ``scavenger``, ``low-priority``),
    then verifies they have GPU GRES and are accessible.

    On many clusters, ``PreemptMode=REQUEUE`` is set globally for all
    partitions, so we can't rely on that alone — we use naming conventions
    to identify the actual preemptible/scavenger partition.

    Returns
    -------
    str or None
        Partition name, or None if no preemptible GPU partition found.
    """
    # Common names for preemptible/scavenger partitions
    PREEMPT_KEYWORDS = ("preempt", "scaveng", "low", "opportun", "backfill")

    try:
        result = subprocess.run(
            ["scontrol", "show", "partitions"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        # Parse partition blocks — collect name + state
        partitions: list[tuple[str, str]] = []  # (name, state)
        current_name: Optional[str] = None
        current_state: str = ""

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("PartitionName="):
                if current_name:
                    partitions.append((current_name, current_state))
                current_name = line.split()[0].split("=", 1)[1]
                current_state = ""
            if "State=" in line:
                for part in line.split():
                    if part.startswith("State="):
                        current_state = part.split("=", 1)[1]
        if current_name:
            partitions.append((current_name, current_state))

        # Filter to partitions with preemptible-sounding names + State=UP
        candidates = [
            name
            for name, state in partitions
            if state.upper() == "UP"
            and any(kw in name.lower() for kw in PREEMPT_KEYWORDS)
        ]

        # Filter to partitions with GPU GRES
        for partition in candidates:
            if _partition_has_gpu(partition):
                return partition

    except FileNotFoundError:
        pass  # Slurm is optional; no scheduler means no preemptible partition.
    except subprocess.TimeoutExpired:
        pass  # A busy Slurm controller is an expected transient fallback.
    except Exception as exc:
        _warn_probe_failure_once("detect_preemptible_gpu_partition", exc)
    return None


def validate_partition_access(partition: str) -> bool:
    """Check if a partition exists and is UP.

    Uses ``sinfo -p <partition>`` to verify availability.
    """
    try:
        result = subprocess.run(
            ["sinfo", "-p", partition, "--format=%a", "--noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        # Check if partition is available (not down/drain)
        for line in result.stdout.splitlines():
            if line.strip().lower() == "up":
                return True
        return False
    except FileNotFoundError:
        pass  # A partial Slurm client install may omit sinfo.
    except subprocess.TimeoutExpired:
        pass  # A busy Slurm controller is an expected transient fallback.
    except Exception as exc:
        _warn_probe_failure_once("validate_partition_access", exc)
    return False


def _cuda_build_info_path() -> Optional[Path]:
    """Return the optional CUDA build metadata path."""
    try:
        cuda_spec = find_spec("luxar.gsplats.models.gsplats.cuda")
        if cuda_spec is None or cuda_spec.origin is None:
            return None
        return Path(cuda_spec.origin).parent / "cuda_build_info.json"
    except ImportError:
        return None  # The optional CUDA package or one of its dependencies is absent.
    except Exception as exc:
        _warn_probe_failure_once("_cuda_build_info_path", exc)
        return None


def read_cuda_build_info() -> Dict:
    """Return the CUDA build metadata written by build.py, or an empty dict."""
    info_path = _cuda_build_info_path()
    if info_path is None:
        return {}
    if info_path.exists():
        try:
            with open(info_path) as f:
                return dict(json.load(f))
        except Exception as exc:
            _warn_probe_failure_once("read_cuda_build_info", exc)
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
            "\n⚠  The CUDA extension was built with these modules, which are not\n"
            "   currently loaded.  They will be added to the sbatch preamble:\n"
            + "".join(f"     module load {m}\n" for m in missing)
            + "   To silence this warning, load them now:\n"
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
        env.luxar_version = version("luxar")
    except PackageNotFoundError:
        env.luxar_version = "unknown"
    except Exception as exc:
        _warn_probe_failure_once("capture_environment", exc)
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

    # Env vars BEFORE module loads — modules (gcc, cuda) append/prepend
    # their own paths to LD_LIBRARY_PATH, and those must take priority
    # over the captured (potentially stale) paths from plan time.
    for var, val in env.env_vars.items():
        # Skip CONDA_PREFIX/VIRTUAL_ENV — handled by activation
        if var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
            continue
        # LD_LIBRARY_PATH: set as baseline; module loads will prepend theirs.
        if var == "LD_LIBRARY_PATH":
            lines.append(f"export {var}={shlex.quote(val)}:${{{var}:-}}")
        else:
            lines.append(f"export {var}={shlex.quote(val)}")

    # Module loads AFTER env vars — modules prepend to LD_LIBRARY_PATH,
    # so freshly-loaded CUDA/GCC libraries take priority over captured paths.
    for mod in env.loaded_modules:
        lines.append(f"module load {shlex.quote(mod)}")

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
