#!/usr/bin/env python3
"""
Submit a CUDA extension build job to Slurm.

This script is called by 'make build-cuda SLURM=1'.  It:
  1. Detects the PyTorch CUDA version already installed in the hatch env.
  2. Finds the best matching 'cuda/X.Y...' module available on this system.
  3. Detects the active virtual environment (hatch-managed).
  4. Generates a self-contained sbatch script with step-by-step diagnostics.
  5. Submits it (or prints a dry-run preview with --dry-run).
  6. Tells you exactly how to monitor the job and what to do next.

Usage (via make):
    make build-cuda SLURM=1
    make build-cuda SLURM=1 SLURM_PARTITION=gpu
    make build-cuda SLURM=1 SLURM_PARTITION=gpu CUDA_MODULE=cuda/12.8.0_570.86.10
    make build-cuda SLURM=1 SLURM_PARTITION=gpu SLURM_ACCOUNT=myaccount SLURM_TIME=02:00:00

Direct usage:
    python scripts/build_cuda_slurm.py --partition gpu
    python scripts/build_cuda_slurm.py --partition gpu --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def die(msg: str) -> None:
    print(f"\n❌  {msg}\n", file=sys.stderr)
    sys.exit(1)


def banner(msg: str) -> None:
    bar = "━" * 60
    print(f"\n{bar}\n  {msg}\n{bar}")


def detect_pytorch_cuda_version() -> str | None:
    """Return the CUDA version PyTorch was built for (e.g. '12.8'), or None."""
    result = run([sys.executable, "-c", "import torch; print(torch.version.cuda)"])
    if result.returncode != 0 or not result.stdout.strip():
        return None
    v = result.stdout.strip()
    return None if v == "None" else v


def list_available_cuda_modules() -> list[str]:
    """Return sorted list of 'cuda/X.Y.Z...' module names available via 'module spider cuda'."""
    result = run(["bash", "-c", "module spider cuda 2>&1"])
    modules = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("cuda/"):
            modules.append(stripped)
    return sorted(modules)


def best_cuda_module(torch_cuda_version: str, available: list[str]) -> str | None:
    """
    Pick the highest available module whose major.minor matches torch_cuda_version.

    torch_cuda_version is e.g. '12.8'.
    Module names look like 'cuda/12.8.0_570.86.10'.
    """
    major_minor = torch_cuda_version.strip()  # e.g. "12.8"
    candidates = [m for m in available if m.startswith(f"cuda/{major_minor}.")]
    if not candidates:
        return None
    return candidates[-1]  # highest version (list is sorted)


def list_available_gcc_modules() -> list[str]:
    """Return sorted list of 'gcc/X.Y...' module names available via 'module spider gcc'."""
    result = run(["bash", "-c", "module spider gcc 2>&1"])
    modules = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("gcc/"):
            modules.append(stripped)
    return sorted(modules)


def best_gcc_module(available: list[str]) -> str | None:
    """
    Pick the highest available GCC module that is >= 10 (required for C++20).

    Module names look like 'gcc/11.3' or 'gcc/14.2'.
    Returns None if no suitable module is found (system GCC may already be >=10).
    """
    candidates = []
    for m in available:
        ver_str = m.split("/")[1].split(".")[0]  # e.g. "11" from "gcc/11.3"
        try:
            major = int(ver_str)
            if major >= 10:
                candidates.append((major, m))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]  # highest major version


def _highest_gcc_module(available: list[str]) -> str | None:
    """Return the highest versioned GCC module, regardless of compiler floor."""
    candidates = []
    for module in available:
        match = re.fullmatch(r"gcc/(\d+(?:\.\d+)*)", module)
        if match is None:
            continue
        version = tuple(int(part) for part in match.group(1).split("."))
        candidates.append((version, module))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def get_virtual_env() -> str | None:
    """Return the path to the active virtualenv, or None."""
    return os.environ.get("VIRTUAL_ENV")


def get_project_root() -> Path:
    """Return the absolute path to the luxar project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def check_sbatch_available() -> None:
    if not shutil.which("sbatch"):
        die(
            "sbatch not found in PATH.\n"
            "  This script must be run on an HPC login node where Slurm is available.\n"
            "  If you are on the right machine, ensure the Slurm module is loaded."
        )


def list_gpu_partitions() -> list[str]:
    """Return names of partitions that appear to have GPU nodes."""
    result = run(["sinfo", "-h", "-o", "%P %G"])
    partitions = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            name = parts[0].rstrip("*")
            gpus = parts[1]
            if gpus not in ("(null)", "0"):
                partitions.append(name)
    return partitions


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def generate_sbatch_script(
    *,
    partition: str,
    cuda_module: str,
    gcc_module: str,
    virtual_env: str,
    project_root: Path,
    account: str,
    qos: str,
    time_limit: str,
    cpus: int,
    mem_gb: int,
) -> str:
    """Return a complete, heavily-commented sbatch script string."""

    log_dir = project_root / "build-cuda-logs"
    log_out = log_dir / "build_%j.out"
    log_err = log_dir / "build_%j.err"
    # SLURM=0 prevents recursion: the Slurm job environment may have SLURM
    # set to a non-empty value, which would cause make build-cuda to
    # re-submit another job instead of actually building.
    make_cmd = f"make -C {project_root} build-cuda SLURM=0"

    account_line = (
        f"#SBATCH --account={account}"
        if account
        else "# (no --account set; add SLURM_ACCOUNT=... to make command if needed)"
    )
    qos_line = (
        f"#SBATCH --qos={qos}"
        if qos
        else "# (no --qos set; add SLURM_QOS=... to make command if needed)"
    )

    # gcc_load_block is injected at column 0 in the template (see placeholder
    # below).  textwrap.dedent strips the MINIMUM leading whitespace across all
    # lines.  To ensure it still strips the 8-space indent that every other
    # template line has, every line of gcc_load_block must also start with 8
    # spaces — that way the minimum stays 8 and dedent strips uniformly.
    _i = "        "  # 8 spaces — matches the template's indent level
    if gcc_module:
        gcc_load_block = (
            f"{_i}# The shipped C++20 build requires GCC >= 10.  Load a newer GCC module.\n"
            f'{_i}echo "    Loading GCC module: {gcc_module}"\n'
            f"{_i}module load {gcc_module}\n"
            f'{_i}echo "    g++ version: $(g++ --version | head -1)"\n'
        )
    else:
        gcc_load_block = (
            f"{_i}# GCC module: none needed (system GCC is assumed to be >= 10)\n"
            f'{_i}echo "    g++ version: $(g++ --version | head -1)"\n'
        )

    script = textwrap.dedent(f"""\
        #!/bin/bash
        # =============================================================
        # Luxar CUDA Extension Build Job
        # Generated by: make build-cuda SLURM=1
        # Project:      {project_root}
        # =============================================================
        #
        # This job compiles the CUDA splatting extension (.so) that
        # enables GPU-accelerated Gaussian splat fitting.
        # After the job finishes, you can use:
        #   make test-cuda        — verify the extension works
        #   make benchmark-cuda   — measure GPU throughput
        #   hatch run luxar gsplat fit ...  — fit splats on a GPU
        #
        # ── Slurm directives ─────────────────────────────────────────
        #SBATCH --job-name=luxar-build-cuda
        #SBATCH --partition={partition}
        #SBATCH --ntasks=1
        #SBATCH --gpus-per-task=1
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mem={mem_gb}G
        #SBATCH --time={time_limit}
        #SBATCH --output={log_out}
        #SBATCH --error={log_err}
        {account_line}
        {qos_line}

        set -uo pipefail   # Exit on unbound variable / pipe failure (but NOT -e, so we can capture build exit code)

        echo "============================================================"
        echo " Luxar CUDA Extension Build"
        echo " Job ID : $SLURM_JOB_ID"
        echo " Node   : $SLURMD_NODENAME"
        echo " Started: $(date)"
        echo "============================================================"
        echo ""

        # ── Step 1: Load CUDA toolkit (and GCC >= 10 if needed) ──────────
        echo ">>> Step 1/5: Loading CUDA module ({cuda_module})"
        module load {cuda_module}
        echo "    nvcc version: $(nvcc --version | grep 'release' | awk '{{print $6}}')"
        echo "    nvcc path   : $(which nvcc)"
{gcc_load_block}        echo ""

        # ── Step 2: Verify GPU is accessible ─────────────────────────
        echo ">>> Step 2/5: Verifying GPU access"
        if ! nvidia-smi >/dev/null 2>&1; then
            echo ""
            echo "ERROR: GPU not accessible on this node."
            echo "  This usually means:"
            echo "    - The partition '{partition}' doesn't have GPUs"
            echo "    - The NVIDIA driver is not loaded"
            echo "  Try: srun --partition={partition} --gpus=1 nvidia-smi"
            exit 1
        fi
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | \\
            awk -F',' '{{printf "    GPU: %s | Driver: %s | VRAM: %s\\n", $1, $2, $3}}'
        echo ""

        # ── Step 3: Activate Python environment ───────────────────────
        echo ">>> Step 3/5: Activating Python environment"
        ACTIVATE="{virtual_env}/bin/activate"
        if [ ! -f "$ACTIVATE" ]; then
            echo ""
            echo "ERROR: Python virtual environment not found at:"
            echo "  {virtual_env}"
            echo ""
            echo "  This env was captured from the machine where you ran"
            echo "  'make build-cuda SLURM=1'. If you deleted or moved the"
            echo "  hatch environment, re-run 'make setup-dev' and try again."
            exit 1
        fi
        # Save CUDA library path BEFORE activating the venv — some activate
        # scripts reset LD_LIBRARY_PATH, which would make the linker unable to
        # find libcuda.so / libcudart.so when compiling the C++ extension.
        CUDA_LIB_DIR="$(dirname "$(which nvcc)")/../lib64"
        CUDA_LIB_DIR="$(realpath "$CUDA_LIB_DIR" 2>/dev/null || echo "$CUDA_LIB_DIR")"
        source "$ACTIVATE"
        # Re-export LD_LIBRARY_PATH so the CUDA shared libraries are visible to
        # the compiler and runtime inside the venv.
        export LD_LIBRARY_PATH="${{CUDA_LIB_DIR}}:${{LD_LIBRARY_PATH:-}}"
        echo "    Python : $(python --version)"
        echo "    PyTorch: $(python -c 'import torch; print(torch.__version__)')"
        echo "    CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
        echo "    LD_LIBRARY_PATH prefix: $CUDA_LIB_DIR"
        echo ""

        # ── Step 4: Verify PyTorch can see the GPU ────────────────────
        echo ">>> Step 4/5: Verifying PyTorch CUDA"
        python -c "
        import torch, sys
        if not torch.cuda.is_available():
            print('ERROR: torch.cuda.is_available() returned False.')
            print('  CUDA module loaded: yes')
            print('  GPU accessible: yes (nvidia-smi passed)')
            print('  Possible causes:')
            print('    - PyTorch was built for a different CUDA version')
            print(f'    - PyTorch CUDA version: {{torch.version.cuda}}')
            print(f'    - Loaded module: {cuda_module}')
            print('  Fix: ensure the cuda module version matches torch.version.cuda')
            print('  Tip: run   make check-cuda-deps   for a full diagnosis')
            sys.exit(1)
        dev = torch.cuda.current_device()
        print(f'    torch.cuda.is_available(): True')
        print(f'    Device: {{torch.cuda.get_device_name(dev)}}')
        print(f'    Compute capability: {{torch.cuda.get_device_capability(dev)}}')
        "
        echo ""

        # ── Step 5: Build the extension ───────────────────────────────
        echo ">>> Step 5/5: Building CUDA extension"
        echo "    Project: {project_root}"
        echo "    This compiles .cu/.cpp sources via torch.utils.cpp_extension."
        echo "    Expected time: 1-5 minutes depending on GPU node CPU speed."
        echo "    (stderr merged into this log so compiler errors are visible)"
        echo ""
        # Run with 2>&1 so compiler errors (from nvcc/g++) appear here and not
        # in the separate .err file.  We capture the exit code manually since we
        # removed -e from set flags above.
        {make_cmd} 2>&1
        BUILD_EXIT=$?

        if [ "$BUILD_EXIT" -ne 0 ]; then
            echo ""
            echo "============================================================"
            echo " BUILD FAILED (exit code $BUILD_EXIT)"
            echo "============================================================"
            echo ""
            echo " Possible causes and fixes:"
            echo ""
            echo " 1. GCC version too old (C++20 required):"
            echo "      g++ --version"
            echo "    If GCC < 10, load a newer toolchain:"
            echo "      module load gcc/10   # or highest available"
            echo "      make build-cuda SLURM=1 SLURM_PARTITION={partition}"
            echo ""
            echo " 2. CUDA/PyTorch version mismatch:"
            echo "      python -c 'import torch; print(torch.version.cuda)'"
            echo "      nvcc --version"
            echo "    They must match.  Loaded module: {cuda_module}"
            echo ""
            echo " 3. Missing CUDA headers (nvcc found but toolkit incomplete):"
            echo "      ls $(dirname $(which nvcc))/../include/cuda_runtime.h"
            echo ""
            echo " 4. Check full compiler output above for the specific error."
            echo ""
            echo " Tip: run   make check-cuda-deps   on a GPU node for a full diagnosis."
            exit $BUILD_EXIT
        fi

        echo ""
        echo "============================================================"
        echo " Build completed successfully at: $(date)"
        echo " Logs: {log_dir}/"
        echo ""
        echo " Next steps:"
        echo "   make test-cuda       — run unit tests"
        echo "   make benchmark-cuda  — measure throughput"
        echo "   hatch run luxar gsplat fit <input> <output> --preset standard"
        echo "============================================================"
    """)

    return script


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a CUDA extension build job to Slurm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              make build-cuda SLURM=1
              make build-cuda SLURM=1 SLURM_PARTITION=gpu
              make build-cuda SLURM=1 SLURM_PARTITION=gpu CUDA_MODULE=cuda/12.8.0_570.86.10
              python scripts/build_cuda_slurm.py --partition gpu --dry-run
        """),
    )
    parser.add_argument(
        "--partition",
        "-p",
        default="gpu",
        help="Slurm partition to submit to (default: gpu)",
    )
    parser.add_argument(
        "--cuda-module",
        default="auto",
        help="CUDA module to load, e.g. cuda/12.8.0_570.86.10 "
        "(default: auto-detect from torch.version.cuda)",
    )
    parser.add_argument("--account", "-A", default="", help="Slurm account (optional)")
    parser.add_argument("--qos", default="", help="Slurm QOS (optional)")
    parser.add_argument(
        "--time", default="01:00:00", help="Wall-time limit (default: 01:00:00)"
    )
    parser.add_argument(
        "--cpus", type=int, default=4, help="CPUs per task (default: 4)"
    )
    parser.add_argument(
        "--mem", type=int, default=16, help="Memory in GB (default: 16)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch script but do not submit",
    )
    args = parser.parse_args()

    project_root = get_project_root()

    banner("Luxar CUDA Build — Slurm Submission")
    print(f"  Project root : {project_root}")
    print(f"  Partition    : {args.partition}")

    # ── Check sbatch is available ──────────────────────────────────────────
    if not args.dry_run:
        check_sbatch_available()

    # ── Detect PyTorch CUDA version ────────────────────────────────────────
    print("\n  Detecting PyTorch CUDA version...", end=" ", flush=True)
    torch_cuda = detect_pytorch_cuda_version()
    if torch_cuda is None:
        print("FAILED")
        die(
            "Could not determine PyTorch CUDA version.\n\n"
            "  Make sure PyTorch with CUDA support is installed in the hatch env:\n"
            "    hatch run pip install torch --index-url https://download.pytorch.org/whl/cu128\n\n"
            "  Then run again: make build-cuda SLURM=1"
        )
    print(f"PyTorch was built for CUDA {torch_cuda}")

    # ── Resolve CUDA module ────────────────────────────────────────────────
    print("  Finding matching CUDA module...", end=" ", flush=True)
    available_modules = list_available_cuda_modules()

    if args.cuda_module == "auto":
        cuda_module = best_cuda_module(torch_cuda, available_modules)
        if cuda_module is None:
            print("NOT FOUND")
            print(f"\n⚠️  No available cuda/ module matches PyTorch CUDA {torch_cuda}.")
            print("\n  Available CUDA modules on this system:")
            for m in available_modules:
                print(f"    {m}")
            print(
                f"\n  PyTorch is built for CUDA {torch_cuda}.  "
                f"You need a cuda/{torch_cuda}.x module.\n"
                f"\n  Options:\n"
                f"    1. Ask your HPC admins to install cuda/{torch_cuda}.x\n"
                f"    2. Reinstall PyTorch to match an available module, e.g.:\n"
            )
            # Suggest closest available
            for m in reversed(available_modules):
                ver = m.split("/")[1].split("_")[0]  # e.g. "12.6.3"
                maj_min = ".".join(ver.split(".")[:2])  # e.g. "12.6"
                major = int(maj_min.split(".")[0])
                if major >= 11:
                    whl_tag = "cu" + maj_min.replace(".", "")
                    print(
                        f"       hatch run pip install torch "
                        f"--index-url https://download.pytorch.org/whl/{whl_tag}"
                    )
                    print(f"       make build-cuda SLURM=1 CUDA_MODULE={m}")
                    break
            sys.exit(1)
        print(f"{cuda_module}  (auto-selected)")
    else:
        cuda_module = args.cuda_module
        if cuda_module not in available_modules:
            print("WARNING")
            print(
                f"\n⚠️  Module '{cuda_module}' was not found in 'module spider cuda' output."
            )
            print("  Available modules:")
            for m in available_modules:
                print(f"    {m}")
            print(
                "\n  Continuing anyway — module load will fail at runtime if it's wrong."
            )
        else:
            print(f"{cuda_module}  (user-specified)")

    # ── Detect virtual environment ─────────────────────────────────────────
    print("  Detecting virtual environment...", end=" ", flush=True)
    virtual_env = get_virtual_env()
    if virtual_env is None:
        print("NOT FOUND")
        die(
            "VIRTUAL_ENV is not set.  The build job needs to know which Python\n"
            "  environment to activate on the compute node.\n\n"
            "  Run this command from inside the hatch environment:\n"
            "    hatch run make build-cuda SLURM=1\n\n"
            "  Or activate the env manually first:\n"
            "    source $(hatch env find)/bin/activate\n"
            "    make build-cuda SLURM=1"
        )
    activate_path = Path(virtual_env) / "bin" / "activate"
    if not activate_path.exists():
        print("BROKEN")
        die(
            f"VIRTUAL_ENV is set to '{virtual_env}' but\n"
            f"  {activate_path}\n"
            f"  does not exist.  The hatch environment may be corrupted.\n\n"
            f"  Recreate it:\n"
            f"    hatch env remove\n"
            f"    hatch env create\n"
            f"  Then try again."
        )
    print(f"{virtual_env}")

    # ── Detect GCC module ─────────────────────────────────────────────────
    print("  Finding GCC >= 10 module...", end=" ", flush=True)
    available_gcc = list_available_gcc_modules()
    gcc_module = best_gcc_module(available_gcc)
    if gcc_module:
        print(f"{gcc_module}  (auto-selected; system GCC 8.5.0 is too old for C++20)")
    elif available_gcc:
        highest_gcc = _highest_gcc_module(available_gcc)
        found = highest_gcc or ", ".join(available_gcc)
        print(
            f"none suitable (highest found: {found}; "
            "cannot compile the shipped C++20 build)"
        )
    else:
        print("none needed (system GCC >= 10 assumed)")

    # ── Validate partition ─────────────────────────────────────────────────
    if not args.dry_run:
        print("  Checking partition...", end=" ", flush=True)
        result = run(["sinfo", "-h", "-p", args.partition, "-o", "%P"])
        if result.returncode != 0 or not result.stdout.strip():
            print("NOT FOUND")
            gpu_parts = list_gpu_partitions()
            hint = ""
            if gpu_parts:
                hint = f"\n  GPU-capable partitions on this cluster: {', '.join(gpu_parts)}"
                hint += (
                    f"\n  Try: make build-cuda SLURM=1 SLURM_PARTITION={gpu_parts[0]}"
                )
            die(
                f"Partition '{args.partition}' not found.{hint}\n\n"
                f"  Run 'sinfo' to see all available partitions."
            )
        print(f"'{args.partition}' exists")

    # ── Create log directory ───────────────────────────────────────────────
    log_dir = project_root / "build-cuda-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate sbatch script ─────────────────────────────────────────────
    script = generate_sbatch_script(
        partition=args.partition,
        cuda_module=cuda_module,
        gcc_module=gcc_module or "",
        virtual_env=virtual_env,
        project_root=project_root,
        account=args.account,
        qos=args.qos,
        time_limit=args.time,
        cpus=args.cpus,
        mem_gb=args.mem,
    )

    script_path = log_dir / "build_cuda_job.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    print(f"\n  Sbatch script written to: {script_path}")

    # ── Dry run: just print ────────────────────────────────────────────────
    if args.dry_run:
        banner("DRY RUN — sbatch script contents")
        print(script)
        print("\n  To submit for real, run:")
        print(f"    make build-cuda SLURM=1 SLURM_PARTITION={args.partition}")
        return

    # ── Submit ────────────────────────────────────────────────────────────
    print("\n  Submitting to Slurm...", end=" ", flush=True)
    result = run(["sbatch", str(script_path)])
    if result.returncode != 0:
        print("FAILED")
        die(
            f"sbatch returned exit code {result.returncode}.\n\n"
            f"  Error output:\n"
            + textwrap.indent(result.stderr.strip(), "    ")
            + f"\n\n  The script is saved at:\n    {script_path}\n"
            f"  You can inspect it and submit manually:\n"
            f"    sbatch {script_path}"
        )

    # Parse job ID from "Submitted batch job 12345"
    output = result.stdout.strip()
    print("submitted!")
    job_id = output.split()[-1] if output else "unknown"

    banner(f"✅  Build job submitted — Job ID {job_id}")
    print(f"  Partition   : {args.partition}")
    print(f"  CUDA module : {cuda_module}")
    print(f"  Log directory: {log_dir}/")
    print()
    print("  Monitor the job:")
    print(f"    squeue --job {job_id}              # check status")
    print(f"    tail -f {log_dir}/build_{job_id}.out   # live output")
    print(f"    scancel {job_id}                   # cancel if needed")
    print()
    print("  When the job completes, verify the build:")
    print("    make test-cuda")
    print("    make check-cuda-deps")
    print()
    print("  Then run splat fitting on a GPU node:")
    print(
        f"    hatch run luxar gsplat batch-fit submit <input.zarr> <output/> --partition {args.partition}"
    )
    print()


if __name__ == "__main__":
    main()
