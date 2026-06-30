"""``luxar gsplat benchmark`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_benchmark_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    pass


def benchmark_gpu(
    force: bool = typer.Option(
        False, "--force", help="Re-run even if a profile exists for this GPU"
    ),
    list_gpus: bool = typer.Option(False, "--list", help="List profiled GPUs and exit"),
    sweep: bool = typer.Option(
        True, "--sweep/--no-sweep", help="Include splat count sweep"
    ),
    shape: Optional[str] = typer.Option(
        None, "--shape", help="Sweep volume shape (e.g., '512,512,512')"
    ),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a one-shot Slurm job"),
    partition: Optional[str] = typer.Option(
        None, "--partition", help="Slurm partition (for --slurm)"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
) -> None:
    """Benchmark CUDA kernels and build a GPU performance profile.

    Runs the CUDA benchmark suite to characterize GPU throughput, OOM
    boundaries, and optimal operating points. Results are stored in
    ~/.luxar/gpu_profiles.yaml for use by `luxar gsplat batch-fit`.

    Multiple runs on the same GPU are aggregated (averaged throughput,
    conservative OOM boundaries).

    Examples:
        luxar gsplat benchmark
        luxar gsplat benchmark --list
        luxar gsplat benchmark --force
        luxar gsplat benchmark --slurm --partition gpu
    """
    from luxar.gsplats.gpu_profile import (
        PROFILE_PATH,
        load_profiles,
    )

    # --list mode
    if list_gpus:
        profiles = load_profiles()
        gpus = profiles.get("gpus", {})
        if not gpus:
            aprint("No GPU profiles found.")
            aprint("Run `luxar gsplat benchmark` to create one.")
            raise typer.Exit(0)

        with asection("Profiled GPUs"):
            for name, entry in gpus.items():
                info = entry.get("info", {})
                summary = entry.get("summary", {})
                n_runs = len(entry.get("runs", []))
                recs = summary.get("recommendations", {})
                peak = recs.get("peak_throughput_3d", {})

                aprint(f"\n{name}")
                aprint(f"  Memory: {info.get('total_memory_gb', '?')} GB")
                aprint(f"  Compute: sm_{info.get('compute_capability', '?')}")
                aprint(f"  Benchmark runs: {n_runs}")
                if peak:
                    shape_str = "x".join(str(s) for s in peak.get("shape", []))
                    aprint(
                        f"  Peak 3D: {peak.get('gvoxel_per_s', '?')} GV/s "
                        f"at {shape_str}"
                    )
                oom = summary.get("oom_boundaries", {}).get("3d", {})
                if oom.get("max_successful_shape"):
                    shape_str = "x".join(str(s) for s in oom["max_successful_shape"])
                    aprint(f"  Max safe 3D: {shape_str}")

        raise typer.Exit(0)

    # --slurm mode: submit a one-shot job
    if slurm:
        if not partition:
            aprint("Error: --partition is required with --slurm")
            raise typer.Exit(1)

        import subprocess
        import tempfile

        from luxar.gsplats.batch.env_capture import (
            capture_environment,
            generate_env_preamble,
        )

        env = capture_environment()
        preamble = generate_env_preamble(env)

        script = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name=luxar-benchmark\n"
            f"#SBATCH --partition={partition}\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --gpus-per-task=1\n"
            "#SBATCH --cpus-per-task=4\n"
            "#SBATCH --mem=32G\n"
            "#SBATCH --time=00:30:00\n"
            "#SBATCH --output=luxar-benchmark.out\n"
            "#SBATCH --error=luxar-benchmark.err\n\n"
            f"{preamble}\n\n"
            "luxar gsplat benchmark --force"
            f"{' --no-sweep' if not sweep else ''}"
            f"{' --shape ' + shape if shape else ''}\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            aprint(f"Submitting benchmark job to partition '{partition}'...")
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                aprint(f"Error submitting job: {result.stderr}")
                raise typer.Exit(1)
            aprint(result.stdout.strip())
            aprint(f"Profile will be saved to: {PROFILE_PATH}")
        finally:
            import os

            os.unlink(script_path)
        raise typer.Exit(0)

    # Direct benchmark run
    try:
        import torch
    except ImportError:
        aprint("Error: PyTorch is required for GPU benchmarking.")
        raise typer.Exit(1)

    if not torch.cuda.is_available():
        aprint("Error: CUDA is not available on this system.")
        aprint("Run `luxar gsplat benchmark --slurm` to benchmark on a GPU node.")
        raise typer.Exit(1)

    gpu_name_detected = torch.cuda.get_device_properties(0).name

    # Check if profile already exists
    if not force:
        profiles = load_profiles()
        if gpu_name_detected in profiles.get("gpus", {}):
            n_runs = len(profiles["gpus"][gpu_name_detected].get("runs", []))
            aprint(f"Profile already exists for {gpu_name_detected} ({n_runs} runs).")
            aprint("Use --force to add another run, or --list to view.")
            raise typer.Exit(0)

    try:
        from luxar.gsplats.models.gsplats.cuda.benchmark import (
            generate_profile,
            run_benchmark,
            run_splat_sweep,
        )
    except ImportError:
        aprint("Error: CUDA splatting backend is not compiled.")
        aprint("Build it with: make build-cuda")
        raise typer.Exit(1)

    with asection(f"Benchmarking GPU: {gpu_name_detected}"):
        results = run_benchmark(verbose=verbose)

        sweep_results = None
        sweep_shape_parsed = None
        if sweep:
            if shape:
                sweep_shape_parsed = tuple(int(s) for s in shape.split(","))
            elif results:
                best_gvs = 0.0
                for _label, r in results.items():
                    if r["dim"] == 3 and r.get("gvoxel_per_s_fp32") is not None:
                        if r["gvoxel_per_s_fp32"] > best_gvs:
                            best_gvs = r["gvoxel_per_s_fp32"]
                            sweep_shape_parsed = r["shape"]
            if sweep_shape_parsed is None:
                sweep_shape_parsed = (512, 512, 512)

            sweep_results = run_splat_sweep(shape=sweep_shape_parsed, verbose=verbose)

        profile_path = generate_profile(
            benchmark_results=results or {},
            sweep_results=sweep_results,
            sweep_shape=sweep_shape_parsed,
        )

    aprint(f"\nProfile saved to: {profile_path}")
    aprint("View with: luxar gsplat benchmark --list")


def register_benchmark_commands(app: typer.Typer) -> None:
    """Register the benchmark commands onto ``app_gsplat``."""
    app.command("benchmark")(benchmark_gpu)
