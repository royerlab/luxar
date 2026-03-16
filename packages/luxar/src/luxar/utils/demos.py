"""luxar.demos – Demo scene generators and precomputed-data helpers for Luxar.

This module provides functions to generate various demo scenes for testing
and demonstration purposes, plus helpers for loading precomputed GSplat data
from Git LFS (shipped with the package) or a local cache.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from arbol import aprint, asection

from ..core.dimensions import Dimension, Dimensions
from ..io.compiler import LuxarZarrCompiler
from ..typing_utils.config import check_dataset_size_warning
from ..typing_utils.protocols import PathLike


def detect_device(verbose: bool = True) -> str:
    """Auto-detect the best available compute device (cuda > mps > cpu).

    Args:
        verbose: If True, print detected device via arbol.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        if verbose:
            aprint("Using CUDA device")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        if verbose:
            aprint("Using MPS device (Metal acceleration)")
    else:
        device = "cpu"
        if verbose:
            aprint("Using CPU device")
    return device


def warn_if_no_cuda_gpu() -> None:
    """Print a warning if no CUDA GPU is available.

    GSplat demos require significant GPU compute for fitting.  Running on
    CPU is orders of magnitude slower and generally impractical for
    production runs.  This function prints a prominent warning so users
    understand the hardware requirements before waiting hours for a CPU run.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return  # All good
        device = (
            "MPS"
            if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "CPU"
        )
    except ImportError:
        device = "CPU (PyTorch not installed)"

    aprint("")
    aprint("=" * 70)
    aprint("WARNING: No CUDA GPU detected — running on " + device)
    aprint("=" * 70)
    aprint("GSplat demos require a CUDA GPU for practical performance.")
    aprint("Without one, fitting can take hours instead of minutes.")
    if device.startswith("MPS"):
        aprint("MPS (Apple Metal) provides some acceleration but is much")
        aprint("slower than CUDA for Gaussian splatting workloads.")
    aprint("")
    aprint("Options:")
    aprint("  - Use a machine with an NVIDIA GPU (CUDA)")
    aprint("  - Use --timepoints=2 for a quick test run")
    aprint("  - Use --serve-only if a scene was already generated")
    aprint("=" * 70)
    aprint("")


# =============================================================================
# Precomputed Data Helpers
# =============================================================================

# Directory containing precomputed data shipped with the package (via Git LFS)
_DEMOS_DATA_DIR = Path(__file__).resolve().parent.parent / "demos" / "data"

# Default user-level cache
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "luxar"


def is_lfs_pointer(path: Path) -> bool:
    """Check whether *path* is an unpulled Git LFS pointer file.

    LFS pointers are small text files (< 1 KB) whose first line is
    ``version https://git-lfs.github.com/spec/v1``.
    """
    if not path.exists():
        return False
    if path.stat().st_size > 1024:
        return False
    try:
        with open(path, "r") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except (UnicodeDecodeError, OSError):
        return False


def _validate_lfs_files(paths: list[Path]) -> None:
    """Raise a helpful error if any *paths* are missing or are LFS pointers."""
    missing = [p for p in paths if not p.exists()]
    pointers = [p for p in paths if p.exists() and is_lfs_pointer(p)]

    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Precomputed data files not found: {names}\n"
            "Run 'git lfs pull' to download the data files.\n"
            "Alternatively, use --recompute to fit from scratch (requires GPU)."
        )
    if pointers:
        names = ", ".join(p.name for p in pointers)
        raise FileNotFoundError(
            f"Precomputed data files are Git LFS pointers (not pulled): {names}\n"
            "Run 'git lfs pull' to download the actual data files.\n"
            "Alternatively, use --recompute to fit from scratch (requires GPU)."
        )


def parse_demo_flags() -> dict:
    """Parse common GSplat demo command-line flags from ``sys.argv``.

    Returns a dict with keys: ``recompute``, ``no_serve``, ``serve_only``.

    Handles the ``--no-cache`` → ``--recompute`` deprecation.
    """
    recompute = "--recompute" in sys.argv
    if "--no-cache" in sys.argv:
        aprint(
            "WARNING: --no-cache is deprecated, use --recompute instead. "
            "Treating as --recompute."
        )
        recompute = True

    return {
        "recompute": recompute,
        "no_serve": "--no-serve" in sys.argv,
        "serve_only": "--serve-only" in sys.argv,
    }


def load_precomputed_gsplats(
    demo_name: str,
    file_names: list[str],
    *,
    recompute: bool = False,
) -> list | None:
    """Load precomputed GSplat data from Git LFS / local cache.

    Resolution order (per file):
      1. If ``recompute`` is True, return ``None`` immediately.
      2. Check the local cache ``~/.cache/luxar/<demo_name>/``.
      3. Copy from ``demos/data/<demo_name>/`` (LFS) to local cache.
      4. Load from local cache.

    Args:
        demo_name: Subdirectory name under ``demos/data/`` (e.g. ``"tribolium"``).
        file_names: File basenames to load (e.g. ``["tribolium_gsplats.gsplats.zarr.zip"]``).
        recompute: If True, skip precomputed data entirely and return None.

    Returns:
        List of :class:`GSplatData` in the same order as *file_names*,
        or ``None`` when the caller should recompute.
    """
    from ..gsplats.gsplat_data import GSplatData

    if recompute:
        return None

    cache_dir = _DEFAULT_CACHE_ROOT / demo_name
    lfs_dir = _DEMOS_DATA_DIR / demo_name

    with asection(f"Loading precomputed GSplats ({demo_name})"):
        # Ensure cache dir exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Copy any missing files from LFS source to cache
        for fname in file_names:
            cache_file = cache_dir / fname
            if not cache_file.exists():
                lfs_file = lfs_dir / fname
                _validate_lfs_files([lfs_file])
                aprint(f"Copying {fname} from package data to cache")
                shutil.copy2(lfs_file, cache_file)

        # Load all
        results = []
        for fname in file_names:
            cache_file = cache_dir / fname
            gsplats = GSplatData.load(cache_file, include_stats=False)
            aprint(f"Loaded {fname}: {len(gsplats.amplitudes):,} splats")
            results.append(gsplats)

        return results


def load_precomputed_bundle(
    demo_name: str,
    bundle_name: str,
    file_names: list[str],
    *,
    recompute: bool = False,
) -> list | None:
    """Load precomputed GSplat data from a bundled zip archive in Git LFS.

    For timelapse demos where many per-frame ``.gsplats.zarr.zip`` files are
    bundled into a single outer ``.zip`` stored via Git LFS.

    Args:
        demo_name: Subdirectory name under ``demos/data/`` (e.g. ``"zebrafish"``).
        bundle_name: Filename of the outer bundle zip (e.g. ``"zebrafish_gsplats.zip"``).
        file_names: Basenames of per-frame files *inside* the bundle to load,
            in the desired order.
        recompute: If True, return None.

    Returns:
        List of :class:`GSplatData`, or ``None`` when the caller should recompute.
    """
    from ..gsplats.gsplat_data import GSplatData

    if recompute:
        return None

    cache_dir = _DEFAULT_CACHE_ROOT / demo_name
    bundle_path = _DEMOS_DATA_DIR / demo_name / bundle_name

    with asection(f"Loading precomputed GSplats bundle ({demo_name})"):
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if all files already in cache
        missing = [f for f in file_names if not (cache_dir / f).exists()]

        if missing:
            # Extract from bundle
            _validate_lfs_files([bundle_path])
            aprint(f"Extracting {len(missing)} files from {bundle_name}")
            with zipfile.ZipFile(bundle_path, "r") as zf:
                members = zf.namelist()
                for fname in missing:
                    # Files may be at top level or inside a directory in the zip
                    matching = [m for m in members if m.endswith(fname)]
                    if not matching:
                        raise FileNotFoundError(
                            f"{fname} not found in bundle {bundle_name}. "
                            f"Available: {members[:5]}..."
                        )
                    zf.extract(matching[0], cache_dir)
                    # If extracted into a subdirectory, move to cache root
                    extracted = cache_dir / matching[0]
                    target = cache_dir / fname
                    if extracted != target:
                        shutil.move(str(extracted), str(target))

        # Load all
        results = []
        for fname in file_names:
            cache_file = cache_dir / fname
            gsplats = GSplatData.load(cache_file, include_stats=False)
            aprint(f"Loaded {fname}: {len(gsplats.amplitudes):,} splats")
            results.append(gsplats)

        return results


def launch_viewer(output_path: Union[str, Path], open_browser: bool = True) -> None:
    """Launch the Luxar viewer to display a dataset.

    This function uses sys.executable to ensure it works regardless of how
    the demo script was launched (hatch run, hatch shell, conda, etc.).

    Args:
        output_path: Path to the .zarr dataset to view
        open_browser: Whether to automatically open a browser window

    Raises:
        SystemExit: If the viewer fails to launch
    """
    cmd = [sys.executable, "-m", "luxar", "serve", str(output_path), "--viewer"]
    if open_browser:
        cmd.append("--open")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        aprint("\n🛑 Stopping demo...")
    except subprocess.CalledProcessError as e:
        # Check if it's a "module not found" error
        if e.returncode == 1:
            aprint(f"\n❌ Error running luxar: {e}")
            aprint("Make sure luxar is installed in your Python environment.")
            aprint("If using hatch: run this demo with 'hatch run python <demo.py>'")
        else:
            aprint(f"\n❌ Error: {e}")
            aprint(
                "Make sure the viewer is built: cd packages/luxar-viewer && pnpm build"
            )
        sys.exit(1)


def create_lorenz_attractor(
    store_path: PathLike,
    n_points: int = 10_000,
    seed: Optional[int] = None,
) -> None:
    """Create a demo scene with a Lorenz attractor visualization.

    This creates a beautiful butterfly-shaped 3D structure with colors
    that transition smoothly over time, demonstrating the Luxar scene format
    with an aesthetically pleasing mathematical visualization.

    Args:
        store_path: Path to the Zarr store to create
        n_points: Number of points to generate along the attractor
        seed: Random seed for reproducible results

    Example:
        >>> from luxar.utils import create_lorenz_attractor
        >>> create_lorenz_attractor('demo.zarr', n_points=50000)
    """
    aprint(f"Creating Lorenz attractor demo scene with {n_points:,} points.")

    # Check for performance warnings
    warning = check_dataset_size_warning(n_points)
    if warning:
        aprint(f"Performance warning: {warning}")

    # Lorenz attractor parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dt = 0.01

    # Initialize arrays
    positions = np.zeros((n_points, 3), dtype=np.float32)

    # Starting point (with small random perturbation if seed is provided)
    rng = np.random.default_rng(seed)
    x, y, z = 0.1, 0.0, 0.0
    if seed is not None:
        x += rng.uniform(-0.01, 0.01)

    # Generate Lorenz attractor points
    for i in range(n_points):
        # Lorenz equations
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        x += dx
        y += dy
        z += dz

        positions[i] = [x, y, z]

    # Scale positions to fit nicely in view
    positions *= 0.1

    # Center the attractor at its center of mass
    center_of_mass = np.mean(positions, axis=0)
    positions -= center_of_mass

    # Create time-based colors with smooth transitions
    # Using HSV color space for smooth color transitions
    t = np.linspace(0, 1, n_points)
    hue = (t * 2) % 1.0  # Cycle through hues twice

    # Convert HSV to RGB using vectorized operations
    # Full saturation and value for vibrant colors
    s, v = 1.0, 1.0

    # HSV to RGB vectorized conversion
    # Based on standard HSV→RGB algorithm, vectorized for performance
    c = v * s  # Chroma
    h_prime = hue * 6.0  # Hue in [0, 6) range
    x = c * (1 - np.abs(h_prime % 2 - 1))  # Intermediate value
    m = v - c  # Match value

    # Initialize RGB arrays
    r = np.zeros(n_points, dtype=np.float32)
    g = np.zeros(n_points, dtype=np.float32)
    b = np.zeros(n_points, dtype=np.float32)

    # Apply RGB values based on hue sector (0-5)
    # Each sector represents 60° of the color wheel
    sector = np.floor(h_prime).astype(int)

    # Sector 0: Red to Yellow (R=max, G=rising, B=0)
    mask = sector == 0
    r[mask], g[mask], b[mask] = c, x[mask], 0.0

    # Sector 1: Yellow to Green (R=falling, G=max, B=0)
    mask = sector == 1
    r[mask], g[mask], b[mask] = x[mask], c, 0.0

    # Sector 2: Green to Cyan (R=0, G=max, B=rising)
    mask = sector == 2
    r[mask], g[mask], b[mask] = 0.0, c, x[mask]

    # Sector 3: Cyan to Blue (R=0, G=falling, B=max)
    mask = sector == 3
    r[mask], g[mask], b[mask] = 0.0, x[mask], c

    # Sector 4: Blue to Magenta (R=rising, G=0, B=max)
    mask = sector == 4
    r[mask], g[mask], b[mask] = x[mask], 0.0, c

    # Sector 5: Magenta to Red (R=max, G=0, B=falling)
    mask = sector == 5
    r[mask], g[mask], b[mask] = c, 0.0, x[mask]

    # Add match value to get final RGB (adjust for brightness)
    colors = np.column_stack([r + m, g + m, b + m]).astype(np.float32)

    # Create scene with new API
    with LuxarZarrCompiler(store_path) as compiler:
        # Define 3D dimensions
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        compiler.create_scene(dimensions=dims)

        # Write the attractor data
        # NOTE: radii must be scaled by same factor as positions!
        compiler.write_points(
            "LorenzAttractor",
            positions * 100.0 - 50.0,  # Scale up for better visibility
            colors=colors,
            radii=2.0,  # Uniform radius for visibility
            opacity=0.9,
            blending_mode="additive",
        )

    aprint(f"✓ Lorenz attractor demo scene created at {store_path}")


def create_random_spheres(
    store_path: PathLike,
    n_spheres: int = 100,
    points_per_sphere: int = 1000,
    seed: Optional[int] = None,
) -> None:
    """Create a demo scene with random colored spheres.

    Args:
        store_path: Path to the Zarr store to create
        n_spheres: Number of spheres to generate
        points_per_sphere: Points per sphere
        seed: Random seed for reproducible results
    """
    rng = np.random.default_rng(seed)

    aprint(f"Creating random spheres demo with {n_spheres} spheres")

    with LuxarZarrCompiler(store_path) as compiler:
        from luxar import Dimensions

        compiler.create_scene(dimensions=Dimensions.default_3d())

        for i in range(n_spheres):
            # Random center position
            center = rng.uniform(-10, 10, 3)

            # Generate points on sphere surface
            phi = rng.uniform(0, 2 * np.pi, points_per_sphere)
            costheta = rng.uniform(-1, 1, points_per_sphere)
            u = rng.uniform(0, 1, points_per_sphere)

            theta = np.arccos(costheta)

            radius = rng.uniform(0.5, 2.0)
            r = radius * u ** (1 / 3)

            x = r * np.sin(theta) * np.cos(phi) + center[0]
            y = r * np.sin(theta) * np.sin(phi) + center[1]
            z = r * np.cos(theta) + center[2]

            positions = np.column_stack([x, y, z]).astype(np.float32)

            # Random HDR color for each sphere - use tuple for scalar passthrough
            color_arr = rng.uniform(0.5, 2.0, 3)
            color_tuple = tuple(float(c) for c in color_arr)  # Convert to tuple

            compiler.write_points(
                f"sphere_{i:03d}",
                positions,
                colors=color_tuple,  # Single color for whole sphere (tuple for broadcasting)
                radii=0.05,  # Scalar passthrough
                opacity=0.8,
            )

    aprint(f"✓ Random spheres demo created at {store_path}")


def create_time_series_demo(
    store_path: PathLike,
    n_timepoints: int = 10,
    n_points_per_time: int = 1000,
    seed: Optional[int] = None,
) -> None:
    """Create a 4D time series demo scene.

    Args:
        store_path: Path to the Zarr store to create
        n_timepoints: Number of time points
        n_points_per_time: Points per time step
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    aprint(f"Creating 4D time series demo with {n_timepoints} time points")

    # Ensure at least 2 timepoints for valid dimension range
    if n_timepoints < 2:
        n_timepoints = 2
        aprint("  Note: Minimum 2 time points required, using 2")

    # Create 4D dimensions
    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
            Dimension(
                "time",
                unit="s",
                display=False,
                discrete=True,
                range=(0, n_timepoints - 1),
            ),
        ]
    )

    with LuxarZarrCompiler(store_path) as compiler:
        compiler.create_scene(dimensions=dims)

        # Generate all time points
        all_positions = []
        all_colors = []

        for t in range(n_timepoints):
            # Generate expanding sphere over time
            radius = 1.0 + t * 0.5

            # Random points in sphere
            phi = rng.uniform(0, 2 * np.pi, n_points_per_time)
            costheta = rng.uniform(-1, 1, n_points_per_time)
            u = rng.uniform(0, 1, n_points_per_time)

            theta = np.arccos(costheta)
            r = radius * u ** (1 / 3)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Add time dimension
            time_coord = np.full(n_points_per_time, t, dtype=np.float32)

            positions_4d = np.column_stack([x, y, z, time_coord]).astype(np.float32)
            all_positions.append(positions_4d)

            # Color changes over time
            color = np.array(
                [1.0 - t / n_timepoints, 0.5, t / n_timepoints], dtype=np.float32
            )
            colors = np.tile(color, (n_points_per_time, 1))
            all_colors.append(colors)

        # Concatenate all time points
        positions_array = np.vstack(all_positions)
        colors_array = np.vstack(all_colors)

        # Write as single 4D dataset
        compiler.write_points(
            "time_series",
            positions_array,
            colors=colors_array,
            radii=0.1,  # Scalar passthrough
        )

    aprint(f"✓ Time series demo created at {store_path}")
