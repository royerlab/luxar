#!/usr/bin/env python3
"""
Compare Initialization Methods Demo

Compares convergence speed and quality of different initialization methods
(pyramid, uniform, coarse, finest) for multi-scale image decomposition.

Generates console-friendly ASCII plots showing loss curves over iterations.
"""

import sys
from typing import Dict, List

import numpy as np
from arbol import aprint, asection
from skimage import color, data, img_as_float32

from luxar.gsplats.multiscale import decompose_image

# Check for --no-viz flag
NO_VIZ = "--no-viz" in sys.argv

# ======= Demo knobs =======
SCALES = [1, 2, 4, 8]  # Scale factors to use
N_ITERS = 500  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
# ==========================


def plot_ascii_convergence(
    histories: Dict[str, List[Dict]],
    title: str = "Convergence Comparison",
    height: int = 20,
    width: int = 80
) -> None:
    """
    Create ASCII plot of convergence curves.

    Parameters
    ----------
    histories : dict
        Dictionary mapping method name to list of stats dicts
    title : str
        Plot title
    height : int
        Plot height in characters
    width : int
        Plot width in characters
    """
    # Extract reconstruction loss over iterations
    data = {}
    max_iters = 0
    min_loss = float('inf')
    max_loss = float('-inf')

    for method, history in histories.items():
        losses = [h['recon_loss'] for h in history]
        data[method] = losses
        max_iters = max(max_iters, len(losses))
        min_loss = min(min_loss, min(losses))
        max_loss = max(max_loss, max(losses))

    # Use log scale if range is large
    loss_range = max_loss - min_loss
    use_log = loss_range > 100 or (max_loss / (min_loss + 1e-12) > 100)

    if use_log:
        min_loss = np.log10(min_loss + 1e-12)
        max_loss = np.log10(max_loss + 1e-12)
        for method in data:
            data[method] = [np.log10(x + 1e-12) for x in data[method]]

    # Add some padding to range
    loss_range = max_loss - min_loss
    min_loss -= loss_range * 0.05
    max_loss += loss_range * 0.05

    # Symbols for different methods
    symbols = {
        'pyramid': '●',
        'uniform': '■',
        'coarse': '▲',
        'finest': '◆'
    }

    # Colors (ANSI escape codes)
    colors = {
        'pyramid': '\033[92m',  # Green
        'uniform': '\033[94m',  # Blue
        'coarse': '\033[93m',   # Yellow
        'finest': '\033[91m',   # Red
    }
    reset = '\033[0m'

    aprint("\n" + "=" * width)
    aprint(title.center(width))
    aprint("=" * width)

    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot each method
    for method, losses in data.items():
        symbol = symbols.get(method, '○')

        for i, loss in enumerate(losses):
            # Map to grid coordinates
            x = int((i / max_iters) * (width - 1))
            y = height - 1 - int(((loss - min_loss) / (max_loss - min_loss)) * (height - 1))
            y = max(0, min(height - 1, y))  # Clamp

            if grid[y][x] == ' ':
                grid[y][x] = symbol

    # Print grid with y-axis labels
    for i, row in enumerate(grid):
        # Y-axis value
        y_frac = 1.0 - (i / (height - 1))
        y_value = min_loss + y_frac * (max_loss - min_loss)

        if use_log:
            y_value = 10 ** y_value

        if i % 5 == 0:  # Label every 5th row
            label = f"{y_value:.2e}"
        else:
            label = ""

        aprint(f"{label:>10s} │ {''.join(row)}")

    # X-axis
    aprint(" " * 11 + "└" + "─" * width)
    aprint(" " * 11 + "0" + " " * (width - 20) + f"{max_iters} iterations")

    # Legend
    aprint("\nLegend:")
    for method in sorted(data.keys()):
        symbol = symbols.get(method, '○')
        color = colors.get(method, '')
        final_loss = histories[method][-1]['recon_loss']
        aprint(f"  {color}{symbol}{reset} {method:8s} (final loss: {final_loss:.6e})")

    aprint("")


def create_summary_table(
    results: Dict[str, Dict],
    width: int = 80
) -> None:
    """
    Create a summary table comparing methods.

    Parameters
    ----------
    results : dict
        Dictionary mapping method name to results dict
    width : int
        Table width in characters
    """
    aprint("\n" + "=" * width)
    aprint("Summary Comparison".center(width))
    aprint("=" * width)

    # Headers
    header = f"{'Method':12s} │ {'Init Loss':>12s} │ {'Final Loss':>12s} │ {'Time (s)':>8s} │ {'Coarse %':>8s}"
    aprint(header)
    aprint("─" * width)

    # Sort by final loss (best first)
    sorted_methods = sorted(results.keys(), key=lambda m: results[m]['final_loss'])

    for method in sorted_methods:
        r = results[method]
        init_loss = r['history'][0]['recon_loss']
        final_loss = r['final_loss']
        time_s = r['time_seconds']
        energy_dist = r['energy_distribution']
        coarse_pct = energy_dist[-1] * 100  # Last scale is coarsest

        row = f"{method:12s} │ {init_loss:12.6e} │ {final_loss:12.6e} │ {time_s:8.2f} │ {coarse_pct:7.1f}%"
        aprint(row)

    aprint("─" * width)

    # Find winner
    best_method = sorted_methods[0]
    aprint(f"\n🏆 Best convergence: {best_method} (lowest final loss)")

    # Analyze initialization quality
    init_losses = {m: results[m]['history'][0]['recon_loss'] for m in results}
    best_init = min(init_losses, key=init_losses.get)
    aprint(f"🎯 Best initialization: {best_init} (lowest initial loss)")

    # Fastest convergence (steepest improvement in first 50 iterations)
    improvements = {}
    for method, r in results.items():
        if len(r['history']) >= 50:
            init = r['history'][0]['recon_loss']
            iter50 = r['history'][49]['recon_loss']
            improvements[method] = (init - iter50) / init  # Fractional improvement

    if improvements:
        fastest = max(improvements, key=improvements.get)
        aprint(f"⚡ Fastest early convergence: {fastest} ({improvements[fastest]:.1%} improvement in 50 iters)")


def plot_ascii_energy_bars(
    results: Dict[str, Dict],
    width: int = 80
) -> None:
    """
    Create ASCII bar chart of energy distribution for each method.

    Parameters
    ----------
    results : dict
        Dictionary mapping method name to results dict
    width : int
        Chart width in characters
    """
    aprint("\n" + "=" * width)
    aprint("Energy Distribution by Method (Coarse → Fine)".center(width))
    aprint("=" * width)

    # Get scales from first result
    scales = list(results.values())[0]['scales']

    for method in sorted(results.keys()):
        energy_dist = results[method]['energy_distribution']

        aprint(f"\n{method.capitalize():12s}")
        for scale, energy_frac in zip(scales, energy_dist):
            energy_pct = energy_frac * 100
            bar_length = int(energy_pct / 2)  # Scale for 50 chars max
            bar = "█" * bar_length
            aprint(f"  Scale {scale:2d}x: {energy_pct:5.1f}% {bar}")


with asection("Multi-Scale Initialization Method Comparison"):
    aprint("Comparing 4 initialization methods: pyramid, uniform, coarse, finest")
    aprint(f"Test image: Astronaut (scikit-image)")
    aprint(f"Scales: {SCALES}")
    aprint(f"Iterations: {N_ITERS}")

    with asection("Loading test image"):
        # Load astronaut image and convert to grayscale
        img = data.astronaut()
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = color.rgb2gray(img)
        V = img_as_float32(img)

        # Crop to smaller size for faster demo
        V = V[:256, :256]

        aprint(f"Image shape: {V.shape}")
        aprint(f"Data range: [{V.min():.4f}, {V.max():.4f}]")

    # Run decomposition with each initialization method
    methods = ['pyramid', 'uniform', 'coarse', 'finest']
    results = {}
    histories = {}

    for method in methods:
        with asection(f"Testing '{method}' initialization"):
            scales_list, stats = decompose_image(
                V,
                scales=SCALES,
                n_iters=N_ITERS,
                init_method=method,
                energy_weight=0.001,
                alpha=1.5,
                device=DEVICE,
                verbose=True
            )

            results[method] = {
                'scales_list': scales_list,
                'final_loss': stats['final_error'],
                'time_seconds': stats['time_seconds'],
                'energy_distribution': stats['energy_distribution'],
                'scales': stats['scales'],
                'history': stats['history']
            }
            histories[method] = stats['history']

            aprint(f"✓ {method}: final loss = {stats['final_error']:.6e}, time = {stats['time_seconds']:.2f}s")

    # Visualization
    if not NO_VIZ:
        with asection("Convergence Analysis"):
            # Plot convergence curves
            plot_ascii_convergence(histories, title="Reconstruction Loss Over Iterations")

            # Summary table
            create_summary_table(results)

            # Energy distribution bars
            plot_ascii_energy_bars(results)

    aprint("\n✨ Comparison complete!")
    aprint("\nKey Observations:")
    aprint("  • 'pyramid' initialization typically starts with lowest loss")
    aprint("  • 'uniform' provides balanced starting point, works well with energy loss")
    aprint("  • 'coarse' strongly biases toward coarse scales from the start")
    aprint("  • 'finest' starts worst but shows optimization dynamics clearly")
    aprint("  • All methods converge to similar final quality with enough iterations")
