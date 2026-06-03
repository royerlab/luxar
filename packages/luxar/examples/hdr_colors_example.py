#!/usr/bin/env python3
"""HDR Colors Example — emission > 1.0 with additive blending.

This example demonstrates Luxar's **HDR color** support:

- Float colors with **values > 1.0** are interpreted as high-dynamic-range
  emission. They are stored as ``float32`` instead of quantized to
  ``uint8``, and additive blending then sums their contributions
  faithfully — bright sources can saturate the tone-mapping curve and
  create visible glow without clipping the spectrum.
- The HDR / SDR mode is auto-detected at write time: any value > 1.0 in
  the colors array (or list/tuple) bumps the node into HDR mode. The
  log message ``✓ Detected HDR colors (values > 1.0)`` confirms it.
- Pairs nicely with a tone-mapping ``ViewerConfig`` (e.g. ACES) so the
  bright sources don't just blow out white.

Scene: two identical clouds of "stars" side-by-side. Left cloud uses
SDR colors (everything in [0, 1]). Right cloud uses HDR colors (peak ~5x
white) on the same ``additive`` blending mode. The bright cores of HDR
points show up obviously brighter; the dim peripheries match.

Educational value:
- Understand when (and why) to use HDR colors: emissive sources,
  multi-source accumulation, anything that would clip in SDR.
- See the difference HDR makes side-by-side with the SDR baseline.
- Combine with ``ViewerConfig(tone_mapping='ACES')`` from
  ``viewer_config_example.py`` for the full curated look.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import (
    CameraConfig,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
)
from luxar.utils.paths import get_examples_output_dir


def make_star_cluster(
    n_points: int, x_offset: float, intensity_peak: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cluster of points whose colors taper from peak at center to 0 at edge.

    Args:
        n_points: number of points
        x_offset: spatial offset along X (so the two clusters don't overlap)
        intensity_peak: brightness at the center (≤1 → SDR, >1 → HDR)
    """
    positions = rng.normal(0.0, 1.0, (n_points, 3)).astype(np.float32)
    positions[:, 0] += x_offset

    # Color tapers radially. With intensity_peak <= 1, every value
    # stays in [0, 1] (SDR). With intensity_peak > 1, the central
    # points exceed 1 and trigger HDR mode.
    radius = np.linalg.norm(positions - np.array([x_offset, 0.0, 0.0]), axis=1)
    falloff = np.exp(-(radius**2) / 2.0)
    base = np.array([1.0, 0.8, 0.4], dtype=np.float32)  # warm star colour
    colors = (intensity_peak * falloff[:, None] * base[None, :]).astype(np.float32)
    return positions, colors


def main() -> None:
    """Render an SDR and HDR cluster side-by-side under additive blending."""
    output_path = get_examples_output_dir() / "hdr_colors_example.zarr"
    aprint(f"Writing HDR-colors example to {output_path}")

    rng = np.random.default_rng(seed=0)
    sdr_positions, sdr_colors = make_star_cluster(
        2_000, x_offset=-3.0, intensity_peak=1.0, rng=rng
    )
    hdr_positions, hdr_colors = make_star_cluster(
        2_000, x_offset=3.0, intensity_peak=5.0, rng=rng
    )
    aprint(f"SDR cluster: max color = {sdr_colors.max():.2f}")
    aprint(
        f"HDR cluster: max color = {hdr_colors.max():.2f}  (>1.0 → HDR auto-detected)"
    )

    # ACES tone-mapping prevents the HDR cores from clipping straight
    # to white; it compresses the bright end into displayable range.
    viewer_config = ViewerConfig(
        tone_mapping="ACES",
        exposure=0.0,
        background_color="#05050a",
        camera=CameraConfig(position=(0.0, 0.0, 14.0), target=(0.0, 0.0, 0.0)),
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=viewer_config,
        )

        scene.add_points(
            "sdr_cluster",
            sdr_positions,
            colors=sdr_colors,
            radii=0.08,
            sharpness=1.0,
            blending_mode="additive",
            layer=True,
        )
        scene.add_points(
            "hdr_cluster",
            hdr_positions,
            colors=hdr_colors,
            radii=0.08,
            sharpness=1.0,
            blending_mode="additive",
            layer=True,
        )

        # Explainer overlay describing what to look for in the viewer.
        add_explainer(
            scene,
            title="HDR Colors",
            body=(
                "Two identical <code>additive</code> star clusters differ only "
                "in peak brightness: the left stays in SDR [0,1], the right "
                "peaks ~5x white as <strong>HDR</strong> (values &gt;1.0 are "
                "stored as float32). ACES tone mapping keeps cores from clipping."
            ),
            observe=[
                "Right (HDR) cluster cores render visibly hotter than the left.",
                "Dim peripheries of both clusters look essentially identical.",
                "HDR cores roll off via ACES instead of flat white clipping.",
                "Console logs HDR auto-detection for the right cluster.",
            ],
            observe_label="Notice",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Compare the cluster cores: HDR (right) renders visibly hotter.")


if __name__ == "__main__":
    main()
