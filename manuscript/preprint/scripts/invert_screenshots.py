#!/usr/bin/env python3
"""Invert viewer screenshots for paper: dark→white background, 180° hue rotation.

The inversion flips all pixel values (white↔black), and the 180° hue rotation
restores the original color impression on the now-white background.
"""

from pathlib import Path

import numpy as np
from PIL import Image

FIGS_DIR = Path(__file__).parent.parent / "figs"

SCREENSHOTS = [
    "viewer_gsplats_organoid.png",
    "viewer_gsplats_tribolium.png",
    "viewer_lorenz_attractor.png",
    "viewer_lsystem_forest.png",
    "viewer_spiral_galaxy.png",
    "viewer_storm_microtubules.png",
]


def invert_and_rotate_hue(img_array: np.ndarray) -> np.ndarray:
    """Invert image and rotate hue by 180 degrees.

    1. Invert: pixel = 255 - pixel (flips dark↔light)
    2. Convert to HSV, rotate H by 180°, convert back to RGB
    """
    # Step 1: Invert
    inverted = 255 - img_array[:, :, :3]  # Only RGB channels

    # Step 2: Convert to float [0,1] for HSV conversion
    h, w, _ = inverted.shape
    result = np.zeros_like(inverted, dtype=np.uint8)

    # Vectorized HSV conversion
    rgb_float = inverted.astype(np.float64) / 255.0

    # Use vectorized approach for speed
    r, g, b = rgb_float[:,:,0], rgb_float[:,:,1], rgb_float[:,:,2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / maxc, 0)

    # Compute hue
    delta = maxc - minc
    delta_safe = np.where(delta == 0, 1, delta)  # avoid division by zero

    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe

    h_val = np.where(r == maxc, bc - gc,
            np.where(g == maxc, 2.0 + rc - bc,
                     4.0 + gc - rc))
    h_val = (h_val / 6.0) % 1.0
    h_val = np.where(delta == 0, 0, h_val)

    # Rotate hue by 180 degrees
    h_val = (h_val + 0.5) % 1.0

    # Convert back to RGB
    i = (h_val * 6.0).astype(int)
    f = (h_val * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i % 6

    conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r_vals = [v, q, p, p, t, v]
    g_vals = [t, v, v, q, p, p]
    b_vals = [p, p, t, v, v, q]

    r_out = np.select(conditions, r_vals)
    g_out = np.select(conditions, g_vals)
    b_out = np.select(conditions, b_vals)

    result[:,:,0] = np.clip(r_out * 255, 0, 255).astype(np.uint8)
    result[:,:,1] = np.clip(g_out * 255, 0, 255).astype(np.uint8)
    result[:,:,2] = np.clip(b_out * 255, 0, 255).astype(np.uint8)

    return result


def main():
    for fname in SCREENSHOTS:
        path = FIGS_DIR / fname
        if not path.exists():
            print(f"  Skipping {fname} (not found)")
            continue

        img = Image.open(path).convert('RGB')
        arr = np.array(img)

        result = invert_and_rotate_hue(arr)

        out_name = fname.replace('.png', '_inverted.png')
        out_path = FIGS_DIR / out_name
        Image.fromarray(result).save(out_path, quality=95)
        print(f"  {fname} -> {out_name}")

    print("Done!")


if __name__ == '__main__':
    main()
