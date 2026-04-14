#!/usr/bin/env python3
"""Crop inverted screenshots to remove viewer UI elements (titles, controls)."""

from pathlib import Path

from PIL import Image

FIGS_DIR = Path(__file__).parent.parent / "figs"

# Crop specs: (left%, top%, right%, bottom%) as fractions of image size
# Removes title bar at top and status bar at bottom
CROPS = {
    "viewer_gsplats_organoid_inverted.png": (0.10, 0.05, 0.90, 0.95),
    "viewer_gsplats_tribolium_inverted.png": (0.10, 0.08, 0.90, 0.92),
    "viewer_lorenz_attractor_inverted.png": (0.10, 0.05, 0.90, 0.95),
    "viewer_lsystem_forest_inverted.png": (0.05, 0.10, 0.95, 0.95),
    "viewer_spiral_galaxy_inverted.png": (0.10, 0.10, 0.90, 0.95),
    "viewer_storm_microtubules_inverted.png": (0.08, 0.12, 0.92, 0.90),
}

def main():
    for fname, (l, t, r, b) in CROPS.items():
        path = FIGS_DIR / fname
        if not path.exists():
            print(f"  Skipping {fname}")
            continue

        img = Image.open(path)
        w, h = img.size
        cropped = img.crop((int(l*w), int(t*h), int(r*w), int(b*h)))

        out = FIGS_DIR / fname.replace('_inverted.png', '_paper.png')
        cropped.save(out, quality=95)
        print(f"  {fname} -> {out.name} ({cropped.size[0]}x{cropped.size[1]})")

    print("Done!")

if __name__ == '__main__':
    main()
