#!/usr/bin/env python3
"""Score gallery stills for over-exposure (blown highlights).

For each ``docs/images/gallery/<id>.png``, compute how much of the *subject*
(the lit foreground, ignoring the dark background) is clipped to near-white.
A hero shot reads badly when a large, desaturated white blob swallows the
structure/colour — that's what we flag.

Metrics per image (over the lit foreground, luma > LIT_THRESHOLD):
  - clipped_frac : fraction of lit pixels with luma > CLIP_LUMA AND low
                   saturation (near-white, not a saturated bright colour)
  - p99          : 99th-percentile luma of lit pixels
  - lit_frac     : fraction of the whole frame that is lit

Verdict: OVER if clipped_frac > CLIP_FRAC_MAX (too much of the subject is
blown to white). Printed sorted worst-first so tuning can focus there.

Usage:
    hatch run python scripts/gallery/score_exposure.py
    hatch run python scripts/gallery/score_exposure.py --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
GALLERY_DIR = REPO_ROOT / "docs" / "images" / "gallery"

LIT_THRESHOLD = 0.05  # luma below this is background
CLIP_LUMA = 0.95  # luma above this counts as "blown" (if also desaturated)
SAT_MAX = 0.15  # (max-min)/max below this = near-white / desaturated
# Matches the harness clip-guard (generate-gallery.spec.ts CLIP_FRAC_MAX): a
# hero shot should have <5% of its subject blown to near-white.
CLIP_FRAC_MAX = 0.05  # > this fraction of the subject blown → over-exposed


def score_image(path: Path) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lit = luma > LIT_THRESHOLD
    n_lit = int(lit.sum())
    if n_lit == 0:
        return {"clipped_frac": 0.0, "p99": 0.0, "lit_frac": 0.0, "verdict": "EMPTY"}

    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    # "blown" = bright AND desaturated (a saturated bright colour is fine).
    blown = lit & (luma > CLIP_LUMA) & (sat < SAT_MAX)
    clipped_frac = float(blown.sum()) / n_lit
    p99 = float(np.percentile(luma[lit], 99))
    lit_frac = n_lit / luma.size
    verdict = "OVER" if clipped_frac > CLIP_FRAC_MAX else "ok"
    return {
        "clipped_frac": round(clipped_frac, 3),
        "p99": round(p99, 3),
        "lit_frac": round(lit_frac, 3),
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    pngs = sorted(GALLERY_DIR.glob("*.png"))
    rows = [{"id": p.stem, **score_image(p)} for p in pngs]
    rows.sort(key=lambda r: r["clipped_frac"], reverse=True)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"\nExposure scan of {len(rows)} gallery stills  (worst-first)")
    print(f"{'id':<34} {'clipped':>8} {'p99':>6} {'lit%':>6}  verdict")
    print("-" * 66)
    over = 0
    for r in rows:
        if r["verdict"] == "OVER":
            over += 1
        print(
            f"{r['id']:<34} {r['clipped_frac']:>8.3f} {r['p99']:>6.3f} "
            f"{r['lit_frac'] * 100:>5.1f}%  {r['verdict']}"
        )
    print("-" * 66)
    print(f"{over}/{len(rows)} flagged OVER (clipped_frac > {CLIP_FRAC_MAX})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
