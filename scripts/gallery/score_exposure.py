#!/usr/bin/env python3
"""Score gallery stills for over-exposure (blown highlights and flat subjects).

For each ``docs/images/gallery/<id>.png``, compute how the *subject* (the lit
foreground, ignoring the dark background) sits on the tone curve. A hero shot
reads badly in two distinct ways, matching the two failure modes the capture
harness guards against:

  1. A large, desaturated white blob swallows the structure/colour — a TAIL
     failure, caught by ``clipped_frac``.
  2. The whole subject sits high on the curve with no internal dynamic range —
     a headlit shaded surface (mesh) driven into the ACES shoulder, which
     desaturates toward white by design. The tail test cannot see this: a
     histogram only ~0.06 wide sitting at 0.87 has no tail above CLIP_LUMA.

RELATION TO THE HARNESS: the thresholds below are HAND-SYNCED copies of the
ones in ``packages/luxar-viewer/src/tests/screenshots/exposure-policy.ts`` —
nothing asserts they agree, so change both together. Two deliberate
differences: this scorer reads a lossless PNG while the harness measures a
JPEG-q70 screenshot (so the two can disagree at the margin), and the FLAT rule
here adds a ``p50 > FLAT_MID_MIN`` term that the harness's gate does not have
(the harness converges *onto* TARGET_MID, so the scorer must not flag its own
successful output).

Metrics per image (over the lit foreground, luma > LIT_THRESHOLD):
  - clipped_frac : fraction of lit pixels with luma > CLIP_LUMA AND low
                   saturation (near-white, not a saturated bright colour)
  - p10/p50/p99  : 10th / 50th / 99th-percentile luma of lit pixels
  - spread       : p99 - p10, the lit histogram's width
  - lit_frac     : fraction of the whole frame that is lit

Verdict: OVER if clipped_frac > CLIP_FRAC_MAX (too much of the subject is
blown to white); else FLAT if the frame actually has a subject
(lit_frac >= MIN_LIT_FRAC) whose histogram is narrow (spread <
NARROW_SPREAD_MAX) and parked high (p50 > FLAT_MID_MIN). OVER takes
precedence. Rows print worst-first: OVER, then FLAT, then ok, then EMPTY, and
within each class by the metric that class is ranked on.

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

# Background cutoff; hand-synced with exposure-policy.ts LIT_THRESHOLD (used by
# the harness that produces these stills), so the scores reflect what it
# actually optimizes.
LIT_THRESHOLD = 0.04
CLIP_LUMA = 0.95  # luma above this counts as "blown" (if also desaturated)
SAT_MAX = 0.15  # (max-min)/max below this = near-white / desaturated
# Matches the harness clip-guard (exposure-policy.ts CLIP_FRAC_MAX): a hero shot
# should have <5% of its subject blown to near-white.
CLIP_FRAC_MAX = 0.05  # > this fraction of the subject blown → over-exposed
# Matches the harness flat-subject gate (exposure-policy.ts NARROW_SPREAD_MAX):
# a lit histogram narrower than this has no internal dynamic range.
NARROW_SPREAD_MAX = 0.12
TARGET_MID = 0.5  # the harness's mid-tone target (exposure-policy.ts TARGET_MID)
# TARGET_MID plus margin. The harness's early exit is a SYMMETRIC band in log2
# (|log2(TARGET_MID / p50)| < MID_EXPOSURE_TOL ⇒ p50 ∈ [0.483, 0.518]), so a
# flat subject that phase 1 left below target converges from below just as
# readily as from above. In a band sweep on the simulated headlit subject the
# converged tiles landed at p50 0.505-0.509, and runs that exhausted
# MID_EXPOSURE_ITERS (or hit the EXPOSURE_MIN clamp) reached 0.586. 0.65 clears
# all of those, so the scorer never flags the harness's own successful output.
FLAT_MID_MIN = 0.65
# Mirrors exposure-policy.ts MIN_LIT_FRACTION: below this the frame has
# effectively nothing lit, so its percentiles describe a handful of stray pixels
# rather than a subject. The harness gates its flat-subject decision on the same
# floor; without it here, a frame with a single bright speck reports spread 0.0
# and a high p50 and sorts to the top of the worst-first table.
MIN_LIT_FRAC = 0.0005


def score_image(path: Path) -> dict:
    """Return exposure metrics + a verdict for one gallery still."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lit = luma > LIT_THRESHOLD
    n_lit = int(lit.sum())
    if n_lit == 0:
        return {
            "clipped_frac": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "spread": 0.0,
            "lit_frac": 0.0,
            "verdict": "EMPTY",
        }

    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    # "blown" = bright AND desaturated (a saturated bright colour is fine).
    blown = lit & (luma > CLIP_LUMA) & (sat < SAT_MAX)
    clipped_frac = float(blown.sum()) / n_lit
    p10, p50, p99 = (float(v) for v in np.percentile(luma[lit], [10, 50, 99]))
    spread = p99 - p10
    lit_frac = n_lit / luma.size
    # OVER (a blown tail) takes precedence; FLAT catches the narrow, uniformly
    # over-exposed subject the tail test structurally cannot see.
    if clipped_frac > CLIP_FRAC_MAX:
        verdict = "OVER"
    elif lit_frac >= MIN_LIT_FRAC and spread < NARROW_SPREAD_MAX and p50 > FLAT_MID_MIN:
        verdict = "FLAT"
    else:
        verdict = "ok"
    return {
        "clipped_frac": round(clipped_frac, 3),
        "p10": round(p10, 3),
        "p50": round(p50, 3),
        "p99": round(p99, 3),
        "spread": round(spread, 3),
        "lit_frac": round(lit_frac, 3),
        "verdict": verdict,
    }


# Worst-first ordering: an OVER row always outranks a FLAT one, and both
# outrank ok/EMPTY. A FLAT row has clipped_frac ≈ 0 by definition, so ranking
# the whole table on clipped_frac alone would bury every FLAT row at the bottom.
_VERDICT_RANK = {"OVER": 0, "FLAT": 1, "ok": 2, "EMPTY": 3}


def _sort_key(row: dict) -> tuple[int, float]:
    """Rank by verdict class, then by the metric that class is judged on."""
    rank = _VERDICT_RANK.get(row["verdict"], len(_VERDICT_RANK))
    # OVER is ranked by how blown it is; the rest by how high the subject sits.
    within = row["clipped_frac"] if row["verdict"] == "OVER" else row["p50"]
    return (rank, -within)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    pngs = sorted(GALLERY_DIR.glob("*.png"))
    rows = [{"id": p.stem, **score_image(p)} for p in pngs]
    rows.sort(key=_sort_key)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"\nExposure scan of {len(rows)} gallery stills  (worst-first)")
    print(
        f"{'id':<34} {'clipped':>8} {'p10':>6} {'p50':>6} {'p99':>6} "
        f"{'spread':>7} {'lit%':>6}  verdict"
    )
    print("-" * 92)
    over = 0
    flat = 0
    for r in rows:
        if r["verdict"] == "OVER":
            over += 1
        elif r["verdict"] == "FLAT":
            flat += 1
        print(
            f"{r['id']:<34} {r['clipped_frac']:>8.3f} {r['p10']:>6.3f} "
            f"{r['p50']:>6.3f} {r['p99']:>6.3f} {r['spread']:>7.3f} "
            f"{r['lit_frac'] * 100:>5.1f}%  {r['verdict']}"
        )
    print("-" * 92)
    print(f"{over}/{len(rows)} flagged OVER (clipped_frac > {CLIP_FRAC_MAX})")
    print(
        f"{flat}/{len(rows)} flagged FLAT "
        f"(lit_frac >= {MIN_LIT_FRAC}, spread < {NARROW_SPREAD_MAX}, "
        f"p50 > {FLAT_MID_MIN})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
