"""Shared cross-validation optimal point detection for manuscript analysis scripts.

Provides a single canonical implementation of the CV-optimal splat count
algorithm used across figure generation and analysis scripts.
"""

from __future__ import annotations

import numpy as np

# Algorithm thresholds (dB)
_PEAK_MARGIN = 0.1  # both sides of the peak must be this far below it
_KNEE_MARGIN = 0.3  # fallback margin for plateau-onset detection


def find_cv_optimal_idx(held_psnr_values) -> int:
    """Find the CV-optimal index in a held-out PSNR curve.

    Uses a hybrid approach:
      1. **Clear peak**: argmax where the average of values *before* and the
         average of values *after* are both at least 0.1 dB below the peak —
         indicating a true local maximum (overfitting sets in after the peak).
      2. **Plateau / monotonic**: if no clear peak is found, returns the first
         point within 0.3 dB of the maximum (plateau onset) — a conservative
         stopping criterion when overfitting is absent.

    Parameters
    ----------
    held_psnr_values : array-like
        Held-out PSNR values, ordered by increasing splat count.

    Returns
    -------
    int
        Positional index into *held_psnr_values* of the optimal point.
    """
    vals = np.asarray(held_psnr_values, dtype=float)

    peak_idx = int(np.argmax(vals))
    peak_val = vals[peak_idx]

    # Check if peak is a true local maximum (not just endpoint of a plateau)
    avg_before_gap = (peak_val - np.mean(vals[:peak_idx])) if peak_idx > 0 else 0.0
    avg_after_gap = (peak_val - np.mean(vals[peak_idx + 1:])) if peak_idx < len(vals) - 1 else 0.0
    has_clear_peak = avg_before_gap >= _PEAK_MARGIN and avg_after_gap >= _PEAK_MARGIN

    if has_clear_peak:
        return peak_idx

    # Plateau: first point within _KNEE_MARGIN dB of the max
    threshold = peak_val - _KNEE_MARGIN
    for i in range(len(vals)):
        if vals[i] >= threshold:
            return i
    return peak_idx  # fallback (should not be reached)
