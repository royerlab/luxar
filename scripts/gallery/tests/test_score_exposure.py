"""Unit tests for scripts/gallery/score_exposure.py.

Collected by the default suite: ``scripts/gallery/tests`` is listed both on
pytest's ``testpaths`` AND on the explicit path arguments of the ``test`` /
``test-cov`` / ``test-cov-all`` hatch scripts — the latter matters, because an
explicit path argument OVERRIDES ``testpaths``. Run in isolation with:
    hatch run pytest scripts/gallery/tests/test_score_exposure.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import score_exposure as se  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_TS = (
    REPO_ROOT
    / "packages"
    / "luxar-viewer"
    / "src"
    / "tests"
    / "screenshots"
    / "exposure-policy.ts"
)

# Thresholds the scorer mirrors from the capture harness, as
# {name in exposure-policy.ts: name in score_exposure.py}. Deliberately NOT
# listed: FLAT_MID_MIN (a scorer-only margin, see its comment) and the harness's
# iteration/exposure-range knobs (nothing offline consumes them).
MIRRORED_THRESHOLDS = {
    "LIT_THRESHOLD": "LIT_THRESHOLD",
    "CLIP_LUMA": "CLIP_LUMA",
    "CLIP_SAT_MAX": "SAT_MAX",
    "CLIP_FRAC_MAX": "CLIP_FRAC_MAX",
    "NARROW_SPREAD_MAX": "NARROW_SPREAD_MAX",
    "TARGET_MID": "TARGET_MID",
    "MIN_LIT_FRACTION": "MIN_LIT_FRAC",
}

_TS_CONST = re.compile(
    r"^\s*(?:export\s+)?const\s+([A-Z][A-Z0-9_]*)\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*;",
    re.MULTILINE,
)


def _write(tmp_path: Path, name: str, rgb: np.ndarray) -> Path:
    """Save a float RGB array in [0,1] as an 8-bit PNG and return its path."""
    path = tmp_path / f"{name}.png"
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)).save(path)
    return path


def _solid(h: int, w: int, color: tuple[float, float, float]) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.float32), (h, w, 1))


def _emissive(h: int = 64, w: int = 64) -> np.ndarray:
    """Wide-histogram, emissive-like subject: a ramp from just-lit to bright."""
    ramp = np.linspace(0.06, 0.90, h * w, dtype=np.float32).reshape(h, w)
    # Slightly warm so nothing counts as "desaturated near-white".
    return np.stack([ramp, ramp * 0.85, ramp * 0.55], axis=-1)


def _blown(h: int = 64, w: int = 64) -> np.ndarray:
    """Half the frame pushed to pure white — the classic blown-highlight case."""
    img = _emissive(h, w).copy()
    img[: h // 2] = 1.0
    return img


def _flat_bright(h: int = 64, w: int = 64) -> np.ndarray:
    """Headlit-mesh case: a colour driven into the tone-curve shoulder.

    Luma sits ~0.86-0.90 with a spread of only a few hundredths. The blown-tail
    test sees nothing here simply because the LUMA term already excludes these
    pixels (max 0.9029 < CLIP_LUMA = 0.95) — the saturation term never even gets
    a chance to matter. That is the whole point: the failure has no tail.
    """
    green = np.linspace(0.85, 0.91, h * w, dtype=np.float32).reshape(h, w)
    return np.stack([np.ones_like(green), green, np.full_like(green, 0.55)], axis=-1)


def _converged_midtone(h: int = 64, w: int = 64) -> np.ndarray:
    """What a CORRECTLY fixed flat subject looks like after the mid-tone pass.

    The harness converges onto TARGET_MID from above and stops within
    MID_EXPOSURE_TOL, so a fixed tile is a narrow band at p50 ~ 0.52. The
    scorer must NOT flag its own successful output (hence FLAT_MID_MIN).
    """
    green = np.linspace(0.490, 0.535, h * w, dtype=np.float32).reshape(h, w)
    return np.stack(
        [np.full_like(green, 0.62), green, np.full_like(green, 0.30)], axis=-1
    )


def test_emissive_ramp_scores_ok(tmp_path: Path) -> None:
    row = se.score_image(_write(tmp_path, "emissive", _emissive()))
    assert row["verdict"] == "ok"
    assert row["clipped_frac"] == pytest.approx(0.0, abs=1e-3)
    # A wide histogram: p10 near the lit floor, p99 near the top of the ramp.
    assert row["p10"] < 0.2
    assert row["p99"] > 0.7
    assert row["spread"] > se.NARROW_SPREAD_MAX


def test_blown_image_scores_over(tmp_path: Path) -> None:
    row = se.score_image(_write(tmp_path, "blown", _blown()))
    assert row["verdict"] == "OVER"
    assert row["clipped_frac"] > se.CLIP_FRAC_MAX
    assert row["p99"] > se.CLIP_LUMA


def test_narrow_bright_image_scores_flat(tmp_path: Path) -> None:
    row = se.score_image(_write(tmp_path, "flat", _flat_bright()))
    assert row["verdict"] == "FLAT"
    # The failure the tail test structurally cannot see: nothing is blown...
    assert row["clipped_frac"] == pytest.approx(0.0, abs=1e-3)
    assert row["p99"] < se.CLIP_LUMA
    # ...yet the whole subject is a narrow band parked well above the mid-tone
    # target, so it is over-exposed in the way the tail test cannot express.
    assert row["spread"] < se.NARROW_SPREAD_MAX
    assert row["p50"] > se.FLAT_MID_MIN
    assert row["p10"] == pytest.approx(0.86, abs=0.02)
    assert row["p50"] == pytest.approx(0.88, abs=0.02)


def test_over_takes_precedence_over_flat(tmp_path: Path) -> None:
    """A narrow-but-blown subject (near-white everywhere) reads OVER, not FLAT."""
    row = se.score_image(_write(tmp_path, "white", _solid(32, 32, (0.99, 0.99, 0.99))))
    assert row["spread"] < se.NARROW_SPREAD_MAX
    assert row["p50"] > se.FLAT_MID_MIN
    assert row["verdict"] == "OVER"


def test_narrow_but_dark_image_is_not_flat(tmp_path: Path) -> None:
    """A narrow histogram BELOW the mid-tone target is correctly exposed, not FLAT."""
    row = se.score_image(_write(tmp_path, "dark", _solid(32, 32, (0.40, 0.35, 0.20))))
    assert row["spread"] < se.NARROW_SPREAD_MAX
    assert row["p50"] < se.TARGET_MID
    assert row["verdict"] == "ok"


def test_converged_midtone_scores_ok(tmp_path: Path) -> None:
    """REGRESSION GUARD: the harness's own converged output must not read FLAT.

    p50 lands just above TARGET_MID (0.5) because the mid-tone pass approaches
    from above; comparing against TARGET_MID with no margin would flag every
    correctly-fixed mesh tile.
    """
    row = se.score_image(_write(tmp_path, "converged", _converged_midtone()))
    assert row["spread"] < se.NARROW_SPREAD_MAX  # still a narrow subject...
    assert row["p50"] > se.TARGET_MID  # ...sitting just above the target...
    assert row["p50"] == pytest.approx(0.52, abs=0.02)
    assert row["verdict"] == "ok"  # ...and correctly NOT flagged.


def test_rows_sort_over_then_flat_then_ok() -> None:
    """Worst-first ordering ranks by verdict class, not by clipped_frac alone."""
    rows = [
        {"verdict": "ok", "clipped_frac": 0.0, "p50": 0.30},
        {"verdict": "EMPTY", "clipped_frac": 0.0, "p50": 0.0},
        {"verdict": "FLAT", "clipped_frac": 0.0, "p50": 0.88},
        {"verdict": "OVER", "clipped_frac": 0.2, "p50": 0.95},
        {"verdict": "OVER", "clipped_frac": 0.9, "p50": 0.99},
    ]
    ordered = sorted(rows, key=se._sort_key)
    assert [r["verdict"] for r in ordered] == ["OVER", "OVER", "FLAT", "ok", "EMPTY"]
    # Within the OVER class, the more blown row comes first.
    assert ordered[0]["clipped_frac"] == 0.9


def test_single_speck_frame_is_not_flat(tmp_path: Path) -> None:
    """A frame with almost nothing lit must not be flagged FLAT.

    Its percentiles describe a handful of stray pixels, not a subject: spread
    collapses to ~0 and p50 sits wherever the speck is, which would otherwise
    sort a non-problem to the TOP of the worst-first table. The harness gates
    its own flat-subject decision on the same lit-fraction floor.

    The frame must be realistically sized for the floor to bite: MIN_LIT_FRAC
    is 5e-4, so one lit pixel needs a frame of >~2000 px (a real gallery still
    is ~1e6). On a toy 8x8 frame a single pixel is 1.6% of the frame and would
    still clear the floor.
    """
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[0, 0] = (1.0, 0.88, 0.55)  # one bright, saturated speck
    row = se.score_image(_write(tmp_path, "speck", img))
    assert row["lit_frac"] < se.MIN_LIT_FRAC
    # The other two FLAT conditions DO hold — only the lit-fraction floor saves it.
    assert row["spread"] < se.NARROW_SPREAD_MAX
    assert row["p50"] > se.FLAT_MID_MIN
    assert row["verdict"] == "ok"


def test_thresholds_match_the_capture_harness() -> None:
    """The scorer's mirrored thresholds must equal the harness's own.

    ``score_exposure.py`` re-derives the harness's metrics offline from
    hand-written copies of its constants. If the two drift, the scorer silently
    scores against a policy the harness no longer implements — so pin them here
    rather than in a comment. Change a threshold in ``exposure-policy.ts`` and
    this test names the Python constant that needs the same edit.
    """
    if not POLICY_TS.exists():
        pytest.skip(f"viewer sources not present: {POLICY_TS}")
    ts_values = {
        name: float(value) for name, value in _TS_CONST.findall(POLICY_TS.read_text())
    }
    missing = sorted(set(MIRRORED_THRESHOLDS) - set(ts_values))
    assert not missing, (
        f"not found in {POLICY_TS.name} (renamed or removed?): {missing}"
    )
    mismatched = {
        ts_name: (ts_values[ts_name], getattr(se, py_name))
        for ts_name, py_name in MIRRORED_THRESHOLDS.items()
        if getattr(se, py_name) != ts_values[ts_name]
    }
    assert not mismatched, (
        f"score_exposure.py has drifted from exposure-policy.ts (ts, py): {mismatched}"
    )


def test_black_image_scores_empty(tmp_path: Path) -> None:
    row = se.score_image(_write(tmp_path, "black", _solid(16, 16, (0.0, 0.0, 0.0))))
    assert row["verdict"] == "EMPTY"
    assert row["p10"] == 0.0
    assert row["p50"] == 0.0
    assert row["p99"] == 0.0
    assert row["spread"] == 0.0
    assert row["lit_frac"] == 0.0
