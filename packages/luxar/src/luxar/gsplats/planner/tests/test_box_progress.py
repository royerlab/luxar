"""Per-box progress lines of a content-tiled (planned) fit.

The inner box fits run quiet, so without these a long content fit printed
nothing between "Fitting N boxes" and the end.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from luxar.gsplats.planner import box_progress
from luxar.gsplats.planner.box_progress import (
    BoxProgress,
    OncePerRunWarnings,
    format_duration,
)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0.4, "0s"), (42, "42s"), (187, "3m07s"), (3725, "1h02m"), (float("nan"), "?")],
)
def test_format_duration(seconds: float, text: str) -> None:
    assert format_duration(seconds) == text


def _fake_clock(monkeypatch: pytest.MonkeyPatch, *ticks: float) -> None:
    clock = iter(ticks)
    monkeypatch.setattr(box_progress.time, "perf_counter", lambda: next(clock))


def test_eta_is_budget_weighted(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _fake_clock(monkeypatch, 0.0, 10.0, 10.0, 10.0)
    progress = BoxProgress({0: 100, 1: 300})
    # 10 s for 100 of 400 budget units -> 30 s for the remaining 300.
    progress.box_done(0, n_kept=12, psnr_db=31.234)
    line = capsys.readouterr().out
    assert "box 0 done (1/2): kept 12 splats, PSNR 31.23 dB (padded crop)" in line
    assert "10s elapsed, ETA 30s" in line


def test_resumed_boxes_count_as_done_but_not_towards_timing(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A checkpoint reload is neither fitting time nor remaining budget."""
    # t0, reset by the up-front resume, reset by the lazy resume, box_done x2.
    _fake_clock(monkeypatch, 0.0, 50.0, 60.0, 70.0, 70.0)
    progress = BoxProgress({0: 100, 1: 100, 2: 100, 3: 100}, resumed=[0])
    progress.box_resumed(1)
    progress.box_done(2)
    line = capsys.readouterr().out
    assert "box 1 resumed from checkpoint (2/4)" in line
    # 10 s fitting 100 units, 100 units left (resumed boxes excluded) -> 10 s.
    assert "box 2 done (3/4): 10s elapsed, ETA 10s" in line


def test_the_start_line_counts_this_runs_boxes(capsys) -> None:
    progress = BoxProgress({4: 1500, 9: 10})
    progress.box_started(9)
    assert "box 9 (1/2): fitting, budget 10 splats" in capsys.readouterr().out


def test_last_box_has_no_eta_and_a_failure_says_so(capsys) -> None:
    progress = BoxProgress({4: 10})
    progress.box_done(4, failed=True, detail="exit 3")
    line = capsys.readouterr().out
    assert "box 4 FAILED (1/1): exit 3" in line
    assert "ETA" not in line


def test_iter_callback_reports_each_quarter_once(capsys) -> None:
    progress = BoxProgress({0: 10})
    cb = progress.iter_callback(0, 100)
    for it in (5, 25, 50, 50, 75, 100):
        cb(it, None, {"loss": 0.5})
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "iter" in ln]
    assert [ln.split("iter ")[1].split(",")[0] for ln in lines] == [
        "25/100",
        "50/100",
        "75/100",
    ]
    assert "box 0:" in lines[0] and "box ETA" in lines[0]


def test_each_distinct_warning_is_shown_once_per_run() -> None:
    run_warnings = OncePerRunWarnings()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        for _ in range(2):
            with run_warnings.once():
                warnings.warn("slow CPU fit", UserWarning, stacklevel=1)
                warnings.warn("other", RuntimeWarning, stacklevel=1)
    assert sorted(str(w.message) for w in shown) == ["other", "slow CPU fit"]


def test_a_box_that_raises_still_shows_its_warnings() -> None:
    """The warning that explains a failure must not vanish with it."""
    run_warnings = OncePerRunWarnings()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="box blew up"):
            with run_warnings.once():
                warnings.warn("slow CPU fit", UserWarning, stacklevel=1)
                raise ValueError("box blew up")
    assert [str(w.message) for w in shown] == ["slow CPU fit"]


def test_a_reissue_turned_error_never_masks_the_original_exception() -> None:
    run_warnings = OncePerRunWarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # recorded inside, then...
        with pytest.raises(ValueError, match="original"):
            with run_warnings.once():
                warnings.warn("w", UserWarning, stacklevel=1)
                warnings.simplefilter("error")  # ...re-issue would raise
                raise ValueError("original")


def _two_box_plan():
    from luxar.gsplats.planner.spec import FitPlan, PlanBox

    return FitPlan(
        volume_shape=[16, 16, 32],
        boxes=[
            PlanBox(box=[0, 16, 0, 16, j * 16, (j + 1) * 16], n_features=5, budget=20)
            for j in range(2)
        ],
        overlap=2,
        feature_method="peaks",
        min_leaf=8,
        max_leaf=16,
        density={"saturation_cap": 1_000},
    )


def _fit(progress_callback=None, verbose: bool = True) -> None:
    from luxar.gsplats.planner import fit_planned

    rng = np.random.default_rng(0)
    V = (rng.random((16, 16, 32)) ** 8).astype(np.float32)
    fit_planned(
        V,
        _two_box_plan(),
        device="cpu",
        n_iters=100,
        early_stop_patience=1_000,
        use_cuda=False,
        use_metal=False,
        verbose=verbose,
        progress_callback=progress_callback,
    )


def test_verbose_fit_planned_prints_one_quiet_progress_stream(capsys) -> None:
    """End to end on CPU: start, iteration and completion lines per box only."""
    _fit()
    out = capsys.readouterr().out
    assert "box 0 (1/2): fitting, budget" in out
    assert "box 0: iter 25/100" in out
    assert "box 1: iter 75/100" in out
    assert "box 0 done (1/2): kept" in out and "dB (padded crop)" in out
    assert "box 1 done (2/2):" in out
    # ...and none of the inner fits' narration.
    assert "Fitting Gaussian Splats" not in out
    assert "Generating seeds" not in out


def test_a_callers_progress_callback_replaces_the_stream(capsys) -> None:
    """A library caller with its own callback never sees a box twice."""
    calls: list[str] = []
    _fit(progress_callback=lambda i, n, msg: calls.append(msg))
    out = capsys.readouterr().out
    assert [m.split(" budget=")[0] for m in calls] == ["box 1/2", "box 2/2"]
    assert "done (" not in out and ": iter " not in out


def test_a_quiet_fit_planned_is_quieter_than_a_verbose_one(capsys) -> None:
    """`verbose=False` (the CLI's `--quiet`) quiets the box fits too."""
    _fit(verbose=False)
    out = capsys.readouterr().out
    assert "Fitting Gaussian Splats" not in out
    assert "Generating seeds" not in out
    assert "done (" not in out and ": iter " not in out
