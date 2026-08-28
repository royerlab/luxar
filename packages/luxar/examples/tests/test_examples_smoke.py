"""Smoke tests for ``packages/luxar/examples/*.py``.

For each example we redirect ``luxar.utils.paths.get_examples_output_dir``
to a per-test ``tmp_path`` (so the shared ``datasets/examples/``
location isn't clobbered), import the example module by file path, run
its ``main()``, and assert that an output zarr exists.

Heavy examples are explicitly excluded:

- ``dense_cubic_gradient_example`` — 1.5M points; same rationale.
- ``rainbow_sphere_spiral_example`` — 200K points; slow on CI.
- ``performance_benchmark_example`` — runs 100 nodes × 1K points;
  intentionally a benchmark, not a smoke target.
- ``metal_acceleration_example`` — needs MPS or CUDA torch backend
  with non-trivial setup; out of scope for a smoke test.
- ``gsplats_fit_volume_example`` / ``gsplats_lod_example`` — run a
  small fitter on CPU. Included but marked slow.

The full smoke suite runs in roughly 30-90 seconds depending on the
machine. Each test is independent, so failures isolate to one example.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

import luxar.utils.paths as luxar_paths

EXAMPLES_DIR = Path(__file__).resolve().parent.parent

HEAVY_EXAMPLES = frozenset(
    {
        "dense_cubic_gradient_example",
        "rainbow_sphere_spiral_example",
        "performance_benchmark_example",
        "metal_acceleration_example",
    }
)

# Run on CPU, take ~3-10s each; mark slow so they can be opt-out under
# `pytest -m 'not slow'`.
SLOW_EXAMPLES = frozenset(
    {
        "gsplats_fit_volume_example",
        "gsplats_lod_example",
    }
)


def _discover_example_stems() -> list[str]:
    """All ``*_example.py`` stems in ``packages/luxar/examples/`` (sorted)."""
    return sorted(
        path.stem for path in EXAMPLES_DIR.glob("*_example.py") if path.is_file()
    )


def _load_example(stem: str):
    """Import an example by file path, without registering it on ``sys.path``.

    Importing by file path avoids name clashes — the examples directory
    is not a package — and lets each test load its own fresh module
    object even if other tests have already loaded a sibling module.
    """
    path = EXAMPLES_DIR / f"{stem}.py"
    if not path.exists():
        pytest.skip(f"Example missing on disk: {path}")
    # Examples import the shared explainer-overlay helper with
    # ``from _overlay_style import add_explainer``. When run as scripts the
    # examples dir is sys.path[0] automatically; under pytest we import by
    # file path, so make the helper importable here too.
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(f"_smoke_{stem}", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parametrize_stems(stems: Iterable[str]) -> list[pytest.param]:
    """Build pytest parameters, applying ``slow`` marks where appropriate."""
    params: list[pytest.param] = []
    for stem in stems:
        marks = ()
        if stem in SLOW_EXAMPLES:
            marks = (pytest.mark.slow,)
        params.append(pytest.param(stem, marks=marks))
    return params


def test_temporal_spiral_sphere_stays_within_preflight_budget():
    """The 4D navigation example must remain practical for ``run-examples``."""
    module = _load_example("temporal_spiral_sphere_4d_example")

    point_records = module.N_POINTS_PER_FRAME * module.N_FRAMES

    assert module.N_POINTS_PER_FRAME >= 1_000
    assert module.N_FRAMES >= 32
    assert point_records <= 1_000_000


@pytest.mark.parametrize("n_clusters", [8, 19, 20, 40])
def test_spatial_index_demo_discrete_coordinates_stay_navigable(n_clusters):
    """Discrete values stay navigable and large radii span every channel."""
    module = _load_example("spatial_index_demo_example")

    positions, colors, radii = module.create_5d_clusters(n_clusters, 100)
    repeated = module.create_5d_clusters(n_clusters, 100)

    for actual, expected in zip((positions, colors, radii), repeated, strict=True):
        np.testing.assert_array_equal(actual, expected)

    time_offset = np.abs(
        positions[:, 3]
        - (
            module.TIME_RANGE[0]
            + np.round((positions[:, 3] - module.TIME_RANGE[0]) / module.TIME_STEP)
            * module.TIME_STEP
        )
    )
    channel_offset = np.abs(
        positions[:, 4]
        - (
            module.CHANNEL_RANGE[0]
            + np.round(
                (positions[:, 4] - module.CHANNEL_RANGE[0]) / module.CHANNEL_STEP
            )
            * module.CHANNEL_STEP
        )
    )
    initial_slice = (
        np.abs(positions[:, 3] - module.TIME_RANGE[0]) <= module.TIME_STEP / 4
    ) & (np.abs(positions[:, 4] - module.CHANNEL_RANGE[0]) <= module.CHANNEL_STEP / 4)

    assert np.all(time_offset <= module.TIME_STEP / 4)
    assert np.all(channel_offset <= module.CHANNEL_STEP / 4)
    assert positions[:, 3].min() >= module.TIME_RANGE[0]
    assert positions[:, 3].max() <= module.TIME_RANGE[1]
    assert positions[:, 4].min() >= module.CHANNEL_RANGE[0]
    assert positions[:, 4].max() <= module.CHANNEL_RANGE[1]
    assert np.count_nonzero(initial_slice) > 0

    large_radius_channels = (
        module.CHANNEL_RANGE[0]
        + np.round(
            (positions[radii >= 3.0, 4] - module.CHANNEL_RANGE[0]) / module.CHANNEL_STEP
        )
        * module.CHANNEL_STEP
    )
    expected_channels = np.arange(
        module.CHANNEL_RANGE[0],
        module.CHANNEL_RANGE[1] + module.CHANNEL_STEP,
        module.CHANNEL_STEP,
    )
    np.testing.assert_array_equal(np.unique(large_radius_channels), expected_channels)


def test_spatial_index_demo_authored_dimensions_use_shared_constants():
    """Serialized authored metadata stays coupled to shared constants."""
    module = _load_example("spatial_index_demo_example")

    dimensions = {
        dimension["name"]: dimension
        for dimension in module.create_dimensions().to_dict()["dimensions"]
    }

    assert dimensions["time"]["range"] == list(module.TIME_RANGE)
    assert dimensions["time"]["step"] == module.TIME_STEP
    assert dimensions["time"]["discrete"] is True
    assert dimensions["channel"]["range"] == list(module.CHANNEL_RANGE)
    assert dimensions["channel"]["step"] == module.CHANNEL_STEP
    assert dimensions["channel"]["discrete"] is True


_ALL_STEMS = [s for s in _discover_example_stems() if s not in HEAVY_EXAMPLES]


@pytest.fixture
def redirected_examples_dir(tmp_path, monkeypatch):
    """Redirect ``get_examples_output_dir`` for the duration of one test.

    Each example writes ``<output_dir>/<name>_example.luxar.zarr``; routing to
    ``tmp_path`` keeps tests hermetic and parallel-safe.
    """
    monkeypatch.setattr(luxar_paths, "get_examples_output_dir", lambda: tmp_path)
    # Also patch any modules that have already imported the symbol by
    # name. Most examples use ``from luxar.utils.paths import
    # get_examples_output_dir``, which captures the symbol at import
    # time, so a plain ``monkeypatch.setattr`` on the module is not
    # enough — we need to clear cached examples too.
    cached_modules = [name for name in list(sys.modules) if name.startswith("_smoke_")]
    for name in cached_modules:
        del sys.modules[name]
    return tmp_path


@pytest.mark.parametrize("stem", _parametrize_stems(_ALL_STEMS))
def test_example_runs_and_writes_zarr(stem, redirected_examples_dir, monkeypatch):
    """Each example's ``main()`` runs without exception and produces a zarr.

    A handful of historical examples don't follow the ``<stem>.luxar.zarr``
    naming convention (``build_example`` emits ``build_example_manual.luxar.zarr``
    + ``build_example_structured.luxar.zarr``; ``memory_optimization_example``
    emits three encoding-mode variants). To keep this test useful as a
    smoke check across the full example surface, we assert only that at
    least one ``*.luxar.zarr`` was created — the stricter naming contract is
    a separate concern documented in ``TEMPLATE.md``.
    """
    if "CI" in os.environ and stem in SLOW_EXAMPLES:
        pytest.skip("Skipping slow example under CI; run locally with -m slow")

    # Isolate ``sys.argv`` so examples that use ``argparse``
    # (``spatial_index_demo_example``) don't see pytest's own argv.
    monkeypatch.setattr(sys, "argv", [f"{stem}.py"])

    module = _load_example(stem)
    if not hasattr(module, "main"):
        pytest.fail(f"{stem}.py has no top-level main() function")

    module.main()

    zarrs = sorted(redirected_examples_dir.glob("*.luxar.zarr"))
    assert zarrs, (
        f"Expected at least one *.luxar.zarr in {redirected_examples_dir} after "
        f"running {stem}.main(); directory contents: "
        f"{list(redirected_examples_dir.iterdir())}"
    )
