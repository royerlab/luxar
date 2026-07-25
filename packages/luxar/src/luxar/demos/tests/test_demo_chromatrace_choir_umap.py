"""Regression tests for the Chromatrace CHOIR UMAP demos' data loading.

Both ``demo_chromatrace_choir_umap`` and its ``_sequence`` sibling read the
UMAP coordinates out of a parquet file and immediately recenter them in place.
The parquet stores ``UMAP1..3`` as float32, so a same-dtype
``DataFrame.to_numpy(dtype=np.float32)`` hands back a READ-ONLY *view* onto the
DataFrame's block under pandas' copy-on-write (always on in pandas >= 3) — and
``coords -= center`` then died with ``ValueError: output array is read-only``
on every single run.

These tests exercise the real ``load_chromatrace_data`` code path against a
real (tiny) parquet bundle written to ``tmp_path`` — no mocking of the function
under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")  # parquet engine

from luxar.demos.demo_chromatrace_choir_umap import (  # noqa: E402
    COLORMAP_NAME,
    PARQUET_NAME,
)
from luxar.demos.demo_chromatrace_choir_umap import (  # noqa: E402
    load_chromatrace_data as load_static,
)
from luxar.demos.demo_chromatrace_choir_umap_sequence import (  # noqa: E402
    load_chromatrace_data as load_sequence,
)

UMAP_COLUMNS = ["UMAP1", "UMAP2", "UMAP3"]

# Deliberately off-center so recentering is observable: the midpoint of each
# axis is 10, 20 and -5 respectively.
_X = np.array([9.0, 11.0, 10.0, 9.5], dtype=np.float32)
_Y = np.array([18.0, 22.0, 20.0, 21.0], dtype=np.float32)
_Z = np.array([-6.0, -4.0, -5.0, -5.5], dtype=np.float32)
_EXPECTED_CENTER = np.array([10.0, 20.0, -5.0], dtype=np.float32)


def _make_frame() -> Any:
    """A DataFrame shaped like the real bundle (float32 UMAP columns)."""
    return pd.DataFrame(
        {
            "UMAP1": _X,
            "UMAP2": _Y,
            "UMAP3": _Z,
            "choir_bio_term": pd.Categorical(
                ["Neuron", "Neuron", "Fibroblast", "Neuron"]
            ),
            "choir_bio_group": ["Neuroectoderm", "Neuroectoderm", "Mesoderm", "Other"],
        }
    )


def _write_bundle(root: Path) -> Path:
    """Write a minimal ``<root>/{parquet,colormap}`` bundle and return ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    _make_frame().to_parquet(root / PARQUET_NAME, index=False)
    (root / COLORMAP_NAME).write_text(
        json.dumps(
            {
                "colors": {"Neuron": "#ff0080", "Fibroblast": "#00ff00"},
                "groups": [
                    {"name": "Neuroectoderm", "cell_types": ["Neuron"]},
                    {"name": "Mesoderm", "cell_types": ["Fibroblast"]},
                ],
            }
        )
    )
    return root


class TestReadOnlyPrecondition:
    """Document the pandas behaviour that caused the crash."""

    def test_same_dtype_to_numpy_is_read_only(self, tmp_path: Path) -> None:
        _make_frame().to_parquet(tmp_path / PARQUET_NAME, index=False)
        df = pd.read_parquet(tmp_path / PARQUET_NAME)
        assert all(df[c].dtype == np.float32 for c in UMAP_COLUMNS)

        view = df[UMAP_COLUMNS].to_numpy(dtype=np.float32)
        if view.flags.writeable:
            pytest.skip(
                "this pandas returns a writeable copy from to_numpy(); the "
                "read-only condition guarded by copy=True cannot be reproduced"
            )
        # This is exactly what the demos used to do — and why they crashed.
        with pytest.raises(ValueError, match="read-only"):
            view -= view.mean(axis=0)

        # copy=True (the fix) removes the aliasing, so in-place ops are safe.
        owned = df[UMAP_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        assert owned.flags.writeable
        owned -= owned.mean(axis=0)


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param(load_static, id="demo_chromatrace_choir_umap"),
        pytest.param(load_sequence, id="demo_chromatrace_choir_umap_sequence"),
    ],
)
class TestLoadChromatraceData:
    """``load_chromatrace_data`` must survive a read-only ``to_numpy`` result."""

    def test_recenters_without_read_only_error(
        self, loader: Callable[..., Any], tmp_path: Path
    ) -> None:
        root = _write_bundle(tmp_path / "bundle")

        # Pre-fix this raises "ValueError: output array is read-only".
        coords, attributes, category_maps, term_colors, groups = loader(root)

        assert coords.dtype == np.float32
        assert coords.shape == (4, 3)
        # The returned array must be independently writeable so later stages
        # (and any caller) can keep transforming it in place.
        assert coords.flags.writeable

        raw = np.stack([_X, _Y, _Z], axis=1)
        np.testing.assert_allclose(coords, raw - _EXPECTED_CENTER, atol=1e-6)
        # Recentered: the bounding-box midpoint sits on the origin.
        np.testing.assert_allclose(
            0.5 * (coords.min(axis=0) + coords.max(axis=0)),
            np.zeros(3, dtype=np.float32),
            atol=1e-6,
        )

        assert set(attributes) == {"bio_term", "bio_group"}
        assert attributes["bio_term"].dtype == np.int32
        assert category_maps["bio_term"] == ["Fibroblast", "Neuron"]
        assert term_colors["Neuron"] == "#ff0080"
        assert [g["name"] for g in groups] == ["Neuroectoderm", "Mesoderm"]

    def test_returned_coords_are_not_aliased_to_the_frame(
        self, loader: Callable[..., Any], tmp_path: Path
    ) -> None:
        """Mutating the result must not need a defensive copy at the callsite."""
        root = _write_bundle(tmp_path / "bundle")
        coords, *_ = loader(root)

        coords *= 2.0  # would raise if `coords` were still a read-only view
        assert float(coords[0, 0]) == pytest.approx(-2.0)
