"""GSplats amplitudes in SOURCE DATA UNITS plus a separate ``display_gain`` (WP16).

Two contracts:

* **The opt-in attrs.** ``display_gain`` / ``amplitude_unit`` /
  ``amplitude_offset`` reach the store only when passed, ``display_gain`` composes
  (it lands on a kind=lod / kind=partition wrapper, like ``intensity``), bad values
  are refused before anything is written, and the three are refused on the other
  geometry types, which have no amplitude.
* **Precision.** Every amplitude encoding is anchored on the array's own range
  (geometric-log / bounded-linear quantisation, float32 passthrough), so storing
  ``k × a`` instead of ``a`` loses nothing: the encoder picked and the RELATIVE
  error are identical, and the decoded values scale by exactly ``k``. Detector
  counts up to 65535 and float data of any magnitude round-trip without overflow;
  no amplitude path emits float16.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import luxar
from luxar import LuxarZarrCompiler
from luxar._zarr_compat import open_group, read_node_attrs
from luxar.core.dimension_inference import build_dimensions_from_data
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.tree import iter_leaves
from luxar.io._compiler.gsplat_tree import read_gsplat_node

_RNG = np.random.default_rng(16)
_N = 4000


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    with luxar.verbosity("silent"):
        yield


def _data(amplitudes: np.ndarray) -> GSplatData:
    n = amplitudes.shape[0]
    rng = np.random.default_rng(3)
    return GSplatData(
        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
        amplitudes=amplitudes.astype(np.float32),
        cholesky_factors=np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (n, 1)),
    )


def _write(
    out: Path, data: GSplatData, *, mode: str = "auto", bits: Any = 16, **attrs: Any
) -> Path:
    with LuxarZarrCompiler(
        out, encoding_mode=mode, gsplat_amplitude_bits=bits
    ) as compiler:
        scene = compiler.create_scene(
            dimensions=build_dimensions_from_data(data.centers, None)
        )
        scene.add_gsplats_from_data(name="g", result=data, **attrs)
    return out


def _decoded(out: Path) -> np.ndarray:
    root = open_group(str(out), mode="r")
    node = read_gsplat_node(root["g"], root)
    return np.concatenate(
        [
            np.asarray(s.amplitudes)
            for leaf in iter_leaves(node)
            for s in leaf.additive_sublods
        ]
    )


def _encoder(out: Path) -> str:
    root = open_group(str(out), mode="r")
    return str(root["g"]["amplitudes"].attrs["encoding"]["name"])


# --------------------------------------------------------------------------- #
# The opt-in attrs
# --------------------------------------------------------------------------- #


def test_default_insertion_writes_none_of_the_data_unit_attrs(tmp_path: Path) -> None:
    out = _write(tmp_path / "s.luxar.zarr", _data(_RNG.uniform(1, 4000, _N)))
    attrs = read_node_attrs(out / "g")
    assert not {"display_gain", "amplitude_unit", "amplitude_offset"} & set(attrs)


def test_opt_in_keeps_data_units_and_stamps_the_attrs(tmp_path: Path) -> None:
    amps = _RNG.uniform(1, 4000, _N)
    out = _write(
        tmp_path / "s.luxar.zarr",
        _data(amps),
        mode="precision",
        normalize_amplitudes=False,
        display_gain=2.5e-4,
        amplitude_unit="counts above background",
        amplitude_offset=112.0,
    )
    attrs = read_node_attrs(out / "g")
    assert attrs["display_gain"] == 2.5e-4
    assert attrs["amplitude_unit"] == "counts above background"
    assert attrs["amplitude_offset"] == 112.0
    np.testing.assert_array_equal(
        np.sort(_decoded(out)), np.sort(amps.astype(np.float32))
    )
    assert attrs["amplitude_data_range"][1] == pytest.approx(
        float(np.percentile(amps.astype(np.float32), 99.9)), rel=1e-4
    )


def test_display_gain_composes_onto_a_lod_wrapper(tmp_path: Path) -> None:
    """Like ``intensity``: one gain on the wrapper, never copied per level."""
    out = _write(
        tmp_path / "s.luxar.zarr",
        _data(_RNG.uniform(1, 4000, _N)),
        normalize_amplitudes=False,
        substitutive_lod={"compression_factor": 4, "levels": 2},
        display_gain=0.01,
        amplitude_unit="counts",
    )
    wrapper = read_node_attrs(out / "g")
    assert wrapper["kind"] == "lod"
    assert wrapper["display_gain"] == 0.01
    for child in ("child_0", "child_1"):
        leaf = read_node_attrs(out / "g" / child)
        assert "display_gain" not in leaf
        assert leaf["amplitude_unit"] == "counts"


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"display_gain": 0.0}, "positive"),
        ({"display_gain": -1.0}, "positive"),
        ({"display_gain": float("nan")}, "finite"),
        ({"display_gain": True}, "bool"),
        ({"amplitude_unit": ""}, "non-empty"),
        ({"amplitude_unit": 3}, "non-empty"),
        ({"amplitude_offset": float("inf")}, "finite"),
    ],
)
def test_bad_values_are_refused_before_writing(
    tmp_path: Path, attrs: Dict[str, Any], match: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        _write(
            tmp_path / "s.luxar.zarr",
            _data(_RNG.uniform(1, 10, 50)),
            normalize_amplitudes=False,
            **attrs,
        )


@pytest.mark.parametrize("geometry", ["points", "lines"])
def test_refused_on_geometry_without_an_amplitude(
    tmp_path: Path, geometry: str
) -> None:
    positions = np.random.default_rng(1).uniform(0, 1, (10, 3)).astype(np.float32)
    with LuxarZarrCompiler(tmp_path / "s.luxar.zarr") as compiler:
        scene = compiler.create_scene(
            dimensions=build_dimensions_from_data(positions, None)
        )
        with pytest.raises(ValueError, match="gsplats-only"):
            if geometry == "points":
                scene.add_points(name="p", positions=positions, display_gain=2.0)
            else:
                scene.add_lines(
                    name="l", vertices=positions, widths=0.1, amplitude_unit="counts"
                )


# --------------------------------------------------------------------------- #
# Precision: data units lose nothing
# --------------------------------------------------------------------------- #

_DISTRIBUTIONS = {
    # 16-bit detector counts, the largest integer source a fit records.
    "uint16_counts": lambda: _RNG.integers(1, 65536, _N).astype(np.float64),
    # Float data spanning twelve decades.
    "float_wide": lambda: np.exp(_RNG.uniform(np.log(1e-6), np.log(1e6), _N)),
    # A float volume whose values are all tiny (a float source in [0, 1e-5]).
    "float_tiny": lambda: _RNG.uniform(1e-7, 1e-5, _N),
}


@pytest.mark.parametrize("dist", sorted(_DISTRIBUTIONS))
@pytest.mark.parametrize(("mode", "bits"), [("auto", 8), ("auto", 16), ("memory", 16)])
def test_data_unit_encoding_matches_the_normalized_one(
    tmp_path: Path, dist: str, mode: str, bits: int
) -> None:
    """``k × a`` encodes with the same encoder and decodes to ``k × decode(a)``."""
    data_units = _DISTRIBUTIONS[dist]().astype(np.float32)
    k = float(np.percentile(data_units, 99.9))
    normalized = (data_units / k).astype(np.float32)
    raw = _write(
        tmp_path / "raw.luxar.zarr",
        _data(data_units),
        mode=mode,
        bits=bits,
        normalize_amplitudes=False,
    )
    norm = _write(
        tmp_path / "norm.luxar.zarr",
        _data(normalized),
        mode=mode,
        bits=bits,
        normalize_amplitudes=False,
    )
    assert _encoder(raw) == _encoder(norm)
    assert "float16" not in _encoder(raw)
    got_raw = np.sort(_decoded(raw).astype(np.float64))
    got_norm = np.sort(_decoded(norm).astype(np.float64))
    assert np.isfinite(got_raw).all()
    # The relative error against the source values is the encoder's own,
    # independent of the scale (the quantisation grid is anchored on the data).
    ref = np.sort(data_units.astype(np.float64))
    ref_norm = np.sort(normalized.astype(np.float64))
    rel_raw = float(np.max(np.abs(got_raw - ref) / ref))
    rel_norm = float(np.max(np.abs(got_norm - ref_norm) / ref_norm))
    assert rel_raw == pytest.approx(rel_norm, rel=0.05, abs=1e-6)
    # And the decoded values agree up to one quantisation step: the float32
    # division ``a / k`` can move a value across a code boundary, nothing more.
    np.testing.assert_allclose(got_raw, got_norm * k, rtol=2.5 * rel_norm + 1e-5)


@pytest.mark.parametrize("dist", sorted(_DISTRIBUTIONS))
def test_precision_mode_is_exact_in_data_units(tmp_path: Path, dist: str) -> None:
    amps = _DISTRIBUTIONS[dist]().astype(np.float32)
    out = _write(
        tmp_path / "s.luxar.zarr",
        _data(amps),
        mode="precision",
        normalize_amplitudes=False,
    )
    assert _encoder(out) == "float32"
    np.testing.assert_array_equal(np.sort(_decoded(out)), np.sort(amps))


def test_uint16_range_survives_the_gsplats_store(tmp_path: Path) -> None:
    """A fit-sized store (``amplitude_bits`` 16 and 8) holds 0-65535 counts."""
    amps = np.concatenate([[1.0, 65535.0], _RNG.integers(1, 65536, _N - 2)])
    for bits in (8, 16):
        path = tmp_path / f"fit{bits}.gsplats.zarr"
        _data(amps.astype(np.float32)).save(path, amplitude_bits=bits)
        got = GSplatData.load(path).amplitudes.astype(np.float64)
        assert np.isfinite(got).all()
        assert got.max() == pytest.approx(65535.0, rel=0.03)
        assert got.min() == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize("normalize", [True, "auto", 1.0])
def test_data_unit_attrs_refused_while_normalising(
    tmp_path: Path, normalize: Any
) -> None:
    """The attrs describe data-unit amplitudes; a normalising insertion would
    rescale them and every label (and the gain) would be wrong."""
    with pytest.raises(ValueError, match="normalize_amplitudes=False"):
        _write(
            tmp_path / "s.luxar.zarr",
            _data(_RNG.uniform(1, 4000, 50)),
            normalize_amplitudes=normalize,
            display_gain=1e-3,
        )


def test_data_unit_attrs_allowed_with_the_identity_scale(tmp_path: Path) -> None:
    from luxar.core.group.gsplats_pipeline.amplitude_norm import AmplitudeScale

    out = _write(
        tmp_path / "s.luxar.zarr",
        _data(_RNG.uniform(1, 4000, 50)),
        normalize_amplitudes=AmplitudeScale(1.0),
        amplitude_unit="counts",
    )
    assert read_node_attrs(out / "g")["amplitude_unit"] == "counts"
