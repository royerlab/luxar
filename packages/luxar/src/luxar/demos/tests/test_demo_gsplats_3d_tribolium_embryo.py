"""Smoke test for the Tribolium embryo demo's authored appearance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import GSplatData

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_tribolium_embryo.py"


def _load_demo_module():
    name = "_luxar_demo_tribolium_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DEMO_MODULE = _load_demo_module()
create_luxar_scene = _DEMO_MODULE.create_luxar_scene


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats, sufficient to build the scene offline."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


def test_scene_bakes_volumetric_appearance(tmp_path) -> None:
    out = create_luxar_scene(_tiny_gsplat_data(), tmp_path / "tribolium.luxar.zarr")
    node = zarr.open_group(str(out), mode="r")["tribolium_embryo"]
    attrs = dict(node.attrs)

    assert attrs["blending_mode"] == "volumetric"
    # Hand-tuned in the hosted viewer's Layers panel on 2026-09-10 (colour
    # range 0 - 0.533, opacity 1.0, absorption 2.53) with exposure back at the
    # 0-stop identity.
    assert attrs["absorption"] == pytest.approx(2.53)
    assert attrs["opacity"] == pytest.approx(1.0)
    assert attrs["intensity"] == pytest.approx(1.0 / 0.533)


def test_fit_normalises_counts_and_floor_without_mutating_input(
    monkeypatch, tmp_path
) -> None:
    volume = np.array([[[675.0, 1675.0, 3675.0]]], dtype=np.float32)
    original = volume.copy()
    captured: dict[str, object] = {}

    def fake_fit_gaussian_splats(image, **kwargs):
        captured["image"] = image
        captured["fit_kwargs"] = kwargs
        return _tiny_gsplat_data()

    monkeypatch.setattr("luxar.gsplats.fit_gaussian_splats", fake_fit_gaussian_splats)
    monkeypatch.setattr(_DEMO_MODULE, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_DEMO_MODULE, "DEVICE", "cpu")

    stored = _DEMO_MODULE.fit_tribolium(volume)

    fitted_image = captured["image"]
    fit_kwargs = captured["fit_kwargs"]
    assert isinstance(fitted_image, np.ndarray)
    assert isinstance(fit_kwargs, dict)
    assert fitted_image.dtype == np.float32
    assert fitted_image.max() == pytest.approx(1.0)
    assert fit_kwargs["floor"] == pytest.approx(
        _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS / float(volume.max())
    )
    assert (tmp_path / "tribolium.gsplats.zarr.zip").exists()
    assert stored.n_splats == 8
    assert stored.n_additive_sublods == 4
    np.testing.assert_array_equal(volume, original)


def test_roundtrip_comparison_uses_the_fit_output_scale() -> None:
    volume = np.array([[[675.0, 1675.0, 3675.0]]], dtype=np.float32)
    vmax = float(volume.max())
    recon = (volume - _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS) / vmax

    reference, scaled_recon = _DEMO_MODULE._prepare_roundtrip_comparison(
        volume, recon.copy()
    )

    expected = np.clip(
        (volume - _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS)
        / (vmax - _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS),
        0.0,
        1.0,
    )
    np.testing.assert_allclose(reference, expected)
    np.testing.assert_allclose(scaled_recon, expected)


def test_specimen_floor_is_a_count_level_inside_the_data_range() -> None:
    """The floor is stated in camera counts, so it must read as one.

    Pinned because the value is only checkable as a count: the medium in this
    stack sits at ~204 and the specimen background at ~675, and expressing the
    same level against the normalised volume (0.0424) hides which of the two it
    is. A floor that drifted below the medium peak, or above the data range,
    would silently stop suppressing the haze.
    """
    assert _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS == pytest.approx(675.0)
    # Comfortably above the ~204-count detector offset it must NOT be confused
    # with, and far below the stack's ~15900-count maximum.
    assert 400.0 < _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS < 5000.0


@pytest.mark.parametrize("maximum", [0.0, 400.0, 675.0])
def test_volume_not_exceeding_specimen_floor_is_rejected_before_fitting(
    maximum,
) -> None:
    """An empty or below-floor volume must fail loudly, not fit nonsense.

    Normalising by the volume's own maximum reintroduced a division the previous
    min-max form guarded with `+ 1e-8`. A positive maximum at or below the floor
    is equally invalid: it makes the surviving range non-positive and the fitter
    ignores the floor. Both cases indicate an empty or corrupt source, and the
    guard runs before any fitting, so this needs no GPU.
    """
    volume = np.full((4, 4, 4), maximum, dtype=np.float32)
    with pytest.raises(ValueError, match="does not exceed the specimen"):
        _DEMO_MODULE.fit_tribolium(volume)


@pytest.mark.parametrize(
    ("floor", "should_warn", "why"),
    [
        (None, True, "a pre-floor fit, or a reduction pass that rewrote pipeline/"),
        (
            204.8 / _DEMO_MODULE.EXPECTED_VOLUME_MAX_COUNTS,
            True,
            "the ~205-count detector offset `auto` resolves to",
        ),
        (
            675.0 / _DEMO_MODULE.EXPECTED_VOLUME_MAX_COUNTS,
            False,
            "the specimen floor this demo now asks for",
        ),
        (675.0 / 1828.0, False, "--downsample lowers vmax, so the ratio RISES"),
    ],
)
def test_stale_cache_warning_fires_only_on_a_pre_floor_fit(
    monkeypatch, tmp_path, capsys, floor, should_warn, why
) -> None:
    """The staleness check must catch an old fit and never flag a good one.

    Otherwise the background fix silently applies only to whoever happens to have
    a cold cache. The `--downsample` row is the one-sidedness guard: downsampling
    lowers the volume maximum, which RAISES the normalised floor, so it must not
    be mistaken for a stale fit.
    """
    (tmp_path / "tribolium.gsplats.zarr.zip").write_bytes(b"placeholder")
    monkeypatch.setattr(_DEMO_MODULE, "CACHE_DIR", tmp_path)

    class FakeData:
        stats = {} if floor is None else {"floor": floor}

    monkeypatch.setattr(
        _DEMO_MODULE.GSplatData, "load", classmethod(lambda cls, *a, **k: FakeData())
    )
    _DEMO_MODULE.warn_if_cached_tribolium_fit_predates_floor()
    warned = "predates the specimen background floor" in capsys.readouterr().out
    assert warned is should_warn, f"floor={floor!r} ({why})"


def test_stale_cache_check_is_silent_without_a_cache(
    monkeypatch, tmp_path, capsys
) -> None:
    """No cache is the fresh-clone case, not a stale one."""
    monkeypatch.setattr(_DEMO_MODULE, "CACHE_DIR", tmp_path)
    _DEMO_MODULE.warn_if_cached_tribolium_fit_predates_floor()
    assert capsys.readouterr().out == ""


def test_stale_cache_check_never_breaks_the_demo(monkeypatch, tmp_path, capsys) -> None:
    """A diagnostic that cannot read the store must not take the demo down."""
    (tmp_path / "tribolium.gsplats.zarr.zip").write_bytes(b"not a zarr archive")
    monkeypatch.setattr(_DEMO_MODULE, "CACHE_DIR", tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("unreadable store")

    monkeypatch.setattr(_DEMO_MODULE.GSplatData, "load", classmethod(boom))
    _DEMO_MODULE.warn_if_cached_tribolium_fit_predates_floor()  # must not raise
    assert "could not read the cached fit's floor" in capsys.readouterr().out
