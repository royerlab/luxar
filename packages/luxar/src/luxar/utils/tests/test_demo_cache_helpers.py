"""Tests for the shared demo cache/arg/color helpers in ``luxar.utils.demos``.

Covers ``cache_computed`` (hit/miss/corrupt-quarantine/version), ``cached_download``
(skip-if-present, no network in the cached path), ``parse_int_arg``, and the
vectorized ``hsv_to_rgb``. Network is never touched: ``cached_download`` is only
exercised on the already-cached branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.utils import demos as demo_utils


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the module cache root at a temp dir for every test."""
    monkeypatch.setattr(demo_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")


def test_cache_computed_miss_then_hit():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"arr": np.arange(5), "label": "x"}

    r1 = demo_utils.cache_computed("demoX", "k1", compute)
    r2 = demo_utils.cache_computed("demoX", "k1", compute)
    assert calls["n"] == 1  # second call served from cache
    assert r2["label"] == "x"
    np.testing.assert_array_equal(r1["arr"], r2["arr"])


def test_cache_computed_version_and_recompute():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    assert demo_utils.cache_computed("demoX", "k", compute, version=1) == 1
    # A new version key must not read the v1 cache.
    assert demo_utils.cache_computed("demoX", "k", compute, version=2) == 2
    # recompute=True bypasses the cache.
    assert demo_utils.cache_computed("demoX", "k", compute, version=2, recompute=True) == 3


def test_cache_computed_corrupt_is_quarantined(tmp_path):
    def compute():
        return 42

    demo_utils.cache_computed("demoY", "kk", compute)
    cache_dir = (tmp_path / "cache") / "demoY"
    pkl = next(cache_dir.glob("*.pkl"))
    pkl.write_bytes(b"not a pickle")  # corrupt it

    # Should quarantine and recompute rather than raise.
    assert demo_utils.cache_computed("demoY", "kk", compute) == 42
    assert list(cache_dir.glob("*.pkl.corrupt")), "corrupt cache was not quarantined"


def test_cached_download_returns_present_file_without_network(tmp_path, monkeypatch):
    # Pre-seed the cache file; a hard failure if the network is touched.
    cache_dir = (tmp_path / "cache") / "demoZ"
    cache_dir.mkdir(parents=True)
    (cache_dir / "data.bin").write_bytes(b"hello world")

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("network download attempted for a cached file")

    monkeypatch.setattr("luxar.utils.download.robust_download", _boom)
    monkeypatch.setattr("luxar.utils.download.download_with_checksum", _boom)

    p = demo_utils.cached_download("http://example.invalid/data.bin", "demoZ", "data.bin")
    assert p.read_bytes() == b"hello world"


def test_parse_int_arg_equals_space_and_default():
    assert demo_utils.parse_int_arg("points", 100, ["--points=4000"]) == 4000
    assert demo_utils.parse_int_arg("points", 100, ["--points", "2000"]) == 2000
    assert demo_utils.parse_int_arg("points", 100, ["--other=1"]) == 100
    assert demo_utils.parse_int_arg("points", 100, ["--points=bad"]) == 100


def test_hsv_to_rgb_primaries_and_shape():
    # Red at hue 0, green at 1/3, blue at 2/3.
    hues = np.array([0.0, 1 / 3, 2 / 3], dtype=np.float32)
    rgb = demo_utils.hsv_to_rgb(hues)
    assert rgb.shape == (3, 3)
    np.testing.assert_allclose(rgb[0], [1, 0, 0], atol=1e-5)
    np.testing.assert_allclose(rgb[1], [0, 1, 0], atol=1e-5)
    np.testing.assert_allclose(rgb[2], [0, 0, 1], atol=1e-5)
    # Value/saturation scaling.
    grey = demo_utils.hsv_to_rgb(np.array([0.0]), s=0.0, v=0.5)
    np.testing.assert_allclose(grey[0], [0.5, 0.5, 0.5], atol=1e-5)
