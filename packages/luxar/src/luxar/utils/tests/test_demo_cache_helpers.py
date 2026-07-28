"""Tests for the shared demo cache/arg/color helpers in ``luxar.utils.demos``.

Covers ``cache_computed`` (hit/miss/corrupt-quarantine/version), ``cached_download``
(skip-if-present, no network in the cached path), ``parse_int_arg``, and the
vectorized ``hsv_to_rgb``. Network is never touched: ``cached_download`` is only
exercised on the already-cached branch.
"""

from __future__ import annotations

from pathlib import Path

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
    assert (
        demo_utils.cache_computed("demoX", "k", compute, version=2, recompute=True) == 3
    )


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

    p = demo_utils.cached_download(
        "http://example.invalid/data.bin", "demoZ", "data.bin"
    )
    assert p.read_bytes() == b"hello world"


def test_parse_int_arg_equals_space_and_default():
    assert demo_utils.parse_int_arg("points", 100, ["--points=4000"]) == 4000
    assert demo_utils.parse_int_arg("points", 100, ["--points", "2000"]) == 2000
    assert demo_utils.parse_int_arg("points", 100, ["--other=1"]) == 100
    assert demo_utils.parse_int_arg("points", 100, ["--points=bad"]) == 100


def test_stack_colorings_shapes_and_alignment():
    n = 5
    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    red = np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    blue = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    out = demo_utils.stack_colorings(
        coords,
        [
            {"label": "A", "colors": red, "labels": [f"a{i}" for i in range(n)]},
            {"label": "B", "colors": blue, "labels": [f"b{i}" for i in range(n)]},
        ],
    )
    assert out.positions.shape == (2 * n, 4)  # leading coloring-index column
    assert out.colors.shape == (2 * n, 3)
    assert out.categories == ["A", "B"]
    # Block 0 = coloring index 0 (red), block 1 = index 1 (blue).
    assert (out.positions[:n, 0] == 0).all() and (out.positions[n:, 0] == 1).all()
    np.testing.assert_array_equal(out.positions[:n, 1:], coords)
    np.testing.assert_array_equal(out.colors[:n], red)
    np.testing.assert_array_equal(out.colors[n:], blue)
    assert out.labels == [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]


def test_stack_colorings_labels_none_if_any_view_missing():
    n = 3
    coords = np.zeros((n, 3), dtype=np.float32)
    c = np.zeros((n, 3), dtype=np.float32)
    out = demo_utils.stack_colorings(
        coords,
        [
            {"label": "A", "colors": c, "labels": ["x", "y", "z"]},
            {"label": "B", "colors": c},  # no labels → combined labels is None
        ],
    )
    assert out.labels is None


def test_stack_colorings_rejects_bad_shape():
    coords = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        demo_utils.stack_colorings(
            coords, [{"label": "A", "colors": np.zeros((3, 3), dtype=np.float32)}]
        )


def test_stack_colorings_builds_a_real_scene(tmp_path):
    """End-to-end: a stacked coloring cloud writes a valid scene with a
    categorical `coloring` dimension (the pattern all 4 landscape demos use)."""
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    n = 40
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((n, 3)).astype(np.float32)
    out = demo_utils.stack_colorings(
        coords,
        [
            {
                "label": "View A",
                "colors": rng.random((n, 3)).astype(np.float32),
                "labels": [f"a{i}" for i in range(n)],
            },
            {
                "label": "View B",
                "colors": rng.random((n, 3)).astype(np.float32),
                "labels": [f"b{i}" for i in range(n)],
            },
        ],
    )
    scene_path = tmp_path / "coloring.luxar.zarr"
    dims = Dimensions(
        [
            Dimension("coloring", unit="", categories=out.categories, display=False),
            Dimension("x", unit="u", display=True),
            Dimension("y", unit="u", display=True),
            Dimension("z", unit="u", display=True),
        ]
    )
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points(
            "points",
            out.positions,
            colors=out.colors,
            radii=np.full(len(out.positions), 0.02, np.float32),
            labels=out.labels,
        )
    assert scene_path.exists()


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


def test_cached_download_quarantines_a_checksum_failing_cache(tmp_path, monkeypatch):
    """A complete-but-wrong cached file must move aside BEFORE re-downloading.

    robust_download resumes onto whatever bytes are at the destination, so
    leaving them appends the new download to the old garbage (or trips a 416).
    """
    import hashlib

    from luxar.utils.download import find_quarantined_files

    monkeypatch.setattr(demo_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
    cache_dir = (tmp_path / "cache") / "demoQ"
    cache_dir.mkdir(parents=True)
    dest = cache_dir / "data.bin"
    dest.write_bytes(b"wrong bytes")
    good = b"right bytes"

    def _fake(url, output_path, expected_sha256=None, **kw):
        assert not Path(output_path).exists(), "would have resumed onto stale bytes"
        Path(output_path).write_bytes(good)
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.download_with_checksum", _fake)

    out = demo_utils.cached_download(
        "http://example.invalid/data.bin",
        "demoQ",
        "data.bin",
        sha256=hashlib.sha256(good).hexdigest(),
    )

    assert out.read_bytes() == good
    assert find_quarantined_files(dest)


def test_cached_download_keeps_a_short_file_so_it_can_resume(tmp_path, monkeypatch):
    """Shorter than expected IS a resumable partial download — do not touch it."""
    monkeypatch.setattr(demo_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
    cache_dir = (tmp_path / "cache") / "demoQ3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "data.bin").write_bytes(b"half")
    seen: dict = {}

    def _fake(url, output_path, **kw):
        seen["existed"] = Path(output_path).exists()
        Path(output_path).write_bytes(b"halfhalf")
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.robust_download", _fake)

    demo_utils.cached_download(
        "http://example.invalid/data.bin", "demoQ3", "data.bin", expected_size=8
    )

    assert seen["existed"] is True, "partial file must survive for the resume"


def test_cached_download_keeps_an_overlong_file_for_robust_download(
    tmp_path, monkeypatch
):
    """Longer than expected must NOT be quarantined.

    ``expected_size`` is only a skip-if-matches hint and may be a stale/wrong
    client-side guess (e.g. an API byte count for a gzip-decoded response). A
    complete-but-oversized cached file must be left in place for
    ``robust_download`` to reconcile — it restarts from scratch when the local
    copy is larger than the true remote size.
    """
    from luxar.utils.download import find_quarantined_files

    monkeypatch.setattr(demo_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
    cache_dir = (tmp_path / "cache") / "demoQ4"
    cache_dir.mkdir(parents=True)
    dest = cache_dir / "data.bin"
    dest.write_bytes(b"way too many bytes")
    seen: dict = {}

    def _fake(url, output_path, **kw):
        # robust_download owns the reconcile; the file is handed to it untouched.
        seen["existed"] = Path(output_path).exists()
        Path(output_path).write_bytes(b"12345678")
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.robust_download", _fake)

    demo_utils.cached_download(
        "http://example.invalid/data.bin", "demoQ4", "data.bin", expected_size=8
    )

    assert not find_quarantined_files(dest), "oversized file must not be quarantined"
    assert seen["existed"] is True, "oversized file must survive for robust_download"


def test_cached_download_no_redownload_loop_on_stale_expected_size(
    tmp_path, monkeypatch
):
    """A stale ``expected_size`` must not cause a re-download-every-launch loop.

    Reproduces the review's regression: the cached file is COMPLETE but larger
    than the (wrong) ``expected_size``. ``robust_download`` reconciles against
    the true remote size — which is still larger than the stale guess — so it
    keeps overwriting with those true bytes. If ``cached_download`` quarantined
    the complete file, every launch would re-fetch it forever. With the fix the
    file is never quarantined, so repeated launches don't churn.
    """
    from luxar.utils.download import find_quarantined_files

    monkeypatch.setattr(demo_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
    cache_dir = (tmp_path / "cache") / "demoQ6"
    cache_dir.mkdir(parents=True)
    dest = cache_dir / "data.bin"

    stale_expected = 8  # wrong: true bytes are longer than this guess
    true_bytes = b"the true remote bytes, longer than the stale guess"
    calls = {"downloads": 0}

    def _fake(url, output_path, **kw):
        # Model the real robust_download reconcile: local > remote -> restart
        # from scratch, writing the true (still-oversized) bytes.
        calls["downloads"] += 1
        Path(output_path).write_bytes(true_bytes)
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.robust_download", _fake)

    for _ in range(3):
        demo_utils.cached_download(
            "http://example.invalid/data.bin",
            "demoQ6",
            "data.bin",
            expected_size=stale_expected,
        )
        # The complete file is never renamed to .corrupt on any run.
        assert not find_quarantined_files(dest), "complete file was quarantined"

    # cached_download hands the file to robust_download on every run (the fake
    # always rewrites), but it must never DESTROY the complete file first.
    assert dest.read_bytes() == true_bytes
    assert calls["downloads"] == 3


def test_cached_download_quarantines_an_lfs_pointer(tmp_path, monkeypatch):
    """A pointer stub is not data, and is exactly what a resume would append to."""
    from luxar.utils.download import find_quarantined_files

    monkeypatch.setattr(demo_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
    cache_dir = (tmp_path / "cache") / "demoQ5"
    cache_dir.mkdir(parents=True)
    dest = cache_dir / "data.bin"
    dest.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:"
        + b"0" * 64
        + b"\nsize 11\n"
    )

    def _fake(url, output_path, **kw):
        assert not Path(output_path).exists()
        Path(output_path).write_bytes(b"real bytes")
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.robust_download", _fake)

    out = demo_utils.cached_download(
        "http://example.invalid/data.bin", "demoQ5", "data.bin"
    )

    assert out.read_bytes() == b"real bytes"
    assert find_quarantined_files(dest)
