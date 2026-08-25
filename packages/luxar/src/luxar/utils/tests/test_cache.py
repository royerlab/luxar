"""Tests for demo download and computed-value caches."""

from pathlib import Path

import numpy as np
import pytest

from luxar.utils import cache as cache_utils


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the module cache root at a temp dir for every test."""
    monkeypatch.setattr(cache_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")


def test_cache_computed_miss_then_hit():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"arr": np.arange(5), "label": "x"}

    r1 = cache_utils.cache_computed("demoX", "k1", compute)
    r2 = cache_utils.cache_computed("demoX", "k1", compute)
    assert calls["n"] == 1  # second call served from cache
    assert r2["label"] == "x"
    np.testing.assert_array_equal(r1["arr"], r2["arr"])


def test_cache_computed_version_and_recompute():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    assert cache_utils.cache_computed("demoX", "k", compute, version=1) == 1
    # A new version key must not read the v1 cache.
    assert cache_utils.cache_computed("demoX", "k", compute, version=2) == 2
    # recompute=True bypasses the cache.
    assert (
        cache_utils.cache_computed("demoX", "k", compute, version=2, recompute=True)
        == 3
    )


def test_cache_computed_explicit_cache_dir_is_used_verbatim(tmp_path):
    """``cache_dir=`` writes/reads there, and never touches the default root.

    The three graph demos take a ``--cache-dir`` override and must be able to put
    a derived result beside the raw downloads it came from — so the directory has
    to be usable verbatim, with ``name`` demoted to a log label.
    """
    calls = {"n": 0}
    explicit = tmp_path / "elsewhere" / "caida"

    def compute():
        calls["n"] += 1
        return {"coords": np.arange(3)}

    r1 = cache_utils.cache_computed(
        "caida", "layout3d_abc", compute, cache_dir=explicit
    )
    r2 = cache_utils.cache_computed(
        "caida", "layout3d_abc", compute, cache_dir=explicit
    )

    assert calls["n"] == 1  # second call served from the explicit directory
    np.testing.assert_array_equal(r1["coords"], r2["coords"])
    assert [p.name for p in explicit.glob("*.pkl")] == ["layout3d_abc_v1.pkl"]
    # The namespaced default root is not created, let alone written to.
    assert not (tmp_path / "cache").exists()


def test_cache_computed_corrupt_is_quarantined(tmp_path):
    def compute():
        return 42

    cache_utils.cache_computed("demoY", "kk", compute)
    cache_dir = (tmp_path / "cache") / "demoY"
    pkl = next(cache_dir.glob("*.pkl"))
    pkl.write_bytes(b"not a pickle")  # corrupt it

    # Should quarantine and recompute rather than raise.
    assert cache_utils.cache_computed("demoY", "kk", compute) == 42
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

    p = cache_utils.cached_download(
        "http://example.invalid/data.bin", "demoZ", "data.bin"
    )
    assert p.read_bytes() == b"hello world"


def test_cached_download_checksum_hit_is_silent_when_quiet(
    tmp_path, monkeypatch, capsys
):
    """A checksum-verified cache hit under ``verbose=False`` must print nothing.

    ``cached_download`` re-hashes the cached file on every call, so a
    ``verify_file_checksum`` that ignores ``verbose`` logs a per-file
    "Verifying …" block for a caller that asked for silence.
    """
    import hashlib

    cache_dir = (tmp_path / "cache") / "demoQuiet"
    cache_dir.mkdir(parents=True)
    payload = b"hello world"
    (cache_dir / "data.bin").write_bytes(payload)

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("network download attempted for a cached file")

    monkeypatch.setattr("luxar.utils.download.robust_download", _boom)
    monkeypatch.setattr("luxar.utils.download.download_with_checksum", _boom)

    p = cache_utils.cached_download(
        "http://example.invalid/data.bin",
        "demoQuiet",
        "data.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        verbose=False,
    )

    assert p.read_bytes() == payload
    assert capsys.readouterr().out == ""


def test_cached_download_quarantines_a_checksum_failing_cache(tmp_path, monkeypatch):
    """A complete-but-wrong cached file must move aside BEFORE re-downloading.

    robust_download resumes onto whatever bytes are at the destination, so
    leaving them appends the new download to the old garbage (or trips a 416).
    """
    import hashlib

    from luxar.utils.download import find_quarantined_files

    monkeypatch.setattr(cache_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
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

    out = cache_utils.cached_download(
        "http://example.invalid/data.bin",
        "demoQ",
        "data.bin",
        sha256=hashlib.sha256(good).hexdigest(),
    )

    assert out.read_bytes() == good
    assert find_quarantined_files(dest)


def test_cached_download_keeps_a_short_file_so_it_can_resume(tmp_path, monkeypatch):
    """Shorter than expected IS a resumable partial download — do not touch it."""
    monkeypatch.setattr(cache_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
    cache_dir = (tmp_path / "cache") / "demoQ3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "data.bin").write_bytes(b"half")
    seen: dict = {}

    def _fake(url, output_path, **kw):
        seen["existed"] = Path(output_path).exists()
        Path(output_path).write_bytes(b"halfhalf")
        return Path(output_path)

    monkeypatch.setattr("luxar.utils.download.robust_download", _fake)

    cache_utils.cached_download(
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

    monkeypatch.setattr(cache_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
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

    cache_utils.cached_download(
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

    monkeypatch.setattr(cache_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
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
        cache_utils.cached_download(
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

    monkeypatch.setattr(cache_utils, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")
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

    out = cache_utils.cached_download(
        "http://example.invalid/data.bin", "demoQ5", "data.bin"
    )

    assert out.read_bytes() == b"real bytes"
    assert find_quarantined_files(dest)
