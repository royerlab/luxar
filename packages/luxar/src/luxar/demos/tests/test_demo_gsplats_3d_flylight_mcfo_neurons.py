"""Tests for demo_gsplats_3d_flylight_mcfo_neurons.

Covers the two pieces of this demo that are real logic rather than glue:

* ``splat_colors`` — the physical-to-voxel conversion, channel balancing and
  hue normalisation. The conversion in particular is worth pinning: centres
  come back from the fit in micrometres, and indexing them as voxels samples
  the wrong places *silently* (every colour is wrong, nothing raises).
* ``_HttpRangeFile`` — the seekable file that lets one sample be pulled out of
  a 7.1 GB remote zip. Its seek/read arithmetic is exercised against an
  in-memory server, with no network.

No network, no GPU fit.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_flylight_mcfo_neurons.py"
)


def _load_demo_module():
    name = "_luxar_demo_flylight_mcfo_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
splat_colors = _demo.splat_colors
_HttpRangeFile = _demo._HttpRangeFile


# ---------------------------------------------------------------------------
# splat_colors
# ---------------------------------------------------------------------------


BRIGHT = ((1, 2, 3), (4, 5, 6), (5, 7, 9))


def _channels(shape=(6, 8, 10)):
    """Three volumes, each with a single distinct bright voxel.

    Bright voxels are clamped into ``shape`` so the helper stays valid for the
    smaller volumes some tests use.
    """
    chans = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    for chan, idx in zip(chans, BRIGHT):
        chan[tuple(min(i, n - 1) for i, n in zip(idx, shape))] = 1.0
    return chans


def test_centers_are_interpreted_as_physical_units() -> None:
    """Centres in micrometres must be divided by voxel size before indexing.

    This is the regression that matters: with the conversion missing the
    sampled voxel is off by 1/voxel_size and every colour is quietly wrong.
    """
    chans = _channels()
    voxel = (0.5, 0.5, 0.5)
    # Physical coordinates of the voxel that is bright in channel 1 only.
    centers = np.array([[4 * 0.5, 5 * 0.5, 6 * 0.5]], dtype=np.float32)

    rgb = splat_colors(centers, chans, voxel_size=voxel)

    assert rgb.shape == (1, 3)
    # Green channel dominates; had the centres been indexed as voxels we would
    # have landed on (4, 5, 6)/0.5 -> out of range and clipped to a dark voxel.
    assert rgb[0].argmax() == 1
    assert rgb[0, 1] == pytest.approx(1.0)


def test_channels_are_balanced_before_compositing() -> None:
    """A dim channel must still be able to win a splat it dominates."""
    shape = (4, 4, 4)
    chans = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    # ch0 is bright everywhere; ch2 is 100x dimmer but is the only channel
    # present at the sampled voxel.
    chans[0][:] = 0.5
    chans[0][1, 1, 1] = 0.0
    chans[2][1, 1, 1] = 0.005

    centers = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    rgb = splat_colors(centers, chans, voxel_size=(1.0, 1.0, 1.0))

    assert rgb[0].argmax() == 2, "dim-but-dominant channel should win the splat"


def test_hue_is_normalised_per_splat() -> None:
    """Each splat's strongest channel saturates; brightness lives in amplitude."""
    chans = _channels()
    centers = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    rgb = splat_colors(centers, chans, voxel_size=(1.0, 1.0, 1.0))

    assert np.allclose(rgb.max(axis=1), 1.0)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_out_of_range_centers_are_clipped_not_wrapped() -> None:
    """Negative indices must clamp, not wrap around to the far side."""
    chans = _channels(shape=(4, 4, 4))
    chans[0][0, 0, 0] = 1.0
    centers = np.array([[-50.0, -50.0, -50.0]], dtype=np.float32)

    rgb = splat_colors(centers, chans, voxel_size=(1.0, 1.0, 1.0))

    # Clamped to (0, 0, 0), which is bright in channel 0 only.
    assert rgb[0].argmax() == 0


def test_zero_signal_splat_gets_black_not_nan() -> None:
    chans = _channels(shape=(4, 4, 4))
    centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    rgb = splat_colors(centers, chans, voxel_size=(1.0, 1.0, 1.0))

    assert np.all(np.isfinite(rgb))
    assert np.allclose(rgb, 0.0)


# ---------------------------------------------------------------------------
# _HttpRangeFile
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 206):
        self.content = content
        self.status_code = status_code


class _FakeSession:
    """Serves byte ranges out of an in-memory blob and counts requests."""

    def __init__(self, blob: bytes, status_code: int = 206):
        self.blob = blob
        self.status_code = status_code
        self.requests: list[tuple[int, int]] = []

    def get(self, url, headers=None, timeout=None):  # noqa: D102
        rng = (headers or {})["Range"].split("=", 1)[1]
        start, end = (int(x) for x in rng.split("-"))
        self.requests.append((start, end))
        return _FakeResponse(self.blob[start : end + 1], self.status_code)


def _blob(n: int = 4096) -> bytes:
    return bytes((i * 7 + 11) % 256 for i in range(n))


def test_sequential_read_matches_source() -> None:
    blob = _blob()
    session = _FakeSession(blob)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=256)

    assert f.read(100) == blob[:100]
    assert f.read(300) == blob[100:400]
    assert f.tell() == 400


def test_seek_end_and_read_tail() -> None:
    """zipfile finds the central directory by seeking from the end."""
    blob = _blob()
    session = _FakeSession(blob)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=256)

    f.seek(-64, io.SEEK_END)
    assert f.read() == blob[-64:]


def test_seek_cur_and_clamping() -> None:
    blob = _blob(512)
    f = _HttpRangeFile("http://x", len(blob), _FakeSession(blob), chunk_size=128)

    f.seek(100)
    f.seek(50, io.SEEK_CUR)
    assert f.tell() == 150
    # Seeking past either end clamps rather than raising or going negative.
    f.seek(-10_000, io.SEEK_SET)
    assert f.tell() == 0
    f.seek(10_000, io.SEEK_SET)
    assert f.tell() == len(blob)
    assert f.read(10) == b""


def test_readahead_block_avoids_a_request_per_read() -> None:
    """Adjacent small reads must be served from one fetched block."""
    blob = _blob(2048)
    session = _FakeSession(blob)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=1024)

    for _ in range(32):
        f.read(8)

    assert len(session.requests) == 1, session.requests


def test_non_206_response_is_a_clear_error() -> None:
    """A host that stops honouring ranges must fail loudly, not download 7 GB."""
    blob = _blob(256)
    session = _FakeSession(blob, status_code=200)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=64)

    with pytest.raises(RuntimeError, match="206"):
        f.read(10)


def test_read_spanning_multiple_blocks() -> None:
    blob = _blob(1024)
    session = _FakeSession(blob)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=100)

    assert f.read(450) == blob[:450]
    assert len(session.requests) >= 4


# ---------------------------------------------------------------------------
# Archive extraction safety (zip-slip) and --recompute replacement
# ---------------------------------------------------------------------------

_safe_extract_path = _demo._safe_extract_path


def test_safe_extract_path_accepts_normal_members(tmp_path) -> None:
    root = (tmp_path / "store").resolve()
    dest = _safe_extract_path(root, "volumes/raw/0.0.0.0", "m")
    assert dest == root / "volumes" / "raw" / "0.0.0.0"


@pytest.mark.parametrize(
    "rel",
    [
        "../escape.txt",
        "volumes/../../escape.txt",
        "a/b/../../../escape.txt",
    ],
)
def test_safe_extract_path_rejects_traversal(tmp_path, rel: str) -> None:
    """A zip member must not be able to write outside the extraction root."""
    root = (tmp_path / "store").resolve()
    with pytest.raises(RuntimeError, match="escapes the extraction root"):
        _safe_extract_path(root, rel, f"member:{rel}")


@pytest.mark.parametrize("rel", ["/etc/passwd", "C:\\\\windows\\\\system32\\\\x"])
def test_safe_extract_path_rejects_absolute(tmp_path, rel: str) -> None:
    root = (tmp_path / "store").resolve()
    with pytest.raises(RuntimeError, match="absolute path"):
        _safe_extract_path(root, rel, f"member:{rel}")


def test_recompute_replaces_a_populated_target(tmp_path, monkeypatch) -> None:
    """`--recompute` must replace an existing store, not trip over it.

    Renaming the freshly extracted directory onto a populated target raises
    ENOTEMPTY, which broke the documented recompute path on every second run.
    """
    import zipfile

    sample = "SAMPLE_X"
    cache = tmp_path / "cache"
    cache.mkdir()

    # An existing, non-empty store — what --recompute finds on a second run.
    target = cache / f"{sample}.zarr"
    (target / "volumes").mkdir(parents=True)
    (target / "volumes" / "stale").write_bytes(b"old")

    archive = tmp_path / "a.zip"
    prefix = f"{_demo.SAMPLE_MEMBER_PREFIX}/{sample}.zarr/"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}volumes/raw/.zarray", b"{}")
        zf.writestr(f"{prefix}volumes/raw/0.0.0.0", b"new")

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", True)
    monkeypatch.setattr(
        _demo, "_open_remote_zip", lambda url: (zipfile.ZipFile(archive), None)
    )

    out = _demo.fetch_sample(sample)

    assert out == target
    assert (target / "volumes" / "raw" / "0.0.0.0").read_bytes() == b"new"
    assert not (target / "volumes" / "stale").exists(), "old store was not replaced"
    assert not target.with_suffix(".zarr.partial").exists()
    assert not target.with_suffix(".zarr.stale").exists()


def test_extraction_loop_rejects_a_traversing_member(tmp_path, monkeypatch) -> None:
    """The guard must be wired into the extraction loop, not merely exist.

    Testing ``_safe_extract_path`` alone would still pass if the loop stopped
    calling it, so this drives a malicious archive through ``fetch_sample``.
    """
    import zipfile

    sample = "EVIL"
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = tmp_path / "evil.zip"
    prefix = f"{_demo.SAMPLE_MEMBER_PREFIX}/{sample}.zarr/"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}volumes/raw/.zarray", b"{}")
        zf.writestr(f"{prefix}../../../pwned.txt", b"escaped")

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", False)
    monkeypatch.setattr(
        _demo, "_open_remote_zip", lambda url: (zipfile.ZipFile(archive), None)
    )

    with pytest.raises(RuntimeError, match="escapes the extraction root"):
        _demo.fetch_sample(sample)

    assert not (tmp_path / "pwned.txt").exists()
    assert not (cache.parent / "pwned.txt").exists()
