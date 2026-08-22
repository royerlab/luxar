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
    """Balancing must flip which channel wins, not merely rescale it.

    Constructed so the UNBALANCED winner and the BALANCED winner differ: ch0 is
    the larger raw value at the sampled voxel but has a far higher gain, so
    after dividing by each channel's own robust maximum ch2 must win. A test
    whose winner is the same either way passes even if balancing is deleted.
    """
    shape = (8, 8, 8)
    chans = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    # Gains: ch0 is a bright channel (p99.99 ~ 1.0), ch2 a faint one (~0.01).
    chans[0][:] = 1.0
    chans[2][:] = 0.01
    # At the sampled voxel ch0 is numerically larger...
    chans[0][2, 2, 2] = 0.50
    chans[2][2, 2, 2] = 0.01
    centers = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

    raw_winner = int(np.argmax([0.50, 0.0, 0.01]))
    assert raw_winner == 0, "fixture must have ch0 winning before balancing"

    rgb = splat_colors(centers, chans, voxel_size=(1.0, 1.0, 1.0))

    # ...but after balancing ch2 uses all of its dynamic range and ch0 half.
    assert rgb[0].argmax() == 2, "balancing did not change the winning channel"


def test_hue_is_normalised_per_splat() -> None:
    """Each splat's strongest channel saturates; brightness lives in amplitude.

    The balanced triplet here is deliberately well below 1.0, so returning it
    un-normalised would fail — a fixture whose triplet already peaks at 1.0
    cannot tell the two apart.
    """
    shape = (6, 6, 6)
    chans = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
    # Each channel's gain is set by a bright voxel elsewhere; the sampled voxel
    # sits at a fraction of it, so balanced values are ~0.25/0.10/0.05.
    # A whole plane carries the gain value so p99.99 lands exactly on it —
    # with a single bright voxel the percentile interpolates and the expected
    # ratios below would be approximate rather than exact.
    for c, here in enumerate((0.25, 0.10, 0.05)):
        chans[c][0] = 1.0
        chans[c][3, 3, 3] = here
    centers = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)

    rgb = splat_colors(centers, chans, voxel_size=(1.0, 1.0, 1.0))

    assert np.isclose(rgb[0].max(), 1.0), "strongest channel must saturate"
    # Ratios survive normalisation: 0.10/0.25 and 0.05/0.25.
    assert np.allclose(rgb[0], [1.0, 0.4, 0.2], atol=1e-5)
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
    def __init__(self, content: bytes, status_code: int = 206, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False
        self.consumed = 0

    def iter_content(self, chunk_size: int = 1):
        """Yield the body in chunks, recording how much was actually read."""
        for i in range(0, len(self.content), chunk_size):
            part = self.content[i : i + chunk_size]
            self.consumed += len(part)
            yield part

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    """Serves byte ranges out of an in-memory blob and counts requests."""

    def __init__(self, blob: bytes, status_code: int = 206, served=None):
        self.blob = blob
        self.status_code = status_code
        # ``served(start, end)`` overrides what the host answers with, as
        # ``(body, Content-Range)``; the default is the honest range.
        self.served = served
        self.requests: list[tuple[int, int]] = []
        self.streamed: list[bool] = []
        self.responses: list[_FakeResponse] = []
        self.closed = False

    def get(self, url, headers=None, timeout=None, stream=None):  # noqa: D102
        rng = (headers or {})["Range"].split("=", 1)[1]
        start, end = (int(x) for x in rng.split("-"))
        self.requests.append((start, end))
        self.streamed.append(bool(stream))
        if self.served is None:
            body = self.blob[start : end + 1]
            crange = f"bytes {start}-{start + len(body) - 1}/{len(self.blob)}"
        else:
            body, crange = self.served(start, end)
        resp = _FakeResponse(body, self.status_code, {"Content-Range": crange})
        self.responses.append(resp)
        return resp

    def close(self) -> None:  # noqa: D102
        self.closed = True


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


def test_range_fetches_are_streamed_and_released() -> None:
    """Every range fetch must ask for a streamed body and close it.

    Failing loudly on a non-206 is only cheap if the body was not already
    read: a host that ignores ``Range`` answers 200 with the WHOLE 7.1 GB
    archive, and a non-streaming ``get`` buffers all of it in memory before the
    status check can refuse it. So the request has to carry ``stream=True`` —
    and then be closed, or the connection is never returned to the pool.
    """
    blob = _blob(512)
    session = _FakeSession(blob)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=128)

    f.read(200)

    assert session.streamed and all(session.streamed), session.streamed
    assert all(r.closed for r in session.responses)


def test_a_206_for_the_wrong_range_is_refused() -> None:
    """A 206 is only usable if it answers the interval that was asked for.

    The block is cached at the offset that was REQUESTED, so a host answering
    from somewhere else (a caching proxy serving a stale range, say) would have
    its bytes filed at the wrong place and the archive assembled out of the
    wrong data — surfacing much later as an unreadable zip rather than as this.
    """
    blob = _blob(512)
    session = _FakeSession(
        blob, served=lambda s, e: (blob[:128], f"bytes 0-127/{len(blob)}")
    )
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=128)
    f.seek(256)

    with pytest.raises(RuntimeError, match="wrong offset"):
        f.read(16)


@pytest.mark.parametrize("crange", ["", "bytes */4096", "kilobytes 0-127/4096"])
def test_a_206_without_a_usable_content_range_is_refused(crange: str) -> None:
    """No parseable range means no way to know where the bytes belong."""
    blob = _blob(256)
    session = _FakeSession(blob, served=lambda s, e: (blob[s : e + 1], crange))
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=128)

    with pytest.raises(RuntimeError, match="Content-Range"):
        f.read(16)


def test_a_body_longer_than_the_advertised_range_is_refused_early() -> None:
    """The body is bounded, not trusted: a 206 can carry the whole archive.

    ``resp.content`` would buffer whatever arrives, which is the 7.1 GB
    download the range access exists to avoid — a status check alone does not
    prevent it. So the read has to stop as soon as the body runs past the range
    the response itself advertised.
    """
    blob = _blob(200_000)
    session = _FakeSession(
        blob, served=lambda s, e: (blob, f"bytes {s}-{e}/{len(blob)}")
    )
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=128)

    with pytest.raises(RuntimeError, match="more than the 128 bytes"):
        f.read(16)
    # Refused part-way through rather than after buffering all of it, which is
    # the whole point — the real body here is the archive, not 200 KB.
    assert 0 < session.responses[0].consumed < len(blob)


def test_a_short_but_honest_range_is_completed_by_a_refetch() -> None:
    """A host may legally answer with LESS than was asked for.

    That is not a misbehaving host and must not be refused: the read loop just
    fetches the remainder from where the short block ended.
    """
    blob = _blob(512)
    half = 64
    session = _FakeSession(
        blob,
        served=lambda s, e: (
            blob[s : s + half],
            f"bytes {s}-{s + half - 1}/{len(blob)}",
        ),
    )
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=128)

    assert f.read(200) == blob[:200]
    assert len(session.requests) == 4, session.requests


def test_closing_the_range_file_releases_the_session() -> None:
    """``ZipFile`` never closes a file it was handed, so the handle must.

    Nothing else in the chain owns the session: without this the pooled
    connection stays open for the whole run (a fit lasting minutes) and is
    reclaimed only whenever the collector gets round to it.
    """
    blob = _blob(256)
    session = _FakeSession(blob)
    f = _HttpRangeFile("http://x", len(blob), session, chunk_size=64)
    f.read(10)

    f.close()

    assert session.closed, "the HTTP session outlived the file over it"
    assert f.closed
    # Idempotent: ``fetch_sample`` closes it explicitly and the finaliser may
    # run again later.
    f.close()


class _ProbeResponse:
    def __init__(self, status_code: int, headers: dict, url: str = "http://x/a.zip"):
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self.content = b"\x00"
        self.closed = False

    def raise_for_status(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _ProbeSession:
    """A session whose only job is to answer the size-probing ranged GET."""

    def __init__(self, status_code: int, headers: dict):
        self._status_code = status_code
        self._headers = headers
        self.streamed: list[bool] = []
        self.responses: list[_ProbeResponse] = []
        self.closed = False

    def get(  # noqa: D102
        self, url, headers=None, allow_redirects=None, timeout=None, stream=None
    ):
        self.streamed.append(bool(stream))
        resp = _ProbeResponse(self._status_code, self._headers, url)
        self.responses.append(resp)
        return resp

    def close(self) -> None:  # noqa: D102
        self.closed = True


def _probe_with(monkeypatch, status_code: int, headers: dict) -> _ProbeSession:
    session = _ProbeSession(status_code, headers)
    monkeypatch.setattr(_demo.requests, "Session", lambda: session)
    return session


@pytest.mark.parametrize("headers", [{"Content-Range": "bytes 0-0/*"}, {}])
def test_unknown_total_size_is_refused_with_the_range_explanation(
    monkeypatch, headers: dict
) -> None:
    """``bytes 0-0/*`` is a legal 206 carrying no total — and unusable here.

    Without the size there is nothing to seek against, so it must produce the
    "did not honour a byte-range request" guidance rather than a bare ValueError
    out of ``int('*')``.
    """
    _probe_with(monkeypatch, 206, headers)

    with pytest.raises(RuntimeError, match="did not honour a byte-range"):
        _demo._open_remote_zip("http://x/a.zip")


def test_non_206_probe_is_refused(monkeypatch) -> None:
    """A host answering 200 to a ranged GET would mean a 7.1 GB download."""
    session = _probe_with(monkeypatch, 200, {"Content-Range": "bytes 0-0/4096"})

    with pytest.raises(RuntimeError, match="did not honour a byte-range"):
        _demo._open_remote_zip("http://x/a.zip")

    # The refusal must be free: a 200 to this request carries the ENTIRE
    # archive, so the probe has to stream (headers only) and then release it.
    # Without ``stream=True`` requests would have buffered 7.1 GB before the
    # check above ever ran.
    assert session.streamed == [True]
    assert all(r.closed for r in session.responses)
    # A refusal returns no handle, so nothing downstream can close the session
    # for us — the failure path has to do it itself.
    assert session.closed, "the refused probe left its session open"


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


class _RecordingHandle:
    """Stands in for the HTTP-backed file ``_open_remote_zip`` hands back."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        """Record that the caller released the handle (and its session)."""
        self.closed = True


def _stub_remote_zip(monkeypatch, archive: Path) -> _RecordingHandle:
    """Serve ``fetch_sample`` from a local archive, tracking handle closure."""
    import zipfile

    handle = _RecordingHandle()
    monkeypatch.setattr(
        _demo, "_open_remote_zip", lambda url: (zipfile.ZipFile(archive), handle)
    )
    return handle


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
    with pytest.raises(RuntimeError, match="absolute path|backslash"):
        _safe_extract_path(root, rel, f"member:{rel}")


@pytest.mark.parametrize("rel", ["..\\\\..\\\\pwned.txt", "a\\\\..\\\\..\\\\b"])
def test_safe_extract_path_rejects_backslash_traversal(tmp_path, rel: str) -> None:
    """Backslash members traverse on Windows but not POSIX — refuse everywhere.

    Without this the guard's behaviour depends on the host OS: the same archive
    is inert on Linux and an escape on Windows.
    """
    root = (tmp_path / "store").resolve()
    with pytest.raises(RuntimeError, match="backslash"):
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
    prefix = f"{_demo.SAMPLE_MEMBER_ROOT}/train/{sample}.zarr/"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}volumes/raw/.zarray", b"{}")
        zf.writestr(f"{prefix}volumes/raw/0.0.0.0", b"new")

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", True)
    handle = _stub_remote_zip(monkeypatch, archive)

    out = _demo.fetch_sample(sample)

    assert out == target
    assert (target / "volumes" / "raw" / "0.0.0.0").read_bytes() == b"new"
    assert not (target / "volumes" / "stale").exists(), "old store was not replaced"
    assert not target.with_suffix(".zarr.partial").exists()
    assert not target.with_suffix(".zarr.stale").exists()
    assert handle.closed, "the remote handle (and its session) was left open"


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
    prefix = f"{_demo.SAMPLE_MEMBER_ROOT}/train/{sample}.zarr/"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}volumes/raw/.zarray", b"{}")
        zf.writestr(f"{prefix}../../../pwned.txt", b"escaped")

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", False)
    handle = _stub_remote_zip(monkeypatch, archive)

    with pytest.raises(RuntimeError, match="escapes the extraction root"):
        _demo.fetch_sample(sample)

    assert not (tmp_path / "pwned.txt").exists()
    assert not (cache.parent / "pwned.txt").exists()
    assert handle.closed, "an aborted extraction left the remote handle open"


# ---------------------------------------------------------------------------
# merge_for_render — one node, per-splat RGBA
# ---------------------------------------------------------------------------

merge_for_render = _demo.merge_for_render


class _Splats:
    """Minimal stand-in for GSplatData (only the fields merging touches)."""

    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.centers = rng.uniform(0, 10, (n, 3)).astype(np.float32)
        self.amplitudes = rng.uniform(0.2, 1.0, n).astype(np.float32)
        self.cholesky_factors = np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32)


def test_merge_without_neuropil_is_neurons_only() -> None:
    neurons = _Splats(5)
    rgb = np.tile([1.0, 0.5, 0.0], (5, 1)).astype(np.float32)

    centers, amps, chol, rgba = merge_for_render(neurons, rgb, None)

    assert len(centers) == len(amps) == len(chol) == len(rgba) == 5
    assert rgba.shape[1] == 4
    # Neurons carry the amplitude->alpha ramp, not a constant: the brightest
    # splat saturates and none exceeds the valid range.
    assert 0.0 <= rgba[:, 3].min() and rgba[:, 3].max() <= 1.0
    assert np.isclose(rgba[:, 3].max(), 1.0)


def test_merge_concatenates_and_tags_alpha() -> None:
    neurons, neuropil = _Splats(4, 1), _Splats(7, 2)
    rgb = np.tile([1.0, 1.0, 1.0], (4, 1)).astype(np.float32)

    centers, amps, chol, rgba = merge_for_render(neurons, rgb, neuropil)

    assert len(centers) == len(amps) == len(chol) == len(rgba) == 11
    # Neuropil is emitted first, so the split is at len(neuropil).
    assert np.allclose(rgba[:7, 3], _demo.NEUROPIL_ALPHA)
    # Neurons carry the ramp; the brightest reaches 1.0.
    assert np.isclose(rgba[7:, 3].max(), 1.0)
    assert rgba[7:, 3].min() >= 0.0
    assert np.allclose(rgba[:7, :3], np.asarray(_demo.NEUROPIL_RGB, dtype=np.float32))


def test_merge_scales_neuropil_amplitude_only() -> None:
    """The dimming multiplier must not touch the neuron amplitudes."""
    neurons, neuropil = _Splats(3, 3), _Splats(3, 4)
    rgb = np.zeros((3, 3), dtype=np.float32)

    _, amps, _, _ = merge_for_render(neurons, rgb, neuropil)

    assert np.allclose(amps[:3], neuropil.amplitudes * _demo.NEUROPIL_AMP)
    assert np.allclose(amps[3:], neurons.amplitudes)


def test_merge_outputs_are_float32() -> None:
    """Mixed dtypes here would silently upcast the whole scene to float64."""
    neurons, neuropil = _Splats(2, 5), _Splats(2, 6)
    rgb = np.zeros((2, 3), dtype=np.float32)

    for arr in merge_for_render(neurons, rgb, neuropil):
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Scene authoring: one volumetric node, per-splat RGBA, the film look
# ---------------------------------------------------------------------------


def _tiny_scene(tmp_path, alpha: float = 0.5):
    """Build the demo's scene from six splats and return the opened store."""
    import zarr as _zarr

    rng = np.random.default_rng(0)
    n = 6
    centers = rng.uniform(0, 10, (n, 3)).astype(np.float32)
    amplitudes = rng.uniform(0.2, 1.0, n).astype(np.float32)
    cholesky = np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32)
    rgba = np.concatenate(
        [rng.uniform(0.1, 0.9, (n, 3)), np.full((n, 1), alpha)], axis=1
    ).astype(np.float32)

    out = _demo.create_luxar_scene(
        centers, amplitudes, cholesky, rgba, tmp_path / "mcfo.luxar.zarr"
    )
    return _zarr.open_group(str(out), mode="r")


def test_scene_is_one_volumetric_node_with_per_splat_rgba(tmp_path) -> None:
    """The whole point of the demo: one node, volumetric, alpha per splat.

    Under additive compositing the neuropil would glow through the neurites
    instead of occluding them, and splitting the two into separate nodes would
    reintroduce the draw-order problem the merge exists to avoid.
    """
    scene = _tiny_scene(tmp_path)

    nodes = [k for k in scene.group_keys() if k != "overlays"]
    assert nodes == ["gsplats_mcfo"], f"expected exactly one splat node, got {nodes}"
    node = scene["gsplats_mcfo"]
    assert node.attrs["blending_mode"] == "volumetric"
    assert node.attrs["absorption"] == 1.0
    # Four columns: the alpha one is the optical depth that distinguishes
    # neuropil from neurons inside the single node.
    assert node["colors"].shape[1] == 4


def test_scene_describes_a_neuropil_only_when_there_is_one() -> None:
    """A neurons-only run must not advertise a counterstain it never fitted.

    Missing ffmpeg/h5py, an unmapped sample, a grid that does not register and
    --no-neuropil all produce that scene, and the description is where a reader
    finds out which of the two they are looking at.
    """
    with_pil = _demo.scene_description(True)
    assert "Volume-Rendered Neuropil" in with_pil
    assert "FlyLight Gen1 MCFO reference channel" in with_pil

    without = _demo.scene_description(False)
    assert "neurons only, no neuropil" in without
    assert "There is no neuropil in this scene" in without
    assert "FlyLight Gen1 MCFO reference channel" not in without
    # Both keep the citations: the neurons are FISBe imagery either way.
    for text in (with_pil, without):
        assert "Mais et al." in text and _demo.SAMPLE in text


def test_scene_bakes_the_film_look_effect_by_effect(tmp_path) -> None:
    """``cinematic_mode`` alone renders nothing — the effects must be authored.

    The viewer applies its cinematic preset from the C-key toggle; a scene's
    flag only sets the panel's summary state. So the scene has to carry the
    individual effects, or the demo ships plain ACES and the documented look
    silently never happens.
    """
    config = dict(_tiny_scene(tmp_path).attrs)["viewer_config"]

    assert config["tone_mapping"] == "ACES"
    for key in ("bloom_enabled", "vignette_enabled", "detector_noise_enabled"):
        assert config.get(key) is True, f"{key} missing from the authored scene"


# ---------------------------------------------------------------------------
# --sample validation and H5J decode integrity
# ---------------------------------------------------------------------------

validate_sample_name = _demo.validate_sample_name


@pytest.mark.parametrize(
    "name",
    [
        "VT047848-20171020_66_I3",
        "JRC_SS04989-20160318_24_B1",
        "R14A02-20180905_65_A6",
    ],
)
def test_real_sample_names_are_accepted(name: str) -> None:
    assert validate_sample_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "a/../../etc/passwd",
        "/absolute/path",
        "back\\slash",
        "..",
        "",
        "has space",
        "semi;colon",
    ],
)
def test_path_bearing_sample_names_are_rejected(name: str) -> None:
    """--sample lands in cache paths and the output filename, so it must be a
    bare basename: otherwise reads and writes escape the demo's directories."""
    with pytest.raises(ValueError, match="Invalid --sample"):
        validate_sample_name(name)


class _FakeH5Group:
    """Stands in for the ``Channels`` group of an H5J file."""

    def __init__(self, attrs, payload_len=8):
        self.attrs = attrs
        self._payload_len = payload_len

    def __getitem__(self, key):
        return np.zeros(self._payload_len, dtype=np.uint8)


class _FakeH5File:
    def __init__(self, attrs, channel_attrs):
        self.attrs = attrs
        self._grp = _FakeH5Group(channel_attrs)

    def __getitem__(self, key):
        assert key == "Channels"
        return self._grp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_h5py(n, h, w, pr, pb, spec=b"sssr"):
    """A minimal h5py stand-in, so these tests run without h5py installed."""
    import types

    mod = types.SimpleNamespace()
    mod.File = lambda path, mode: _FakeH5File(
        {"channel_spec": spec if isinstance(spec, bytes) else spec.encode()},
        {
            "width": np.array([w]),
            "height": np.array([h]),
            "frames": np.array([n]),
            "pad_right": np.array([pr]),
            "pad_bottom": np.array([pb]),
        },
    )
    return mod


def _stub_decode_env(monkeypatch, n, h, w, pr, pb, stdout, returncode=0):
    monkeypatch.setattr(
        _demo, "require_module", lambda name: _fake_h5py(n, h, w, pr, pb)
    )

    class _Proc:
        pass

    _Proc.returncode = returncode
    _Proc.stdout = stdout
    _Proc.stderr = b""
    monkeypatch.setattr(_demo.subprocess, "run", lambda *a, **k: _Proc())


def test_decode_rejects_a_truncated_stream(tmp_path, monkeypatch) -> None:
    """A short decode must raise, not yield a thinner (misregistered) volume."""
    n, h, w = 4, 6, 5
    # One frame short of the declared four.
    _stub_decode_env(monkeypatch, n, h, w, 0, 0, bytes((n - 1) * h * w))

    with pytest.raises(RuntimeError, match="decoded .* expected"):
        _demo.decode_h5j_channel(tmp_path / "fake.h5j", 3)


def test_decode_rejects_a_ragged_stream(tmp_path, monkeypatch) -> None:
    """Trailing partial-frame bytes must not be silently discarded either."""
    n, h, w = 3, 4, 4
    _stub_decode_env(monkeypatch, n, h, w, 0, 0, bytes(n * h * w + 7))

    with pytest.raises(RuntimeError, match="decoded .* expected"):
        _demo.decode_h5j_channel(tmp_path / "ragged.h5j", 3)


def test_decode_returns_cropped_volume_on_a_complete_stream(
    tmp_path, monkeypatch
) -> None:
    """Codec padding is cropped off; a complete stream returns (n, h, w)."""
    n, h, w, pr, pb = 3, 5, 4, 2, 1
    ew, eh = w + pr, h + pb
    payload = np.arange(n * eh * ew, dtype=np.uint8).tobytes()
    _stub_decode_env(monkeypatch, n, h, w, pr, pb, payload)

    vol = _demo.decode_h5j_channel(tmp_path / "ok.h5j", 3)

    assert vol.shape == (n, h, w)
    expected = np.frombuffer(payload, dtype=np.uint8).reshape(n, eh, ew)[:, :h, :w]
    assert np.array_equal(vol, expected)


def _import_demo_with_argv(argv):
    """Import a fresh copy of the demo under a given argv.

    ``SAMPLE`` is parsed at module scope, so proving the validation is actually
    *wired* to ``--sample`` (rather than merely existing as a helper) means
    re-importing the module with that argv in place.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_flylight_argv_probe", _DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = argv
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old
    return module


def test_bad_sample_argument_is_rejected_at_parse_time() -> None:
    with pytest.raises(ValueError, match="Invalid --sample"):
        _import_demo_with_argv(["demo", "--sample=../../escape"])


def test_good_sample_argument_is_accepted_at_parse_time() -> None:
    mod = _import_demo_with_argv(["demo", "--sample=R14A02-20180905_65_A6"])
    assert mod.SAMPLE == "R14A02-20180905_65_A6"


# ---------------------------------------------------------------------------
# Cache durability: never lose the last good copy
# ---------------------------------------------------------------------------


def test_failed_install_restores_the_previous_sample(tmp_path, monkeypatch) -> None:
    """If installing the new store fails, the old one must survive.

    Deleting the moved-aside copy in a `finally` would lose BOTH the new
    (incomplete) and the old (working) store, leaving no usable sample.
    """
    import zipfile

    sample = "SAMPLE_Y"
    cache = tmp_path / "cache"
    cache.mkdir()

    target = cache / f"{sample}.zarr"
    (target / "volumes").mkdir(parents=True)
    (target / "volumes" / "precious").write_bytes(b"the only good copy")

    archive = tmp_path / "a.zip"
    prefix = f"{_demo.SAMPLE_MEMBER_ROOT}/train/{sample}.zarr/"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}volumes/raw/.zarray", b"{}")

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", True)
    _stub_remote_zip(monkeypatch, archive)

    real_rename = Path.rename

    def _fail_installing_new(self, dest):
        # Fail only the final install; the move-aside must still work.
        if str(self).endswith(".zarr.partial"):
            raise OSError("simulated failure installing the new store")
        return real_rename(self, dest)

    monkeypatch.setattr(Path, "rename", _fail_installing_new)

    with pytest.raises(OSError, match="simulated failure"):
        _demo.fetch_sample(sample)

    assert target.exists(), "the previous sample was destroyed"
    assert (target / "volumes" / "precious").read_bytes() == b"the only good copy"
    assert not target.with_suffix(".zarr.stale").exists()


def test_h5j_and_npy_caches_are_published_atomically(tmp_path) -> None:
    """Caches land via a temporary sibling, so no partial file is ever trusted."""
    payload = b"x" * 1024
    dest = tmp_path / "thing.h5j"
    _demo._atomic_write(dest, payload)
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".h5j.part").exists()

    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    npy = tmp_path / "vol.npy"
    _demo._atomic_save_npy(npy, arr)
    assert np.array_equal(np.load(npy), arr)
    assert not npy.with_suffix(".npy.part").exists()


def test_atomic_write_leaves_no_partial_on_failure(tmp_path, monkeypatch) -> None:
    """A failed write publishes nothing AND strands no temporary sibling.

    The interesting failure is the one that dies part-way through (disk full):
    a stub that raises before creating anything passes even when the cleanup is
    missing entirely, so this one writes half the payload first.
    """
    dest = tmp_path / "broken.h5j"
    real_write_bytes = Path.write_bytes

    def _boom(self, data):
        real_write_bytes(self, data[: len(data) // 2])
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    with pytest.raises(OSError, match="disk full"):
        _demo._atomic_write(dest, b"payload" * 100)

    assert not dest.exists()
    assert not dest.with_suffix(".h5j.part").exists(), "partial left in the cache"


def test_atomic_save_npy_leaves_no_partial_on_failure(tmp_path, monkeypatch) -> None:
    """Same for the decoded volume, where the partial is hundreds of MB."""
    dest = tmp_path / "vol.npy"

    def _boom(fh, array):
        fh.write(b"partially written")
        raise OSError("disk full")

    monkeypatch.setattr(_demo.np, "save", _boom)

    with pytest.raises(OSError, match="disk full"):
        _demo._atomic_save_npy(dest, np.zeros(4, dtype=np.uint8))

    assert not dest.exists()
    assert not dest.with_suffix(".npy.part").exists(), "partial left in the cache"


# ---------------------------------------------------------------------------
# Optional-dependency fallback and parameter-keyed fit caches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["ffmpeg", "h5py"])
def test_neuropil_degrades_when_an_optional_dep_is_missing(
    tmp_path, monkeypatch, missing: str
) -> None:
    """Both neuropil dependencies must degrade to a neurons-only scene.

    The demo documents that fallback, so a missing optional dependency has to
    return None rather than raise out of the run.
    """
    # An empty cache, not the developer's own: on a machine that has actually
    # run this demo the real cache holds a decoded neuropil, which is returned
    # before the dependencies are ever consulted (see the warm-cache test).
    monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_demo, "RECOMPUTE", False)
    monkeypatch.setattr(
        _demo.shutil, "which", lambda name: None if name == "ffmpeg" else "/bin/x"
    )
    real_find_spec = _demo.importlib.util.find_spec
    monkeypatch.setattr(
        _demo.importlib.util,
        "find_spec",
        lambda name: None if name == "h5py" else real_find_spec(name),
    )
    if missing == "ffmpeg":
        monkeypatch.setattr(_demo.importlib.util, "find_spec", lambda name: object())
    else:
        monkeypatch.setattr(_demo.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    assert _demo.fetch_neuropil(_demo.DEFAULT_SAMPLE) is None


def test_matching_grids_produce_no_registration_warning() -> None:
    assert _demo.neuropil_grid_mismatch((390, 1058, 907), (390, 1058, 907)) is None


def test_mismatched_grids_are_reported() -> None:
    """Two independently released files on different grids cannot register.

    Both clouds are fitted in physical units from the origin, so a reference
    channel of a different shape lands off its own neurons — worth saying out
    loud, because it looks like a fitting artefact rather than wrong inputs.
    """
    warning = _demo.neuropil_grid_mismatch((389, 1058, 907), (390, 1058, 907))

    assert warning is not None
    assert "(389, 1058, 907)" in warning and "(390, 1058, 907)" in warning


def test_mismatched_grids_skip_the_neuropil(monkeypatch) -> None:
    """Detecting the mismatch is not enough — it has to stop the merge.

    Warning and then fitting anyway spends the neuropil fit to publish a brain
    sitting off its own neurons, which is the exact scene the warning says is
    wrong. A mismatch degrades like a missing dependency: neurons only.
    """
    ref = np.zeros((389, 8, 8), dtype=np.uint8)
    monkeypatch.setattr(_demo, "fetch_neuropil", lambda sample: ref)

    assert _demo.neuropil_reference(_demo.DEFAULT_SAMPLE, (390, 8, 8)) is None


def test_matching_grids_pass_the_reference_through(monkeypatch) -> None:
    """The registration check must not cost the neuropil when it registers."""
    ref = np.zeros((390, 8, 8), dtype=np.uint8)
    monkeypatch.setattr(_demo, "fetch_neuropil", lambda sample: ref)

    assert _demo.neuropil_reference(_demo.DEFAULT_SAMPLE, (390, 8, 8)) is ref


def test_unavailable_reference_stays_none(monkeypatch) -> None:
    """An absent reference channel must not trip the shape comparison."""
    monkeypatch.setattr(_demo, "fetch_neuropil", lambda sample: None)

    assert _demo.neuropil_reference(_demo.DEFAULT_SAMPLE, (390, 8, 8)) is None


def test_neuropil_skipped_for_an_unmapped_sample() -> None:
    assert _demo.fetch_neuropil("SOME_UNMAPPED_SAMPLE") is None


def test_fit_cache_key_changes_with_every_fit_parameter() -> None:
    """Editing a tuning constant must not silently reuse the old fit."""
    base = _demo.fit_cache_key(1_200_000, "p99", 0.999)

    assert _demo.fit_cache_key(600_000, "p99", 0.999) != base, "seeds ignored"
    assert _demo.fit_cache_key(1_200_000, "p98", 0.999) != base, "floor ignored"
    assert _demo.fit_cache_key(1_200_000, "p99", 0.95) != base, "retention ignored"
    assert _demo.fit_cache_key(1_200_000, "auto", 0.999) != base


def test_fit_cache_key_is_stable_for_identical_parameters() -> None:
    assert _demo.fit_cache_key(1_200_000, "p99", 0.999) == _demo.fit_cache_key(
        1_200_000, "p99", 0.999
    )


def test_fit_cache_key_tracks_the_optimiser_schedule() -> None:
    """The schedule changes splat SHAPES while leaving the count identical.

    That is what makes it easy to miss: seeds, floor and retention are all
    unchanged when the schedule is retuned, so a key blind to it hands back the
    old under-converged, beaded fit and the retune looks like a no-op.
    """
    args = (1_200_000, "auto", 0.9999)
    bare = _demo.fit_cache_key(*args)
    tuned = _demo.fit_cache_key(*args, _demo.NEURON_FIT_SCHEDULE)
    assert tuned != bare, "schedule ignored — a retuned fit would reuse the old file"

    for knob, other in (
        ("n_iters", 5_000),
        ("patience", 15),
        ("early_stop_patience", 300),
        ("enable_dynamic_ops", True),
        ("max_eccentricity", 10.0),
        ("l1_diag", 1e-4),
    ):
        variant = dict(_demo.NEURON_FIT_SCHEDULE)
        variant[knob] = other
        assert _demo.fit_cache_key(*args, variant) != tuned, f"{knob} ignored"


def test_fit_cache_key_ignores_schedule_dict_ordering() -> None:
    """Same knobs in a different insertion order are the same fit.

    Digesting ``str(dict)`` would key on insertion order and force a needless
    (very expensive) refit after a purely cosmetic edit.
    """
    args = (1_200_000, "auto", 0.9999)
    reordered = dict(reversed(list(_demo.NEURON_FIT_SCHEDULE.items())))
    assert list(reordered) != list(_demo.NEURON_FIT_SCHEDULE)
    assert _demo.fit_cache_key(*args, reordered) == _demo.fit_cache_key(
        *args, _demo.NEURON_FIT_SCHEDULE
    )


def test_neuron_schedule_overrides_the_defaults_that_cause_beading() -> None:
    """Guard every knob in the schedule, each of which was measured to matter.

    ``fit_gaussian_splats`` defaults to n_iters=1000 — below the CLI's own
    ``draft`` preset — which leaves splats at their isotropic seed shape and
    renders thin axons as bead chains. The two patience knobs are in here for a
    reason that is easy to overlook: at the defaults (15 and 300) the fit decays
    its shape learning rate and then stops long before iteration 10,000, so
    reverting either one quietly undoes most of what ``n_iters`` bought.
    """
    s = _demo.NEURON_FIT_SCHEDULE
    assert s["n_iters"] >= 5_000, "1000 (the API default) leaves splats at seed shape"
    assert s["patience"] >= 200, "15 decays the shape LR away before shapes settle"
    assert s["early_stop_patience"] >= 2_000, "300 stops the fit before convergence"
    assert s["enable_dynamic_ops"] is False, "relocation re-isotropises splats mid-fit"
    assert s["max_eccentricity"] is None, "the default 10.0 caps the axis ratio"
    assert s["l1_diag"] == 0.0, "the default penalty pulls shapes toward isotropy"


def test_every_schedule_knob_is_a_real_fitter_parameter() -> None:
    """A typo in the schedule would be SWALLOWED, not raised.

    ``fit_volume`` splats the schedule into ``fit_gaussian_splats``, which ends
    in ``**seed_kwargs`` and forwards anything it does not recognise on to the
    seeder — where an unknown key is a ``UserWarning``, not an error. Buried in
    a verbose multi-minute fit that reads as "the retune did nothing", which is
    the same silent failure the cache-key digest exists to prevent.
    """
    import inspect

    from luxar.gsplats import fit_gaussian_splats

    named = {
        name
        for name, p in inspect.signature(fit_gaussian_splats).parameters.items()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    unknown = sorted(set(_demo.NEURON_FIT_SCHEDULE) - named)
    assert not unknown, f"not fit_gaussian_splats parameters: {unknown}"


def test_fit_volume_forwards_the_schedule_to_the_fitter(tmp_path, monkeypatch) -> None:
    """A schedule that never reaches the fitter would be silent and useless.

    The acquisition dtype travels the same route and fails the same way: both
    components arrive here already widened to float32, so a declaration that
    stops short of the fitter leaves the recorded provenance describing the
    working copy and overstating compression by the cast.
    """
    seen: dict = {}

    class _Result:
        amplitudes = np.zeros(3, dtype=np.float32)

        def save(self, *a, **k):
            seen["saved"] = True

    def _fake_fit(volume, **kwargs):
        seen.update(kwargs)
        return _Result()

    import luxar.gsplats as _gs

    monkeypatch.setattr(_gs, "fit_gaussian_splats", _fake_fit)
    monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)

    _demo.fit_volume(
        np.zeros((4, 4, 4), dtype=np.float32),
        tmp_path / "absent.gsplats.zarr.zip",
        1000,
        "auto",
        0.99,
        "neurons",
        schedule=_demo.NEURON_FIT_SCHEDULE,
        source_dtype="uint16",
    )

    for knob, value in _demo.NEURON_FIT_SCHEDULE.items():
        assert seen[knob] == value, f"{knob} never reached fit_gaussian_splats"
    assert seen["source_dtype"] == "uint16", "the acquisition dtype was dropped"


def test_fit_volume_without_a_schedule_passes_no_overrides(
    tmp_path, monkeypatch
) -> None:
    """The neuropil is deliberately left on the library defaults."""
    seen: dict = {}

    class _Result:
        amplitudes = np.zeros(3, dtype=np.float32)

        def save(self, *a, **k):
            pass

    def _fake_fit(volume, **kwargs):
        seen.update(kwargs)
        return _Result()

    import luxar.gsplats as _gs

    monkeypatch.setattr(_gs, "fit_gaussian_splats", _fake_fit)
    monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)

    _demo.fit_volume(
        np.zeros((4, 4, 4), dtype=np.float32),
        tmp_path / "absent2.gsplats.zarr.zip",
        1000,
        "auto",
        0.95,
        "neuropil",
    )

    for knob in _demo.NEURON_FIT_SCHEDULE:
        assert knob not in seen, f"{knob} leaked into the neuropil fit"


def test_fit_cache_paths_differ_between_components_and_settings() -> None:
    neurons = _demo._fit_cache_path("neurons", 1_200_000, "p99", 0.999)
    neuropil = _demo._fit_cache_path("neuropil", 600_000, "auto", 0.95)
    retuned = _demo._fit_cache_path("neurons", 1_200_000, "p99", 0.95)

    assert neurons != neuropil
    assert neurons != retuned, "a retuned fit must not reuse the old cache file"
    assert neurons.name.startswith(f"{_demo.SAMPLE}_neurons_")
    assert neurons.suffixes[-2:] == [".zarr", ".zip"]


def test_warm_neuropil_cache_is_used_without_either_dependency(
    tmp_path, monkeypatch
) -> None:
    """A decoded .npy needs neither ffmpeg nor h5py, so it must still load.

    Gating the dependencies first would discard a perfectly good warm cache and
    silently drop the scene to neurons-only.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    sample = _demo.DEFAULT_SAMPLE
    vol = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    np.save(cache / f"{sample}_neuropil.npy", vol)

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", False)
    # Both decoding dependencies absent.
    monkeypatch.setattr(_demo.shutil, "which", lambda name: None)
    monkeypatch.setattr(_demo.importlib.util, "find_spec", lambda name: None)

    got = _demo.fetch_neuropil(sample)

    assert got is not None, "warm cache ignored when dependencies are missing"
    assert np.array_equal(got, vol)


def test_recompute_still_needs_the_dependencies(tmp_path, monkeypatch) -> None:
    """--recompute must re-decode, so it degrades when the deps are missing."""
    cache = tmp_path / "cache"
    cache.mkdir()
    sample = _demo.DEFAULT_SAMPLE
    np.save(cache / f"{sample}_neuropil.npy", np.zeros((2, 2, 2), dtype=np.uint8))

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", True)
    monkeypatch.setattr(_demo.shutil, "which", lambda name: None)
    monkeypatch.setattr(_demo.importlib.util, "find_spec", lambda name: None)

    assert _demo.fetch_neuropil(sample) is None


@pytest.mark.parametrize(
    "spec,expected", [("sssr", 3), ("ssr", 2), ("sr", 1), ("rsss", 0)]
)
def test_reference_channel_index_is_read_from_the_file(
    tmp_path, monkeypatch, spec: str, expected: int
) -> None:
    """The neuropil channel must be located by channel_spec, not hard-coded.

    A sample whose spec is not 'sssr' would otherwise have a *signal* channel
    decoded as the neuropil — a plausible-looking but wrong scene.
    """
    monkeypatch.setattr(
        _demo, "require_module", lambda name: _fake_h5py(1, 1, 1, 0, 0, spec=spec)
    )
    assert _demo.reference_channel_index(tmp_path / "x.h5j") == expected


def test_reference_channel_index_rejects_a_spec_without_one(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        _demo, "require_module", lambda name: _fake_h5py(1, 1, 1, 0, 0, spec="sss")
    )
    with pytest.raises(RuntimeError, match="no reference"):
        _demo.reference_channel_index(tmp_path / "x.h5j")


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_sample_is_found_in_any_split(split: str) -> None:
    """All 30 completely-labelled samples must be reachable, not just train's.

    The archive splits them 18/5/7 across train/val/test; pinning one split
    silently made 12 of the 30 unusable via --sample.
    """
    names = [
        f"{_demo.SAMPLE_MEMBER_ROOT}/{split}/SAMPLE_A.zarr/volumes/raw/.zarray",
        f"{_demo.SAMPLE_MEMBER_ROOT}/{split}/SAMPLE_A.zarr/volumes/raw/0.0.0.0",
    ]
    got = _demo.find_member_prefix(names, "SAMPLE_A")
    assert got == f"{_demo.SAMPLE_MEMBER_ROOT}/{split}/SAMPLE_A.zarr/"


def test_sample_lookup_does_not_match_a_longer_name() -> None:
    """`ABC` must not match `XYZ_ABC.zarr` — the prefix is anchored."""
    names = [f"{_demo.SAMPLE_MEMBER_ROOT}/train/XYZ_ABC.zarr/volumes/raw/.zarray"]
    assert _demo.find_member_prefix(names, "ABC") is None
    assert _demo.find_member_prefix(names, "XYZ_ABC") is not None


def test_sample_lookup_returns_none_when_absent() -> None:
    names = [f"{_demo.SAMPLE_MEMBER_ROOT}/train/OTHER.zarr/volumes/raw/.zarray"]
    assert _demo.find_member_prefix(names, "MISSING") is None


# ---------------------------------------------------------------------------
# Alpha ramp, per-specimen tilt, and camera roll
# ---------------------------------------------------------------------------


def test_alpha_ramp_spans_min_to_one_and_is_monotone() -> None:
    """Faint splats go transparent, bright ones opaque — that is the ramp."""
    amp = np.linspace(0.0, 1.0, 1001).astype(np.float32)
    a = _demo.neuron_alpha(amp)

    assert np.isclose(a.min(), _demo.NEURON_ALPHA_MIN) and a.max() == 1.0
    assert a.shape == amp.shape
    assert np.all(np.diff(a) >= -1e-7), "a brighter splat is never more transparent"


def test_alpha_ramp_never_reaches_zero() -> None:
    """A zero-alpha splat is discarded in the shader, display range or not.

    The suppressed splats are only worth fitting because they can be brought
    back; clipping the bottom of the ramp to 0 makes them as unrecoverable as a
    floor would have, which is the entire argument for doing this at render
    time.
    """
    assert 0.0 < _demo.NEURON_ALPHA_MIN < 0.1
    rng = np.random.default_rng(0)
    amp = np.concatenate([rng.exponential(0.002, 9000), rng.exponential(0.2, 1000)])
    a = _demo.neuron_alpha(amp.astype(np.float32))
    assert a.min() > 0.0, "the faint splats must still emit something"


def test_alpha_ramp_makes_the_bulk_transparent() -> None:
    """With the configured percentiles most splats end near-invisible.

    That is the mechanism replacing the floor: suppress background at render
    time instead of deleting it before the fit.
    """
    rng = np.random.default_rng(0)
    amp = np.concatenate([rng.exponential(0.002, 9000), rng.exponential(0.2, 1000)])
    a = _demo.neuron_alpha(amp.astype(np.float32))
    assert (a < 0.05).mean() > 0.8


def test_alpha_ramp_handles_empty_and_constant_input() -> None:
    assert _demo.neuron_alpha(np.zeros(0, dtype=np.float32)).shape == (0,)
    assert np.all(np.isfinite(_demo.neuron_alpha(np.full(50, 0.3, dtype=np.float32))))


@pytest.mark.parametrize("deg", [0.0, 30.0, -52.0, 75.0])
def test_measure_tilt_recovers_a_known_rotation(deg: float) -> None:
    """The tilt must be measured from the data, not assumed.

    FISBe ships the unaligned stack, so mounting angle differs per specimen; a
    hard-coded angle would be wrong for every --sample but one.
    """
    rng = np.random.default_rng(1)
    t = np.radians(deg)
    long_axis = rng.normal(0, 100, 4000)
    short_axis = rng.normal(0, 3, 4000)
    y = long_axis * np.sin(t) + short_axis * np.cos(t)
    x = long_axis * np.cos(t) - short_axis * np.sin(t)
    centers = np.stack([rng.normal(0, 5, 4000), y, x], axis=1).astype(np.float32)

    got = _demo.measure_tilt_deg(centers, np.ones(4000, dtype=np.float32))
    diff = (got - deg + 90) % 180 - 90  # a principal axis is defined up to 180 deg
    assert abs(diff) < 3.0, f"measured {got:.1f}, expected {deg}"


def test_measure_tilt_is_degenerate_safe() -> None:
    z3 = np.zeros((1, 3), np.float32)
    assert _demo.measure_tilt_deg(z3, np.ones(1, np.float32)) == 0.0
    assert (
        _demo.measure_tilt_deg(np.zeros((0, 3), np.float32), np.zeros(0, np.float32))
        == 0.0
    )
    assert (
        _demo.measure_tilt_deg(np.ones((10, 3), np.float32), np.zeros(10, np.float32))
        == 0.0
    )


def _tilted_cloud(deg: float, n: int = 600):
    """A thin (Z, Y, X) cloud whose long axis sits at ``deg`` in the Y/X plane."""
    rng = np.random.default_rng(2)
    t = np.radians(deg)
    long_axis = rng.normal(0, 100, n)
    short_axis = rng.normal(0, 3, n)
    y = long_axis * np.sin(t) + short_axis * np.cos(t)
    x = long_axis * np.cos(t) - short_axis * np.sin(t)
    centers = np.stack([rng.normal(0, 5, n), y, x], axis=1).astype(np.float32)
    return centers, np.full(n, 0.5, dtype=np.float32)


@pytest.mark.parametrize("deg", [30.0, -52.0, 75.0])
def test_authored_camera_up_is_perpendicular_to_the_specimen_axis(
    tmp_path, deg: float
) -> None:
    """The roll must put the long axis horizontal, not merely rotate it.

    `(sin, cos)` instead of `(-sin, cos)` rolls the wrong way: still diagonal,
    just mirrored, which looks plausible enough to ship. Read back out of the
    authored scene, so the check covers the measurement, the sign and the
    serialization rather than restating the formula.
    """
    import zarr as _zarr

    centers, amps = _tilted_cloud(deg)
    chol = np.tile([1, 0, 1, 0, 0, 1], (len(centers), 1)).astype(np.float32)
    rgba = np.ones((len(centers), 4), dtype=np.float32)

    out = _demo.create_luxar_scene(
        centers, amps, chol, rgba, tmp_path / f"tilt{deg}.luxar.zarr"
    )
    cam = dict(_zarr.open_group(str(out), mode="r").attrs)["viewer_config"]["camera"]

    # The scene's dimensions are (x, y, z), so the up vector's first two
    # components are the world x/y the specimen's long axis lives in.
    up = np.asarray(cam["up"], dtype=np.float64)
    t = np.radians(deg)
    long_axis = np.array([np.cos(t), np.sin(t)])
    assert abs(float(np.dot(up[:2], long_axis))) < 0.06, f"up={up} rolls the wrong way"
    assert up[2] == 0.0
    assert np.isclose(np.linalg.norm(up), 1.0)
    # Framed from +z at a measured distance, looking at the recentred cloud.
    assert cam["position"][0] == cam["position"][1] == 0.0
    assert cam["position"][2] > 0.0
    assert cam["target"] == [0.0, 0.0, 0.0]
    assert "fov" not in cam
