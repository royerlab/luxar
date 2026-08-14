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
    assert np.allclose(rgba[:, 3], 1.0), "neurons must be fully opaque"


def test_merge_concatenates_and_tags_alpha() -> None:
    neurons, neuropil = _Splats(4, 1), _Splats(7, 2)
    rgb = np.tile([1.0, 1.0, 1.0], (4, 1)).astype(np.float32)

    centers, amps, chol, rgba = merge_for_render(neurons, rgb, neuropil)

    assert len(centers) == len(amps) == len(chol) == len(rgba) == 11
    # Neuropil is emitted first, so the split is at len(neuropil).
    assert np.allclose(rgba[:7, 3], _demo.NEUROPIL_ALPHA)
    assert np.allclose(rgba[7:, 3], 1.0)
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


def _fake_h5py(n, h, w, pr, pb):
    """A minimal h5py stand-in, so these tests run without h5py installed."""
    import types

    mod = types.SimpleNamespace()
    mod.File = lambda path, mode: _FakeH5File(
        {"channel_spec": b"sssr"},
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
    prefix = f"{_demo.SAMPLE_MEMBER_PREFIX}/{sample}.zarr/"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}volumes/raw/.zarray", b"{}")

    monkeypatch.setattr(_demo, "CACHE_DIR", cache)
    monkeypatch.setattr(_demo, "RECOMPUTE", True)
    monkeypatch.setattr(
        _demo, "_open_remote_zip", lambda url: (zipfile.ZipFile(archive), None)
    )

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
    """A failed write must not publish anything at the destination."""
    dest = tmp_path / "broken.h5j"

    def _boom(self, data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    with pytest.raises(OSError, match="disk full"):
        _demo._atomic_write(dest, b"payload")

    assert not dest.exists()


# ---------------------------------------------------------------------------
# Optional-dependency fallback and parameter-keyed fit caches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["ffmpeg", "h5py"])
def test_neuropil_degrades_when_an_optional_dep_is_missing(
    monkeypatch, missing: str
) -> None:
    """Both neuropil dependencies must degrade to a neurons-only scene.

    The demo documents that fallback, so a missing optional dependency has to
    return None rather than raise out of the run.
    """
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
