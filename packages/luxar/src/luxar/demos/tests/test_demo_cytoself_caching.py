"""Tests for the CytoSelf demo's download staging and thumbnail resumability.

These exercise the real code paths in ``demo_cytoself_protein_landscape`` with a
fake ``requests.Session`` and a fake Google-Drive fetch, so nothing touches the
network. They cover the download contract (identity encoding, ``.part`` staging,
HTML-quota-page rejection, content-length verification, the completion sidecar
and the legacy no-sidecar fallback), the consumer-side self-heal, and the
thumbnail pipeline (per-source-file part caches keyed to the row mapping, a
quarantined-and-rebuilt bundle, legacy-bundle adoption, and the match-rate
tripwire).

The fake response models content NEGOTIATION, not just a body: a client that
accepts gzip is served a compressed byte count in ``Content-Length`` while
``iter_content`` hands back the decoded bytes, exactly as ``requests`` behaves.
That is the only way a test can see the difference the identity header makes.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import types
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest

# The demo imports ``luxar.utils._umap_utils`` at module scope, which imports
# Pillow; Pillow ships in the ``demos`` extra, not core.
pytest.importorskip("PIL")

import requests  # noqa: E402

from luxar.demos import demo_cytoself_protein_landscape as demo  # noqa: E402

# =============================================================================
# Fake HTTP plumbing
# =============================================================================


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response.

    ``iter_content`` deliberately ignores the caller's ``chunk_size`` and yields
    small chunks so a mid-stream failure can be simulated on a small body. It
    also yields the DECODED body: ``content_length`` is set independently so a
    gzipped response (compressed length, inflated body) can be modelled.
    """

    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        content_length: int | None = -1,
        content_encoding: str | None = None,
        fail_after_chunks: int | None = None,
        on_fail: Callable[[], None] | None = None,
        chunk: int = 4096,
    ) -> None:
        self._body = body
        self._chunk = chunk
        self._fail_after_chunks = fail_after_chunks
        self._on_fail = on_fail
        self.headers: dict[str, str] = {"content-type": content_type}
        declared = len(body) if content_length == -1 else content_length
        if declared is not None:
            self.headers["content-length"] = str(declared)
        if content_encoding is not None:
            self.headers["content-encoding"] = content_encoding
        self.cookies: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size: int = 1 << 20):
        for n, start in enumerate(range(0, len(self._body), self._chunk)):
            if self._fail_after_chunks is not None and n >= self._fail_after_chunks:
                if self._on_fail is not None:
                    # Inspect the filesystem AT the moment of failure, which is
                    # the only way to see where the bytes were being written.
                    self._on_fail()
                raise ConnectionError("connection reset mid-stream")
            yield self._body[start : start + self._chunk]


class _SessionRecorder:
    """What the production code configured on, and asked of, its session."""

    def __init__(self) -> None:
        # Seeded with the header a real ``requests.Session`` carries by default,
        # so a test can tell an override from an absence.
        self.headers: dict[str, str] = {"Accept-Encoding": "gzip, deflate"}
        self.urls: list[str] = []


_Factory = Callable[[str, Mapping[str, str]], _FakeResponse]


def _install_session(
    monkeypatch: pytest.MonkeyPatch, factory: _Factory
) -> _SessionRecorder:
    """Patch ``requests.Session``; return a recorder of headers and URLs."""
    recorder = _SessionRecorder()

    class _Session:
        def __init__(self) -> None:
            self.headers = recorder.headers

        def get(
            self,
            url: str,
            params: dict | None = None,
            stream: bool = False,
            timeout: int | None = None,
        ) -> _FakeResponse:
            recorder.urls.append(url)
            return factory(url, recorder.headers)

    monkeypatch.setattr(requests, "Session", _Session)
    return recorder


def _serve(body: bytes, **kwargs: object) -> _Factory:
    """A factory serving one fixed body, ignoring the request headers."""
    return lambda url, headers: _FakeResponse(body, **kwargs)  # type: ignore[arg-type]


def _serve_negotiated(body: bytes) -> _Factory:
    """Serve *body* the way a real server and ``requests`` pair up.

    A client that accepts gzip gets a COMPRESSED ``Content-Length`` while
    ``requests`` transparently inflates the stream; a client that asks for
    identity gets bytes and length that agree.
    """

    def factory(url: str, headers: Mapping[str, str]) -> _FakeResponse:
        if "gzip" in headers.get("Accept-Encoding", ""):
            return _FakeResponse(
                body,
                content_type="text/csv",
                content_length=len(gzip.compress(body)),
                content_encoding="gzip",
            )
        return _FakeResponse(body, content_type="text/csv")

    return factory


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to open an HTTP session fail loudly."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(requests, "Session", _boom)


@pytest.fixture(autouse=True)
def _never_touch_the_real_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the module's default cache dir at ``tmp_path`` for every test.

    A complete cytoself cache is ~186 GB of real data. One forgotten
    ``cache_dir=`` kwarg would have the suite reading — or quarantining — it.
    """
    monkeypatch.setattr(demo, "DEFAULT_CACHE_DIR", tmp_path / "default_cache")


def _leftovers(dest: Path) -> set[str]:
    """Everything in the cache dir other than *dest* itself.

    Scans the directory rather than probing fixed temp names, so staging files
    stay covered whatever they are called.
    """
    return {p.name for p in dest.parent.iterdir()} - {dest.name}


# =============================================================================
# Download staging, verification and the completion sidecar
# =============================================================================


def test_gzipped_response_is_not_mistaken_for_a_truncated_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A gzipped text/csv: Content-Length counts the compressed bytes while the
    # stream delivers the inflated ones. Only an identity request makes the two
    # comparable — without it a complete download looks short and is destroyed.
    body = b"ensg,name\nENSG00000141510,TP53\n" * 15_000
    assert len(gzip.compress(body)) < len(body) // 10
    recorder = _install_session(monkeypatch, _serve_negotiated(body))
    dest = tmp_path / "label.csv"

    demo._download_from_google_drive("ABC123", dest, expected_min_size=100_000)

    assert recorder.headers["Accept-Encoding"] == "identity"
    assert dest.read_bytes() == body
    assert json.loads(demo._sidecar_path(dest).read_text(encoding="utf-8"))[
        "size"
    ] == len(body)


# Exact byte counts obtained by HTTP Range probes of the actual Google Drive
# files, so these are the sizes to calibrate against — not `du` on a cache that
# may itself be incomplete. For the two families the SMALLEST member is what
# matters: a floor above it rejects a genuine file and bricks the demo.
MEASURED_MIN_BYTES = {
    "MIN_SIZE_EMBEDDINGS": 4_232_208_512,  # Global_representation.npy
    "MIN_SIZE_LABELS_CSV": 6_553_361,  # label.csv
    "MIN_SIZE_LABEL_DATA_CSV": 3_950_974,  # smallest of Label_data00..09
    "MIN_SIZE_IMAGE_DATA": 11_282_720_128,  # smallest of Image_data00..09
}


@pytest.mark.parametrize("name", sorted(MEASURED_MIN_BYTES))
def test_size_floors_match_the_measured_artifacts(name: str) -> None:
    """Each floor is calibrated against the real smallest file of its kind.

    Two bars at once, because both directions are harmful. After the ``* 0.9``
    slack the bar must sit BELOW the smallest real file — a floor above it would
    reject the genuine artifact forever — and it must sit high enough that a
    large fragment is still refused. The original 400 MB guess for the
    Image_data files failed the second bar by ~40x; a 4 MB floor for the
    Label_data CSVs failed the first, with only the 0.9 slack in between.
    """
    measured = MEASURED_MIN_BYTES[name]
    bar = getattr(demo, name) * 0.9

    assert bar < measured, f"{name} would reject the real {measured:,}-byte file"
    margin = 1.0 - bar / measured
    assert margin >= 0.10, f"{name} leaves only {margin:.1%} headroom"
    assert bar >= 0.5 * measured, f"{name} accepts under half a real file"


def test_atomic_writes_do_not_disturb_another_runs_staging_file(
    tmp_path: Path,
) -> None:
    # Two runs assembling the same bundle must not rename each other's bytes
    # into place: the loser's result is a perfectly VALID npz with the wrong
    # blob count, which nothing downstream can detect.
    target = tmp_path / demo.THUMBNAIL_CACHE_NAME
    foreign_bundle = tmp_path / (target.name + ".tmp")
    foreign_bundle.write_bytes(b"another run's half-written bundle")

    demo._write_npz_atomic(target, blobs=np.array([b"ours"], dtype=object))

    assert foreign_bundle.read_bytes() == b"another run's half-written bundle"
    with np.load(target, allow_pickle=True) as data:
        assert [bytes(b) for b in data["blobs"]] == [b"ours"]

    # Same shape, same treatment, for the completion sidecar.
    dest = tmp_path / "Image_data00.npy"
    dest.write_bytes(b"x" * 16)
    foreign_sidecar = tmp_path / (demo._sidecar_path(dest).name + ".tmp")
    foreign_sidecar.write_bytes(b"another run's sidecar")

    demo._write_completion_sidecar(dest, 16)

    assert foreign_sidecar.read_bytes() == b"another run's sidecar"
    assert (
        json.loads(demo._sidecar_path(dest).read_text(encoding="utf-8"))["size"] == 16
    )


def test_dead_runs_npz_staging_files_are_reclaimed(tmp_path: Path) -> None:
    # A killed `np.savez` on the ~114k-blob bundle strands a large `.tmp`, and a
    # PID-private name means nothing would ever look at it again.
    target = tmp_path / demo.THUMBNAIL_CACHE_NAME
    dead = tmp_path / f"{target.name}.999999{demo.TMP_SUFFIX}"
    dead.write_bytes(b"a half-written bundle from a run that is long gone")
    assert not demo._pid_is_alive(999999)

    demo._write_npz_atomic(target, blobs=np.array([b"ours"], dtype=object))

    assert not dead.exists()
    assert target.exists()


def test_interrupted_stream_leaves_nothing_at_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"\x93NUMPY" + b"\x11" * 40_000
    dest = tmp_path / "Image_data00.npy"

    def _check_mid_stream() -> None:
        # The criterion is about a SIGKILL, which runs no cleanup handler, so
        # asserting only on the post-hoc state would also pass an implementation
        # that writes to the destination and unlinks it on the way out.
        assert not dest.exists(), "partial bytes must never sit at the destination"
        assert demo._part_path(dest).exists(), "bytes must be going to the .part"

    _install_session(
        monkeypatch, _serve(body, fail_after_chunks=2, on_fail=_check_mid_stream)
    )

    with pytest.raises(ConnectionError):
        demo._download_from_google_drive("FILEID", dest, expected_min_size=10_000)

    assert not dest.exists()
    assert not demo._sidecar_path(dest).exists()

    # A healthy retry must succeed despite whatever the failed attempt staged.
    _install_session(monkeypatch, _serve(body))
    demo._download_from_google_drive("FILEID", dest, expected_min_size=10_000)

    assert dest.read_bytes() == body
    assert _leftovers(dest) == {dest.name + demo.COMPLETE_SUFFIX}


def test_google_drive_quota_html_page_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real quota page is 2-3 KB of HTML: it clears the old ``< 1_000`` floor.
    page = (
        b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded"
        b"</title></head><body>" + b"sorry. " * 350 + b"</body></html>"
    )
    assert 1_000 < len(page) < 5_000
    recorder = _install_session(
        monkeypatch, _serve(page, content_type="text/html; charset=utf-8")
    )
    dest = tmp_path / "Global_representation.npy"

    with pytest.raises(RuntimeError) as excinfo:
        demo._download_from_google_drive("ABC123", dest, expected_min_size=1_000_000)

    message = str(excinfo.value)
    assert "https://drive.google.com/file/d/ABC123/view?usp=sharing" in message
    assert str(dest) in message
    # The confirmation dance ran (and still ended in a rejection).
    assert any("drive.usercontent.google.com" in url for url in recorder.urls)
    assert not dest.exists()
    assert _leftovers(dest) == set()


def test_small_non_html_body_hits_the_absolute_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pins the branch ordering: this body is binary, so the HTML sniff must miss
    # it and the size floor must be what rejects it.
    _install_session(monkeypatch, _serve(b"\x00\x01\x02" * 166))
    dest = tmp_path / "label.csv"

    with pytest.raises(RuntimeError, match="too small"):
        demo._download_from_google_drive("ABC123", dest, expected_min_size=1_000_000)

    assert not dest.exists()
    assert _leftovers(dest) == set()


def test_content_length_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"\x93NUMPY" + b"\x00" * 8_000
    _install_session(monkeypatch, _serve(body, content_length=len(body) + 4_096))
    dest = tmp_path / "label.csv"

    with pytest.raises(RuntimeError, match="incomplete"):
        demo._download_from_google_drive("ABC123", dest, expected_min_size=1_000)

    assert not dest.exists()
    assert _leftovers(dest) == set()


def test_short_body_without_content_length_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No declared length means the equality check cannot run. Promoting the file
    # anyway would stamp a sidecar certifying the truncation as complete, which
    # is worse than the heuristic it replaced — so the floor has to catch it.
    # The margin mirrors production: a stream cut at half the expected size.
    body = b"\x93NUMPY" + b"\x00" * 5_000_000
    _install_session(monkeypatch, _serve(body, content_length=None))
    dest = tmp_path / "Global_representation.npy"

    with pytest.raises(RuntimeError, match="too short"):
        demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000_000)

    assert not dest.exists()
    assert _leftovers(dest) == set()


def test_short_body_with_an_honest_content_length_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drive also serves small non-HTML "cannot access this file" bodies. They
    # agree with their own Content-Length and clear the 1 KB absolute floor, so
    # only the expected-size floor stands between them and a sidecar certifying
    # 2 KB as a complete 11 GB artifact.
    body = b"Sorry, you cannot access this file at this time.\n" * 43
    assert demo.MIN_DOWNLOAD_BYTES < len(body) < 4_000
    assert not demo._looks_like_html(body[:512])
    _install_session(monkeypatch, _serve(body, content_type="text/plain"))
    dest = tmp_path / "Image_data00.npy"

    with pytest.raises(RuntimeError, match="too short"):
        demo._download_from_google_drive(
            "ABC123", dest, expected_min_size=demo.MIN_SIZE_IMAGE_DATA
        )

    assert not dest.exists()
    assert _leftovers(dest) == set()


def test_body_without_content_length_above_the_floor_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The companion of the test above: a chunked host that declares nothing is
    # still a legitimate source when the bytes clear the expected floor.
    body = b"\x93NUMPY" + b"\x00" * 40_000
    _install_session(monkeypatch, _serve(body, content_length=None))
    dest = tmp_path / "Global_representation.npy"

    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    assert dest.read_bytes() == body
    assert _leftovers(dest) == {dest.name + demo.COMPLETE_SUFFIX}


def test_happy_path_writes_sidecar_and_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"ensg,name\n" + b"ENSG,PROT\n" * 500
    _install_session(monkeypatch, _serve(body))
    dest = tmp_path / "label.csv"

    returned = demo._download_from_google_drive("ABC123", dest, expected_min_size=1_000)

    assert returned == dest
    assert dest.read_bytes() == body
    assert _leftovers(dest) == {dest.name + demo.COMPLETE_SUFFIX}
    sidecar = demo._sidecar_path(dest)
    assert json.loads(sidecar.read_text(encoding="utf-8"))["size"] == len(body)

    _forbid_network(monkeypatch)
    demo._download_from_google_drive("ABC123", dest, expected_min_size=1_000)
    assert dest.read_bytes() == body


def test_truncation_behind_the_sidecar_forces_a_refetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"\x93NUMPY" + b"\xab" * 20_000
    _install_session(monkeypatch, _serve(body))
    dest = tmp_path / "Image_data00.npy"
    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    # Truncate by well under 10%: the legacy size heuristic would wave it
    # through, but the sidecar no longer agrees with the file.
    dest.write_bytes(body[: len(body) - 500])
    assert dest.stat().st_size > 10_000 * 0.9

    fresh = b"\x93NUMPY" + b"\xcd" * 20_000
    _install_session(monkeypatch, _serve(fresh))
    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    assert dest.read_bytes() == fresh
    assert json.loads(demo._sidecar_path(dest).read_text(encoding="utf-8"))[
        "size"
    ] == len(fresh)


def test_a_failed_refetch_does_not_launder_the_truncated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sidecar is the only evidence the cached file is bad. If it were
    # cleared before the replacement existed, an offline retry would leave a
    # truncated file with no sidecar — which the legacy heuristic then trusts.
    body = b"\x93NUMPY" + b"\xab" * 20_000
    _install_session(monkeypatch, _serve(body))
    dest = tmp_path / "label.csv"
    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)
    dest.write_bytes(body[: len(body) - 500])

    def _offline(url: str, headers: Mapping[str, str]) -> _FakeResponse:
        raise ConnectionError("offline")

    _install_session(monkeypatch, _offline)
    with pytest.raises(ConnectionError):
        demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    assert demo._sidecar_path(dest).exists()
    assert demo._cached_file_is_complete(dest, 10_000) is False

    fresh = b"\x93NUMPY" + b"\xcd" * 20_000
    _install_session(monkeypatch, _serve(fresh))
    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)
    assert dest.read_bytes() == fresh


def test_unreadable_sidecar_forces_a_refetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"\x93NUMPY" + b"\xab" * 20_000
    _install_session(monkeypatch, _serve(body))
    dest = tmp_path / "Image_data00.npy"
    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    demo._sidecar_path(dest).write_text("not json", encoding="utf-8")

    fresh = b"\x93NUMPY" + b"\xcd" * 20_000
    _install_session(monkeypatch, _serve(fresh))
    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    assert dest.read_bytes() == fresh


def test_staging_does_not_touch_another_runs_in_flight_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cache dir is shared between runs, so the staging name is keyed on the
    # PID and each run deletes only its own. The foreign file is placed at the
    # name a FIXED-name implementation would stage into, which is exactly the
    # file such an implementation unlinks on the way in.
    body = b"\x93NUMPY" + b"\xab" * 20_000
    _install_session(monkeypatch, _serve(body))
    dest = tmp_path / "Image_data00.npy"
    foreign = tmp_path / (dest.name + demo.PART_SUFFIX)
    foreign.write_bytes(b"another run's in-flight bytes")

    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    assert dest.read_bytes() == body
    assert foreign.read_bytes() == b"another run's in-flight bytes"
    assert _leftovers(dest) == {foreign.name, dest.name + demo.COMPLETE_SUFFIX}


def test_staging_files_of_dead_runs_are_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Process-private names would otherwise strand a multi-gigabyte fragment on
    # every Ctrl-C, since nothing else ever looks at them again.
    body = b"\x93NUMPY" + b"\xab" * 20_000
    _install_session(monkeypatch, _serve(body))
    dest = tmp_path / "Image_data00.npy"

    dead = tmp_path / f"{dest.name}.999999{demo.PART_SUFFIX}"
    dead.write_bytes(b"9 GB of abandoned bytes, in spirit")
    live = tmp_path / f"{dest.name}.{os.getpid()}x{demo.PART_SUFFIX}"
    live.write_bytes(b"not a pid-shaped name; not ours to delete")
    other_file = tmp_path / f"Image_data01.npy.999999{demo.PART_SUFFIX}"
    other_file.write_bytes(b"another destination's staging file")
    assert not demo._pid_is_alive(999999)

    demo._download_from_google_drive("ABC123", dest, expected_min_size=10_000)

    assert not dead.exists()
    assert live.exists()
    assert other_file.exists()


def test_legacy_cache_without_sidecar_is_still_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cache written before staging existed has no sidecar; it must not force a
    # multi-gigabyte re-download.
    dest = tmp_path / "Global_representation.npy"
    dest.write_bytes(b"x" * 2_000)
    _forbid_network(monkeypatch)

    demo._download_from_google_drive("ABC123", dest, expected_min_size=1_000)

    assert dest.read_bytes() == b"x" * 2_000
    assert not demo._sidecar_path(dest).exists()


# =============================================================================
# Consumer-side self-heal
# =============================================================================


def test_unreadable_artifact_is_quarantined_and_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "Global_representation.npy"
    path.write_bytes(b"truncated garbage")
    good = np.arange(12, dtype=np.float32)
    fetched: list[str] = []

    def fake_download(
        file_id: str, output_path: Path, expected_min_size: int = 0
    ) -> Path:
        fetched.append(file_id)
        np.save(output_path, good)
        return output_path

    monkeypatch.setattr(demo, "_download_from_google_drive", fake_download)

    loaded = demo._load_downloaded_artifact(
        path, np.load, demo.GDRIVE_EMBEDDINGS_ID, expected_min_size=4_000_000_000
    )

    assert np.array_equal(loaded, good)
    assert fetched == [demo.GDRIVE_EMBEDDINGS_ID]
    assert (tmp_path / "Global_representation.npy.corrupt").exists()


def test_memory_error_is_not_treated_as_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The embeddings need ~16 GB of RAM. Quarantining a healthy 3.9 GB cache and
    # re-downloading it would fail identically, at the cost of the bytes twice.
    path = tmp_path / "Global_representation.npy"
    path.write_bytes(b"a perfectly good file we simply cannot fit in RAM")

    def _oom(_path: Path) -> np.ndarray:
        raise MemoryError("Unable to allocate 3.94 GiB")

    monkeypatch.setattr(
        demo,
        "_download_from_google_drive",
        lambda *a, **k: pytest.fail("must not re-download on MemoryError"),
    )

    with pytest.raises(MemoryError):
        demo._load_downloaded_artifact(path, _oom, demo.GDRIVE_EMBEDDINGS_ID)

    assert path.exists()
    assert not (tmp_path / "Global_representation.npy.corrupt").exists()


def test_import_error_is_not_treated_as_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reader reaching for an engine that is not installed (pandas does this)
    # says nothing about the bytes on disk.
    path = tmp_path / "label.csv"
    path.write_bytes(b"ensg,name\nENSG,PROT\n")

    def _no_engine(_path: Path) -> object:
        raise ImportError("Missing optional dependency 'pyarrow'")

    monkeypatch.setattr(
        demo,
        "_download_from_google_drive",
        lambda *a, **k: pytest.fail("must not re-download on ImportError"),
    )

    with pytest.raises(ImportError):
        demo._load_downloaded_artifact(path, _no_engine, demo.GDRIVE_LABELS_ID)

    assert path.exists()
    assert not (tmp_path / "label.csv.corrupt").exists()


def test_load_cytoself_data_self_heals_a_corrupt_embeddings_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embeddings load is wired to the self-heal.

    ``_download_from_google_drive``, ``cache_computed`` and the umap import are
    stubbed; everything between them is the real function.
    """
    pd = pytest.importorskip("pandas")

    n_rows = 6
    labels = pd.DataFrame(
        {
            "ensg": [f"ENSG{i:05d}" for i in range(n_rows)],
            "name": [f"PROT{i}" for i in range(n_rows)],
            "loc_grade1": ["nucleus;speckle"] * n_rows,
        }
    )
    embeddings = np.arange(n_rows * 4, dtype=np.float32).reshape(n_rows, 4)
    fetched: list[str] = []

    def fake_download(
        file_id: str, output_path: Path, expected_min_size: int = 0
    ) -> Path:
        fetched.append(file_id)
        if file_id == demo.GDRIVE_LABELS_ID:
            labels.to_csv(output_path, index=False)
        elif fetched.count(file_id) == 1:
            # First fetch lands a file that passes every size check and still
            # cannot be parsed — the shape of a legacy truncated cache.
            output_path.write_bytes(b"\x93NUMPY truncated")
        else:
            np.save(output_path, embeddings)
        return output_path

    monkeypatch.setattr(demo, "_download_from_google_drive", fake_download)

    class _FakeUMAP:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, data: np.ndarray) -> np.ndarray:
            return np.tile(np.arange(len(data), dtype=np.float32)[:, None], (1, 3))

    real_require = demo.require_module

    def fake_require(name: str) -> object:
        # umap-learn is an external optional dep and is not needed to test the
        # cache wiring; everything else resolves for real.
        if name == "umap":
            return types.SimpleNamespace(UMAP=_FakeUMAP)
        return real_require(name)

    monkeypatch.setattr(demo, "require_module", fake_require)
    # cache_computed would write into the real ~/.cache/luxar; run the closure.
    monkeypatch.setattr(
        demo,
        "cache_computed",
        lambda namespace, key, fn, version=1, recompute=False: fn(),
    )

    coordinates, attributes, category_maps = demo.load_cytoself_data(cache_dir=tmp_path)

    assert coordinates.shape == (n_rows, 3)
    assert "localization" in attributes and "protein_name" in category_maps
    assert fetched.count(demo.GDRIVE_EMBEDDINGS_ID) == 2
    assert (tmp_path / "Global_representation.npy.corrupt").exists()


# =============================================================================
# Thumbnail pipeline: per-source-file part caches
# =============================================================================

_CROPS_PER_FILE = 3
_N_IMAGE_FILES = len(demo.GDRIVE_IMAGE_IDS)
_N_GLOBAL = _CROPS_PER_FILE * _N_IMAGE_FILES


def _synthetic_crops(seed: int) -> np.ndarray:
    """A tiny stand-in for one ``Image_data`` file: (N, 100, 100, 4) uint8."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(_CROPS_PER_FILE, 100, 100, 4), dtype=np.uint8)


def _blobs_in_global_order() -> list[bytes]:
    """Ground truth: every crop of every file, encoded, in global row order.

    ``_encode_crops_to_webp`` normalizes per image, so encoding the whole stack
    gives byte-identical results to encoding it in per-file subsets.
    """
    stack = np.concatenate([_synthetic_crops(i) for i in range(_N_IMAGE_FILES)])
    return demo._encode_crops_to_webp(stack)


def _install_fake_image_downloads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    corrupt_first: str | None = None,
) -> list[str]:
    """Patch the Drive fetch to write synthetic ``Image_data`` files locally.

    Returns the (mutable) list of downloaded file names. ``corrupt_first`` names
    a file whose FIRST fetch writes unreadable bytes, to exercise the
    quarantine-and-refetch self-heal.
    """
    names = list(demo.GDRIVE_IMAGE_IDS)
    id_to_index = {fid: i for i, fid in enumerate(demo.GDRIVE_IMAGE_IDS.values())}
    downloaded: list[str] = []

    def fake_download(
        file_id: str, output_path: Path, expected_min_size: int = 0
    ) -> Path:
        downloaded.append(output_path.name)
        index = id_to_index[file_id]
        assert output_path.name == names[index]
        if corrupt_first is not None and output_path.name == corrupt_first:
            if downloaded.count(output_path.name) == 1:
                output_path.write_bytes(b"not a numpy file at all")
                return output_path
        np.save(output_path, _synthetic_crops(index))
        return output_path

    monkeypatch.setattr(demo, "_download_from_google_drive", fake_download)
    return downloaded


def _install_mapping(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[int, int], n_test: int
) -> None:
    monkeypatch.setattr(
        demo,
        "_build_test_index_mapping",
        lambda cache_dir: (dict(mapping), n_test),
    )


def _identity_mapping() -> dict[int, int]:
    return {i: i for i in range(_N_GLOBAL)}


def test_thumbnail_parts_make_a_rerun_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(monkeypatch)

    blobs_full = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs_full == _blobs_in_global_order()
    assert len(downloaded) == _N_IMAGE_FILES
    for i in range(_N_IMAGE_FILES):
        assert demo._thumbnail_part_path(tmp_path, i).exists()
    assert (tmp_path / demo.THUMBNAIL_CACHE_NAME).exists()

    # Drop the assembled bundle and exactly one part cache.
    (tmp_path / demo.THUMBNAIL_CACHE_NAME).unlink()
    demo._thumbnail_part_path(tmp_path, 3).unlink()
    downloaded.clear()

    blobs_resumed = demo.load_cytoself_images(cache_dir=tmp_path)

    assert downloaded == ["Image_data03.npy"]
    assert blobs_resumed == blobs_full

    # And the assembled bundle short-circuits everything on the next run.
    downloaded.clear()
    assert demo.load_cytoself_images(cache_dir=tmp_path) == blobs_full
    assert downloaded == []


def test_parts_built_under_another_mapping_are_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dangerous case: a repaired Label_data CSV shifts the row mapping
    # without changing its length, so a range check cannot see the difference.
    # Reusing the parts would paste every blob onto the wrong row.
    all_blobs = _blobs_in_global_order()
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(monkeypatch)
    demo.load_cytoself_images(cache_dir=tmp_path)

    shifted = {t: (t + 7) % _N_GLOBAL for t in range(_N_GLOBAL)}
    (tmp_path / demo.THUMBNAIL_CACHE_NAME).unlink()
    _install_mapping(monkeypatch, shifted, _N_GLOBAL)
    downloaded.clear()

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs == [all_blobs[shifted[t]] for t in range(_N_GLOBAL)]
    assert len(downloaded) == _N_IMAGE_FILES


def test_part_cache_from_a_smaller_test_table_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(monkeypatch)
    demo.load_cytoself_images(cache_dir=tmp_path)

    (tmp_path / demo.THUMBNAIL_CACHE_NAME).unlink()
    _install_mapping(monkeypatch, {i: i for i in range(5)}, 5)
    downloaded.clear()

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs == _blobs_in_global_order()[:5]
    # Part 0 only ever held rows 0-2, which the new mapping still asks for in
    # the same order, so it is reused; every later part disagrees and is rebuilt.
    assert downloaded == list(demo.GDRIVE_IMAGE_IDS)[1:]
    part1 = demo._thumbnail_part_path(tmp_path, 1)
    assert part1.with_name(part1.name + ".corrupt").exists()


def test_garbage_part_cache_is_quarantined_and_only_that_file_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(monkeypatch)
    blobs_full = demo.load_cytoself_images(cache_dir=tmp_path)

    (tmp_path / demo.THUMBNAIL_CACHE_NAME).unlink()
    part5 = demo._thumbnail_part_path(tmp_path, 5)
    part5.write_bytes(b"PK\x03\x04 truncated npz")
    downloaded.clear()

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs == blobs_full
    assert downloaded == ["Image_data05.npy"]
    assert part5.with_name(part5.name + ".corrupt").exists()


def test_part_cache_with_mismatched_column_lengths_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = demo._thumbnail_part_path(tmp_path, 0)
    demo._write_npz_atomic(
        part,
        blobs=np.array([b"a", b"b"], dtype=object),
        test_indices=np.asarray([0], dtype=np.int64),
        n_crops=np.int64(3),
    )

    assert demo._load_thumbnail_part(part) is None
    assert not part.exists()
    assert part.with_name(part.name + ".corrupt").exists()


def test_failed_npz_write_leaves_no_stray_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The realistic trigger is ENOSPC part-way through the ~114k-blob bundle,
    # where a full-size stray is exactly what the disk cannot afford.
    def _no_space(handle: object, **arrays: object) -> None:
        handle.write(b"partial")  # type: ignore[attr-defined]
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(demo.np, "savez", _no_space)
    target = tmp_path / demo.THUMBNAIL_CACHE_NAME

    with pytest.raises(OSError):
        demo._write_npz_atomic(target, blobs=np.array([b"a"], dtype=object))

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_corrupt_thumbnail_bundle_is_quarantined_and_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(monkeypatch)
    blobs_full = demo.load_cytoself_images(cache_dir=tmp_path)

    bundle = tmp_path / demo.THUMBNAIL_CACHE_NAME
    bundle.write_bytes(b"PK\x03\x04 truncated npz")
    downloaded.clear()

    rebuilt = demo.load_cytoself_images(cache_dir=tmp_path)

    assert rebuilt == blobs_full
    assert (tmp_path / (demo.THUMBNAIL_CACHE_NAME + ".corrupt")).exists()
    # Rebuilt entirely from the surviving part caches — no re-download.
    assert downloaded == []


def test_legacy_bundle_is_adopted_instead_of_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The v1 rename must not cost an existing user a multi-gigabyte rebuild:
    # the contents are byte-identical, so the file is simply adopted.
    blobs = [b"webp-one", b"webp-two", b"webp-three"]
    demo._write_npz_atomic(
        tmp_path / demo.LEGACY_THUMBNAIL_CACHE_NAME,
        blobs=np.array(blobs, dtype=object),
    )
    monkeypatch.setattr(
        demo,
        "_build_test_index_mapping",
        lambda cache_dir: pytest.fail("must not rebuild an adoptable cache"),
    )
    monkeypatch.setattr(
        demo,
        "_download_from_google_drive",
        lambda *a, **k: pytest.fail("must not download an adoptable cache"),
    )

    adopted = demo.load_cytoself_images(cache_dir=tmp_path)

    assert adopted == blobs
    assert (tmp_path / demo.THUMBNAIL_CACHE_NAME).exists()
    assert not (tmp_path / demo.LEGACY_THUMBNAIL_CACHE_NAME).exists()


def test_a_future_version_does_not_adopt_the_legacy_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adoption is only sound while the encoding is unchanged. A version bump has
    # to disable it by itself — a comment saying "do not adopt" cannot.
    monkeypatch.setattr(demo, "THUMBNAIL_CACHE_NAME", "image_labels_test_webp_v2.npz")
    legacy = tmp_path / demo.LEGACY_THUMBNAIL_CACHE_NAME
    demo._write_npz_atomic(legacy, blobs=np.array([b"v1-era blob"], dtype=object))
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(monkeypatch)

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs == _blobs_in_global_order()
    assert len(downloaded) == _N_IMAGE_FILES
    assert legacy.exists(), "an un-adoptable bundle is left alone, not laundered"


def test_corrupt_image_file_is_quarantined_and_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mapping(monkeypatch, _identity_mapping(), _N_GLOBAL)
    downloaded = _install_fake_image_downloads(
        monkeypatch, corrupt_first="Image_data00.npy"
    )

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs == _blobs_in_global_order()
    assert downloaded.count("Image_data00.npy") == 2
    assert (tmp_path / "Image_data00.npy.corrupt").exists()


def test_image_loop_uses_the_real_download_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop's cache check, staging and sidecar are the real ones.

    Every other thumbnail test stubs ``_download_from_google_drive`` wholesale,
    so this is the one that exercises the seam between the sidecar and the loop:
    a verified file on disk must skip the network entirely while its neighbour
    is fetched through the full staging path.
    """
    two_files = {
        "Image_data00.npy": "IMGID0",
        "Image_data01.npy": "IMGID1",
    }
    monkeypatch.setattr(demo, "GDRIVE_IMAGE_IDS", two_files)
    # The real floor is 10 GB; these synthetic files are ~120 KB.
    monkeypatch.setattr(demo, "MIN_SIZE_IMAGE_DATA", 10_000)
    n_global = 2 * _CROPS_PER_FILE
    _install_mapping(monkeypatch, {i: i for i in range(n_global)}, n_global)

    def _npy_bytes(index: int) -> bytes:
        buf = io.BytesIO()
        np.save(buf, _synthetic_crops(index))
        return buf.getvalue()

    # File 0 is already cached AND certified; file 1 is absent.
    cached = tmp_path / "Image_data00.npy"
    cached.write_bytes(_npy_bytes(0))
    demo._write_completion_sidecar(cached, cached.stat().st_size)

    recorder = _install_session(monkeypatch, _serve(_npy_bytes(1)))

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert blobs == demo._encode_crops_to_webp(
        np.concatenate([_synthetic_crops(0), _synthetic_crops(1)])
    )
    assert recorder.urls, "file 1 should have been fetched"
    assert all("IMGID1" in url for url in recorder.urls)
    assert not any("IMGID0" in url for url in recorder.urls)
    # File 1 went through staging and came out certified.
    fetched = tmp_path / "Image_data01.npy"
    assert json.loads(demo._sidecar_path(fetched).read_text(encoding="utf-8"))[
        "size"
    ] == len(_npy_bytes(1))
    assert not demo._part_path(fetched).exists()


def test_low_match_rate_returns_blobs_but_does_not_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only 4 of 30 test rows match — far below MIN_MATCH_FRACTION. That reads as
    # a regression, so the bundle must not be frozen; but a partly-working
    # tooltip still beats none, so the blobs are returned and nothing raises.
    _install_mapping(monkeypatch, {i: i for i in range(4)}, _N_GLOBAL)
    _install_fake_image_downloads(monkeypatch)

    blobs = demo.load_cytoself_images(cache_dir=tmp_path)

    assert len(blobs) == _N_GLOBAL
    assert blobs[:4] == _blobs_in_global_order()[:4]
    assert not (tmp_path / demo.THUMBNAIL_CACHE_NAME).exists()
    # The expensive per-file work is kept, so a retry is cheap.
    assert demo._thumbnail_part_path(tmp_path, 0).exists()


@pytest.mark.parametrize(
    ("n_matched", "expect_cached"),
    [(15, True), (14, False)],
)
def test_match_fraction_boundary_is_inclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_matched: int,
    expect_cached: bool,
) -> None:
    # Pins `<` rather than `<=`: exactly MIN_MATCH_FRACTION (15/30 = 0.5) is
    # acceptable, one row fewer is not.
    assert demo.MIN_MATCH_FRACTION == 0.5
    _install_mapping(monkeypatch, {i: i for i in range(n_matched)}, _N_GLOBAL)
    _install_fake_image_downloads(monkeypatch)

    demo.load_cytoself_images(cache_dir=tmp_path)

    assert (tmp_path / demo.THUMBNAIL_CACHE_NAME).exists() is expect_cached
