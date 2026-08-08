"""Regression tests for the CytoSelf demo's hover-overlay layout.

The demo defines a BESPOKE two-panel hover layout: a thumbnail panel anchored
top-right at ``(0.98, 0.02)`` and a text label deliberately parked at
``x = 0.82`` so it sits to the LEFT of that panel. Defining any ``hover=True``
overlay suppresses the compiler's auto-injected default hover overlay
(``auto_inject_hover_overlay``).

When the image thumbnails are unavailable — the Google-Drive download failed,
the count-mismatch guard dropped them, or ``--without-images`` was passed — the
image half was silently skipped while the text half stayed pinned at
``x = 0.82``, i.e. floating in an empty slot next to a panel that does not
exist. Hovering then "appeared to do nothing". The fix relocates the label to
the centre-left slot ``(0.02, 0.5)`` (the house convention for a text-only
hover tooltip, shared with ``demo_chromatrace_choir_umap`` and six siblings)
rather than falling back to the auto-injected default, which is the same
top-right corner only further into it.

The reporting is covered too: which of the two absence messages is printed
(``images_expected``), and the ``recompute`` pass-through that is the only way
to rebuild the cached thumbnail bundle from the CLI.

These tests drive the REAL ``create_cytoself_scene`` on a tiny synthetic input
and read the built zarr store back. Nothing here touches the network.
"""

from __future__ import annotations

import io
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

# ``demo_cytoself_protein_landscape`` imports ``luxar.utils._umap_utils`` at
# module scope, which imports PIL. Pillow also encodes the fixture thumbnails.
pytest.importorskip("PIL")

from luxar.demos import MissingDependencyError  # noqa: E402
from luxar.demos import demo_cytoself_protein_landscape as demo  # noqa: E402
from luxar.demos.demo_cytoself_protein_landscape import (  # noqa: E402
    THUMBNAILS_CACHE_NAME,
    create_cytoself_scene,
    load_cytoself_images,
)

N_POINTS = 4


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything in the scene-building path opens a socket."""

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the test path must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)


def _make_webp(color: tuple[int, int, int]) -> bytes:
    """A real (tiny) WebP blob, mirroring the demo's own encoder output."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="webp")
    return buf.getvalue()


def _inputs() -> tuple[np.ndarray, dict, dict]:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.5],
            [-1.0, 0.25, 0.75],
            [0.5, -1.0, 0.25],
        ],
        dtype=np.float32,
    )
    attributes = {
        "localization": np.array([0, 1, 0, 2], dtype=np.int32),
        "protein_name": np.array([1, 0, 1, 2], dtype=np.int32),
    }
    category_maps = {
        "localization": ["nucleoplasm", "cytoplasm", "mitochondria"],
        "protein_name": ["CDC27", "TUBB", "ACTB"],
    }
    return coordinates, attributes, category_maps


def _build(
    tmp_path: Path,
    image_labels: list[bytes] | None,
    *,
    images_expected: bool = True,
) -> dict[str, dict]:
    """Build the real scene and return ``{overlay_name: attrs}``."""
    coordinates, attributes, category_maps = _inputs()
    output_path = tmp_path / "cytoself_test.luxar.zarr"

    n_points = create_cytoself_scene(
        output_path,
        coordinates,
        attributes,
        category_maps,
        image_labels=image_labels,
        images_expected=images_expected,
    )
    assert n_points == N_POINTS

    store = zarr.open_group(str(output_path), mode="r")
    overlays = store["overlays"]
    return {name: dict(overlays[name].attrs) for name in overlays.group_keys()}


def _hover_overlays(overlays: dict[str, dict]) -> dict[str, dict]:
    return {n: a for n, a in overlays.items() if a.get("hover")}


class TestHoverLayoutWithThumbnails:
    """With aligned image labels the bespoke two-panel layout is built."""

    def test_custom_pair_is_defined_and_autoinject_is_suppressed(
        self, tmp_path: Path
    ) -> None:
        blobs = [_make_webp((i * 40, 0, 0)) for i in range(N_POINTS)]
        overlays = _build(tmp_path, blobs)
        hover = _hover_overlays(overlays)

        images = [a for a in hover.values() if a["type"] == "overlay_html"]
        assert len(images) == 1
        assert "{hover_image_label}" in images[0]["html"]
        assert list(images[0]["position"]) == [0.98, 0.02]
        assert images[0]["anchor"] == "top-right"

        texts = [a for a in hover.values() if a["type"] == "overlay_text"]
        assert len(texts) == 1
        assert texts[0]["text"] == "{hover_label}"
        assert texts[0]["position"][0] == pytest.approx(0.82)

        # The custom pair must suppress the compiler's default hover overlays.
        assert "__hover_text" not in overlays
        assert "__hover_image" not in overlays


# The three ways the thumbnails can be absent. The relocated layout must not
# depend on WHICH of them happened.
_NO_THUMBNAIL_CASES = pytest.mark.parametrize(
    "case",
    ["none", "count-mismatch", "without-images"],
)


def _build_without_thumbnails(tmp_path: Path, case: str) -> dict[str, dict]:
    if case == "none":
        return _build(tmp_path, None)
    if case == "count-mismatch":
        # Longer than n_points — the guard in create_cytoself_scene drops these.
        blobs = [_make_webp((10, 20, 30))] * (N_POINTS + 3)
        return _build(tmp_path, blobs)
    if case == "without-images":
        # The deliberate `--without-images` shape.
        return _build(tmp_path, None, images_expected=False)
    raise AssertionError(f"unknown case {case!r}")


class TestHoverLayoutWithoutThumbnails:
    """Without thumbnails the label moves to the centre-left slot."""

    @_NO_THUMBNAIL_CASES
    def test_no_overlay_is_parked_at_x_082(self, tmp_path: Path, case: str) -> None:
        overlays = _build_without_thumbnails(tmp_path, case)

        # Nothing may sit in the slot reserved for "left of the thumbnail".
        stranded = [
            name
            for name, attrs in overlays.items()
            if attrs.get("position") and attrs["position"][0] == pytest.approx(0.82)
        ]
        assert stranded == []

    @_NO_THUMBNAIL_CASES
    def test_hover_text_moves_to_the_centre_left_slot(
        self, tmp_path: Path, case: str
    ) -> None:
        overlays = _build_without_thumbnails(tmp_path, case)
        hover = _hover_overlays(overlays)

        assert len(hover) == 1, f"expected exactly one hover overlay, got {hover}"
        attrs = next(iter(hover.values()))
        assert attrs["type"] == "overlay_text"
        assert attrs["text"] == "{hover_label}"
        # The prominent slot nothing else here uses — the legend is
        # center-RIGHT at 0.98/0.5.
        assert list(attrs["position"]) == [0.02, 0.5]
        assert attrs["anchor"] == "center-left"

    @_NO_THUMBNAIL_CASES
    def test_auto_injection_stays_suppressed(self, tmp_path: Path, case: str) -> None:
        """A custom hover overlay still suppresses the injected defaults."""
        overlays = _build_without_thumbnails(tmp_path, case)

        assert "__hover_text" not in overlays
        assert "__hover_image" not in overlays


class TestMissingThumbnailReporting:
    """``images_expected`` picks WHICH absence message is printed.

    The layout is identical either way, so without this the flag could be
    deleted from the function body and the rest of the suite would stay green.
    """

    def test_unexpected_absence_warns(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        _build(tmp_path, None, images_expected=True)
        out = capfd.readouterr().out

        assert "No image labels" in out
        assert "--without-images" not in out

    def test_deliberate_opt_out_does_not_warn(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        _build(tmp_path, None, images_expected=False)
        out = capfd.readouterr().out

        # A user who passed --without-images must not be warned about an
        # absence they asked for, nor told to pass the flag they just passed.
        assert "--without-images" in out
        assert "No image labels" not in out


class _RebuildEntered(Exception):
    """Raised by the monkeypatched rebuild path to prove it was reached."""


_STUB_IMAGE_FILE = "Image_data00.npy"
_STUB_N_TEST = 2


def _stub_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the REAL rebuild path run, with the network replaced by local data.

    One Image_data file with two crops, so the loop, the WebP encode and the
    bundle write all execute for real without touching Google Drive.
    """
    monkeypatch.setattr(demo, "GDRIVE_IMAGE_IDS", {_STUB_IMAGE_FILE: "stub-id"})
    monkeypatch.setattr(
        demo,
        "_build_test_index_mapping",
        lambda cache_dir: ({0: 0, 1: 1}, _STUB_N_TEST),
    )

    def _fake_download(
        file_id: str, output_path: Path, expected_min_size: int = 0
    ) -> Path:
        crops = np.zeros((_STUB_N_TEST, 100, 100, 4), dtype=np.uint8)
        crops[0, :, :, 0] = 200  # distinct content per crop
        crops[1, :, :, 1] = 90
        np.save(output_path, crops)
        return output_path

    monkeypatch.setattr(demo, "_download_from_google_drive", _fake_download)


class TestThumbnailCacheRecompute:
    """``recompute`` must BYPASS the cached bundle, not merely re-read it.

    Without it a stale ``image_labels_test_webp.npz`` short-circuits before any
    network call, so the count mismatch it causes reproduces forever.
    """

    @staticmethod
    def _seed_cache(cache_dir: Path) -> list[bytes]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        blobs = [_make_webp((1, 2, 3)), _make_webp((4, 5, 6))]
        np.savez(
            cache_dir / THUMBNAILS_CACHE_NAME,
            blobs=np.array(blobs, dtype=object),
        )
        return blobs

    @staticmethod
    def _block_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*args: Any, **kwargs: Any) -> None:
            raise _RebuildEntered("the rebuild path was entered")

        monkeypatch.setattr(demo, "_build_test_index_mapping", _boom)

    def test_cached_bundle_is_reused_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cytoself"
        blobs = self._seed_cache(cache_dir)
        self._block_rebuild(monkeypatch)

        assert load_cytoself_images(cache_dir) == blobs

    def test_recompute_bypasses_the_cached_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cytoself"
        self._seed_cache(cache_dir)
        self._block_rebuild(monkeypatch)

        # Reaching the rebuild proves the cache was skipped, not just re-read.
        with pytest.raises(_RebuildEntered):
            load_cytoself_images(cache_dir, recompute=True)

    def test_recompute_replaces_the_bundle_atomically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--recompute`` is the first path that OVERWRITES a valid bundle.

        ``np.savez`` truncates on open, so writing in place would turn an
        interrupted rebuild into a permanently unreadable cache.
        """
        cache_dir = tmp_path / "cytoself"
        seeded = self._seed_cache(cache_dir)
        _stub_rebuild(monkeypatch)

        blobs = load_cytoself_images(cache_dir, recompute=True)

        assert len(blobs) == _STUB_N_TEST
        assert blobs != seeded, "the stale bundle should have been replaced"

        bundle = cache_dir / THUMBNAILS_CACHE_NAME
        reloaded = np.load(bundle, allow_pickle=True)["blobs"]
        assert [bytes(b) for b in reloaded] == blobs

        # The temp artefact must not survive a successful rename.
        assert list(cache_dir.glob("*.tmp")) == []

    def test_interrupted_rebuild_leaves_the_existing_bundle_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the rename: a half-written bundle is discarded."""
        cache_dir = tmp_path / "cytoself"
        seeded = self._seed_cache(cache_dir)
        _stub_rebuild(monkeypatch)

        def _die_mid_write(file: Any, **kwargs: Any) -> None:
            # The Ctrl-C / OOM shape: some bytes land, then the write dies.
            file.write(b"PK\x03\x04partial")
            raise KeyboardInterrupt("interrupted while writing the bundle")

        monkeypatch.setattr(demo.np, "savez", _die_mid_write)

        with pytest.raises(KeyboardInterrupt):
            load_cytoself_images(cache_dir, recompute=True)

        # Pre-fix this wrote straight to the bundle, so the good one was gone
        # and every later run hit BadZipFile.
        reloaded = np.load(cache_dir / THUMBNAILS_CACHE_NAME, allow_pickle=True)
        assert [bytes(b) for b in reloaded["blobs"]] == seeded
        assert list(cache_dir.glob("*.tmp")) == []


class TestPerFileFailureReporting:
    """A failure inside the per-file loop must name the file AND its type."""

    def test_failure_names_the_file_and_the_original_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cytoself"
        cache_dir.mkdir(parents=True)
        _stub_rebuild(monkeypatch)

        def _oom(*args: Any, **kwargs: Any) -> None:
            # str(MemoryError()) is "" — the type name is the only signal.
            raise MemoryError()

        monkeypatch.setattr(demo, "_download_from_google_drive", _oom)

        with pytest.raises(RuntimeError) as excinfo:
            load_cytoself_images(cache_dir, recompute=True)

        message = str(excinfo.value)
        assert _STUB_IMAGE_FILE in message
        assert "MemoryError" in message

    def test_missing_dependency_is_not_rebranded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It has its own handler (and hint) in ``main()`` — keep the type."""
        cache_dir = tmp_path / "cytoself"
        cache_dir.mkdir(parents=True)
        _stub_rebuild(monkeypatch)

        def _no_pillow(*args: Any, **kwargs: Any) -> None:
            raise MissingDependencyError("Pillow is not installed")

        monkeypatch.setattr(demo, "_encode_crops_to_webp", _no_pillow)

        with pytest.raises(MissingDependencyError):
            load_cytoself_images(cache_dir, recompute=True)
