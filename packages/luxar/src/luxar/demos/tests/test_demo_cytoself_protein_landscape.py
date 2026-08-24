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
(``images_expected``), and — one level up — the ``main()`` wiring that decides
what those two functions are even called with.

These tests drive the REAL ``create_cytoself_scene`` on a tiny synthetic input
and read the built zarr store back. Nothing here touches the network. The
thumbnail CACHE — the bundle, the per-file part caches and the ``recompute``
bypass — is covered next door in ``test_demo_cytoself_caching.py``.
"""

from __future__ import annotations

import io
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

# ``demo_cytoself_protein_landscape`` imports ``luxar.utils._umap_utils`` at
# module scope, which imports PIL. Pillow also encodes the fixture thumbnails.
pytest.importorskip("PIL")

from luxar.demos import demo_cytoself_protein_landscape as demo  # noqa: E402
from luxar.demos.demo_cytoself_protein_landscape import (  # noqa: E402
    create_cytoself_scene,
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


def _decode_keys(node: zarr.Group) -> list[str]:
    offsets = np.asarray(node["key_offsets"][:])
    data = np.asarray(node["key_bytes"][:])
    return [
        bytes(data[int(offsets[i]) : int(offsets[i + 1])]).decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


def test_protein_links_are_serialized_and_tiled_per_view(tmp_path: Path) -> None:
    coordinates, attributes, category_maps = _inputs()
    output_path = tmp_path / "cytoself_links.luxar.zarr"
    create_cytoself_scene(
        output_path,
        coordinates,
        attributes,
        category_maps,
        images_expected=False,
    )

    node = zarr.open_group(str(output_path), mode="r")["Images"]
    assert node.attrs["link"] == "https://www.proteinatlas.org/search/{hover_key}"
    assert node.attrs["copy"] == "{hover_key}"
    assert node.attrs["has_keys"] is True
    assert sorted(_decode_keys(node)) == sorted(["TUBB", "CDC27", "TUBB", "ACTB"] * 2)
    assert list(node.attrs["slice_dims"]) == [0]


def test_protein_links_are_omitted_without_category_maps(tmp_path: Path) -> None:
    coordinates, attributes, _ = _inputs()
    output_path = tmp_path / "cytoself_no_maps.luxar.zarr"
    create_cytoself_scene(
        output_path,
        coordinates,
        attributes,
        None,
        images_expected=False,
    )

    node = zarr.open_group(str(output_path), mode="r")["Images"]
    assert "link" not in node.attrs
    assert "copy" not in node.attrs
    assert "has_keys" not in node.attrs
    assert "key_offsets" not in node


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


class TestMainWiring:
    """``main()`` is the only caller, and reverting any of its kwargs is silent.

    Everything above drives ``create_cytoself_scene`` directly, which pins the
    CALLEE and leaves the caller unguarded: dropping ``recompute=`` from either
    of the two calls that take it, dropping ``expected_count=`` or
    ``images_expected=``, or making the navigation hint unconditional again,
    each left the whole suite green.

    Both ``recompute=`` call sites are asserted. ``--recompute`` discards the
    cached UMAP as well as the thumbnail bundle — a 10-30 minute recompute, and
    the cost ``_resolve_image_labels`` warns about — so a stub that merely
    swallowed the kwarg would leave half the flag unpinned.
    """

    @staticmethod
    def _stub(monkeypatch: pytest.MonkeyPatch, *flags: str) -> dict[str, Any]:
        """Run the real ``main()`` with every expensive edge replaced.

        Returns the kwargs each stub was called with. The viewer path (not
        ``--no-serve``) is the one exercised, because the navigation hint only
        prints there.
        """
        coordinates, attributes, category_maps = _inputs()
        calls: dict[str, Any] = {}

        monkeypatch.setattr(sys, "argv", ["demo_cytoself_protein_landscape", *flags])

        def _fake_data(**kwargs: Any) -> tuple[Any, Any, Any]:
            calls["data"] = kwargs
            return coordinates, attributes, category_maps

        def _fake_images(**kwargs: Any) -> list[bytes]:
            calls["images"] = kwargs
            return [_make_webp((7, 8, 9))] * N_POINTS

        def _fake_scene(*args: Any, **kwargs: Any) -> int:
            calls["scene"] = kwargs
            return N_POINTS

        monkeypatch.setattr(demo, "load_cytoself_data", _fake_data)
        monkeypatch.setattr(demo, "load_cytoself_images", _fake_images)
        monkeypatch.setattr(demo, "create_cytoself_scene", _fake_scene)
        monkeypatch.setattr(demo, "generate_all_legends", lambda *a, **k: None)
        monkeypatch.setattr(demo, "launch_viewer", lambda *a, **k: None)
        return calls

    def test_recompute_and_the_point_count_reach_the_loader(
        self, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
    ) -> None:
        calls = self._stub(monkeypatch, "--recompute")

        demo.main()

        # --recompute has to reach BOTH caches: the UMAP (the 10-30 min half)
        # and the thumbnail bundle. The point count is what lets a stale bundle
        # be detected at all.
        assert calls["data"] == {"recompute": True}
        assert calls["images"] == {"expected_count": N_POINTS, "recompute": True}
        assert calls["scene"]["images_expected"] is True
        assert "fluorescence image" in capfd.readouterr().out

    def test_without_images_skips_the_loader_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
    ) -> None:
        calls = self._stub(monkeypatch, "--without-images")

        demo.main()

        assert calls["data"] == {"recompute": False}
        assert "images" not in calls, "--without-images must not fetch thumbnails"
        # A deliberate opt-out builds quietly rather than warning about an
        # absence the user asked for.
        assert calls["scene"]["images_expected"] is False
        out = capfd.readouterr().out
        assert "localization and protein" in out
        assert "fluorescence image" not in out


class TestProteinLinkKeys:
    """The UniProt-ish click-through, and the code that must not become a link.

    `protein_name` is a pandas categorical code, so an unannotated cell carries
    -1. Guarding only the upper bound let `names[-1]` return the LAST protein —
    a real one — so that cell linked confidently to the wrong page. This is the
    regression test for the lower bound; it is here rather than beside the demo
    because `_inputs` is the only seam that can fabricate a -1.
    """

    @staticmethod
    def _keys(scene_path) -> list[str]:
        root = zarr.open_group(str(scene_path), mode="r")["Images"]
        node = root
        if not dict(root.attrs).get("has_keys"):
            for name in sorted(root.keys()):
                child = root[name]
                if hasattr(child, "attrs") and dict(child.attrs).get("has_keys"):
                    node = child
                    break
        return _decode_keys(node)

    def test_keys_are_the_bare_protein_name(self, tmp_path: Path) -> None:
        coordinates, attributes, category_maps = _inputs()
        out = tmp_path / "cytoself_keys.luxar.zarr"
        create_cytoself_scene(
            out,
            coordinates,
            attributes,
            category_maps,
            image_labels=None,
            images_expected=False,
        )
        keys = self._keys(out)
        # One per point per attribute view, and each is a protein name alone —
        # the label joins localization onto it, which no search wants.
        assert set(keys) <= set(category_maps["protein_name"])
        assert "TUBB" in keys

    def test_a_missing_code_yields_no_key_rather_than_the_last_protein(
        self, tmp_path: Path
    ) -> None:
        coordinates, attributes, category_maps = _inputs()
        # -1 is what pandas stores for an unannotated cell.
        attributes["protein_name"] = np.array([1, -1, 1, 2], dtype=np.int32)
        out = tmp_path / "cytoself_missing.luxar.zarr"
        create_cytoself_scene(
            out,
            coordinates,
            attributes,
            category_maps,
            image_labels=None,
            images_expected=False,
        )
        keys = self._keys(out)
        # The empty one suppresses that element's link. Before the lower bound
        # it was "ACTB" — the last category — and the cell linked to a protein
        # it has nothing to do with.
        assert "" in keys, "a -1 code must produce an empty key"
        # Four points, two attribute views, one unannotated point: its key is
        # empty in each view and nothing else is.
        assert keys.count("") == 2, keys
        # And specifically NOT the last category, which is what the upper-bound
        # guard used to return for -1.
        assert keys.count("ACTB") == 2, "only the genuinely-ACTB point, per view"
