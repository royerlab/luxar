"""Tests for Scene overlay methods (add_text, add_image, add_video, add_html).

Functional tests that verify the full round-trip: Python API -> zarr storage -> metadata.
"""

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler, Overlay
from luxar._zarr_compat import read_raw_bytes
from luxar.core.scene import Scene


def _make_4d_dims():
    """Create 4D dimensions with time + 3 spatial."""
    return Dimensions(
        [
            Dimension("time", range=(0, 100), display=False, discrete=True),
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
        ]
    )


class TestAddText:
    """Test Scene.add_text()."""

    def test_basic_text(self, tmp_path) -> None:
        """Add text with minimal params, verify zarr output."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            overlay = scene.add_text("Hello", position=(0.5, 0.5))

        assert isinstance(overlay, Overlay)
        assert overlay.name == "overlay_0"
        assert overlay.overlay_type == "overlay_text"
        assert overlay.position == (0.5, 0.5)

        # Verify zarr
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["type"] == "overlay_text"
        assert attrs["text"] == "Hello"
        assert attrs["position"] == [0.5, 0.5]
        assert attrs["font"] == "sans"
        assert attrs["color"] == "white"
        assert attrs["opacity"] == 1.0
        assert attrs["anchor"] == "top-left"
        assert attrs["z_index"] == 0

    def test_all_text_params(self, tmp_path) -> None:
        """Add text with every parameter specified."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())
            overlay = scene.add_text(
                "Full params",
                position=(0.1, 0.9),
                name="my_label",
                font_size=0.04,
                font="serif",
                color="red",
                opacity=0.8,
                anchor="bottom-center",
                width=0.3,
                text_align="center",
                line_height=1.6,
                background="rgba(0,0,0,0.5)",
                padding=0.01,
                stroke_color="black",
                stroke_width=0.003,
                visible_range={"time": (5, 10)},
                transition="fade",
                transition_duration=0.5,
                interactive=True,
            )

        assert overlay.name == "my_label"

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/my_label"].attrs)
        assert attrs["font_size"] == 0.04
        assert attrs["font"] == "serif"
        assert attrs["color"] == "red"
        assert attrs["opacity"] == 0.8
        assert attrs["anchor"] == "bottom-center"
        assert attrs["width"] == 0.3
        assert attrs["text_align"] == "center"
        assert attrs["line_height"] == 1.6
        assert attrs["background"] == "rgba(0,0,0,0.5)"
        assert attrs["padding"] == 0.01
        assert attrs["stroke_color"] == "black"
        assert attrs["stroke_width"] == 0.003
        assert attrs["visible_range"] == {"time": [5, 10]}
        assert attrs["transition"] == "fade"
        assert attrs["transition_duration"] == 0.5
        assert attrs["interactive"] is True

    def test_auto_naming_increments(self, tmp_path) -> None:
        """Multiple overlays get incrementing auto-names."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            o1 = scene.add_text("A", position=(0.1, 0.1))
            o2 = scene.add_text("B", position=(0.2, 0.2))
            o3 = scene.add_text("C", position=(0.3, 0.3))

        assert o1.name == "overlay_0"
        assert o2.name == "overlay_1"
        assert o3.name == "overlay_2"

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "overlays/overlay_0" in store
        assert "overlays/overlay_1" in store
        assert "overlays/overlay_2" in store

    def test_z_index_matches_insertion_order(self, tmp_path) -> None:
        """z_index should match the order overlays were added."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("First", position=(0.1, 0.1))
            scene.add_text("Second", position=(0.2, 0.2))
            scene.add_image(b"\x89PNG\r\n", position=(0.3, 0.3))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["z_index"] == 0
        assert store["overlays/overlay_1"].attrs["z_index"] == 1
        assert store["overlays/overlay_2"].attrs["z_index"] == 2

    def test_custom_name(self, tmp_path) -> None:
        """Custom name is used and doesn't affect counter."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            o1 = scene.add_text("A", position=(0.1, 0.1), name="title")
            o2 = scene.add_text("B", position=(0.2, 0.2))

        assert o1.name == "title"
        assert o2.name == "overlay_0"

    def test_text_without_width_no_width_in_attrs(self, tmp_path) -> None:
        """Width should be omitted from attrs when not specified."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("No width", position=(0.5, 0.5))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "width" not in store["overlays/overlay_0"].attrs

    def test_text_with_width_stored(self, tmp_path) -> None:
        """Width should be stored when specified (enables wrapping)."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("Wrapped", position=(0.5, 0.5), width=0.3)

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["width"] == 0.3

    def test_overlays_property(self, tmp_path) -> None:
        """Scene.overlays returns list of all overlays."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("A", position=(0.1, 0.1))
            scene.add_text("B", position=(0.2, 0.2))
            overlays = scene.overlays

        assert len(overlays) == 2
        assert overlays[0].name == "overlay_0"
        assert overlays[1].name == "overlay_1"


class TestAddImage:
    """Test Scene.add_image()."""

    def test_image_from_bytes(self, tmp_path) -> None:
        """Add image from raw bytes, verify file is written."""
        png_bytes = self._make_tiny_png()
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            overlay = scene.add_image(png_bytes, position=(0.9, 0.05))

        assert overlay.overlay_type == "overlay_image"

        # Verify image file exists in zarr directory
        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.png"
        )
        assert image_path.exists()
        assert image_path.read_bytes() == png_bytes

        # Verify metadata
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["type"] == "overlay_image"
        assert attrs["image_file"] == "image.png"

    def test_image_from_numpy(self, tmp_path) -> None:
        """Add image from numpy array — gets encoded to PNG."""
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(arr, position=(0.5, 0.5))

        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.png"
        )
        assert image_path.exists()
        assert len(image_path.read_bytes()) > 0

    def test_image_from_float_numpy(self, tmp_path) -> None:
        """Float numpy arrays (0-1) are converted to uint8 then encoded."""
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(arr, position=(0.5, 0.5))

        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.png"
        )
        assert image_path.exists()

    def test_image_size_accepts_an_auto_height(self, tmp_path) -> None:
        """A None height is stored as null so the viewer keeps the aspect ratio."""
        png_bytes = self._make_tiny_png()
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(png_bytes, position=(0.98, 0.97), size=(0.07, None))
            with pytest.raises(ValueError, match="width > 0"):
                scene.add_image(png_bytes, position=(0.5, 0.5), size=(0.0, None))
            with pytest.raises(ValueError, match="height must be > 0 or None"):
                scene.add_image(png_bytes, position=(0.5, 0.5), size=(0.1, -1.0))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["size"] == [0.07, None]

    def test_image_from_path(self, tmp_path) -> None:
        """Add image from file path."""
        png_bytes = self._make_tiny_png()
        img_file = tmp_path / "logo.png"
        img_file.write_bytes(png_bytes)

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(str(img_file), position=(0.9, 0.05))

        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.png"
        )
        assert image_path.exists()

    def test_jpeg_bytes_use_payload_extension(self, tmp_path) -> None:
        """Pre-encoded bytes use their detected format, not the format option."""
        jpeg_bytes = b"\xff\xd8\xff\xe0"
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(jpeg_bytes, position=(0.9, 0.05), format="png")

        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.jpeg"
        )
        assert image_path.read_bytes() == jpeg_bytes

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["image_file"] == "image.jpeg"

    def test_jpg_path_uses_payload_extension(self, tmp_path) -> None:
        """JPEG paths normalize .jpg to the stored .jpeg member name."""
        jpeg_bytes = b"\xff\xd8\xff\xe0"
        img_file = tmp_path / "logo.jpg"
        img_file.write_bytes(jpeg_bytes)

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(img_file, position=(0.9, 0.05))

        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.jpeg"
        )
        assert image_path.read_bytes() == jpeg_bytes

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["image_file"] == "image.jpeg"

    def test_image_jpeg_format(self, tmp_path) -> None:
        """JPEG format uses .jpeg extension."""
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(arr, position=(0.5, 0.5), format="jpeg")

        image_path = (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "image.jpeg"
        )
        assert image_path.exists()

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["image_file"] == "image.jpeg"

    def test_image_with_size_and_blend(self, tmp_path) -> None:
        """Image with explicit size and blend mode."""
        png_bytes = self._make_tiny_png()
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_image(
                png_bytes,
                position=(0.0, 0.0),
                size=(1.0, 1.0),
                blend_mode="multiply",
                opacity=0.5,
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["size"] == [1.0, 1.0]
        assert attrs["blend_mode"] == "multiply"
        assert attrs["opacity"] == 0.5

    def test_nonexistent_image_path(self, tmp_path) -> None:
        """Non-existent file path raises ValueError."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="not found"):
                scene.add_image("/nonexistent/file.png", position=(0.5, 0.5))

    @staticmethod
    def _make_tiny_png() -> bytes:
        """Create a minimal valid PNG (1x1 red pixel)."""
        try:
            import io

            from PIL import Image as PILImage

            img = PILImage.new("RGB", (1, 1), (255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            # Fallback: raw PNG bytes for a 1x1 red pixel
            return (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03"
                b"\x00\x01\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
            )


class TestAddVideo:
    """Test Scene.add_video()."""

    # A minimal EBML header: enough for the sniffer, which is all the writer checks.
    WEBM_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 60
    MP4_BYTES = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 60

    def test_video_payloads_write_through_a_memory_store(self) -> None:
        class MemoryWriter:
            def __init__(self) -> None:
                self.store = zarr.group()

            def write_group(self, path: str, **attrs: object) -> None:
                group = (
                    self.store if path in ("", "/") else self.store.require_group(path)
                )
                group.attrs.update(attrs)

        writer = MemoryWriter()
        scene = Scene(writer=writer, dimensions=Dimensions.default_3d())  # type: ignore[arg-type]
        scene.add_video(self.WEBM_BYTES, position=(0.1, 0.1))

        group = writer.store["overlays/overlay_0"]
        assert isinstance(group, zarr.Group)
        assert read_raw_bytes(group, "video.webm") == self.WEBM_BYTES

    def test_video_from_bytes_with_poster(self, tmp_path) -> None:
        png_bytes = TestAddImage._make_tiny_png()
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            overlay = scene.add_video(
                self.WEBM_BYTES,
                position=(0.06, 0.5),
                anchor="center-left",
                size=(0.26, None),
                poster=png_bytes,
                playback_rate=1.5,
                transition="fade",
            )

        assert overlay.overlay_type == "overlay_video"
        base = tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0"
        assert (base / "video.webm").read_bytes() == self.WEBM_BYTES
        assert (base / "poster.png").read_bytes() == png_bytes

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["type"] == "overlay_video"
        assert attrs["video_file"] == "video.webm"
        assert attrs["poster_file"] == "poster.png"
        assert (
            attrs["loop"] is True
            and attrs["autoplay"] is True
            and attrs["muted"] is True
        )
        assert attrs["playback_rate"] == 1.5
        # A None height survives as null: the viewer keeps the video's aspect.
        assert attrs["size"] == [0.26, None]
        # An ordinary clip carries no matte layout.
        assert "alpha_matte" not in attrs

    def test_video_alpha_matte_is_recorded_and_validated(self, tmp_path) -> None:
        """`alpha_matte="stacked"` marks a colour-over-matte clip for the viewer.

        Transparency travels as a grey matte stacked below the colour (one opaque
        frame twice as tall) because Safari / WKWebView drop a VP9 alpha plane.
        """
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_video(self.WEBM_BYTES, position=(0.1, 0.5), alpha_matte="stacked")
            with pytest.raises(ValueError, match="alpha_matte must be one of"):
                scene.add_video(
                    self.WEBM_BYTES, position=(0.1, 0.5), alpha_matte="side-by-side"
                )
            with pytest.raises(
                ValueError, match='alpha_matte="stacked" requires autoplay=True'
            ):
                scene.add_video(
                    self.WEBM_BYTES,
                    position=(0.1, 0.5),
                    alpha_matte="stacked",
                    autoplay=False,
                )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["alpha_matte"] == "stacked"

    def test_video_from_path_detects_mp4(self, tmp_path) -> None:
        f = tmp_path / "clip.mp4"
        f.write_bytes(self.MP4_BYTES)
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_video(str(f), position=(0.1, 0.1))
        assert (
            tmp_path / "test.luxar.zarr" / "overlays" / "overlay_0" / "video.mp4"
        ).exists()

    def test_suffix_must_match_payload(self, tmp_path) -> None:
        f = tmp_path / "clip.mp4"
        f.write_bytes(self.WEBM_BYTES)  # says mp4, is webm
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="does not match its payload"):
                scene.add_video(str(f), position=(0.1, 0.1))

    def test_unsupported_payload_is_refused(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="Unsupported video payload"):
                scene.add_video(b"OggS" + b"\x00" * 40, position=(0.1, 0.1))
            with pytest.raises(ValueError, match="autoplay=True requires muted=True"):
                scene.add_video(self.WEBM_BYTES, position=(0.1, 0.1), muted=False)
            with pytest.raises(ValueError, match="playback_rate"):
                scene.add_video(self.WEBM_BYTES, position=(0.1, 0.1), playback_rate=0)

    def test_invalid_size_is_reported_before_an_invalid_payload(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(
                ValueError,
                match=r"size must be \(width, height-or-None\) with width > 0",
            ):
                scene.add_video(b"not a video", position=(0.1, 0.1), size=(0.0, None))

    def test_visible_range_is_validated_like_other_overlays(self, tmp_path) -> None:
        dims = Dimensions(
            [
                Dimension("story", unit="", categories=["a", "b"], display=False),
                Dimension("x", unit="", display=True),
                Dimension("y", unit="", display=True),
                Dimension("z", unit="", display=True),
            ]
        )
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=dims)
            scene.add_video(
                self.WEBM_BYTES, position=(0.1, 0.1), visible_range={"story": 1}
            )
            with pytest.raises(ValueError, match="Unknown dimension"):
                scene.add_video(
                    self.WEBM_BYTES, position=(0.1, 0.1), visible_range={"nope": 1}
                )
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert dict(store["overlays/overlay_0"].attrs)["visible_range"] == {
            "story": 1.0
        }


class TestAddHtml:
    """Test Scene.add_html()."""

    def test_basic_html(self, tmp_path) -> None:
        """Add basic HTML overlay."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            overlay = scene.add_html(
                "<p>Hello <strong>World</strong></p>",
                position=(0.1, 0.5),
            )

        assert overlay.overlay_type == "overlay_html"

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["type"] == "overlay_html"
        assert "<strong>World</strong>" in attrs["html"]

    def test_html_with_interactive(self, tmp_path) -> None:
        """Interactive HTML overlay stores interactive=True."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_html(
                '<a href="https://example.com">Link</a>',
                position=(0.5, 0.5),
                interactive=True,
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["interactive"] is True

    def test_html_sanitization_strips_script(self, tmp_path) -> None:
        """Script tags are stripped from HTML content."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_html(
                '<p>Safe</p><script>alert("xss")</script><p>Also safe</p>',
                position=(0.5, 0.5),
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        html = store["overlays/overlay_0"].attrs["html"]
        assert "<script>" not in html.lower()
        assert "alert" not in html
        assert "<p>Safe</p>" in html

    def test_html_sanitization_strips_event_handlers(self, tmp_path) -> None:
        """Event handler attributes are stripped."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_html(
                '<p onclick="alert(1)">Click me</p>',
                position=(0.5, 0.5),
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        html = store["overlays/overlay_0"].attrs["html"]
        assert "onclick" not in html.lower()
        assert "<p" in html

    def test_html_sanitization_strips_javascript_urls(self, tmp_path) -> None:
        """javascript: URLs are stripped from href/src."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_html(
                '<a href="javascript:alert(1)">Evil</a>',
                position=(0.5, 0.5),
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        html = store["overlays/overlay_0"].attrs["html"]
        assert "javascript:" not in html.lower()

    def test_html_with_width(self, tmp_path) -> None:
        """HTML overlay with width parameter."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_html(
                "<p>Content</p>",
                position=(0.1, 0.1),
                width=0.4,
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["width"] == 0.4


class TestVisibleRange:
    """Test dimension-aware visibility filtering."""

    def test_visible_range_single_value(self, tmp_path) -> None:
        """Exact value match is stored as a float."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())
            scene.add_text(
                "At time 5",
                position=(0.5, 0.5),
                visible_range={"time": 5},
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        vr = store["overlays/overlay_0"].attrs["visible_range"]
        assert vr == {"time": 5.0}

    def test_visible_range_tuple(self, tmp_path) -> None:
        """Range tuple is stored as a [min, max] list."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())
            scene.add_text(
                "Time 5-10",
                position=(0.5, 0.5),
                visible_range={"time": (5, 10)},
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        vr = store["overlays/overlay_0"].attrs["visible_range"]
        assert vr == {"time": [5.0, 10.0]}

    def test_visible_range_invalid_dimension(self, tmp_path) -> None:
        """Unknown dimension name raises ValueError."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())
            with pytest.raises(ValueError, match="Unknown dimension"):
                scene.add_text(
                    "Bad",
                    position=(0.5, 0.5),
                    visible_range={"nonexistent": 5},
                )

    def test_visible_range_min_greater_than_max(self, tmp_path) -> None:
        """Min > max in range raises ValueError."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())
            with pytest.raises(ValueError, match="min.*max"):
                scene.add_text(
                    "Bad range",
                    position=(0.5, 0.5),
                    visible_range={"time": (10, 5)},
                )

    def test_no_visible_range_means_always_visible(self, tmp_path) -> None:
        """Overlay without visible_range should have no visible_range in attrs."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())
            scene.add_text("Always", position=(0.5, 0.5))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "visible_range" not in store["overlays/overlay_0"].attrs


class TestValidation:
    """Test parameter validation errors."""

    def test_position_out_of_range(self, tmp_path) -> None:
        """Position outside [0, 1] raises ValueError."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="position"):
                scene.add_text("Bad", position=(1.5, 0.5))

    def test_position_negative(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="position"):
                scene.add_text("Bad", position=(-0.1, 0.5))

    def test_invalid_anchor(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="anchor"):
                scene.add_text("Bad", position=(0.5, 0.5), anchor="middle")

    def test_invalid_transition(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="transition"):
                scene.add_text("Bad", position=(0.5, 0.5), transition="slide")

    def test_invalid_text_align(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="text_align"):
                scene.add_text("Bad", position=(0.5, 0.5), text_align="middle")

    def test_text_blend_mode_written_to_attrs(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("Title", position=(0.02, 0.02), blend_mode="difference")

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert attrs["blend_mode"] == "difference"

    def test_text_blend_mode_normal_omitted(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("Title", position=(0.02, 0.02))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        attrs = dict(store["overlays/overlay_0"].attrs)
        assert "blend_mode" not in attrs

    def test_invalid_text_blend_mode(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="blend_mode"):
                scene.add_text("Bad", position=(0.5, 0.5), blend_mode="dodge")

    def test_invalid_html_blend_mode(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="blend_mode"):
                scene.add_html("<p>Bad</p>", position=(0.5, 0.5), blend_mode="dodge")

    def test_invalid_blend_mode(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="blend_mode"):
                scene.add_image(b"\x89PNG", position=(0.5, 0.5), blend_mode="dodge")

    def test_invalid_image_format(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="format"):
                scene.add_image(b"\x89PNG", position=(0.5, 0.5), format="bmp")

    def test_name_with_slash(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="/"):
                scene.add_text("Bad", position=(0.5, 0.5), name="a/b")

    def test_position_wrong_length(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="position"):
                scene.add_text("Bad", position=(0.5, 0.5, 0.5))

    def test_duplicate_name_raises(self, tmp_path) -> None:
        """Adding two overlays with the same explicit name raises ValueError."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("First", position=(0.1, 0.1), name="title")
            with pytest.raises(ValueError, match="already exists"):
                scene.add_text("Second", position=(0.9, 0.9), name="title")

    def test_duplicate_name_across_types(self, tmp_path) -> None:
        """Duplicate detection works across overlay types (text vs html)."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("Hello", position=(0.5, 0.5), name="info")
            with pytest.raises(ValueError, match="already exists"):
                scene.add_html("<p>Hi</p>", position=(0.5, 0.5), name="info")


class TestMixedOverlays:
    """Test combinations of different overlay types."""

    def test_mixed_overlay_types(self, tmp_path) -> None:
        """Scene with text, image, and HTML overlays together."""
        png_bytes = TestAddImage._make_tiny_png()

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())

            t = scene.add_text("Title", position=(0.5, 0.02), font_size=0.04)
            i = scene.add_image(png_bytes, position=(0.95, 0.95), anchor="bottom-right")
            h = scene.add_html("<p>Info</p>", position=(0.01, 0.5), width=0.25)

        assert len(scene.overlays) == 3
        assert t.overlay_type == "overlay_text"
        assert i.overlay_type == "overlay_image"
        assert h.overlay_type == "overlay_html"

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["type"] == "overlay_text"
        assert store["overlays/overlay_1"].attrs["type"] == "overlay_image"
        assert store["overlays/overlay_2"].attrs["type"] == "overlay_html"

    def test_overlays_alongside_geometry(self, tmp_path) -> None:
        """Overlays coexist with points/lines geometry."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.randn(100, 3).astype(np.float32)
            scene.add_points("pts", positions)
            scene.add_text("Label", position=(0.5, 0.5))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        # Both geometry and overlays present
        assert "pts" in store
        assert "overlays/overlay_0" in store

    def test_dimension_aware_overlays_multiple(self, tmp_path) -> None:
        """Multiple overlays with different visible_range values."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=_make_4d_dims())

            scene.add_text(
                "Window A",
                position=(0.5, 0.1),
                visible_range={"time": (0, 30)},
                transition="fade",
            )
            scene.add_text(
                "Window B",
                position=(0.5, 0.1),
                visible_range={"time": (31, 70)},
                transition="fade",
            )
            scene.add_text(
                "Window C",
                position=(0.5, 0.1),
                visible_range={"time": (71, 100)},
                transition="fade",
            )
            scene.add_text("Always visible", position=(0.5, 0.95))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["visible_range"] == {"time": [0, 30]}
        assert store["overlays/overlay_1"].attrs["visible_range"] == {"time": [31, 70]}
        assert store["overlays/overlay_2"].attrs["visible_range"] == {"time": [71, 100]}
        assert "visible_range" not in store["overlays/overlay_3"].attrs


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_position_at_corners(self, tmp_path) -> None:
        """Position at exact corners (0,0) and (1,1) should work."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("TL", position=(0.0, 0.0))
            scene.add_text("BR", position=(1.0, 1.0))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["position"] == [0.0, 0.0]
        assert store["overlays/overlay_1"].attrs["position"] == [1.0, 1.0]

    def test_empty_text(self, tmp_path) -> None:
        """Empty text string should be allowed."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text("", position=(0.5, 0.5))

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert store["overlays/overlay_0"].attrs["text"] == ""

    def test_unicode_text(self, tmp_path) -> None:
        """Unicode text (Greek, CJK, emoji) should work."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_text(
                "10 \u03bcm scale bar \u2014 \u7ec6\u80de", position=(0.5, 0.5)
            )

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        text = store["overlays/overlay_0"].attrs["text"]
        assert "\u03bc" in text
        assert "\u7ec6\u80de" in text

    def test_all_nine_anchors(self, tmp_path) -> None:
        """All 9 anchor values should be accepted."""
        anchors = [
            "top-left",
            "top-center",
            "top-right",
            "center-left",
            "center",
            "center-right",
            "bottom-left",
            "bottom-center",
            "bottom-right",
        ]
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            for i, anchor in enumerate(anchors):
                scene.add_text(anchor, position=(0.5, 0.5), anchor=anchor, name=f"a{i}")

        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        for i, anchor in enumerate(anchors):
            assert store[f"overlays/a{i}"].attrs["anchor"] == anchor

    def test_overlay_repr(self, tmp_path) -> None:
        """Overlay __repr__ is human-readable."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            overlay = scene.add_text("Test", position=(0.5, 0.3))

        r = repr(overlay)
        assert "overlay_0" in r
        assert "overlay_text" in r
        assert "0.50" in r
        assert "0.30" in r
