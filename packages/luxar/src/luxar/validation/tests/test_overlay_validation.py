"""Tests for overlay validation functions."""

import numpy as np
import pytest

from luxar.validation.overlays import (
    VALID_ANCHORS,
    VALID_BLEND_MODES,
    VALID_IMAGE_FORMATS,
    VALID_TEXT_ALIGNS,
    VALID_TRANSITIONS,
    sanitize_html,
    validate_anchor,
    validate_blend_mode,
    validate_font,
    validate_image_format,
    validate_image_input,
    validate_position,
    validate_text_align,
    validate_transition,
    validate_visible_range,
)


class TestValidatePosition:
    def test_valid_tuple(self):
        assert validate_position((0.5, 0.5)) == (0.5, 0.5)

    def test_valid_list(self):
        assert validate_position([0.0, 1.0]) == (0.0, 1.0)

    def test_corners(self):
        assert validate_position((0.0, 0.0)) == (0.0, 0.0)
        assert validate_position((1.0, 1.0)) == (1.0, 1.0)

    def test_out_of_range_high(self):
        with pytest.raises(ValueError, match="position"):
            validate_position((1.1, 0.5))

    def test_out_of_range_low(self):
        with pytest.raises(ValueError, match="position"):
            validate_position((-0.1, 0.5))

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="2 floats"):
            validate_position((0.5,))

    def test_wrong_type(self):
        with pytest.raises(ValueError, match="2 floats"):
            validate_position(0.5)


class TestValidateAnchor:
    def test_all_valid_anchors(self):
        for anchor in VALID_ANCHORS:
            assert validate_anchor(anchor) == anchor

    def test_invalid(self):
        with pytest.raises(ValueError, match="anchor"):
            validate_anchor("middle")


class TestValidateFont:
    def test_presets(self):
        for preset in ("sans", "serif", "mono"):
            assert validate_font(preset) == preset

    def test_custom_font(self):
        assert validate_font("Helvetica Neue") == "Helvetica Neue"

    def test_empty_string(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_font("")


class TestValidateBlendMode:
    def test_all_valid(self):
        for mode in VALID_BLEND_MODES:
            assert validate_blend_mode(mode) == mode

    def test_invalid(self):
        with pytest.raises(ValueError, match="blend_mode"):
            validate_blend_mode("dodge")


class TestValidateTransition:
    def test_valid(self):
        for t in VALID_TRANSITIONS:
            assert validate_transition(t) == t

    def test_invalid(self):
        with pytest.raises(ValueError, match="transition"):
            validate_transition("slide")


class TestValidateTextAlign:
    def test_valid(self):
        for align in VALID_TEXT_ALIGNS:
            assert validate_text_align(align) == align

    def test_invalid(self):
        with pytest.raises(ValueError, match="text_align"):
            validate_text_align("middle")


class TestValidateImageFormat:
    def test_valid(self):
        for fmt in VALID_IMAGE_FORMATS:
            assert validate_image_format(fmt) == fmt

    def test_invalid(self):
        with pytest.raises(ValueError, match="format"):
            validate_image_format("bmp")


class TestValidateVisibleRange:
    def test_none(self):
        assert validate_visible_range(None, ["X", "Y"]) is None

    def test_single_value(self):
        result = validate_visible_range({"time": 5}, ["time", "X", "Y"])
        assert result == {"time": 5.0}

    def test_range_tuple(self):
        result = validate_visible_range({"time": (5, 10)}, ["time", "X"])
        assert result == {"time": [5.0, 10.0]}

    def test_range_list(self):
        result = validate_visible_range({"time": [5, 10]}, ["time", "X"])
        assert result == {"time": [5.0, 10.0]}

    def test_unknown_dimension(self):
        with pytest.raises(ValueError, match="Unknown dimension"):
            validate_visible_range({"bad": 5}, ["time", "X"])

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min.*max"):
            validate_visible_range({"time": (10, 5)}, ["time", "X"])

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="dict"):
            validate_visible_range("bad", ["time"])

    def test_invalid_value_type(self):
        with pytest.raises(ValueError, match="number or.*tuple"):
            validate_visible_range({"time": "bad"}, ["time"])

    def test_multiple_dimensions(self):
        result = validate_visible_range(
            {"time": (0, 50), "channel": 2},
            ["time", "channel", "X", "Y"],
        )
        assert result == {"time": [0.0, 50.0], "channel": 2.0}


class TestValidateImageInput:
    def test_bytes_passthrough(self):
        data, fmt = validate_image_input(b"\x89PNG\r\n\x1a\n")
        assert data == b"\x89PNG\r\n\x1a\n"
        assert fmt == "png"

    def test_bytes_ignore_format_kwarg(self):
        data, fmt = validate_image_input(b"\xff\xd8", fmt="png")
        assert data == b"\xff\xd8"
        assert fmt == "jpeg"

    def test_webp_bytes_detected(self):
        data, fmt = validate_image_input(b"RIFF\x04\x00\x00\x00WEBP")
        assert data == b"RIFF\x04\x00\x00\x00WEBP"
        assert fmt == "webp"

    def test_unsupported_bytes_rejected(self):
        with pytest.raises(ValueError, match="encoded image format") as exc:
            validate_image_input(b"not an image")
        assert str(exc.value).endswith("PNG, JPEG, or WebP payload.")

    def test_imageio_failure_preserves_exception_details(self):
        class BrokenArray:
            def __array__(self, *_args, **_kwargs):
                raise RuntimeError("decoder exploded")

        with pytest.raises(
            ValueError,
            match="imageio failed with RuntimeError: decoder exploded",
        ) as exc:
            validate_image_input(BrokenArray())

        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_numpy_rgb(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        data, fmt = validate_image_input(arr)
        assert len(data) > 0
        assert fmt == "png"

    def test_numpy_rgba(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        data, fmt = validate_image_input(arr)
        assert len(data) > 0

    def test_numpy_grayscale(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        data, fmt = validate_image_input(arr)
        assert len(data) > 0

    def test_numpy_float(self):
        arr = np.random.rand(4, 4, 3).astype(np.float32)
        data, fmt = validate_image_input(arr)
        assert len(data) > 0

    def test_file_path(self, tmp_path):
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"\xff\xd8")
        data, fmt = validate_image_input(str(test_file))
        assert data == b"\xff\xd8"
        assert fmt == "jpeg"

    @pytest.mark.parametrize("filename", ["test", "test.tmp"])
    def test_file_path_without_recognized_suffix(self, tmp_path, filename):
        image_bytes, _ = validate_image_input(np.zeros((2, 2, 3), dtype=np.uint8))
        test_file = tmp_path / filename
        test_file.write_bytes(image_bytes)

        data, fmt = validate_image_input(test_file)

        assert data == image_bytes
        assert fmt == "png"

    def test_file_suffix_must_match_payload(self, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\xff\xd8")
        with pytest.raises(ValueError, match="extension.*payload"):
            validate_image_input(test_file)

    def test_unsupported_file_payload_names_path(self, tmp_path):
        test_file = tmp_path / "logo.gif"
        test_file.write_bytes(b"GIF89a")

        with pytest.raises(ValueError, match="Unsupported encoded image format") as exc:
            validate_image_input(test_file)
        assert str(test_file) in str(exc.value)

    def test_nonexistent_file(self):
        with pytest.raises(ValueError, match="not found"):
            validate_image_input("/no/such/file.png")

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="format"):
            validate_image_input(b"\x89PNG", fmt="bmp")

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Cannot process"):
            validate_image_input(12345)


class TestSanitizeHtml:
    def test_safe_html_unchanged(self):
        html = "<p>Hello <strong>world</strong></p>"
        assert sanitize_html(html) == html

    def test_strips_script_tags(self):
        result = sanitize_html("<p>OK</p><script>alert(1)</script>")
        assert "<script>" not in result
        assert "alert" not in result
        assert "<p>OK</p>" in result

    def test_strips_iframe(self):
        result = sanitize_html('<iframe src="evil.html"></iframe>')
        assert "<iframe" not in result

    def test_strips_event_handlers(self):
        result = sanitize_html('<p onclick="alert(1)">Click</p>')
        assert "onclick" not in result
        assert "<p" in result

    def test_strips_javascript_urls(self):
        result = sanitize_html('<a href="javascript:alert(1)">Link</a>')
        assert "javascript:" not in result.lower()

    def test_allows_inline_styles(self):
        html = '<span style="color:red">Red text</span>'
        result = sanitize_html(html)
        assert 'style="color:red"' in result

    def test_allows_safe_links(self):
        html = '<a href="https://example.com">Link</a>'
        result = sanitize_html(html)
        assert 'href="https://example.com"' in result

    def test_strips_form_elements(self):
        result = sanitize_html('<form><input type="text"></form>')
        assert "<form" not in result
        assert "<input" not in result

    def test_not_a_string(self):
        with pytest.raises(ValueError, match="string"):
            sanitize_html(123)

    def test_strips_style_tags(self):
        result = sanitize_html("<style>body{display:none}</style><p>OK</p>")
        assert "<style>" not in result

    def test_case_insensitive_stripping(self):
        result = sanitize_html("<SCRIPT>alert(1)</SCRIPT>")
        assert "SCRIPT" not in result
        assert "alert" not in result

    def test_strips_svg(self):
        result = sanitize_html("<svg onload=alert(1)></svg>")
        assert "<svg" not in result
        assert "onload" not in result
        assert "alert" not in result

    def test_strips_media_and_foreign_tags(self):
        # Previously non-blacklisted tags that slipped straight through.
        for tag in ("video", "audio", "template", "link", "base", "math"):
            result = sanitize_html(f"<{tag}>x</{tag}>")
            assert f"<{tag}" not in result

    def test_strips_template_contents(self):
        result = sanitize_html("<template><script>alert(1)</script></template>")
        assert "<template" not in result
        assert "alert" not in result

    def test_entity_encoded_javascript_scheme(self):
        # &#106; decodes to 'j' -> 'javascript:'; the href must be dropped.
        result = sanitize_html('<a href="&#106;avascript:alert(1)">x</a>')
        assert "javascript:" not in result.lower()
        assert "href" not in result
        assert "x" in result

    def test_entity_encoded_control_char_scheme(self):
        # Browsers strip a tab inside the scheme: jav\tascript: -> javascript:.
        result = sanitize_html('<a href="jav&#9;ascript:alert(1)">x</a>')
        assert "javascript" not in result.lower()
        assert "href" not in result

    def test_allows_safe_data_image(self):
        html = '<img src="data:image/png;base64,AAAA" alt="x">'
        result = sanitize_html(html)
        assert 'src="data:image/png;base64,AAAA"' in result

    def test_rejects_data_svg(self):
        # SVG data URIs are scriptable and must be dropped, unlike raster.
        result = sanitize_html('<img src="data:image/svg+xml;base64,AAAA" alt="x">')
        assert "src" not in result
        assert "svg" not in result

    def test_rejects_data_texthtml(self):
        result = sanitize_html('<a href="data:text/html,<script>1</script>">x</a>')
        assert "href" not in result

    def test_rejects_vbscript(self):
        result = sanitize_html('<a href="vbscript:msgbox(1)">x</a>')
        assert "href" not in result
        assert "vbscript" not in result.lower()

    def test_documented_attributes_survive(self):
        # Attributes documented in LUXAR_ZARR_FORMAT.md / overlay-manager.ts.
        assert 'colspan="2"' in sanitize_html('<td colspan="2">x</td>')
        img = sanitize_html('<img src="x.png" width="120" height="16">')
        assert 'width="120"' in img
        assert 'height="16"' in img
        link = sanitize_html('<a href="https://e.com" title="t" rel="noopener">x</a>')
        assert 'title="t"' in link
        assert 'rel="noopener"' in link

    def test_bare_ampersands_roundtrip(self):
        # No spurious ';' invented, trailing '&' not dropped.
        result = sanitize_html("R&D and Q&A and AT&T")
        assert result == "R&amp;D and Q&amp;A and AT&amp;T"

    def test_void_drop_tag_keeps_following_text(self):
        assert "hello" in sanitize_html("<input>hello")

    def test_event_handler_does_not_destroy_content(self):
        # Regression: the old handler regex ate past the tag close, mangling
        # valid markup. All visible text must survive; only onclick is removed.
        result = sanitize_html("<b>bold</b><div onclick=f()>x</div><span>keep</span>")
        assert "onclick" not in result
        assert "<div>" in result
        assert "<div " not in result  # no leftover truncated/attr-bearing div
        assert "bold" in result
        assert "x" in result
        assert "keep" in result

    def test_unquoted_handler_keeps_text(self):
        result = sanitize_html("<div onclick=doit()>Important text</div>")
        assert "onclick" not in result
        assert "Important text" in result

    def test_preserves_text_entities(self):
        html = "<p>a &amp; b &lt; c</p>"
        assert sanitize_html(html) == html

    def test_malformed_marked_sections_do_not_raise(self):
        # Unpatched CPython's HTMLParser raises AssertionError on unknown
        # marked-section keywords; sanitize_html must never propagate that.
        for bad in ("<![bogus]>", "<![bogus", "<! >", "<!-", "</ x>", "<?php x ?>"):
            result = sanitize_html(bad)
            assert "<!" not in result
            assert "<?" not in result

    def test_parser_failure_falls_back_to_escaped_text(self, monkeypatch):
        # Simulate the stdlib parser blowing up (as it does on Pythons without
        # the 2025 html.parser patch): the input must come back fully escaped.
        from luxar.validation import overlays

        def boom(self, data):
            raise AssertionError("unknown status keyword 'bogus' in marked section")

        monkeypatch.setattr(overlays._HtmlSanitizer, "feed", boom)
        assert sanitize_html("<b>x</b> & y") == "&lt;b&gt;x&lt;/b&gt; &amp; y"
