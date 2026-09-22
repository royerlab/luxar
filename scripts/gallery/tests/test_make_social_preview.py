"""Offline tests for the social-preview banner generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "make_social_preview.py"
SPEC = importlib.util.spec_from_file_location("make_social_preview", SCRIPT)
assert SPEC and SPEC.loader
msp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = msp
SPEC.loader.exec_module(msp)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("file:frame.png", ("file:frame.png", 0.5)),
        ("file:frame.png@0.9", ("file:frame.png", 0.9)),
        ("url:https://h/x.png@0", ("url:https://h/x.png", 0.0)),
        ("url:https://h/x.png@7", ("url:https://h/x.png", 1.0)),  # clamped
        (
            "url:https://user@host/x.png",
            ("url:https://user@host/x.png", 0.5),
        ),  # not an anchor
        ("key:lorenz", ("key:lorenz", 0.5)),
    ],
)
def test_split_source_parses_optional_anchor(
    source: str, expected: tuple[str, float]
) -> None:
    assert msp.split_source(source) == expected


def test_cover_crop_scales_to_cover_and_honours_the_anchor() -> None:
    # 60 wide x 20 tall, one colour per column, cropped to a square: covering the
    # height scales the image to 120 x 40, so the anchor picks which 40 columns.
    image = Image.new("RGB", (60, 20))
    for x in range(60):
        for y in range(20):
            image.putpixel((x, y), (x * 4, 0, 0))
    left = msp.cover_crop(image, 40, 40, anchor=0.0)
    centre = msp.cover_crop(image, 40, 40, anchor=0.5)
    right = msp.cover_crop(image, 40, 40, anchor=1.0)
    assert left.size == centre.size == right.size == (40, 40)
    assert (
        left.getpixel((0, 20))[0]
        < centre.getpixel((0, 20))[0]
        < right.getpixel((0, 20))[0]
    )
    assert left.getpixel((0, 20))[0] <= 4  # starts at the leftmost source column
    assert (
        right.getpixel((39, 20))[0] >= 4 * 59 - 8
    )  # ends at the rightmost source column


def test_regular_weight_never_falls_back_to_a_bold_only_file() -> None:
    assert all("Bold" not in path for path, _ in msp.REGULAR_FONTS)
    regular = msp.font(20, bold=False)
    if hasattr(regular, "getname"):  # a FreeType face; the bitmap fallback has no name
        assert "Bold" not in regular.getname()[1]


def test_non_https_urls_are_refused() -> None:
    with pytest.raises(SystemExit, match="https"):
        msp.load_panel("url:http://example.com/x.png")
    with pytest.raises(SystemExit, match="key:, url: or file:"):
        msp.load_panel("ftp:whatever")


def test_compose_produces_the_github_card_size() -> None:
    blank = Image.new("RGB", (64, 64), (10, 20, 30))
    card = msp.compose([(blank, 0.5)] * 4, scale=1)
    assert card.size == (1280, 640)
    assert msp.compose([(blank, 0.5)] * 4, scale=2).size == (2560, 1280)
