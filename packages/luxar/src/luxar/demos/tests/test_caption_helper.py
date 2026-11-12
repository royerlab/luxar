"""Shared demo captions keep attribution visible and consistently styled."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from luxar.demos._caption import (
    add_demo_caption,
    compression_caption,
    compression_note,
    compression_ratio,
    format_bytes,
    format_demo_caption,
    format_ratio,
)


def test_uncited_caption_is_unchanged() -> None:
    assert format_demo_caption("100³ grid • Depth gradient", None) == (
        "100³ grid • Depth gradient"
    )


def test_compact_short_is_the_default_reference() -> None:
    assert format_demo_caption("8-fold symmetry", {"short": "Bui et al. 2013"}) == (
        "8-fold symmetry • Bui et al. 2013"
    )


def test_explicit_reference_keeps_a_long_byline_out_of_the_caption() -> None:
    citation = {
        "short": "OpenCell (Cho et al. 2022); embeddings by cytoself (Kobayashi et al. 2022)",
        "ref": "Cho et al. 2022",
    }
    assert format_demo_caption("OpenCell • 3D UMAP", citation) == (
        "OpenCell • 3D UMAP • Cho et al. 2022"
    )


def test_existing_reference_tail_is_not_duplicated() -> None:
    caption = "95K cells • 32 cell types • Kim et al. 2024"
    assert format_demo_caption(caption, {"short": "Kim et al. 2024"}) == caption


def test_long_short_requires_an_explicit_reference() -> None:
    with pytest.raises(ValueError, match="citation.ref is required"):
        format_demo_caption("A caption", {"short": "x" * 41})


def test_add_demo_caption_owns_the_bottom_right_overlay_style() -> None:
    scene = Mock()
    add_demo_caption(scene, "8-fold symmetry", {"short": "Bui et al. 2013"})
    scene.add_text.assert_called_once_with(
        "8-fold symmetry • Bui et al. 2013",
        position=(0.98, 0.97),
        font_size=0.015,
        anchor="bottom-right",
        color="rgba(200,200,200,0.5)",
    )


def test_compression_ratio_is_raw_over_compressed() -> None:
    assert compression_ratio(1000, 10) == 100.0
    with pytest.raises(ValueError, match="positive"):
        compression_ratio(1000, 0)


@pytest.mark.parametrize(
    ("ratio", "text"),
    [
        (4.24, "4.2x"),
        (9.94, "9.9x"),
        (9.96, "10x"),
        (42.4, "42x"),
        (314.87, "315x"),
        (999.6, "1,000x"),
        (12_345.0, "12,300x"),
        (1_234_567.0, "1,230,000x"),
    ],
)
def test_format_ratio_rounds_to_what_the_sizes_support(ratio: float, text: str) -> None:
    assert format_ratio(ratio) == text


@pytest.mark.parametrize(
    ("n_bytes", "text"),
    [
        (512, "512 B"),
        (1_500, "1.50 KB"),
        (99_566_681, "99.6 MB"),
        (110_718_328, "111 MB"),
        (999_600_000, "1.00 GB"),
        (99_994_000, "100 MB"),  # rounds to 3 figures before choosing decimals
        (34_861_369_536, "34.9 GB"),
        (2_500_000_000_000_000, "2,500 TB"),
    ],
)
def test_format_bytes_uses_decimal_units(n_bytes: int, text: str) -> None:
    assert format_bytes(n_bytes) == text


def test_compression_texts_state_both_on_disk_sizes() -> None:
    raw, scene = 34_861_369_536, 110_718_328
    assert compression_caption(raw, scene, "raw CZI") == (
        "111 MB, ≈315x smaller than the raw CZI"
    )
    assert compression_note(raw, scene, "raw CZI") == (
        "111 MB on disk, from the 34.9 GB raw CZI (≈315x smaller)"
    )
