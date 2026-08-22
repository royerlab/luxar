"""Shared demo captions keep attribution visible and consistently styled."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from luxar.demos._caption import add_demo_caption, format_demo_caption


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
