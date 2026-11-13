"""Tests for a scene's colour legend (`luxar.core.legend`).

The legend is a root attribute like the citation: validated once, written by
`create_scene(legend=...)`, read back by `luxar export`. These pin the shape,
the validation, and that it reaches -- and round-trips through -- the store.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.legend import Legend, LegendEntry, hex_color, validate_legend

LIME = LegendEntry("Proteobacteria", "#66e680", "lime green", "Pseudomonadota.")
GREY = LegendEntry("Other", "#808080", "grey")
LEGEND = Legend(
    title="What the colours mean", entries=(LIME, GREY), note="Dimmed: dark."
)


def test_hex_color_is_upper_case_and_clamped() -> None:
    assert hex_color((0.4, 0.9, 0.5)) == "#AAF3BC"
    assert hex_color((0.5, 0.5, 0.5)) == "#BCBCBC"
    assert hex_color((0, 0, 0)) == "#000000"
    assert hex_color((1, 1, 1)) == "#FFFFFF"
    assert hex_color((1.5, -0.2, 0.5)) == "#FF00BC"
    with pytest.raises(ValueError, match="RGB triple"):
        hex_color((0.1, 0.2))


def test_an_entry_normalises_its_colour_and_drops_an_empty_description() -> None:
    assert LIME.color == "#66E680"
    assert GREY.to_dict() == {
        "label": "Other",
        "color": "#808080",
        "color_name": "grey",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "", "color": "#000000", "color_name": "black"},
        {"label": "x", "color": "000000", "color_name": "black"},
        {"label": "x", "color": "#00000", "color_name": "black"},
        {"label": "x", "color": "#GGGGGG", "color_name": "black"},
        {"label": "x", "color": "#000000", "color_name": ""},
        {"label": "x\ny", "color": "#000000", "color_name": "black"},
        {"label": "x", "color": "#000000", "color_name": "black", "description": "a‮b"},
    ],
)
def test_a_malformed_entry_is_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="legend"):
        LegendEntry(**kwargs)


def test_a_legend_needs_unique_entries_and_a_title() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        Legend(title="t", entries=())
    with pytest.raises(ValueError, match="unique"):
        Legend(title="t", entries=(LIME, LIME))
    with pytest.raises(ValueError, match="title"):
        Legend(title=" ", entries=(LIME,))
    with pytest.raises(ValueError, match="LegendEntry"):
        Legend(title="t", entries=({"label": "x"},))  # type: ignore[arg-type]


def test_the_dict_form_round_trips() -> None:
    stored = LEGEND.to_dict()
    assert stored["entries"][0]["color"] == "#66E680"
    assert Legend.from_dict(stored) == LEGEND
    assert validate_legend(None) is None
    assert validate_legend(LEGEND) == stored
    with pytest.raises(ValueError, match="Legend"):
        validate_legend(stored)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"title": "t"},
        {"title": "t", "entries": "no"},
        {"title": "t", "entries": [1]},
    ],
)
def test_a_malformed_stored_legend_raises(bad: object) -> None:
    with pytest.raises(ValueError):
        Legend.from_dict(bad)  # type: ignore[arg-type]


def test_the_legend_reaches_the_store_root_and_the_scene(tmp_path) -> None:
    store = tmp_path / "legend.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d(), legend=LEGEND)
        scene.add_points(
            "pts", np.random.default_rng(0).random((16, 3), dtype=np.float32)
        )
        assert scene.legend == LEGEND
    assert (
        Legend.from_dict(dict(zarr.open_group(store, mode="r").attrs["legend"]))
        == LEGEND
    )


def test_no_legend_writes_no_key(tmp_path) -> None:
    store = tmp_path / "plain.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", np.zeros((4, 3), dtype=np.float32))
        assert scene.legend is None
    assert "legend" not in zarr.open_group(store, mode="r").attrs


def test_a_non_string_description_names_its_type() -> None:
    with pytest.raises(ValueError, match="must be a string, got NoneType"):
        LegendEntry("x", "#000000", "black", None)  # type: ignore[arg-type]


def test_the_printed_fields_are_capped() -> None:
    with pytest.raises(ValueError, match="at most 80 fit"):
        LegendEntry("x" * 81, "#000000", "black")
    with pytest.raises(ValueError, match="at most 400 fit"):
        LegendEntry("x", "#000000", "black", "d" * 401)
    with pytest.raises(ValueError, match="at most 64 entries"):
        Legend(
            title="t",
            entries=tuple(LegendEntry(f"g{i}", "#000000", "black") for i in range(65)),
        )
    assert len(Legend(title="t", entries=tuple(
        LegendEntry(f"g{i}", "#000000", "black") for i in range(64)
    )).entries) == 64  # fmt: skip
