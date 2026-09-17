"""Dataset attribution: the validator, and its trip into a scene's root attrs.

The point of these tests is that a credit cannot be silently lost or silently
malformed. A wrong attribution is worse than a missing one, so the validator is
tested for what it *rejects* at least as hard as for what it accepts.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.citation import (
    CITATION_KEYS,
    CITATION_REF_MAX_LENGTH,
    validate_citation,
)

FULL = {
    "short": "Bui et al. 2013",
    "doi": "10.1016/j.cell.2013.10.055",
    "license": "CC BY 4.0",
    "url": "https://example.org/npc",
}


def test_none_means_nothing_to_credit() -> None:
    assert validate_citation(None) is None


def test_minimal_citation_is_just_a_short_form() -> None:
    assert validate_citation({"short": "Yeh 2022"}) == {"short": "Yeh 2022"}


def test_citation_accepts_a_compact_caption_reference() -> None:
    citation = {"short": "A deliberately detailed dataset byline", "ref": "Yeh 2022"}
    assert validate_citation(citation) == citation


def test_caption_reference_has_a_hard_authoring_bound() -> None:
    too_long = "x" * (CITATION_REF_MAX_LENGTH + 1)
    with pytest.raises(ValueError, match=f"at most {CITATION_REF_MAX_LENGTH}"):
        validate_citation({"short": "A dataset", "ref": too_long})


def test_returns_a_copy_not_the_callers_dict() -> None:
    """A demo's DEMO_META literal must not become aliased scene state."""
    source = dict(FULL)
    result = validate_citation(source)
    assert result == FULL
    assert result is not source
    result["short"] = "mutated"
    assert source["short"] == "Bui et al. 2013"


def test_key_order_is_canonical_regardless_of_input_order() -> None:
    shuffled = {"url": FULL["url"], "short": FULL["short"], "doi": FULL["doi"]}
    assert list(validate_citation(shuffled)) == ["short", "doi", "url"]


@pytest.mark.parametrize(
    "bad, expected",
    [
        ("Bui et al. 2013", "must be None or a mapping"),
        (42, "must be None or a mapping"),
        ({}, "short must be a non-empty string"),
        ({"doi": "10.1000/x"}, "short must be a non-empty string"),
        ({"short": ""}, "short must be a non-empty string"),
        ({"short": "   "}, "short must be a non-empty string"),
        ({"short": None}, "short must be a non-empty string"),
        ({"short": 2013}, "short must be a non-empty string"),
        ({"short": "Bui et al.\n2013"}, "short must be a single line"),
        # "Single line" must mean every line break and control character, not
        # just \n -- a lone \r still breaks a line in many renderers.
        ({"short": "Bui et al.\r2013"}, "single line of printable text"),
        ({"short": "Bui et al.\t2013"}, "single line of printable text"),
        # A bidi override can make a rendered credit read differently from the
        # stored string, which in an attribution field is spoofing.
        ({"short": "Bui\u202e et al. 2013"}, "single line of printable text"),
        # DOI near-misses people actually paste.
        ({"short": "A", "doi": "10."}, "must be a bare DOI"),
        ({"short": "A", "doi": "10.1016"}, "must be a bare DOI"),
        ({"short": "A", "doi": "10.1016/"}, "must be a bare DOI"),
        ({"short": "A", "doi": " 10.1016/x "}, "must be a bare DOI"),
        ({"short": "A", "author": "B"}, "unknown keys ['author']"),
        ({"short": "A", "license": ""}, "license must be a non-empty string"),
        ({"short": "A", "ref": ""}, "ref must be a non-empty string"),
        ({"short": "A", "url": None}, "url must be a non-empty string"),
        # The single-line/no-spoofing rule is not specific to `short`: every
        # field here is rendered, and a DOI suffix is otherwise free to hold
        # anything non-whitespace.
        ({"short": "A", "license": "CC BY 4.0\nand also"}, "license must be a single"),
        ({"short": "A", "doi": "10.1000/x\u202ey"}, "doi must be a single"),
        (
            {"short": "A", "url": "https://example.org/\u202egpj.exe"},
            "url must be a single",
        ),
        # A credit's URL becomes a link, so an executable/inline-payload scheme
        # must not be storable in data that is copied and published onward.
        ({"short": "A", "url": "javascript:alert(1)"}, "must be an http"),
        ({"short": "A", "url": "data:text/html,<script>"}, "must be an http"),
        ({"short": "A", "url": "example.org/dataset"}, "must be an http"),
        ({"short": "A", "doi": "https://doi.org/10.1000/x"}, "must be a bare DOI"),
        ({"short": "A", "doi": "doi:10.1000/x"}, "must be a bare DOI"),
    ],
)
def test_rejects_malformed_payloads(bad, expected: str) -> None:
    with pytest.raises(ValueError, match=expected.replace("[", r"\[")):
        validate_citation(bad)


def test_incidental_whitespace_is_stripped() -> None:
    """A stray space in a demo literal would render after the tile's em-dash."""
    got = validate_citation({"short": "  Bui et al. 2013  ", "license": " CC BY 4.0 "})
    assert got == {"short": "Bui et al. 2013", "license": "CC BY 4.0"}


def test_real_world_dois_are_accepted() -> None:
    """The tightened shape must not reject the DOIs the corpus actually uses."""
    for doi in (
        "10.1016/j.cell.2013.10.055",
        "10.5281/zenodo.10875063",
        "10.1051/0004-6361/202243940",  # suffix containing a second slash
        "10.1101/2024.10.18.618987",
        "10.25921/fd45-gt74",
        "10.1126/science.abl4896",
    ):
        assert validate_citation({"short": "A 2020", "doi": doi})["doi"] == doi


def test_citation_keys_matches_what_the_validator_accepts() -> None:
    """The advertised vocabulary and the enforced one cannot drift apart."""
    accepted = dict.fromkeys(CITATION_KEYS, "x")
    accepted["short"] = "A et al. 2020"
    accepted["doi"] = "10.1000/x"
    accepted["url"] = "https://example.org/dataset"
    assert set(validate_citation(accepted)) == set(CITATION_KEYS)


# --------------------------------------------------------------- scene stamping


def _write(tmp_path, name: str, **kwargs):
    store = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d(), **kwargs)
        rng = np.random.default_rng(0)
        scene.add_points("pts", rng.random((64, 3), dtype=np.float32))
    return store


def test_citation_reaches_the_store_root(tmp_path) -> None:
    store = _write(tmp_path, "cited", citation=FULL)
    assert dict(zarr.open_group(store, mode="r").attrs["citation"]) == FULL


def test_citation_and_viewer_config_both_survive(tmp_path) -> None:
    """Two separate root writes, one attrs dict — neither may clobber the other.

    Every credited microscopy demo passes both, so a root write that replaced
    attrs instead of merging them would silently strip whichever came first.
    """
    from luxar.core.viewer_config import ViewerConfig

    store = _write(
        tmp_path,
        "both",
        citation=FULL,
        viewer_config=ViewerConfig(tone_mapping="ACES"),
    )
    attrs = zarr.open_group(store, mode="r").attrs
    assert dict(attrs["citation"]) == FULL
    assert attrs["viewer_config"]["tone_mapping"] == "ACES"


def test_uncredited_scene_carries_no_citation_key(tmp_path) -> None:
    """Absent, not null: a missing key means "unknown", not "has no author"."""
    store = _write(tmp_path, "plain")
    assert "citation" not in zarr.open_group(store, mode="r").attrs


def test_explicit_none_is_the_same_as_omitting_it(tmp_path) -> None:
    store = _write(tmp_path, "procedural", citation=None)
    assert "citation" not in zarr.open_group(store, mode="r").attrs


def test_malformed_citation_never_reaches_the_data(tmp_path) -> None:
    store = tmp_path / "bad.luxar.zarr"
    with pytest.raises(ValueError, match="short must be a non-empty string"):
        with LuxarZarrCompiler(str(store)) as compiler:
            compiler.create_scene(
                dimensions=Dimensions.default_3d(), citation={"short": ""}
            )


def test_citation_survives_a_re_chunk(tmp_path) -> None:
    """`optimize` rebuilds a store's metadata; the credit must come through.

    The format spec promises the attribution "travels with the store", and a
    re-chunk is the journey most likely to break that promise: it rewrites every
    chunk key and stamps a fresh ``content_hash``. A credit silently dropped here
    would leave a published, re-chunked scene uncredited while the demo that
    produced it still claims otherwise.
    """
    from luxar.io.optimize import optimize_store

    src = _write(tmp_path, "before", citation=FULL)
    dst = tmp_path / "after.luxar.zarr"
    optimize_store(src, dst)

    assert dict(zarr.open_group(dst, mode="r").attrs["citation"]) == FULL


def test_scene_exposes_a_defensive_copy(tmp_path) -> None:
    store = tmp_path / "prop.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(), citation=dict(FULL)
        )
        assert scene.citation == FULL
        scene.citation["short"] = "mutated"
        assert scene.citation["short"] == "Bui et al. 2013"
        scene.add_points("pts", np.zeros((8, 3), dtype=np.float32))
