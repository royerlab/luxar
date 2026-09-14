"""Substitutive levels and partitions on Points/Lines must be justified, not default.

``test_lod_policy.py`` gates the GSPLAT half of the policy: a demo that calls a
fitter must choose a topology through :mod:`luxar.demos._lod_policy`. Its universe
is ``_FIT_CALLS``, so it says nothing about a Points or Lines node — and that is
where the machinery had accumulated. Six embedding demos carried three or four
coarse replacement levels apiece for nodes whose RESIDENT slice ran five to
twenty-nine times under its cap, serving a framing the screen-area selector never
picks.

This is the other half. Loic's rule (2026-08-28):

    Subst. LODs don't make sense for most scenes where there is 'one' object or
    when everything can be displayed at once, or progressively via an additive
    ladder! [...] Partitions can also be removed unless there are too many
    elements per node.

So the gate is: any module asking for ``substitutive_lod=`` or ``partition=``, or
hand-building a ``kind=partition`` / ``kind=lod`` group, must appear in
:data:`JUSTIFIED` with a reason. An additive ladder needs no justification — it is
the cheap default the rule prefers, and every demo should have one.

Two things this deliberately does NOT do. It does not check element counts: the
resident slice is not statically knowable (it depends on the data a demo
downloads), and a wrong static estimate is worse than none — see
``_lod_policy`` on the two ways a partitioned node's resident count gets
mismeasured. And it does not forbid anything, because all seven current entries
are correct; it forces the next one to be argued.
"""

from __future__ import annotations

import ast

import pytest

from ._scanned_modules import scanned_demo_modules

#: Keyword arguments that request a structure the rule wants justified.
_GATED_KEYWORDS = frozenset({"substitutive_lod", "partition"})

#: Group builders that hand-build the same structures, bypassing the keywords.
#: ``add_partition_group`` + per-tile ``add_lod_group`` is the Points equivalent
#: of the gsplat ``adaptive`` recipe, and it is how the globe demos are authored,
#: so a gate keyed only on the keywords would miss them entirely.
_GATED_CALLS = frozenset({"add_partition_group", "add_lod_group"})

_CONVERTED_ADDITIVE_TARGETS = {
    "demo_arxiv_embeddings_kaggle.py": "arxiv_papers_kaggle",
    "demo_cellxgene_census_umap.py": "cells",
    "demo_esm3_protein_landscape.py": "proteins",
    "demo_human_multiome_peak_umap.py": "Cells",
    "demo_mouse_multiome_peak_umap.py": "Cells",
    "demo_zebrahub_multiome_peak_umap.py": "Cells",
}

#: Module -> reason, for every module that legitimately builds one of these.
#:
#: A reason must say WHY the cheap default is insufficient for that data, in terms
#: that can be checked against a measurement. "It is a big scene" is not a reason;
#: a resident count against a per-geometry cap, or a correctness constraint, is.
JUSTIFIED: dict[str, str] = {
    "demo_4d_fractals.py": (
        "44,469,233 points span 300 hidden (fractal, w) coordinates, and NO single "
        "leaf can satisfy scripts/check_demo_ladders.py: un-laddered it fails the "
        "no-ladder arm above 200,000, while laddered its rung 0 must be at least "
        "10% of the node (4,446,924 points) AND every increment must clear the "
        "1,000,000 absolute commit cap. The cap's per-coordinate relaxation needs "
        "EXACTLY ONE slice dim, so this leaf's TWO hidden columns leave the whole "
        "node-level cap in force and the 300 stops buy nothing: 4,446,924 > "
        "1,000,000, jointly unsatisfiable. The 297 partition children capped at the "
        "authored 150,000 points per plane fall under the gate's 200,000 threshold "
        "outright, and 37x below the 5,591,040 Points cap."
    ),
    "demo_esm_protein_universe.py": (
        "7,714,508 points RESIDENT in the backdrop (one hidden story coordinate, "
        "extended to every slot, so the resident slice IS the node) — past the "
        "5,591,040 Points cap, so the partition is required: a hand-built "
        "kind=partition of 64 BSP tiles of at most 125,000 points, each its own "
        "hand-authored kind=lod ladder whose coarse level is a 1-in-4 POINTS "
        "subsample (not the writer's synthesised Gaussians, which drew the view "
        "from a knot toward the centre at 6 fps against 144 fps for plain "
        "points). Per-tile selection draws ~1.9M points at the overview and the "
        "whole-map stories; a knot swaps in only its nearest handful of tiles."
    ),
    "demo_desi_galaxies.py": (
        "9,751,955 points RESIDENT with no hidden axis — genuinely past the "
        "5,591,040 Points cap, so the partition is required. The substitutive "
        "levels are required too, and for a separate reason: `partition=` alone "
        "is EAGER (every part is fetched; the GPU only frustum-culls at draw), so "
        "only the coarse levels bound residency. Removing them would load all "
        "9.75M at every framing."
    ),
    "demo_nuclear_pore_complex.py": (
        "the subunits are concave and INTERPENETRATE, so there is no valid "
        "whole-object draw order; the partition's `bsp_tree` split planes are the "
        "only correct traversal, camera-inside-the-channel included. Capacity is a "
        "second, weaker argument: 4,937,064 resident leaves 12% headroom under the "
        "Points cap."
    ),
    "demo_ocean_currents_earth.py": (
        "a globe panned and zoomed rather than orbited, so most of its 16 tiles are "
        "off screen most of the time — which is what PARTS are for, and the "
        "measured payoff is 547,000 elements resident at whole-globe framing "
        "against 9.9M zoomed in, a 33x reduction, with each tile walking its own "
        "level independently. Hand-built rather than `substitutive_lod=` because "
        "that coarsens a line set by SYNTHESISING gsplats, and the ribbon geometry "
        "is the picture; the levels here keep whole elements, with line width "
        "scaled linearly by the thinning factor."
    ),
    "demo_gaia_milky_way_3m.py": (
        "the galaxy is orbited at range AS WELL AS inspected close up, so a coarse "
        "level really is selected and really is fetched — the same case "
        "`_lod_policy` documents for `milky_way_dust`, which pays +39% on purpose."
    ),
    "demo_dmri_tractography.py": (
        "87 per-bundle ladders, and `compression_factor=256` was measured against "
        "the K=4 default at 2.1 GB versus ~200 MB for the whole scene. Removing "
        "them means loading all 87 bundles at once, which the gallery manifest "
        "records as never reaching networkidle inside the 120 s budget."
    ),
    "demo_zebrahub_velocity_streamlines.py": (
        "nothing is shipped to justify: the spec is None unless `--streamline-lod` "
        "is passed, and the published store carries 0 `kind=lod` groups. Listed "
        "only because the keyword appears in the source. If the flag is ever made "
        "default, re-derive first — the published finest level is ~2.71M elements "
        "against a 2,793,472 Lines cap, which is 97% of it and needs checking in "
        "segments rather than vertices before anything is concluded."
    ),
    "demo_biodiversity_planetary_scale.py": (
        "its remaining sites are inside `add_lod_tiles`, which is UNCALLED — the "
        "aggregate layer that needed it was removed for being the densest thing in "
        "the scene. The three shipped layers dropped their partitions (45,000 / "
        "106,026 / 95,418 resident against caps of 5,591,040 and 2,793,472). Listed "
        "so removing the dead helper also removes this entry."
    ),
}


def _is_disabled(node: ast.AST) -> bool:
    """Whether a value is a literal opt-out (``None`` / ``False``).

    ``partition=False`` is a documented no-partition sentinel and
    ``substitutive_lod=None`` is how a demo says "not here", so neither requests
    a structure. Anything else — a dict, a name, a call — does.
    """
    return isinstance(node, ast.Constant) and node.value in (None, False)


def gated_features(source: str, filename: str = "<source>") -> set[str]:
    """Names of the gated structures *source* requests.

    Parsed rather than grepped, for the reason the sibling gates give: the string
    ``"partition="`` appears in prose comments explaining why a demo does NOT use
    one, and several demos now carry exactly that comment.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _GATED_CALLS:
            found.add(name)
        for kw in node.keywords:
            if kw.arg in _GATED_KEYWORDS and not _is_disabled(kw.value):
                found.add(kw.arg)
    return found


def additive_ladder_targets(source: str, filename: str = "<source>") -> set[str]:
    """Literal node names carrying a non-disabled ``additive_lod=`` request."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        target = node.args[0]
        if not isinstance(target, ast.Constant) or not isinstance(target.value, str):
            continue
        if any(
            kw.arg == "additive_lod" and not _is_disabled(kw.value)
            for kw in node.keywords
        ):
            found.add(target.value)
    return found


def _corpus() -> dict[str, set[str]]:
    """Module name -> the gated structures it requests. Empty entries dropped."""
    out: dict[str, set[str]] = {}
    for path in scanned_demo_modules():
        # The policy module itself names these keywords in its own docs and
        # defaults; it is the rule, not a demo applying it.
        if path.name == "_lod_policy.py":
            continue
        features = gated_features(path.read_text(), path.name)
        if features:
            out[path.name] = features
    return out


# ─────────────────────────────────────────────────────────────────────
# The real-environment arm: the shipped corpus
# ─────────────────────────────────────────────────────────────────────


def unjustified_modules(justified: dict[str, str]) -> dict[str, list[str]]:
    """Modules requesting a gated structure with no entry in *justified*.

    Takes the allowlist as an argument so the gate can be run against a MODIFIED
    one — which is how ``test_the_gate_fires_when_an_entry_is_missing`` proves it
    can actually go red against the real corpus, rather than only proving the
    detector works on a synthetic string.
    """
    return {
        name: sorted(features)
        for name, features in _corpus().items()
        if name not in justified
    }


def test_every_substitutive_or_partition_demo_is_justified() -> None:
    unjustified = unjustified_modules(JUSTIFIED)
    assert not unjustified, (
        "these modules request substitutive levels or a partition without a "
        f"recorded reason: {unjustified}. Either add an additive ladder instead "
        "(the cheap default — see `_lod_policy.stream_ladder`), or add an entry "
        "to JUSTIFIED saying why the resident slice of this data needs more. "
        "Compare the RESIDENT count against the per-geometry cap, not the node "
        "total: Lines 2,793,472 / Points 5,591,040 / GSplats 4,194,304."
    )


def test_the_justified_list_does_not_rot() -> None:
    """A shrinking allowlist: an entry whose module stopped using the feature must
    go, or the list slowly stops describing anything and permits everything."""
    corpus = _corpus()
    stale = sorted(set(JUSTIFIED) - set(corpus))
    assert not stale, (
        f"JUSTIFIED lists {stale}, which no longer request substitutive levels "
        "or a partition. Remove the entries — a stale allowlist would silently "
        "re-permit the structure if one were added back."
    )


def test_every_reason_is_substantive() -> None:
    """Guards against an entry added to silence the gate with 'needed'."""
    for name, reason in JUSTIFIED.items():
        assert len(reason) >= 80, f"{name}: reason is too short to be a reason"
        assert any(ch.isdigit() for ch in reason), (
            f"{name}: reason cites no measurement — a resident count, a cap, or a "
            "measured size is what makes it checkable"
        )


def test_the_corpus_still_has_demos_using_these() -> None:
    """If the detector broke, every gate above would pass vacuously."""
    corpus = _corpus()
    assert len(corpus) >= 5, (
        f"only {len(corpus)} modules matched; the detector or the package layout "
        "changed and the gates above would pass by finding nothing"
    )


@pytest.mark.parametrize(
    ("module_name", "target"), sorted(_CONVERTED_ADDITIVE_TARGETS.items())
)
def test_converted_embedding_demo_keeps_its_additive_ladder(
    module_name: str, target: str
) -> None:
    """Removing any converted demo's opt-in ladder must fail in CI."""
    path = next(path for path in scanned_demo_modules() if path.name == module_name)
    assert target in additive_ladder_targets(path.read_text(), path.name)


@pytest.mark.parametrize("dropped", sorted(JUSTIFIED))
def test_the_gate_fires_when_an_entry_is_missing(dropped: str) -> None:
    """Every entry is load-bearing: drop it and the gate goes red.

    The fires-proof, against the REAL corpus rather than a synthetic string. It
    also catches a subtler rot than
    ``test_the_justified_list_does_not_rot``: an entry whose module the DETECTOR
    no longer matches would be silently unnecessary, and this fails on it.
    """
    thinned = {k: v for k, v in JUSTIFIED.items() if k != dropped}
    assert unjustified_modules(thinned) == {dropped: sorted(_corpus()[dropped])}, (
        f"removing {dropped} from JUSTIFIED did not trip the gate"
    )


# ─────────────────────────────────────────────────────────────────────
# Detector self-tests: the negative and positive arms
# ─────────────────────────────────────────────────────────────────────


class TestGatedFeatureDetection:
    @pytest.mark.parametrize(
        "source, expected",
        [
            (
                "scene.add_points('a', p, substitutive_lod=dict(levels=3))",
                {"substitutive_lod"},
            ),
            ("scene.add_points('a', p, partition=dict(max_elements=1))", {"partition"}),
            (
                "scene.add_lines('a', v, 1.0, substitutive_lod=spec)",
                {"substitutive_lod"},
            ),
            (
                "scene.add_points('a', p, partition=P, substitutive_lod=S)",
                {"partition", "substitutive_lod"},
            ),
            (
                "w = scene.add_partition_group('a', max_elements=1)",
                {"add_partition_group"},
            ),
            (
                "g = w.add_lod_group('part_0', selector='screen-area')",
                {"add_lod_group"},
            ),
            # Routed through a helper — still a request.
            (
                "scene.add_points('a', p, substitutive_lod=substitutive_lod_or_flat(S))",
                {"substitutive_lod"},
            ),
        ],
    )
    def test_requests_are_detected(self, source: str, expected: set[str]) -> None:
        assert gated_features(source) == expected

    @pytest.mark.parametrize(
        "source",
        [
            # The cheap default needs no justification.
            "scene.add_points('a', p, additive_lod=stream_ladder(n))",
            "scene.add_lines('a', v, 1.0, additive_lod=stream_ladder(n, geometry='lines'))",
            # Literal opt-outs are not requests.
            "scene.add_points('a', p, substitutive_lod=None)",
            "scene.add_points('a', p, partition=False)",
            # PROSE mentioning the keyword is the case a grep would fail on, and
            # several demos now carry exactly this comment.
            "# No `partition=`. Three parts, and taxon/period are hidden.\nx = 1",
            "'''partition= bounds node size but not residency.'''",
        ],
    )
    def test_non_requests_are_not_detected(self, source: str) -> None:
        assert gated_features(source) == set()

    def test_additive_ladder_targets_ignore_opt_outs_and_dynamic_names(self) -> None:
        source = """
scene.add_points('kept', p, additive_lod=stream_ladder(n))
scene.add_points('disabled', p, additive_lod=False)
scene.add_points(name, p, additive_lod=stream_ladder(n))
"""
        assert additive_ladder_targets(source) == {"kept"}

    def test_a_new_demo_without_a_reason_would_be_caught(self) -> None:
        """The negative arm end-to-end: a module not in JUSTIFIED trips the gate.

        Proves the gate FIRES, rather than only that the current corpus passes —
        a gate that can never fail and one that always passes look identical from
        a green suite.
        """
        features = gated_features(
            "scene.add_points('cells', p, substitutive_lod=dict(levels=3))",
            "demo_a_brand_new_demo.py",
        )
        assert features, "detector found nothing in an obvious request"
        assert "demo_a_brand_new_demo.py" not in JUSTIFIED
