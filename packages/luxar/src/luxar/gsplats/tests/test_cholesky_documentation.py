"""Keep hand-authoring documentation aligned with the Cholesky contract."""

from __future__ import annotations

import re
from inspect import getdoc
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.group.group import Group
from luxar.encoding import ArrayDecoder
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

REPO_ROOT = Path(__file__).resolve().parents[6]
SIGMA = 0.8

PACKING_CLAIM_PATHS = (
    ".agents/skills/luxar-visualization/SKILL.md",
    ".agents/skills/luxar-visualization/references/scene-api.md",
    ".agents/skills/luxar-gsplat-pipeline/SKILL.md",
    "docs/specs/GSPLATS_DIMENSION_MAPPING.md",
)

ISOTROPIC_PACKING = re.compile(r"\[(?P<items>(?:σ|0)(?:\s*,\s*(?:σ|0)){5})\]")
STACKED_AXIS_RECIPE = re.compile(
    r"(?:To hand-author a|For a hand-authored) stacked time/channel axis"
    r".*?(?=\n(?:- |\n))",
    re.DOTALL,
)
ZERO_FILL_SIGMA = re.compile(
    r"regularizes\s+(?:the|that)\s+semantic\s+zero\s+to\s+"
    r"`(?P<epsilon>\d+(?:\.\d+)?e[+-]?\d+)`"
)


def _assert_isotropic_packing_recovers_sigma(text: str, source: str) -> None:
    """Every author-facing packing claim must describe covariance, not precision."""
    compact = re.sub(r"\s+", "", text)
    assert "[1/σ,0,1/σ,0,0,1/σ]" not in compact
    assert "[1/sigma,0,1/sigma,0,0,1/sigma]" not in compact

    matches = list(ISOTROPIC_PACKING.finditer(text))

    assert matches, (
        f"{source} must state the executable isotropic packing [σ, 0, σ, 0, 0, σ]"
    )

    for match in matches:
        items = [item.strip() for item in match["items"].split(",")]
        assert items == ["σ", "0", "σ", "0", "0", "σ"]
        packed = np.array(
            [SIGMA if item == "σ" else 0.0 for item in items], dtype=np.float32
        )
        data = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=packed[None, :],
        )

        np.testing.assert_allclose(data.marginal_sigmas()[0], SIGMA, rtol=1e-6)


@pytest.mark.parametrize("relative_path", PACKING_CLAIM_PATHS)
def test_authoring_doc_packing_claim_recovers_sigma(relative_path: str) -> None:
    """Every authoring-doc packing claim must execute as the covariance."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    _assert_isotropic_packing_recovers_sigma(text, relative_path)


@pytest.mark.parametrize(
    ("source", "docstring"),
    (
        ("Group.add_gsplats", getdoc(Group.add_gsplats)),
        ("AdditiveSubLOD", getdoc(AdditiveSubLOD)),
        ("GSplatData", getdoc(GSplatData)),
    ),
)
def test_source_docstring_packing_recovers_sigma(
    source: str, docstring: str | None
) -> None:
    """The public source docstrings must carry the executable convention."""
    assert docstring is not None

    _assert_isotropic_packing_recovers_sigma(docstring, source)


def test_contributor_reference_names_covariance_convention() -> None:
    """The contributor quick reference must not leave the factor ambiguous."""
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "cholesky_factors (Float32, required; chol(Σ), scale-like diagonal)" in text


def test_viewer_fixture_warning_links_to_authoring_contract() -> None:
    """The historical 1/sigma warning must point back to public authoring docs."""
    text = (
        REPO_ROOT / "packages/luxar-viewer/tests/fixtures/generate_test_data.py"
    ).read_text(encoding="utf-8")
    warning = re.search(r"# IMPORTANT:.*?\n\s*cholesky =", text, re.DOTALL)

    assert warning is not None
    assert "Group.add_gsplats" in warning.group()
    assert "AdditiveSubLOD" in warning.group()


def _stacked_axis_recipe(text: str, source: str) -> str:
    """Extract the hand-authored stacked-axis recipe from an authoring document."""
    match = STACKED_AXIS_RECIPE.search(text)
    assert match is not None, f"{source} must document stacked-axis authoring"
    return match.group()


def test_stacked_axis_docs_distinguish_embedding_from_direct_authoring() -> None:
    """Stacked-axis recipes must opt into embedding rather than broadcasting."""
    dimension_mapping = (
        REPO_ROOT / "docs/specs/GSPLATS_DIMENSION_MAPPING.md"
    ).read_text(encoding="utf-8")
    visualization_skill = (
        REPO_ROOT / ".agents/skills/luxar-visualization/SKILL.md"
    ).read_text(encoding="utf-8")

    for source, text in (
        ("docs/specs/GSPLATS_DIMENSION_MAPPING.md", dimension_mapping),
        (".agents/skills/luxar-visualization/SKILL.md", visualization_skill),
    ):
        recipe = _stacked_axis_recipe(text, source)
        assert 'dim_order=["x", "y", "z"]' in recipe
        assert 'fill_sigma={"time": 0.0}' in recipe
        assert "extend_to_all=[]" in recipe

    scene_api = (
        REPO_ROOT / ".agents/skills/luxar-visualization/references/scene-api.md"
    ).read_text(encoding="utf-8")
    assert 'dim_order=["x", "y", "z"]' in scene_api
    assert 'fill_sigma={"time": 0.0}' in scene_api
    assert "extend_to_all=[]" in scene_api


def _documented_zero_fill_sigma() -> float:
    """Return the shared epsilon documented for semantic zero-width axes."""
    documented_values = []
    for relative_path in (
        "docs/specs/GSPLATS_DIMENSION_MAPPING.md",
        ".agents/skills/luxar-visualization/SKILL.md",
    ):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        recipe = _stacked_axis_recipe(text, relative_path)
        match = ZERO_FILL_SIGMA.search(recipe)
        assert match is not None, f"{relative_path} must document the zero-fill epsilon"
        documented_values.append(float(match["epsilon"]))

    assert documented_values[0] == documented_values[1]
    return documented_values[0]


def test_zero_fill_sigma_writes_positive_epsilon(tmp_path: Path) -> None:
    """The documented stacked-axis recipe must survive the public writer."""
    output = tmp_path / "stacked.luxar.zarr"
    dimensions = Dimensions(
        [
            Dimension("x", display=True),
            Dimension("y", display=True),
            Dimension("z", display=True),
            Dimension("time", display=False, discrete=True, step=1.0),
        ]
    )
    centers = np.zeros((1, 3), dtype=np.float32)
    amplitudes = np.ones(1, dtype=np.float32)
    cholesky_factors = np.array(
        [[SIGMA, 0.0, SIGMA, 0.0, 0.0, SIGMA]], dtype=np.float32
    )

    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)
        scene.add_gsplats(
            "splats",
            centers,
            amplitudes,
            cholesky_factors,
            dim_order=["x", "y", "z"],
            fill={"time": 0.0},
            fill_sigma={"time": 0.0},
            extend_to_all=[],
        )

    group = zarr.open(str(output), mode="r")["splats"]
    diagonal = ArrayDecoder().decode(group["cholesky_factors_diag"], group)
    expected_epsilon = _documented_zero_fill_sigma()
    np.testing.assert_allclose(
        diagonal[0], [SIGMA, SIGMA, SIGMA, expected_epsilon], rtol=1e-6
    )
