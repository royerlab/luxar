"""Keep hand-authoring documentation aligned with the Cholesky contract."""

from __future__ import annotations

import re
from inspect import getdoc
from pathlib import Path

import numpy as np
import pytest

from luxar.core.group.group import Group
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

REPO_ROOT = Path(__file__).resolve().parents[6]
SIGMA = 0.8

PACKING_CLAIM_PATHS = (
    ".agents/skills/luxar-visualization/SKILL.md",
    ".agents/skills/luxar-visualization/references/scene-api.md",
    ".agents/skills/luxar-gsplat-pipeline/SKILL.md",
)

ISOTROPIC_PACKING = re.compile(r"\[(?P<items>(?:σ|0)(?:\s*,\s*(?:σ|0)){5})\]")


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
        packed = np.array(
            [
                SIGMA if item.strip() == "σ" else 0.0
                for item in match["items"].split(",")
            ],
            dtype=np.float32,
        )
        data = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=packed[None, :],
        )

        np.testing.assert_allclose(data.marginal_sigmas()[0], SIGMA, rtol=1e-6)


@pytest.mark.parametrize("relative_path", PACKING_CLAIM_PATHS)
def test_skill_packing_claim_recovers_sigma(relative_path: str) -> None:
    """Every skill packing claim must execute as the documented covariance."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    _assert_isotropic_packing_recovers_sigma(text, relative_path)


@pytest.mark.parametrize(
    ("source", "docstring"),
    (
        ("Group.add_gsplats", getdoc(Group.add_gsplats)),
        ("AdditiveSubLOD", getdoc(AdditiveSubLOD)),
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


def test_basic_example_does_not_claim_disjoint_splats_overlap() -> None:
    """The explainer must describe the corrected, effectively disjoint geometry."""
    text = (REPO_ROOT / "packages/luxar/examples/gsplats_basic_example.py").read_text(
        encoding="utf-8"
    )

    assert "Additive blending makes overlapping splats brighten." not in text


def test_stacked_axis_docs_distinguish_embedding_from_direct_authoring() -> None:
    """Stacked-axis docs must distinguish regularized fill from direct input."""
    dimension_mapping = (
        REPO_ROOT / "docs/specs/GSPLATS_DIMENSION_MAPPING.md"
    ).read_text(encoding="utf-8")
    visualization_skill = (
        REPO_ROOT / ".agents/skills/luxar-visualization/SKILL.md"
    ).read_text(encoding="utf-8")

    for text in (dimension_mapping, visualization_skill):
        assert "stacked time/channel axis" in text
        assert 'fill_sigma={"time": 0.0}' in text
        assert "extend_to_all=[]" in text
        assert "regularizes" in text
        assert "1e-7" in text
        assert "full scene-dimensional" in text
        assert "without `dim_order`" in text
        assert re.search(r"strictly\s+positive", text)
        assert re.search(r"smaller\s+than the coordinate step", text)

    contributor_reference = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "stacked sigma=0 time/channel axis" not in contributor_reference
    assert (
        "stacked time/channel centers column with near-zero Cholesky extent"
        in contributor_reference
    )
