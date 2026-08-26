"""Keep hand-authoring documentation aligned with the Cholesky contract."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData

REPO_ROOT = Path(__file__).resolve().parents[6]
SIGMA = 0.8

PACKING_CLAIM_PATHS = (
    ".agents/skills/luxar-visualization/SKILL.md",
    ".agents/skills/luxar-visualization/references/scene-api.md",
    ".agents/skills/luxar-gsplat-pipeline/SKILL.md",
    "packages/luxar/src/luxar/core/group/group.py",
    "packages/luxar/src/luxar/gsplats/gsplat_data.py",
)

ISOTROPIC_PACKING = re.compile(r"\[(?P<items>(?:σ|0)(?:\s*,\s*(?:σ|0)){5})\]")


@pytest.mark.parametrize("relative_path", PACKING_CLAIM_PATHS)
def test_documented_isotropic_packing_recovers_sigma(relative_path: str) -> None:
    """Every author-facing packing claim must describe covariance, not precision."""
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    match = ISOTROPIC_PACKING.search(text)

    assert match is not None, (
        f"{relative_path} must state the executable isotropic packing "
        "[σ, 0, σ, 0, 0, σ]"
    )

    packed = np.array(
        [SIGMA if item.strip() == "σ" else 0.0 for item in match["items"].split(",")],
        dtype=np.float32,
    )
    data = GSplatData(
        centers=np.zeros((1, 3), dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=packed[None, :],
    )

    np.testing.assert_allclose(data.marginal_sigmas()[0], SIGMA, rtol=1e-6)


def test_contributor_reference_names_covariance_convention() -> None:
    """The contributor quick reference must not leave the factor ambiguous."""
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "cholesky_factors (Float32, required; chol(Σ), scale-like diagonal)" in text


def test_viewer_fixture_warning_links_to_authoring_contract() -> None:
    """The historical 1/sigma warning must point back to public authoring docs."""
    text = (
        REPO_ROOT / "packages/luxar-viewer/tests/fixtures/generate_test_data.py"
    ).read_text(encoding="utf-8")

    assert "Group.add_gsplats" in text
    assert "AdditiveSubLOD" in text


def test_basic_example_does_not_claim_disjoint_splats_overlap() -> None:
    """The explainer must describe the corrected, effectively disjoint geometry."""
    text = (REPO_ROOT / "packages/luxar/examples/gsplats_basic_example.py").read_text(
        encoding="utf-8"
    )

    assert "Additive blending makes overlapping splats brighten." not in text


def test_stacked_axis_docs_require_positive_substep_sigma() -> None:
    """Hand-authored stacked axes need a valid positive covariance diagonal."""
    dimension_mapping = (
        REPO_ROOT / "docs/specs/GSPLATS_DIMENSION_MAPPING.md"
    ).read_text(encoding="utf-8")
    visualization_skill = (
        REPO_ROOT / ".agents/skills/luxar-visualization/SKILL.md"
    ).read_text(encoding="utf-8")

    for text in (dimension_mapping, visualization_skill):
        assert "stacked time/channel axis" in text
        assert "strictly positive" in text
        assert "smaller than the coordinate step" in text
