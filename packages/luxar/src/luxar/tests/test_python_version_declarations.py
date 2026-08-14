"""Guard: every place that names a supported Python version must agree.

Twelve files encode a Python version — packaging metadata, the type checker, the
linter target, two matrices, two bootstrap scans, the HPC smoke test and the docs.
They drifted twice while the 3.12 floor was landing: the classifiers were first
narrowed to the floor alone while ``requires-python`` still admitted anything
newer, and then widened to 3.12-3.14 while CI still tested only 3.12 and 3.14. Both
are the same defect — a version the wheel advertises that nothing exercises — and
neither is visible in a diff, so it is asserted here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
FLOOR = "3.12"


@pytest.fixture(scope="module")
def pyproject() -> str:
    """Raw `pyproject.toml` text — read as TEXT, not parsed.

    The assertions are about the literal declarations a maintainer edits, and a
    TOML parse would happily normalise away the spelling being pinned.
    """
    return (REPO / "pyproject.toml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow() -> str:
    """Raw CI workflow text; the matrix lives in a GitHub expression, not YAML data."""
    return (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def _declared(pyproject: str) -> set[str]:
    """Python versions the wheel's trove classifiers advertise."""
    return set(re.findall(r'"Programming Language :: Python :: (3\.\d+)"', pyproject))


def _ci_legs(workflow: str) -> tuple[set[str], set[str]]:
    """``(pull-request legs, off-PR legs)`` from the python-tests matrix."""
    line = next(ln for ln in workflow.splitlines() if "python-version: ${{" in ln)
    lists = re.findall(r"fromJSON\('(\[[^\]]*\])'\)", line)
    assert len(lists) >= 2, f"matrix expression lost a branch: {line.strip()}"
    return (
        set(re.findall(r"3\.\d+", lists[0])),
        set(re.findall(r"3\.\d+", lists[-1])),
    )


def _hatch_matrix(pyproject: str) -> set[str]:
    """Python versions the `[[tool.hatch.envs.test.matrix]]` block declares."""
    block = re.search(
        r"test\.matrix\]\]\n(?:#[^\n]*\n)*python = (\[[^\]]*\])", pyproject
    )
    assert block, "could not find [[tool.hatch.envs.test.matrix]] python list"
    return set(re.findall(r"3\.\d+", block.group(1)))


def test_requires_python_is_the_floor(pyproject: str) -> None:
    """The packaging floor is the anchor every other declaration is checked against."""
    assert re.search(r'requires-python\s*=\s*">=' + FLOOR + '"', pyproject)


def test_every_declared_version_is_tested_off_pr(pyproject: str, workflow: str) -> None:
    """A classifier the wheel advertises must be a version CI actually runs.

    This is the drift that happened: 3.13 was advertised and never exercised.
    Equality, not containment — testing a version we do NOT advertise is also a
    mismatch worth knowing about.
    """
    declared = _declared(pyproject)
    _, off_pr = _ci_legs(workflow)
    assert declared == off_pr, (
        f"classifiers advertise {sorted(declared)} but CI's off-PR matrix runs "
        f"{sorted(off_pr)}"
    )


def test_the_hatch_matrix_mirrors_ci(pyproject: str, workflow: str) -> None:
    """``hatch run test.pyX.YZ:cov`` must exist for every version CI runs."""
    _, off_pr = _ci_legs(workflow)
    assert _hatch_matrix(pyproject) == off_pr


def test_the_pull_request_leg_is_the_floor_alone(workflow: str) -> None:
    """Branch protection requires ``python-tests (3.12)``, so the floor must run.

    Kept to one leg on purpose: the self-hosted pool is the throughput ceiling
    (see #1484). Widening this is a deliberate cost decision, not a tidy-up.
    """
    pr_legs, _ = _ci_legs(workflow)
    assert pr_legs == {FLOOR}


def test_the_source_gates_agree_with_the_floor(pyproject: str) -> None:
    """mypy and ruff judge the SOURCE, so both must target the floor, not newer.

    Targeting a newer version would let syntax through that the floor cannot run.
    """
    assert re.search(r'python_version\s*=\s*"' + FLOOR + '"', pyproject)
    assert re.search(
        r'target-version\s*=\s*"py' + FLOOR.replace(".", "") + '"', pyproject
    )


def test_no_bootstrap_scan_offers_a_sub_floor_interpreter() -> None:
    """`make setup-dev` and the HPC smoke test must not pick an unsupported Python.

    Both walk a candidate list newest-first; a leftover ``python3.11`` would build
    an environment the project does not support and fail confusingly later.
    """
    floor_minor = int(FLOOR.split(".")[1])
    for rel in ("Makefile", "scripts/check_hpc_setup.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        found = {int(m) for m in re.findall(r"python3\.(\d+)", text)}
        below = [f"3.{v}" for v in sorted(v for v in found if v < floor_minor)]
        assert not below, f"{rel} scans for sub-floor interpreters: {below}"
