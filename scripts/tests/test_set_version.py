"""Tests for release version stamping, consistency, preflight, and package naming.

These release mechanisms run exactly once per release — so a defect surfaces
on launch day, in front of everyone, with no earlier signal. That asymmetry is
why they are tested here rather than trusted.

The gate tests deliberately assert that it *fails*: a consistency check that
cannot go red is indistinguishable from no check at all.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SET_VERSION = REPO / "scripts/set_version.py"
CHECK_VERSIONS = REPO / "scripts/check_version_consistency.py"
RELEASE = REPO / "scripts/release.sh"
METAL_TEST_CONFTEST = (
    REPO / "packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests/conftest.py"
)
CUDA_TEST_CONFTEST = (
    REPO / "packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/conftest.py"
)
NLM_CUDA_TEST_CONFTEST = (
    REPO / "packages/luxar/src/luxar/gsplats/preprocessing/tests/conftest.py"
)
MAKEFILE = REPO / "Makefile"


def _legacy_name_pattern() -> str:
    """An ERE matching every surface where the retired npm name can only mean
    the package.

    Prose is deliberately excluded: the source directory
    ``packages/luxar-viewer/`` and the Cloudflare Pages project of the same name
    are both legitimate, so a general prose alternative would be false positives
    all the way down.
    """
    # Split the retired names so this module never matches its own source.
    bare_name = "luxar" + "-viewer"
    legacy_scope = "@royerlab" + f"/{bare_name}"
    return "|".join(
        (
            legacy_scope,
            # Any module specifier, not only the root and styles.css: a subpath
            # such as <bare-name>/data names the retired package just as much.
            rf"(from|import) ['\"]{bare_name}(/[^'\"]*)?['\"]",
            rf"{bare_name}/styles\.css",
            rf"(npm (install|i)|pnpm add|yarn add) {bare_name}([[:space:]]|$)",
        )
    )


def _git_grep(
    pattern: str,
    *flags: str,
    paths: tuple[str, ...] = (),
    cwd: Path = REPO,
) -> subprocess.CompletedProcess[str]:
    # -e keeps the pattern from being read as a pathspec once paths are given.
    try:
        return subprocess.run(
            ["git", "grep", "-n", "-E", *flags, "-e", pattern, "--", *paths],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - git is a hard dep here
        pytest.fail(f"git is required to run this guard: {exc}")


def test_legacy_name_pattern_matches_the_specifiers_it_claims_to(
    tmp_path: Path,
) -> None:
    """The positive control: a pattern that compiles to something matching
    nothing would let the guard below pass over a tree full of stale names."""
    bare_name = "luxar" + "-viewer"
    stale = tmp_path / "stale.md"
    lines = (
        f"import {{ x }} from '{bare_name}';",
        f"import {{ x }} from '{bare_name}/data';",
        f'import type {{ T }} from "{bare_name}/ui/data-monitor-manager";',
        f'<link href="{bare_name}/styles.css" />',
        f"pnpm add {bare_name}",
        f"yarn add {bare_name}",
        f"npm install {bare_name}",
        f"npm install {bare_name} three",
        f"npm i {bare_name}",
        "@royerlab" + f"/{bare_name}",
    )
    stale.write_text("\n".join(lines) + "\n")

    # --no-index refuses paths outside the cwd's tree, so search from tmp_path.
    result = _git_grep(
        _legacy_name_pattern(), "--no-index", paths=(stale.name,), cwd=tmp_path
    )

    assert result.returncode == 0, (
        f"the guard's pattern matched none of the stale specifiers "
        f"(exit {result.returncode}): {result.stderr}"
    )
    assert len(result.stdout.splitlines()) == len(lines), result.stdout


def test_legacy_npm_package_specifiers_are_absent_from_tracked_files() -> None:
    # The changelog is the one place the retired name belongs: an entry that
    # cannot say what the package used to be called is no use to whoever has to
    # update their install line.
    result = _git_grep(
        _legacy_name_pattern(),
        paths=(".", ":(exclude)CHANGELOG.md", ":(exclude)changelog.d/*"),
    )

    # git grep exits 1 for "no match" and 0 for "matched"; anything else (128
    # outside a work tree, say) is a broken guard, not a clean tree.
    assert result.returncode in (0, 1), (
        f"git grep could not run (exit {result.returncode}): {result.stderr}"
    )
    assert result.returncode == 1, result.stdout


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def set_version_module() -> ModuleType:
    return _load(SET_VERSION, "set_version")


@pytest.fixture
def check_module() -> ModuleType:
    return _load(CHECK_VERSIONS, "check_version_consistency")


def _checkout(tmp_path: Path, version: str = "2026.06.05") -> dict[str, Path]:
    """A miniature repo carrying the three version representations."""
    init = tmp_path / "__init__.py"
    init.write_text(f'"""Luxar."""\n\n__version__ = "{version}"\n\n__all__ = []\n')

    pkg_json = tmp_path / "package.json"
    semver = ".".join(str(int(p)) for p in version.split("."))
    pkg_json.write_text(
        json.dumps({"name": "@luxar/viewer", "version": semver}, indent=2) + "\n"
    )

    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\n"
        'title: "Luxar"\n'
        f'version: "{version}"\n'
        f'date-released: "{version.replace(".", "-")}"\n'
    )
    return {"init": init, "pkg_json": pkg_json, "citation": citation}


def _point_at(
    module: ModuleType,
    paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "INIT", paths["init"])
    monkeypatch.setattr(module, "PKG_JSON", paths["pkg_json"])
    monkeypatch.setattr(module, "CITATION", paths["citation"])


# --------------------------------------------------------------------------- #
# set_version.py
# --------------------------------------------------------------------------- #


def test_stamps_all_three_representations(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0

    assert '__version__ = "2026.09.15"' in paths["init"].read_text()
    # npm/semver forbids leading zeros; the same release, spelled differently.
    assert json.loads(paths["pkg_json"].read_text())["version"] == "2026.9.15"
    cff = paths["citation"].read_text()
    assert 'version: "2026.09.15"' in cff
    assert 'date-released: "2026-09-15"' in cff


def test_zero_padding_is_preserved_where_it_matters(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-digit month/day is the case the two spellings diverge on."""
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.01.02"]) == 0

    assert '__version__ = "2026.01.02"' in paths["init"].read_text()
    assert json.loads(paths["pkg_json"].read_text())["version"] == "2026.1.2"
    assert 'version: "2026.01.02"' in paths["citation"].read_text()
    assert 'date-released: "2026-01-02"' in paths["citation"].read_text()


@pytest.mark.parametrize(
    "bad",
    [
        "2026.6.5",
        "2026-06-05",
        "v2026.06.05",
        "1.2.3",
        "2026.06",
        "2026.02.31",
        "2026.13.01",
        "",
    ],
)
def test_rejects_non_calver(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad: str,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)
    before = {name: path.read_text() for name, path in paths.items()}

    assert set_version_module.main(["set_version.py", bad]) == 2
    assert {name: path.read_text() for name, path in paths.items()} == before


def test_reports_a_citation_missing_its_stampable_lines(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently skipping would reintroduce exactly the drift this closes."""
    paths = _checkout(tmp_path)
    paths["citation"].write_text('cff-version: 1.2.0\ntitle: "Luxar"\n')
    _point_at(set_version_module, paths, tmp_path, monkeypatch)
    before = {name: path.read_text() for name, path in paths.items()}

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 1
    assert {name: path.read_text() for name, path in paths.items()} == before


def test_stamps_citation_lines_without_spaces_after_colons(
    set_version_module: ModuleType,
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    paths["citation"].write_text(
        "cff-version: 1.2.0\nversion:2026.06.05\ndate-released:2026-06-05\n"
    )
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 0
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0
    assert 'version: "2026.09.15"' in paths["citation"].read_text()
    assert 'date-released: "2026-09-15"' in paths["citation"].read_text()


def test_is_idempotent(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0
    once = {k: p.read_text() for k, p in paths.items()}
    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0
    assert {k: p.read_text() for k, p in paths.items()} == once


# --------------------------------------------------------------------------- #
# check_version_consistency.py — each assertion is that the gate goes RED
# --------------------------------------------------------------------------- #


def test_gate_passes_on_a_consistent_checkout(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 0


def test_gate_catches_a_stale_viewer_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _checkout(tmp_path)
    paths["pkg_json"].write_text(json.dumps({"version": "2026.5.1"}) + "\n")
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 1
    assert "make set-version DATE=2026.06.05" in capsys.readouterr().err


@pytest.mark.parametrize("version", ["2026.6.5", "2026.02.31"])
def test_gate_rejects_an_invalid_python_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
) -> None:
    paths = _checkout(tmp_path, version)
    _point_at(check_module, paths, tmp_path, monkeypatch)

    assert check_module.main() == 2
    assert str(paths["init"]) in capsys.readouterr().err


def test_gate_catches_a_stale_citation_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _checkout(tmp_path)
    cff = paths["citation"].read_text().replace("2026.06.05", "2026.05.01")
    paths["citation"].write_text(cff)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 1
    assert "make set-version DATE=2026.06.05" in capsys.readouterr().err


def test_gate_catches_a_date_that_disagrees_with_the_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subtle one: right version, wrong release date."""
    paths = _checkout(tmp_path)
    cff = paths["citation"].read_text().replace("2026-06-05", "2026-06-06")
    paths["citation"].write_text(cff)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 1


def test_gate_reports_a_citation_missing_its_fields(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    paths["citation"].write_text('cff-version: 1.2.0\ntitle: "Luxar"\n')
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 2


@pytest.mark.parametrize(
    "citation",
    [
        "cff-version: 1.2.0\nversion: 2026.06.05\ndate-released: 2026-06-05\n",
        "cff-version: 1.2.0\nversion: '2026.06.05'\ndate-released: '2026-06-05'\n",
        'cff-version: 1.2.0\nversion: "2026.06.05"  # tag v2026.06.05\n'
        'date-released: "2026-06-05"\n',
    ],
)
def test_gate_accepts_valid_yaml_scalar_spellings(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    citation: str,
) -> None:
    """The gate must accept ordinary YAML quoting and inline comments."""
    paths = _checkout(tmp_path)
    paths["citation"].write_text(citation)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 0


# --------------------------------------------------------------------------- #
# The real repo
# --------------------------------------------------------------------------- #


def test_the_committed_tree_is_consistent(check_module: ModuleType) -> None:
    """Guards the actual files, not a fixture — this is what CI runs for."""
    assert check_module.main() == 0
    citation = yaml.safe_load((REPO / "CITATION.cff").read_text())
    init_text = (REPO / "packages/luxar/src/luxar/__init__.py").read_text()
    version_match = check_module.CALVER_RE.search(init_text)
    assert version_match is not None
    version = version_match.group(1)
    assert isinstance(citation["version"], str)
    assert isinstance(citation["date-released"], str)
    assert citation["version"] == version
    assert citation["date-released"] == version.replace(".", "-")


def test_release_remedy_passes_the_version_as_a_make_variable() -> None:
    assert "make set-version DATE=$VERSION" in RELEASE.read_text()


def test_release_handoff_targets_dev_then_promoted_main() -> None:
    assert "open a PR against dev, then release from main once promoted" in (
        MAKEFILE.read_text()
    )
    release_text = RELEASE.read_text()
    assert "promote it to main with CI green" in release_text
    assert "Run 'make set-version' and promote it first." in release_text
    assert "Run 'make set-version DATE=$VERSION' and promote it first." in release_text
    assert "merge it first" not in release_text


def test_set_version_prints_explicit_pr_base_and_release_branch(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0

    output = capsys.readouterr().out
    assert "gh pr create --base dev --fill" in output
    assert "gh pr create --fill" not in output
    assert "switch to main after the bump is promoted" in output


def _run_release_preflight(
    tmp_path: Path,
    *,
    environment: str = "UNSET",
    repository: str = "UNSET",
    organization: str = "UNSET",
    repository_direct: str | None = None,
    changelog_state: str = "ready",
    ci_result: str | None = None,
    check: bool = True,
) -> tuple[str, list[str], int]:
    repo = tmp_path / "release-repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".github/workflows").mkdir(parents=True)
    (repo / "packages/luxar/src/luxar").mkdir(parents=True)
    (repo / "scripts/release.sh").write_text(RELEASE.read_text())
    (repo / "scripts/check_version_consistency.py").write_text("")
    (repo / ".github/workflows/publish.yml").write_text("name: PyPI\n")
    (repo / ".github/workflows/publish-npm.yml").write_text("name: npm\n")
    (repo / "packages/luxar/src/luxar/__init__.py").write_text(
        '__version__ = "2099.01.01"\n'
    )
    # These tests assert the preflight reaches the npm section, which only
    # happens on a tree that is actually ready to tag. Since the preflight now
    # gates on release-prep state, the fixture has to model a folded changelog:
    # no pending fragments, and a CHANGELOG naming the version being released.
    # `changelog_state` lets the two gate tests below break exactly one half.
    if changelog_state != "no-changelog":
        (repo / "CHANGELOG.md").write_text(
            "# Changelog\n\n## [2099.01.01]\n\nFixture release section.\n"
            if changelog_state != "uncut"
            else "# Changelog\n\n## [Unreleased]\n\nFixture.\n"
        )
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d/README.md").write_text("fragments live here\n")
    if changelog_state == "pending-fragments":
        (repo / "changelog.d/1234.md").write_text("#### A fragment\n\nBody.\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = scripts/check_version_consistency.py ]; then exit 0; fi\n'
        'exec "$REAL_PYTHON" "$@"\n'
    )
    (bin_dir / "python3").chmod(0o755)
    gh_log = tmp_path / "gh-calls.log"
    (bin_dir / "gh").write_text(
        """#!/usr/bin/python3
import os
import sys

args = sys.argv[1:]
if args[:1] == ["auth"]:
    raise SystemExit(0)
if args[:1] == ["repo"]:
    print("royerlab/luxar")
    raise SystemExit(0)

endpoint = args[1]
with open(os.environ["GH_CALL_LOG"], "a", encoding="utf-8") as log:
    log.write(f"{endpoint}\\n")

settings = {
    "repos/royerlab/luxar/environments/npm/variables": "NPM_ENV_RESULT",
    "repos/royerlab/luxar/actions/variables": "NPM_REPO_RESULT",
    "repos/royerlab/luxar/actions/organization-variables": "NPM_ORG_RESULT",
}
ci_result = os.environ.get("CI_RESULT")
if endpoint.endswith("/protection/required_status_checks"):
    if ci_result == "protection-error":
        print("gh: Forbidden (HTTP 403)", file=sys.stderr)
        raise SystemExit(1)
    if ci_result == "empty-protection":
        raise SystemExit(0)
    print("python-tests (3.12)")
    raise SystemExit(0)
if endpoint.endswith("/check-runs"):
    if ci_result == "checks-error":
        print("gh: rate limit exceeded (HTTP 403)", file=sys.stderr)
        raise SystemExit(1)
    if ci_result == "green-status":
        raise SystemExit(0)
    status = "in_progress" if ci_result == "pending-check" else "completed"
    conclusion = "null" if status != "completed" else '"success"'
    print('{"name":"python-tests (3.12)","conclusion":' + conclusion + ',"status":"' + status + '"}')
    if ci_result == "duplicate-failure":
        print('{"name":"python-tests (3.12)","conclusion":"failure","status":"completed"}')
    raise SystemExit(0)
if endpoint.endswith("/status"):
    if ci_result == "green-status":
        print('{"context":"python-tests (3.12)","state":"success"}')
    raise SystemExit(0)
if endpoint == "repos/royerlab/luxar/actions/variables/ENABLE_NPM_PUBLISH":
    setting = os.environ.get(
        "NPM_REPO_DIRECT_RESULT", os.environ.get("NPM_REPO_RESULT", "UNSET")
    )
else:
    setting = os.environ.get(settings[endpoint], "UNSET")
    if "--paginate" not in args:
        raise SystemExit(1)
if setting == "ERROR":
    raise SystemExit(1)
if setting == "UNSET":
    raise SystemExit(0)

jq_expression = args[args.index("--jq") + 1]
if jq_expression == ".variables[].name":
    print("ENABLE_NPM_PUBLISH")
elif jq_expression == ".value":
    print(setting)
else:
    print(f"found\\t{setting}\\tvalue-end")
"""
    )
    (bin_dir / "gh").chmod(0o755)

    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--no-verify",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(origin)], check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)], cwd=repo, check=True
    )

    env = os.environ.copy()
    env.update(
        {
            "GH_CALL_LOG": str(gh_log),
            "NPM_ENV_RESULT": environment,
            "NPM_REPO_RESULT": repository,
            "NPM_ORG_RESULT": organization,
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
        }
    )
    if ci_result is None:
        env["SKIP_CI_CHECK"] = "1"
    else:
        env["CI_RESULT"] = ci_result
    if repository_direct is not None:
        env["NPM_REPO_DIRECT_RESULT"] = repository_direct
    result = subprocess.run(
        ["bash", "scripts/release.sh", "--dry-run"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )
    calls = gh_log.read_text().splitlines() if gh_log.exists() else []
    return result.stdout + result.stderr, calls, result.returncode


def test_release_preflight_accepts_a_folded_changelog(tmp_path: Path) -> None:
    """The green path, which the three refusal tests below do NOT pin.

    Without this, a gate that died unconditionally would keep every negative
    test passing — the failure mode those tests exist to prevent, inverted. It
    also pins that a bare README.md in changelog.d is not counted as a pending
    fragment, which is the one piece of the counting loop with a special case.
    """
    output, _, returncode = _run_release_preflight(tmp_path)

    # Exit code too, now the helper reports it: printing the ticks and then
    # failing later would satisfy the string assertions alone.
    assert returncode == 0
    assert "changelog.d/ is empty (all fragments folded)" in output
    assert "CHANGELOG.md names 2099.01.01" in output
    # It must get PAST the gate, not merely print the ticks.
    assert "7. Plan" in output
    assert "DRY RUN complete" in output


def test_release_preflight_refuses_unfolded_changelog_fragments(tmp_path: Path) -> None:
    """A pending fragment means `make changelog` was never run.

    Before this gate the preflight would tag happily with every fragment still
    unfolded, publishing a release whose CHANGELOG.md does not describe it. The
    real tree had 588 pending when the gate was written.
    """
    output, _, returncode = _run_release_preflight(
        tmp_path, changelog_state="pending-fragments", check=False
    )

    assert "unfolded fragment" in output
    assert "make changelog" in output
    # It must stop, not warn and continue into the publish plan.
    assert returncode != 0


def test_release_preflight_refuses_a_changelog_that_omits_the_version(
    tmp_path: Path,
) -> None:
    """Folding the fragments is not the same as cutting the release section.

    The two halves fail independently: fragments can be folded without the cut
    (this case), and the cut can be made before a late fragment lands.
    """
    output, _, returncode = _run_release_preflight(
        tmp_path, changelog_state="uncut", check=False
    )

    assert "does not name version 2099.01.01" in output
    assert "make changelog-release" in output
    assert returncode != 0


def test_release_preflight_reports_a_missing_changelog(tmp_path: Path) -> None:
    output, _, returncode = _run_release_preflight(
        tmp_path, changelog_state="no-changelog", check=False
    )

    assert "CHANGELOG.md is missing" in output
    assert returncode != 0


@pytest.mark.parametrize(
    ("ci_result", "message"),
    [
        ("protection-error", "cannot read branch protection"),
        ("empty-protection", "lists no required status checks"),
        ("checks-error", "cannot read check-runs"),
        ("pending-check", "python-tests (3.12): PENDING"),
        ("duplicate-failure", "python-tests (3.12): failure"),
    ],
)
def test_release_preflight_fails_closed_on_unverified_ci(
    tmp_path: Path, ci_result: str, message: str
) -> None:
    output, _, returncode = _run_release_preflight(
        tmp_path, ci_result=ci_result, check=False
    )

    assert message in output
    assert returncode != 0


def test_release_preflight_accepts_a_green_check_run(tmp_path: Path) -> None:
    """The check-run-backed green path — five of main's six required contexts.

    Its sibling below covers the legacy commit-status fallback, which is the
    path taken when /check-runs reports nothing. That one cannot reach
    `conclusion_of`'s check-run success branch at all, so without this test
    the branch judging most real required contexts green was unpinned: change
    that branch to print anything but "success" and the whole preflight suite
    still passed. The two paths are independent and both need a green case.
    """
    output, _, returncode = _run_release_preflight(
        tmp_path, ci_result="green", check=False
    )

    assert "python-tests (3.12): success" in output
    assert "all required checks green" in output
    assert "DRY RUN complete" in output
    assert returncode == 0


def test_release_preflight_accepts_green_legacy_commit_status(tmp_path: Path) -> None:
    output, _, returncode = _run_release_preflight(
        tmp_path, ci_result="green-status", check=False
    )

    assert "python-tests (3.12): success" in output
    assert "all required checks green" in output
    assert "DRY RUN complete" in output
    assert returncode == 0


def test_release_preflight_honors_environment_precedence(tmp_path: Path) -> None:
    output, calls, _ = _run_release_preflight(
        tmp_path, environment="false", repository="true", organization="true"
    )

    assert "ENABLE_NPM_PUBLISH='false' from npm environment" in output
    assert "the tag will NOT publish to npm" in output
    assert "the tag WILL publish" not in output
    assert calls == ["repos/royerlab/luxar/environments/npm/variables"]


def test_release_preflight_surfaces_native_backend_verification(tmp_path: Path) -> None:
    output, _, _ = _run_release_preflight(tmp_path)

    assert re.search(
        r"Confirm the Apple-silicon native backend release verification ran for "
        r"[0-9a-f]{12}\.",
        output,
    )


def test_required_metal_gate_rejects_an_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(METAL_TEST_CONFTEST, "metal_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_METAL", "1")
    monkeypatch.setattr(module, "is_metal_available", lambda: False)
    monkeypatch.setattr(module, "get_metal_status", lambda: "test backend unavailable")

    with pytest.raises(pytest.UsageError, match="test backend unavailable"):
        module.pytest_configure()


def test_metal_gate_is_inert_without_the_release_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(METAL_TEST_CONFTEST, "optional_metal_test_conftest")
    monkeypatch.delenv("LUXAR_REQUIRE_METAL", raising=False)
    monkeypatch.setattr(module, "is_metal_available", lambda: False)

    module.pytest_configure()


def test_required_metal_gate_accepts_an_available_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(METAL_TEST_CONFTEST, "available_metal_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_METAL", "1")
    monkeypatch.setattr(module, "is_metal_available", lambda: True)

    module.pytest_configure()


def test_required_cuda_gate_rejects_an_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.cuda as cuda_backend

    module = _load(CUDA_TEST_CONFTEST, "required_cuda_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_CUDA", "1")
    monkeypatch.setattr(cuda_backend, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(cuda_backend, "CUDA_BACKEND_AVAILABLE", False)

    with pytest.raises(pytest.UsageError, match="splatting backend is unavailable"):
        module.pytest_configure(Mock())


def test_required_cuda_gate_rejects_an_unavailable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.cuda as cuda_backend

    module = _load(CUDA_TEST_CONFTEST, "required_cuda_device_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_CUDA", "1")
    monkeypatch.setattr(cuda_backend, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(cuda_backend, "CUDA_BACKEND_AVAILABLE", True)

    with pytest.raises(pytest.UsageError, match="cannot access a CUDA device"):
        module.pytest_configure(Mock())


def test_cuda_gate_is_inert_without_the_cadence_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.cuda as cuda_backend

    module = _load(CUDA_TEST_CONFTEST, "optional_cuda_test_conftest")
    monkeypatch.delenv("LUXAR_REQUIRE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backend, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(cuda_backend, "CUDA_BACKEND_AVAILABLE", False)

    module.pytest_configure(Mock())


def test_required_cuda_gate_accepts_an_available_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.models.gsplats.cuda as cuda_backend

    module = _load(CUDA_TEST_CONFTEST, "available_cuda_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_CUDA", "1")
    monkeypatch.setattr(cuda_backend, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(cuda_backend, "CUDA_BACKEND_AVAILABLE", True)

    module.pytest_configure(Mock())


def test_required_nlm_cuda_gate_rejects_an_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.preprocessing.cuda as nlm_cuda_backend

    module = _load(NLM_CUDA_TEST_CONFTEST, "required_nlm_cuda_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_CUDA", "1")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(nlm_cuda_backend, "NLM_CUDA_AVAILABLE", False)

    with pytest.raises(pytest.UsageError, match="NLM CUDA backend is unavailable"):
        module.pytest_configure()


def test_required_nlm_cuda_gate_rejects_an_unavailable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.preprocessing.cuda as nlm_cuda_backend

    module = _load(NLM_CUDA_TEST_CONFTEST, "required_nlm_cuda_device_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_CUDA", "1")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(nlm_cuda_backend, "NLM_CUDA_AVAILABLE", True)

    with pytest.raises(pytest.UsageError, match="cannot access a CUDA device"):
        module.pytest_configure()


def test_nlm_cuda_gate_is_inert_without_the_cadence_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.preprocessing.cuda as nlm_cuda_backend

    module = _load(NLM_CUDA_TEST_CONFTEST, "optional_nlm_cuda_test_conftest")
    monkeypatch.delenv("LUXAR_REQUIRE_CUDA", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(nlm_cuda_backend, "NLM_CUDA_AVAILABLE", False)

    module.pytest_configure()


def test_required_nlm_cuda_gate_accepts_an_available_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import luxar.gsplats.preprocessing.cuda as nlm_cuda_backend

    module = _load(NLM_CUDA_TEST_CONFTEST, "available_nlm_cuda_test_conftest")
    monkeypatch.setenv("LUXAR_REQUIRE_CUDA", "1")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(nlm_cuda_backend, "NLM_CUDA_AVAILABLE", True)

    module.pytest_configure()


def test_release_preflight_matches_case_insensitive_workflow_comparison(
    tmp_path: Path,
) -> None:
    output, _, _ = _run_release_preflight(tmp_path, repository="True")

    assert "ENABLE_NPM_PUBLISH=True from repository" in output
    assert "the tag WILL publish @luxar/viewer" in output


def test_release_preflight_does_not_trim_the_switch_value(tmp_path: Path) -> None:
    output, _, _ = _run_release_preflight(tmp_path, repository="true\n")

    assert "the tag will NOT publish to npm" in output
    assert "the tag WILL publish" not in output


def test_release_preflight_reads_shared_organization_variable(tmp_path: Path) -> None:
    output, calls, _ = _run_release_preflight(tmp_path, organization="true")

    assert "ENABLE_NPM_PUBLISH=true from organization" in output
    assert "the tag WILL publish @luxar/viewer" in output
    assert calls == [
        "repos/royerlab/luxar/environments/npm/variables",
        "repos/royerlab/luxar/actions/variables",
        "repos/royerlab/luxar/actions/organization-variables",
    ]


def test_release_preflight_does_not_turn_a_failed_second_get_into_off(
    tmp_path: Path,
) -> None:
    output, calls, _ = _run_release_preflight(
        tmp_path, repository="true", repository_direct="ERROR"
    )

    assert "ENABLE_NPM_PUBLISH=true from repository" in output
    assert "the tag WILL publish @luxar/viewer" in output
    assert calls == [
        "repos/royerlab/luxar/environments/npm/variables",
        "repos/royerlab/luxar/actions/variables",
    ]


@pytest.mark.parametrize(
    ("environment", "repository", "organization", "failed_level"),
    [
        ("ERROR", "true", "true", "npm environment"),
        ("UNSET", "ERROR", "true", "repository"),
        ("UNSET", "UNSET", "ERROR", "organization"),
    ],
)
def test_release_preflight_reports_lookup_failures_as_unknown(
    tmp_path: Path,
    environment: str,
    repository: str,
    organization: str,
    failed_level: str,
) -> None:
    output, _, _ = _run_release_preflight(
        tmp_path,
        environment=environment,
        repository=repository,
        organization=organization,
    )

    assert f"could not read {failed_level} variables" in output
    assert "cannot tell whether the tag will publish to npm" in output
    assert "the tag WILL publish" not in output
    assert "the tag will NOT publish" not in output


def test_release_preflight_distinguishes_unset_from_unreadable(tmp_path: Path) -> None:
    output, _, _ = _run_release_preflight(tmp_path)

    assert "ENABLE_NPM_PUBLISH is UNSET" in output
    assert "the FIRST publish must be a manual" in output
    assert "cannot tell" not in output


def test_set_version_and_the_gate_agree_on_where_the_files_are() -> None:
    """A path that drifts between the writer and the checker fails silently."""
    writer = _load(SET_VERSION, "set_version_paths")
    checker = _load(CHECK_VERSIONS, "check_versions_paths")
    for attr in ("REPO", "INIT", "PKG_JSON", "CITATION"):
        assert getattr(writer, attr) == getattr(checker, attr), attr
