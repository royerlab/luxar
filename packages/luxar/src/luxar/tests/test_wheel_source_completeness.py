"""No tracked source file may be hidden from the wheel builder by `.gitignore`.

Hatchling selects wheel contents with its own file walk, and that walk honours
`.gitignore` — independently of whether Git is actually tracking the file. So a
source file can be committed, reviewed, imported, covered by tests and green in
CI, and still be **absent from every built wheel**.

That happened. `.gitignore` carried a bare `downloads/`, intended for a
root-level artifact directory. Git patterns without a leading slash match at any
depth, so it also matched
`packages/luxar/src/luxar/demos/_support/downloads/` — a real subpackage. Git
kept tracking those five files, nothing in the repository looked wrong, and
every wheel shipped without them. On a clean `pip install luxar`, the very first
command a user runs died:

    ModuleNotFoundError: No module named 'luxar.demos._support.downloads'

The failure is invisible from the source tree, which is where every other check
looks. This test looks at the one place that matters: what the packaging tool
will be allowed to see.

Deliberately not a wheel build. Building takes minutes and needs the viewer
dist; asking Git which tracked files it would *also* ignore is instant and
catches the same class.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = "packages/luxar/src/luxar"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _tracked_package_files() -> list[str]:
    out = _git("ls-files", "--", PACKAGE_ROOT)
    return [line for line in out.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    try:
        files = _tracked_package_files()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"not a usable Git checkout: {exc}")
    if not files:
        pytest.skip("no tracked package files (shallow or exported tree)")
    return files


def test_no_tracked_package_file_is_gitignored(tracked_files: list[str]) -> None:
    """The guard: a tracked file that `.gitignore` also matches is a wheel hole."""
    # `--no-index` is the whole point — without it Git short-circuits on tracked
    # paths and reports nothing, which is exactly the blind spot that let the
    # `downloads/` rule sit unnoticed.
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--verbose", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(tracked_files),
        capture_output=True,
        text=True,
    )
    # Exit 0 = at least one path matched; 1 = none matched (the healthy case).
    if result.returncode == 1:
        return
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed: {result.stderr.strip()}")

    offenders = [line for line in result.stdout.splitlines() if line.strip()]
    assert not offenders, (
        "These files are tracked by Git but ALSO matched by .gitignore. Hatchling "
        "honours .gitignore when building the wheel, so they will be silently "
        "dropped from the distribution even though the repository looks correct:\n  "
        + "\n  ".join(offenders)
        + "\n\nAnchor the offending pattern (a leading '/' scopes it to the repo "
        "root) rather than deleting it."
    )


def test_the_downloads_subpackage_specifically_survives(
    tracked_files: list[str],
) -> None:
    """The regression that prompted this file, pinned by name.

    Broad guards drift; this one names the module whose absence broke
    `luxar --version` so a future re-broadening is unambiguous.
    """
    downloads = [f for f in tracked_files if "/demos/_support/downloads/" in f]
    assert downloads, "the downloads subpackage vanished from the tracked set"

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--verbose", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(downloads),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        "luxar.demos._support.downloads is gitignored again — it will be dropped "
        f"from the wheel and the CLI will not import:\n{result.stdout}"
    )


def test_the_guard_can_actually_fire(tmp_path: Path) -> None:
    """A guard that cannot go red is indistinguishable from no guard.

    Builds a throwaway repo with the exact defect — a tracked file that an
    unanchored rule also matches — and asserts `check-ignore --no-index` reports
    it. This pins the *mechanism*, so the tests above cannot quietly become
    vacuous if the flag's behaviour changes.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    pkg = tmp_path / "src" / "pkg" / "downloads"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("downloads/\n")
    subprocess.run(["git", "add", "-A", "-f"], cwd=tmp_path, check=True)

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--verbose", "src/pkg/downloads/mod.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "the unanchored rule should have matched"
    assert "downloads/" in result.stdout

    # And anchoring is the fix, not deletion.
    (tmp_path / ".gitignore").write_text("/downloads/\n")
    anchored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--verbose", "src/pkg/downloads/mod.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert anchored.returncode == 1, "anchoring should stop the nested match"
