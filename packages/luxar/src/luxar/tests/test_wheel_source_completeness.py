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
catches the same class. Hatchling applies only the repository-root `.gitignore`
to its project walk, while force-included files bypass that exclusion spec and
are deliberately outside this guard's scope.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = "packages/luxar/src/luxar"


def _git(*args: str) -> str:
    """Run Git in the repository and return its standard output."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _tracked_package_files() -> list[str]:
    """Return tracked package paths without Git's pathname quoting."""
    out = _git("ls-files", "-z", "--", PACKAGE_ROOT)
    return [path for path in out.split("\0") if path]


def _root_gitignore_matches(paths: list[str], *, cwd: Path = REPO_ROOT) -> list[str]:
    """Return root `.gitignore` matches that can exclude Hatchling inputs."""
    if not paths:
        return []

    result = subprocess.run(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--no-index",
            "--verbose",
            "-z",
            "--stdin",
        ],
        cwd=cwd,
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed: {result.stderr.strip()}")

    fields = result.stdout.split("\0")
    if fields[-1:] == [""]:
        fields.pop()
    if len(fields) % 4:
        pytest.fail(f"unexpected git check-ignore output: {result.stdout!r}")

    offenders = []
    for source, line_number, pattern, path in zip(
        fields[0::4], fields[1::4], fields[2::4], fields[3::4], strict=True
    ):
        if source == ".gitignore" and not pattern.startswith("!"):
            offenders.append(f"{source}:{line_number}:{pattern}\t{path}")
    return offenders


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    """Provide tracked package files when tests run from a Git checkout."""
    try:
        files = _tracked_package_files()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"not a usable Git checkout: {exc}")
    if not files:
        pytest.skip("no tracked package files (shallow or exported tree)")
    return files


def test_no_tracked_package_file_is_gitignored(tracked_files: list[str]) -> None:
    """The guard: a tracked file that `.gitignore` also matches is a wheel hole."""
    offenders = _root_gitignore_matches(tracked_files)
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

    offenders = _root_gitignore_matches(downloads)
    assert not offenders, (
        "luxar.demos._support.downloads is gitignored again — it will be dropped "
        f"from the wheel and the CLI will not import:\n{offenders}"
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


def test_the_guard_matches_only_hatchlings_exclusion_source(tmp_path: Path) -> None:
    """Negations and Git-only exclusion sources must not create false failures."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n*.py\n")
    (package / ".gitignore").write_text("nested.txt\n")
    (tmp_path / ".git" / "info" / "exclude").write_text("info.txt\n")
    global_excludes = tmp_path / "global-excludes"
    global_excludes.write_text("global.txt\n")
    subprocess.run(
        ["git", "config", "core.excludesFile", str(global_excludes)],
        cwd=tmp_path,
        check=True,
    )

    paths = [
        "src/pkg/drop.log",
        "src/pkg/keep.log",
        "src/pkg/nested.txt",
        "src/pkg/info.txt",
        "src/pkg/global.txt",
        "src/pkg/snowman-☃\nmodule.py",
    ]
    offenders = _root_gitignore_matches(paths, cwd=tmp_path)

    assert offenders == [
        ".gitignore:1:*.log\tsrc/pkg/drop.log",
        ".gitignore:3:*.py\tsrc/pkg/snowman-☃\nmodule.py",
    ]
