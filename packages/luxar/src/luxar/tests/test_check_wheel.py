"""Tests for the built-wheel inspector (``scripts/check_wheel.py``).

Every arm is exercised against a real ``.whl`` (a zip built in ``tmp_path``)
rather than a mocked ``ZipFile``, because the failures this gate exists to catch
are all "the archive on disk is not what the configuration says it is" — and a
mock would be built from the same belief as the checker.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
_SCRIPT = PROJECT_ROOT / "scripts" / "check_wheel.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="repo script not present (packaged install without scripts/)",
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _load_checker() -> ModuleType:
    """Import ``scripts/check_wheel.py`` as a module by file path."""
    spec = importlib.util.spec_from_file_location("check_wheel", _SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# ---------------------------------------------------------------------------
# glob_to_regex — the `*` vs `**` distinction the exclude patterns rely on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "path", "matches"),
    [
        # `**/x/**` crosses directories in both directions.
        ("**/demos/data/**", "luxar/demos/data/a.npz", True),
        ("**/demos/data/**", "luxar/demos/data/deep/a.npz", True),
        ("**/demos/data/**", "luxar/demos/data", False),
        ("**/demos/data/**", "luxar/demos/scripts/a.py", False),
        # A single `*` must NOT cross a directory separator — this is the whole
        # reason for not using fnmatch.translate, which maps `*` onto `.*`.
        ("**/.gitignore", "luxar/core/.gitignore", True),
        ("*.pyc", "a.pyc", True),
        ("*.pyc", "luxar/a.pyc", True),
        ("__pycache__", "luxar/__pycache__/a.pyc", True),
        ("cache/", "luxar/cache/a.bin", True),
        ("cache/", "luxar/cache", False),
        ("luxar/cache/", "luxar/cache/a.bin", True),
        ("**/demos/data", "luxar/demos/data/a.npz", True),
        ("**/gsplats/demos/**", "luxar/gsplats/demos/run.py", True),
        ("**/gsplats/demos/**", "luxar/gsplats/seeds/demos/run.py", False),
        ("**/gsplats/**/demos/**", "luxar/gsplats/demos/run.py", True),
        ("/luxar/demos/data/**", "luxar/demos/data/a.npz", True),
    ],
)
def test_glob_to_regex_distinguishes_star_from_doublestar(
    pattern: str, path: str, matches: bool
) -> None:
    """`*` stays within a path segment; `**` crosses them."""
    assert bool(checker.glob_to_regex(pattern).match(path)) is matches


# ---------------------------------------------------------------------------
# A synthetic project + wheel, so each arm can be broken in isolation
# ---------------------------------------------------------------------------

_PYPROJECT = """
[tool.hatch.build.targets.wheel]
packages = ["packages/luxar/src/luxar"]
exclude = [
    "**/demos/data/**",
    "**/.gitignore",
]
"""

#: What a healthy wheel holds: two packages, the viewer dist, and metadata.
_GOOD_MEMBERS = {
    "luxar/__init__.py": b'"""root"""\n',
    "luxar/core/__init__.py": b'"""core"""\n',
    "luxar/demos/__init__.py": b'"""demos"""\n',
    "luxar/_viewer_dist/index.html": b"<html></html>",
    "luxar-1.0.dist-info/METADATA": b"Name: luxar\n",
}


def _make_project(tmp_path: Path) -> Path:
    """A project root whose sources match ``_GOOD_MEMBERS``."""
    src = tmp_path / "packages" / "luxar" / "src" / "luxar"
    for package in ("", "core", "demos"):
        directory = src / package if package else src
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text('"""x"""\n')
    # Excluded content in the SOURCE tree: it must not be expected in the wheel.
    (src / "demos" / "data").mkdir(parents=True, exist_ok=True)
    (src / "demos" / "data" / "__init__.py").write_text('"""payload"""\n')
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT)
    return tmp_path


def _make_wheel(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """Write ``members`` into a real zip named like a wheel."""
    wheel = tmp_path / "luxar-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return wheel


def _inspect(tmp_path: Path, members: dict[str, bytes]):
    """Build a synthetic project + wheel and inspect it."""
    root = _make_project(tmp_path)
    return checker.inspect_wheel(_make_wheel(tmp_path, members), root)


def test_a_sound_wheel_reports_no_problems(tmp_path: Path) -> None:
    """The control arm: everything in order yields an empty report.

    Without this, every assertion below could be satisfied by a checker that
    simply reports problems unconditionally.
    """
    report = _inspect(tmp_path, dict(_GOOD_MEMBERS))

    assert report.problems() == 0
    assert report.missing_packages == []
    assert report.excluded_present == []


def test_a_dropped_subpackage_is_caught(tmp_path: Path) -> None:
    """THE bug this gate exists for: a package silently absent from the wheel.

    A `.gitignore` pattern matching a package directory made hatchling skip it.
    Every test stayed green — they import from the source tree — and the break
    surfaced only as `luxar --version` failing on a clean install.
    """
    members = dict(_GOOD_MEMBERS)
    del members["luxar/core/__init__.py"]

    report = _inspect(tmp_path, members)

    assert report.missing_packages == ["luxar/core"]
    assert report.problems() == 1


def test_excluded_content_that_shipped_is_caught(tmp_path: Path) -> None:
    """A regressed exclude ships payload the wheel is configured to omit."""
    members = dict(_GOOD_MEMBERS)
    members["luxar/demos/data/catalog.npz"] = b"\x00" * 64

    report = _inspect(tmp_path, members)

    assert report.excluded_present == ["luxar/demos/data/catalog.npz"]


def test_excluded_source_packages_are_not_expected_in_the_wheel(
    tmp_path: Path,
) -> None:
    """The expectation honours the same excludes the build does.

    `luxar/demos/data` has an `__init__.py` in the synthetic source tree, so a
    naive walk would demand it be present and fail every sound wheel. This is
    the over-correction guard for the check above.
    """
    report = _inspect(tmp_path, dict(_GOOD_MEMBERS))

    assert "luxar/demos/data" not in report.missing_packages


def test_a_git_lfs_pointer_is_caught(tmp_path: Path) -> None:
    """A pointer stub is present, correctly named, and not the data.

    `publish.yml`'s checkout has no `lfs: true`, so this is what a regressed
    exclude actually ships — the plausible-wrong-answer case, not an absence.
    """
    members = dict(_GOOD_MEMBERS)
    members["luxar/core/weights.npz"] = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"0" * 64 + b"\nsize 12345\n"
    )

    report = _inspect(tmp_path, members)

    assert report.lfs_pointers == ["luxar/core/weights.npz"]


def test_a_real_payload_of_pointer_size_is_not_flagged(tmp_path: Path) -> None:
    """Only the LFS magic counts, not the size band it is sniffed in.

    Guards the cheap implementation of the check above: members are only read
    when small enough to be a pointer, so a small REAL file must still pass.
    """
    members = dict(_GOOD_MEMBERS)
    members["luxar/core/small.npz"] = b"\x93NUMPY" + b"\x00" * 100

    report = _inspect(tmp_path, members)

    assert report.lfs_pointers == []


def test_an_oversized_member_is_caught(tmp_path: Path, monkeypatch) -> None:
    """PyPI rejects the upload after the release has otherwise succeeded.

    The limit is monkeypatched rather than writing 100 MB of zeros: the check is
    a comparison, and the constant is asserted separately below.
    """
    monkeypatch.setattr(checker, "PYPI_MAX_FILE_BYTES", 1024)
    members = dict(_GOOD_MEMBERS)
    members["luxar/core/big.bin"] = b"\x00" * 4096

    report = _inspect(tmp_path, members)

    assert [n.split(" (")[0] for n in report.oversized] == ["luxar/core/big.bin"]


def test_the_pypi_limit_is_the_real_one() -> None:
    """The monkeypatch above would hide a wrong constant."""
    assert checker.PYPI_MAX_FILE_BYTES == 100 * 1024 * 1024


def test_a_wheel_over_the_upload_limit_is_caught(tmp_path: Path, monkeypatch) -> None:
    """The limit PyPI actually enforces is on the UPLOADED FILE.

    The per-member check cannot see this case, and this case is the realistic
    one for Luxar: thousands of small test payloads and demo assets summing
    past the cap with nothing individually large. Before this check, such a
    wheel passed every gate and was rejected by PyPI after the tag had been
    pushed — by which time the version is spent.
    """
    # Below the ~700-byte fixture wheel, so the comparison actually trips.
    # The real constant is asserted separately below.
    monkeypatch.setattr(checker, "PYPI_MAX_DIST_BYTES", 100)
    report = _inspect(tmp_path, dict(_GOOD_MEMBERS))

    assert report.dist_too_large
    assert report.problems() >= 1
    # The per-member check must stay silent: no single member is large.
    assert report.oversized == []


def test_a_wheel_inside_the_upload_limit_is_not_flagged(tmp_path: Path) -> None:
    report = _inspect(tmp_path, dict(_GOOD_MEMBERS))

    assert not report.dist_too_large
    assert report.dist_bytes is not None and report.dist_bytes > 0


def test_the_pypi_upload_limit_is_the_real_one() -> None:
    """The monkeypatch above would hide a wrong constant."""
    assert checker.PYPI_MAX_DIST_BYTES == 100 * 1024 * 1024


def test_a_missing_viewer_dist_is_caught(tmp_path: Path) -> None:
    """Mirrors the check publish.yml already makes, so this is a superset."""
    members = {k: v for k, v in _GOOD_MEMBERS.items() if "_viewer_dist" not in k}

    report = _inspect(tmp_path, members)

    assert report.missing_viewer_dist is True


def test_an_empty_wheel_is_an_error_not_a_pass(tmp_path: Path) -> None:
    """A truncated build must not satisfy every check vacuously.

    With no members, `missing_packages` would be the only complaint on a naive
    implementation — and on a project with no packages, none at all.
    """
    root = _make_project(tmp_path)
    wheel = _make_wheel(tmp_path, {})

    with pytest.raises(ValueError, match="no members"):
        checker.inspect_wheel(wheel, root)


def test_a_missing_wheel_is_an_error(tmp_path: Path) -> None:
    """Pointing the gate at nothing must not read as a clean wheel."""
    with pytest.raises(ValueError, match="No such wheel"):
        checker.inspect_wheel(tmp_path / "absent.whl", _make_project(tmp_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_exits_1_and_names_an_oversized_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    """Drive the whole-wheel check through main(), as this file does for the others.

    The report-level test proves the flag flips; this proves the gate actually
    fails the build and says something the operator can act on.
    """
    monkeypatch.setattr(checker, "PYPI_MAX_DIST_BYTES", 100)
    root = _make_project(tmp_path)
    wheel = _make_wheel(tmp_path, dict(_GOOD_MEMBERS))

    code = checker.main([str(wheel), "--project-root", str(root)])

    assert code == 1
    out = capsys.readouterr().out
    # Assert the NUMBERS, not just the heading: the heading survives a wrong
    # limit or a wrong size, which is how the MB/MiB contradiction lived
    # through two rounds. 100 bytes is the patched limit; the fixture wheel is
    # ~700 bytes, so both render as 0.0 MiB — assert the limit token and the
    # units, which is what the operator reads.
    assert "upload limit" in out
    assert "MiB" in out
    assert "MB " not in out.replace("MiB", ""), "decimal MB leaked back in"
    # The per-member section must stay silent: no member is large.
    assert "member(s) exceed" not in out


def test_main_exits_1_and_names_the_dropped_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate's exit code and its message both have to be usable."""
    root = _make_project(tmp_path)
    members = dict(_GOOD_MEMBERS)
    del members["luxar/core/__init__.py"]
    wheel = _make_wheel(tmp_path, members)

    code = checker.main([str(wheel), "--project-root", str(root)])

    out = _ANSI_ESCAPE.sub("", capsys.readouterr().out)
    assert code == 1
    assert "luxar/core" in out
    assert ".gitignore" in out  # the actionable hint


def test_main_exits_0_on_a_sound_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _make_project(tmp_path)
    wheel = _make_wheel(tmp_path, dict(_GOOD_MEMBERS))

    code = checker.main([str(wheel), "--project-root", str(root)])

    assert code == 0
    assert "sound" in _ANSI_ESCAPE.sub("", capsys.readouterr().out)


def test_main_exits_2_on_an_unreadable_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken input is exit 2 — distinct from "the wheel has problems"."""
    root = _make_project(tmp_path)
    bogus = tmp_path / "not-a-wheel.whl"
    bogus.write_bytes(b"this is not a zip archive")

    code = checker.main([str(bogus), "--project-root", str(root)])

    assert code == 2
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Drift guard against the REAL pyproject
# ---------------------------------------------------------------------------


def test_the_checker_imports_only_the_standard_library() -> None:
    """It must run where the wheel is BUILT, not where the project is installed.

    The `wheel-viewer` CI job installs `hatch` and nothing else, and a release
    environment need not have the project's dependencies at all — a wheel
    inspector that requires them cannot inspect a wheel before they exist.

    Not theoretical: the first CI run of this gate died with
    `ModuleNotFoundError: No module named 'arbol'`, because the rest of
    `scripts/` uses arbol for output and this file followed suit. An import
    added for a nicety would break the gate in exactly the same way, and only
    in CI, so it is asserted here rather than left to convention.
    """
    tree = ast.parse(_SCRIPT.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    non_stdlib = sorted(imported - sys.stdlib_module_names)

    assert not non_stdlib, (
        f"scripts/check_wheel.py imports non-stdlib module(s): {non_stdlib}. "
        "It runs in the wheel-viewer CI job, which installs only hatch — a "
        "third-party import here fails the gate with ModuleNotFoundError."
    )


def test_the_real_excludes_are_readable_and_non_empty() -> None:
    """The gate reads its policy from pyproject; an empty read disables it.

    If the table is ever renamed or the key removed, `excluded_present` silently
    stops checking anything, and `expected_packages` starts demanding the
    excluded research demos be in the wheel — a confusing red rather than a
    silent green, but this names the cause directly.
    """
    excludes = checker.read_wheel_excludes(PROJECT_ROOT)

    assert excludes, "no wheel excludes found in pyproject.toml"
    assert "**/demos/data/**" in excludes
