"""Tests for Luxar's custom Hatch wheel build hook."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Pytest imports this file through the ``luxar.tests`` package, so the repository
# root (which owns hatch_build.py) is not otherwise on sys.path. Wheels ship this
# test module but not the root-level hook, so skip outside a source checkout
# (mirroring the installed-wheel guards in other repository-only tests).
_PARENTS = Path(__file__).resolve().parents
PROJECT_ROOT = _PARENTS[5] if len(_PARENTS) > 5 else None
if PROJECT_ROOT is None or not (PROJECT_ROOT / "hatch_build.py").is_file():
    pytest.skip(
        "hatch_build.py only exists in a source checkout, not in installed wheels",
        allow_module_level=True,
    )
sys.path.insert(0, str(PROJECT_ROOT))

from hatch_build import (  # noqa: E402
    REPO_URL,
    LuxarBuildHook,
    LuxarMetadataHook,
    absolutize_readme_links,
)

REPO_BLOB_URL = f"{REPO_URL}/blob/main/"
REPO_TREE_URL = f"{REPO_URL}/tree/main/"


def make_hook(root: Path) -> LuxarBuildHook:
    """Construct the hook with inert Hatchling collaborators."""
    return LuxarBuildHook(
        root=str(root),
        config={},
        build_config=MagicMock(),
        metadata=MagicMock(),
        directory=str(root / "build"),
        target_name="wheel",
    )


def test_editable_build_replaces_viewer_force_include(tmp_path: Path) -> None:
    """Editable builds use a non-empty marker map instead of viewer ``dist``."""
    build_data: dict[str, object] = {}

    make_hook(tmp_path).initialize("editable", build_data)

    assert build_data == {
        "force_include_editable": {
            "hatch_build.py": "_luxar_editable_build_marker",
        }
    }


def test_standard_build_requires_viewer_index(tmp_path: Path) -> None:
    """Standard wheels fail early when the production viewer is absent."""
    with pytest.raises(FileNotFoundError, match=r"make build-viewer"):
        make_hook(tmp_path).initialize("standard", {})


def test_standard_build_accepts_viewer_index(tmp_path: Path) -> None:
    """Standard wheels proceed once the production viewer index exists."""
    viewer_index = tmp_path / "packages" / "luxar-viewer" / "dist" / "index.html"
    viewer_index.parent.mkdir(parents=True)
    viewer_index.touch()
    build_data: dict[str, object] = {}

    make_hook(tmp_path).initialize("standard", build_data)

    assert build_data == {}


def test_unknown_build_version_is_rejected(tmp_path: Path) -> None:
    """Unexpected Hatchling build modes fail instead of bypassing validation."""
    with pytest.raises(ValueError, match=r"expected 'editable' or 'standard'"):
        make_hook(tmp_path).initialize("unexpected", {})


# ---------------------------------------------------------------------------
# The metadata hook: README links must survive the PyPI page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("[a](LICENSE)", f"[a]({REPO_BLOB_URL}LICENSE)"),
        ("[a](./CITATION.cff)", f"[a]({REPO_BLOB_URL}CITATION.cff)"),
        (
            "[a](docs/guides/user/VIEWER_GUIDE.md#cors)",
            f"[a]({REPO_BLOB_URL}docs/guides/user/VIEWER_GUIDE.md#cors)",
        ),
        ("[a](.agents/skills/)", f"[a]({REPO_TREE_URL}.agents/skills/)"),
        ('<a href="SECURITY.md">', f'<a href="{REPO_BLOB_URL}SECURITY.md">'),
        ('<img src="docs/x.png">', f'<img src="{REPO_BLOB_URL}docs/x.png">'),
        # Left alone: absolute, protocol-relative, anchors, mailto, data URIs.
        ("[a](https://x.y/z)", "[a](https://x.y/z)"),
        ('<img src="//cdn/x.png">', '<img src="//cdn/x.png">'),
        ("[a](#gallery)", "[a](#gallery)"),
        ('<a href="mailto:a@b.c">', '<a href="mailto:a@b.c">'),
        (
            '<img src="data:image/png;base64,AA">',
            '<img src="data:image/png;base64,AA">',
        ),
    ],
)
def test_absolutize_readme_links(markdown: str, expected: str) -> None:
    assert absolutize_readme_links(markdown, repository_ref="main") == expected


@pytest.mark.parametrize(
    ("ref_type", "ref_name", "expected_ref"),
    [
        ("tag", "v2026.09.22", "v2026.09.22"),
        ("tag", None, "main"),
        ("tag", "", "main"),
        ("branch", "dev", "main"),
        (None, None, "main"),
    ],
)
def test_metadata_hook_serves_the_readme_with_absolute_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ref_type: str | None,
    ref_name: str | None,
    expected_ref: str,
) -> None:
    (tmp_path / "README.md").write_text(
        "# T\n\nSee [the docs](docs/a.md) and [skills](.agents/skills/).\n"
    )
    metadata: dict[str, object] = {}
    if ref_type is None:
        monkeypatch.delenv("GITHUB_REF_TYPE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_REF_TYPE", ref_type)
    if ref_name is None:
        monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    else:
        monkeypatch.setenv("GITHUB_REF_NAME", ref_name)

    LuxarMetadataHook(str(tmp_path), {}).update(metadata)

    assert metadata == {
        "readme": {
            "content-type": "text/markdown",
            "text": (
                "# T\n\nSee [the docs](https://github.com/royerlab/luxar/"
                f"blob/{expected_ref}/docs/a.md) and "
                "[skills](https://github.com/royerlab/luxar/"
                f"tree/{expected_ref}/.agents/skills/).\n"
            ),
        },
    }


def test_the_real_readme_has_no_relative_link_left_after_rewriting() -> None:
    """The regex the hook uses must cover every link form the README uses."""
    import re  # noqa: PLC0415

    text = absolutize_readme_links(
        (PROJECT_ROOT / "README.md").read_text(), repository_ref="main"
    )
    relative = re.compile(
        r'(?:\]\(|\bhref="|\bsrc=")(?!(?:[a-z][a-z0-9+.-]*:|//|#))([^)"\s]+)'
    )
    assert relative.findall(text) == []
    assert REPO_BLOB_URL + "ACKNOWLEDGMENTS.md" in text
    # Forms the hook does NOT rewrite, so they must not appear with a relative
    # target at all: a relative ``srcset`` or a reference-style definition would
    # pass the hook, pass the wheel check and ship as a dead link on PyPI.
    raw = (PROJECT_ROOT / "README.md").read_text()
    assert re.findall(r'srcset="(?!https?://)[^"]+"', raw) == []
    assert re.findall(r"^\[[^\]]+\]:\s+(?!https?://|#)\S+", raw, re.M) == []


def test_pyproject_wires_the_metadata_hook() -> None:
    """A static ``readme`` would silently bypass the hook."""
    import tomllib  # noqa: PLC0415

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert "readme" in pyproject["project"]["dynamic"]
    assert "readme" not in pyproject["project"]
    assert pyproject["tool"]["hatch"]["metadata"]["hooks"]["custom"] == {
        "path": "hatch_build.py"
    }
