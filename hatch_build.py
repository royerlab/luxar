"""Hatch hooks for Luxar: the prebuilt web viewer and the PyPI README.

Build hook: editable installs use the viewer directly from the source tree and
must not require the gitignored production ``dist`` artifact. Standard wheel
builds do bundle that artifact and fail early with an actionable message when
it is missing.

Metadata hook: ``README.md`` is the PyPI long description, and it links to
``docs/``, ``LICENSE``, ``ACKNOWLEDGMENTS.md`` and the agent skills with
RELATIVE paths, which GitHub resolves against the repository and PyPI resolves
against ``pypi.org/project/luxar/`` -- a 404 for every one of them. The hook
rewrites those links to absolute GitHub URLs in the wheel's metadata only, so
the README in the repository keeps its clean relative links (and GitHub keeps
resolving them per branch) while the PyPI page works. ``scripts/check_wheel.py``
asserts on the built wheel that no relative link survived.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface

#: Where a relative README link points once it leaves the repository. ``main``
#: rather than the release tag: local and editable builds carry versions whose
#: tags may not exist yet, and the README on ``main`` is the one PyPI describes.
REPO_BLOB_URL = "https://github.com/royerlab/luxar/blob/main/"
REPO_TREE_URL = "https://github.com/royerlab/luxar/tree/main/"

#: A Markdown link target or an HTML ``href``/``src`` that is neither absolute
#: (scheme or protocol-relative), an in-page anchor, nor a mailto/data URI.
_RELATIVE_LINK = re.compile(
    r'(\]\(|\bhref="|\bsrc=")(?!(?:[a-z][a-z0-9+.-]*:|//|#))([^)"\s]+)'
)


def absolutize_readme_links(text: str) -> str:
    """Rewrite relative links in Markdown text to absolute GitHub URLs.

    A trailing slash marks a directory and goes to ``tree/``; anything else goes
    to ``blob/``, fragment preserved.
    """

    def _rewrite(match: re.Match[str]) -> str:
        prefix, target = match.group(1), match.group(2)
        path, hash_sign, fragment = target.partition("#")
        path = path.removeprefix("./")
        base = REPO_TREE_URL if path.endswith("/") else REPO_BLOB_URL
        return f"{prefix}{base}{path}{hash_sign}{fragment}"

    return _RELATIVE_LINK.sub(_rewrite, text)


class LuxarMetadataHook(MetadataHookInterface):
    """Serve the README as the long description with PyPI-safe links."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        readme = Path(self.root) / "README.md"
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": absolutize_readme_links(readme.read_text(encoding="utf-8")),
        }


class LuxarBuildHook(BuildHookInterface):
    """Separate editable-install requirements from release-wheel requirements."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            # A non-empty force_include_editable map replaces the wheel target's
            # configured force-include map. The marker is outside the ``luxar``
            # package, so it cannot shadow the source-tree editable package.
            build_data["force_include_editable"] = {
                "hatch_build.py": "_luxar_editable_build_marker"
            }
            return

        if version == "standard":
            viewer_index = (
                Path(self.root) / "packages" / "luxar-viewer" / "dist" / "index.html"
            )
            if not viewer_index.is_file():
                raise FileNotFoundError(
                    "Luxar viewer build missing: expected "
                    "packages/luxar-viewer/dist/index.html. Run "
                    "`make build-viewer` before building a wheel."
                )
            return

        raise ValueError(
            f"Unsupported Luxar wheel build version {version!r}; expected "
            "'editable' or 'standard'"
        )
