"""Hatch build hook for Luxar's prebuilt web viewer.

Editable installs use the viewer directly from the source tree and must not
require the gitignored production ``dist`` artifact. Standard wheel builds do
bundle that artifact and fail early with an actionable message when it is
missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


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
