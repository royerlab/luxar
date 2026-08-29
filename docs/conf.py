# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import sys

from sphinx.util import logging as sphinx_logging

sys.path.insert(0, os.path.abspath("../packages/luxar/src"))

# -- Project information -----------------------------------------------------
project = "Luxar"
copyright = "2025-2026, Luxar Development Team"
author = "Luxar Development Team"

# Derive the version from the installed package metadata (pyproject uses a
# dynamic version), falling back to a placeholder if Luxar is not installed.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("luxar")
except Exception:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",  # For markdown support
]

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock imports for packages that may not be installed (e.g., torch, optional deps)
autodoc_mock_imports = ["torch", "torchvision", "pytorch3d", "scipy"]

# Do not suppress reference warnings: the warning-fatal HTML build is the
# deterministic internal-link gate. External HTTP checking is a separate,
# opt-in linkcheck build because remote sites are not reliable CI dependencies.
# Every exception below is narrow and records why linkcheck cannot verify it.
linkcheck_ignore = [
    # Literal examples emitted by autodoc; ``host`` and ``port`` are placeholders.
    r"^http://host:port(?:/.*)?$",
    # The repository is private, so unauthenticated linkcheck receives 404.
    # Keep this to the two currently referenced endpoints; review new paths.
    r"^https://github\.com/royerlab/luxar(?:/issues)?$",
    # DOI resolves in browsers, but the AIP destination rejects automated probes.
    r"^https://doi\.org/10\.1063/1\.1751381$",
    # Khronos serves this page but rejects automated probes with HTTP 403.
    r"^https://wikis\.khronos\.org/webgl/Debugging$",
    # TypeDoc creates this target after Sphinx; linkcheck cannot see that output.
    r"^viewer/index\.html$",
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx mapping (link to other project's documentation)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}


# The HTML build runs with -W (see the `docs:build` script in pyproject.toml),
# which is what makes it the internal-reference gate. But intersphinx has to
# reach three third-party sites to load the inventories above, and Sphinx
# reports an unreachable inventory as an UNTYPED warning — `suppress_warnings`
# has no name to match it on. Left alone, one bad minute at
# docs.python.org/numpy.org/readthedocs turns a required check red for reasons
# that have nothing to do with the pull request, which is the exact failure mode
# that keeping linkcheck opt-in is meant to avoid.
#
# So demote that single record to informational: it still prints, but it no
# longer counts toward -W. Nothing else is relaxed — unresolved references
# inside our own documentation are still warnings, and still fatal. If
# intersphinx ever renames its logger the filter simply stops matching and we
# are back to today's behavior rather than a broken build.
class _IntersphinxOutageIsInformational(logging.Filter):
    """Keep an unreachable intersphinx inventory out of the -W warning count."""

    def filter(self, record: logging.LogRecord) -> bool:
        if "failed to reach any of the inventories" in str(record.msg):
            record.levelno = logging.INFO
            record.levelname = "INFO"
        return True


# Attach to the logger intersphinx actually emits on. Although the fetch code
# lives in `sphinx.ext.intersphinx._load`, that module deliberately defines its
# LOGGER as `getLogger("sphinx.ext.intersphinx")`, not as a child logger. Sphinx
# then namespaces it under `sphinx.`, so the real name is
# `sphinx.sphinx.ext.intersphinx`; ask Sphinx for it instead of hand-writing that
# doubled prefix, which reads like a typo and invites a well-meaning "fix" that
# would silently stop the filter from matching. It has to be the emitting
# logger, not an ancestor: stdlib only runs a logger's own filters, never a
# parent's, on a record that merely propagates up — and they run before the
# handlers, so the demotion lands before the warning handler (and thus -W) ever
# sees the record.
sphinx_logging.getLogger("sphinx.ext.intersphinx").logger.addFilter(
    _IntersphinxOutageIsInformational()
)

# MyST parser settings (for markdown files)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
# Generate anchors for markdown headings (levels 1-4) so in-document
# TOC links like [Architecture](#architecture) resolve.
myst_heading_anchors = 4

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Benchmark result data, not prose: kept in the repo for the bisection
    # tooling, but it is not documentation and does not belong in the build.
    "benchmarks/**",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"  # Popular Read the Docs theme
# No static assets are shipped; an empty list avoids the missing-dir warning.
html_static_path: list[str] = []
html_title = "Luxar Documentation"
html_short_title = "Luxar"
html_logo = None  # Add logo path if you have one
html_favicon = None  # Add favicon if you have one

# Theme options
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets)
# html_static_path = ['_static']

# Output file base name for HTML help builder
htmlhelp_basename = "Luxardoc"
