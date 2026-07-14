# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

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

# Suppress warnings for missing references
suppress_warnings = ["ref.any"]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx mapping (link to other project's documentation)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}

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
    "templates",
    # Archived/internal docs not part of the main documentation build
    "archive/**",
    "benchmarks/**",
    "bugs/**",
    "code_reviews/**",
    "handoffs/**",
    "reports/**",
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
