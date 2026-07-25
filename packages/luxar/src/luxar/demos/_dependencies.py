"""Single source of truth for gating a demo's OPTIONAL dependencies.

Demos pull in heavyweight extras (``umap-learn``, ``sentence-transformers``,
``torch``, ``esm``, …) that the core package deliberately does not require. Two
rules govern how they are demanded, and both exist because breaking them
produced real bugs:

**1. Gate at the point of use, never at the entry point.** A demo whose
expensive artifact is already cached must run WITHOUT the dependency that
produced it — a warm ``cache_computed``/``cached_download`` cache needs no UMAP
and no model. An entry-point preflight (``try: import umap … except ImportError:
sys.exit(1)`` inside ``main()``) refuses a machine that holds every artifact it
needs, and it silently contradicts the cache-first design those demos advertise.
Because the raw download is itself cached, deferring the gate costs nothing
durable on a cold run either: the download is kept, so re-running after the
install resumes instead of refetching.
:func:`require_module` is the point-of-use gate. ``test_no_entrypoint_dependency
_preflight.py`` fails the build if an entry-point gate reappears.

**2. Advertise the CONSTRAINED requirement, never a bare package name.** A bare
``pip install anndata`` resolves to 0.13+, which requires ``zarr>=3.1`` and
silently upgrades Luxar past its ``zarr>=2.16,<3.0`` pin — breaking every store
on disk. :data:`INSTALL_SPECS` therefore carries the version bound, and
``test_demos_dependencies.py`` fails if a spec here drifts from ``pyproject.toml``.

The gate RAISES and does not print: the demo entry points already report the
exception, and a helper that printed as well showed the user the same message
twice.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple


class DependencySpec(NamedTuple):
    """How to install one optional dependency, and why it is bounded."""

    #: PEP 440 requirement to advertise, e.g. ``anndata>=0.10,<0.13``. MUST stay
    #: equivalent to the corresponding pin in ``pyproject.toml``.
    spec: str
    #: Luxar extra that installs it with the right constraints already applied.
    extra: str
    #: Optional explanation, used when the bound is load-bearing or when a warm
    #: cache makes the dependency skippable.
    note: str = ""


#: Module name -> how to install it. Keyed by the name passed to ``import``,
#: which is not always the distribution name (``umap`` vs ``umap-learn``,
#: ``PIL`` vs ``Pillow``, ``sentence_transformers`` vs ``sentence-transformers``).
INSTALL_SPECS: dict[str, DependencySpec] = {
    "anndata": DependencySpec(
        "anndata>=0.10,<0.13",
        "demos",
        "The upper bound is REQUIRED: anndata >= 0.13 pulls zarr >= 3.1, which "
        "conflicts with Luxar's zarr>=2.16,<3.0 pin and would break every "
        "Luxar store. 0.10/0.11 declare no zarr dependency; 0.12 asks for "
        "zarr>=2.18.7,!=3.0.*, which zarr 2.18.7 satisfies.",
    ),
    "esm": DependencySpec(
        "esm>=3.0.0",
        "demos",
        "Needed only to COMPUTE embeddings; a complete cached embeddings file "
        "skips the model entirely.",
    ),
    "gdown": DependencySpec(
        "gdown",
        "",  # intentionally not in any extra — Google-Drive fetches only
        "Not part of any Luxar extra: it is only needed for the Google-Drive "
        "download path.",
    ),
    "h5py": DependencySpec("h5py>=3.0.0", "demos"),
    "pandas": DependencySpec("pandas>=1.5.0", "demos"),
    "PIL": DependencySpec(
        "Pillow>=9.0.0",
        "demos",
        "Needed only to build image thumbnails; the point cloud itself does "
        "not require it.",
    ),
    "scipy": DependencySpec("scipy>=1.15.0,<2.0", "demos"),
    "sentence_transformers": DependencySpec(
        "sentence-transformers>=2.2.0",
        "demos",
        "Needed only to COMPUTE text embeddings; a cached bundle skips it.",
    ),
    "torch": DependencySpec(
        "torch>=2.2,<3.0",
        "gsplats",
        "The <3.0 cap is load-bearing: the compiled CUDA splatting extension is "
        "built against a specific torch ABI.",
    ),
    "umap": DependencySpec(
        "umap-learn>=0.5.0",
        "demos",
        "Needed only to COMPUTE the projection; a cached UMAP skips it.",
    ),
}


class MissingDependencyError(ImportError):
    """An optional demo dependency is not installed.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working, while callers that care can catch this specifically.
    """


def require_module(module: str, *, pip_name: str | None = None) -> Any:
    """Import an optional dependency, or raise with an actionable install hint.

    Call this AT THE POINT OF USE — immediately before the work that needs the
    module — so a demo running off a warm cache never demands it. See the module
    docstring for why that ordering is a rule rather than a preference.

    Args:
        module: Name to import (``umap``, not ``umap-learn``).
        pip_name: Override the advertised requirement. Defaults to the
            constrained spec from :data:`INSTALL_SPECS`, else the module name.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the import fails. The message names the
            constrained requirement, the Luxar extra that provides it (when
            there is one), and any explanation attached to the spec.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        # A submodule inherits its package's spec, so `require_module("PIL.Image")`
        # still advertises `Pillow>=9.0.0` rather than a bare "PIL.Image".
        known = INSTALL_SPECS.get(module) or INSTALL_SPECS.get(module.split(".")[0])
        if pip_name is not None:
            spec, extra, note = pip_name, "", ""
        elif known is not None:
            spec, extra, note = known
        else:
            spec, extra, note = module, "", ""

        parts = [
            f"Missing dependency: {module}.",
            f"Install with `pip install '{spec}'`",
        ]
        if extra:
            parts[-1] += f" (or the whole extra: `pip install 'luxar[{extra}]'`)"
        parts[-1] += "."
        if note:
            parts.append(note)
        raise MissingDependencyError(" ".join(parts)) from exc
