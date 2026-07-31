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

:func:`survey` reads the same table WITHOUT importing anything, which is what
``luxar demo deps`` reports and installs from. Because both the runtime gate and
the installer are driven by :data:`INSTALL_SPECS`, a dependency cannot be
installable-but-unadvertised or advertised-but-uninstallable.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, NamedTuple


class DependencySpec(NamedTuple):
    """How to install one optional dependency, and why it is bounded."""

    #: PEP 440 requirement to advertise, e.g. ``anndata>=0.10,<0.13``. MUST stay
    #: equivalent to the corresponding pin in ``pyproject.toml``.
    spec: str
    #: Luxar extra this spec is ATTRIBUTED to (``""`` for specs in no extra). A
    #: package pinned in two extras (e.g. ``tifffile`` in both ``io`` and
    #: ``demos``) records only one here, so filtering by extra reports what is
    #: attributed to it, not everything installing that extra would pull in.
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
    "astropy": DependencySpec("astropy>=6.0.0", "demos"),
    "cellxgene_census": DependencySpec("cellxgene-census>=1.0.0", "demos"),
    "gdown": DependencySpec(
        "gdown",
        "",  # intentionally not in any extra — Google-Drive fetches only
        "Not part of any Luxar extra: it is only needed for the Google-Drive "
        "download path.",
    ),
    "h5py": DependencySpec("h5py>=3.0.0", "demos"),
    "imagecodecs": DependencySpec(
        "imagecodecs>=2023.1.0",
        "demos",
        "Never imported directly: tifffile needs it to decode COMPRESSED TIFFs, "
        "so a demo reading one fails inside tifffile without it.",
    ),
    "imageio": DependencySpec("imageio>=2.31.0", "io"),
    "matplotlib": DependencySpec(
        "matplotlib>=3.5.0",
        "demos",
        "Needed only by the gsplat demos' `--show-roundtrip` diagnostic plot; "
        "the demo itself runs without it.",
    ),
    "mrcfile": DependencySpec("mrcfile>=1.4.0", "demos"),
    "networkx": DependencySpec("networkx>=3.0", "demos"),
    "nibabel": DependencySpec("nibabel>=5.0.0", "demos"),
    "pandas": DependencySpec("pandas>=1.5.0", "demos"),
    "PIL": DependencySpec(
        "Pillow>=9.0.0",
        "demos",
        "Needed only to build image thumbnails; the point cloud itself does "
        "not require it.",
    ),
    "pooch": DependencySpec(
        "pooch>=1.6.0",
        "demos",
        "scikit-image's dataset FETCHER. cells3d/kidney are not bundled in the "
        "scikit-image wheel, so the cells3d and kidney demos need pooch to "
        "download them on a cold cache — having scikit-image is not enough.",
    ),
    "pyarrow": DependencySpec(
        "pyarrow>=12.0.0",
        "demos",
        "Never imported directly: it is the engine behind `pandas.read_parquet`, "
        "which several demos use to load their point tables.",
    ),
    "scipy": DependencySpec(
        "scipy>=1.15.0,<2.0",
        "demos",
        "The demos floor is HIGHER than the gsplats extra's (>=1.9.0): "
        "demo_quantum_orbitals uses scipy.special.sph_harm_y, added in 1.15.",
    ),
    "sentence_transformers": DependencySpec(
        "sentence-transformers>=2.2.0",
        "demos",
        "Needed only to COMPUTE text embeddings; a cached bundle skips it.",
    ),
    "shapefile": DependencySpec("pyshp>=2.3.0", "demos"),
    "skimage": DependencySpec(
        "scikit-image>=0.19.0",
        "demos",
        "Pair it with pooch: the cells3d/kidney sample data is fetched, not bundled.",
    ),
    "sklearn": DependencySpec("scikit-learn>=1.0", "demos"),
    "tifffile": DependencySpec("tifffile>=2023.1.0", "demos"),
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


class DependencyStatus(NamedTuple):
    """One row of :func:`survey`: a spec plus whether it is importable here."""

    module: str
    spec: DependencySpec
    installed: bool


def is_installed(module: str) -> bool:
    """Whether ``module`` can be imported, WITHOUT importing it.

    Uses :func:`importlib.util.find_spec`, because actually importing the table
    would be ruinous: ``torch`` alone costs seconds and allocates CUDA context,
    and ``esm``/``sentence_transformers`` pull model machinery. A survey must be
    instant and side-effect free.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # ImportError: a parent package is missing. ValueError: __spec__ is None
        # on some oddly-initialised modules. Both mean "not usable here".
        return False


def survey(extra: str | None = None) -> list[DependencyStatus]:
    """Report install status for every known optional dependency.

    Args:
        extra: Restrict to specs ATTRIBUTED to this Luxar extra (``"demos"``,
            ``"io"``, ``"gsplats"``) — i.e. recorded under it in the table, not
            every spec installing the extra would pull in (a package pinned in
            two extras is attributed to only one). ``None`` surveys the whole
            table, including the specs that deliberately belong to no extra.

    Returns:
        One :class:`DependencyStatus` per spec, sorted case-insensitively by
        module name so the CLI's output order is stable.
    """
    rows = [
        DependencyStatus(module, spec, is_installed(module))
        for module, spec in INSTALL_SPECS.items()
        if extra is None or spec.extra == extra
    ]
    return sorted(rows, key=lambda r: r.module.lower())


def extras_for(rows: list[DependencyStatus]) -> list[str]:
    """The distinct Luxar extras that would install ``rows``, sorted.

    Specs outside every extra (``extra == ""``) contribute nothing — they cannot
    be installed via ``luxar[...]`` and must be named individually.
    """
    return sorted({r.spec.extra for r in rows if r.spec.extra})
