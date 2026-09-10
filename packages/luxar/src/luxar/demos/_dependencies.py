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
``pip install metpy`` happily resolves 1.5.x, which declares only
``numpy>=1.20.0`` and then breaks at runtime against Luxar's ``numpy>=2.0``;
1.6.3 is the release that added NumPy 2 support. :data:`INSTALL_SPECS` therefore
carries the version bound, and ``test_demos_dependencies.py`` fails if a spec
here drifts from ``pyproject.toml``. (The historical example was anndata, capped
at ``<0.13`` because 0.13 required ``zarr>=3.1`` and would have dragged the
project past its old ``zarr<3.0`` pin. That cap is gone — Luxar is on zarr 3 —
but the rule it motivated is not.)

The gate RAISES and does not print: the demo entry points already report the
exception, and a helper that printed as well showed the user the same message
twice. :func:`substitutive_lod_or_flat` is the one exception, and deliberately
so — it gates an OPTIONAL enhancement (LOD coarsening) rather than the demo
itself, so it degrades to a flat scene and prints what was lost.

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

    #: PEP 440 requirement to advertise, e.g. ``metpy>=1.6.3,<2.0``. MUST stay
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
        "anndata>=0.10",
        "demos",
        "Reads the .h5ad inputs for the Tabula Sapiens and Zebrahub-velocity "
        "demos. Formerly capped at <0.13 because anndata 0.13 requires "
        "zarr >= 3.1, which the old zarr<3.0 pin could not satisfy; Luxar is on "
        "zarr 3 now, so the ceiling is gone.",
    ),
    "esm": DependencySpec(
        "esm>=3.0.0",
        "demos",
        "Needed only to COMPUTE embeddings; a complete cached embeddings file "
        "skips the model entirely.",
    ),
    "astropy": DependencySpec("astropy>=6.0.0", "demos"),
    "astroquery": DependencySpec(
        "astroquery>=0.4.7",
        "demos",
        "Queries the ESA Gaia archive only when the Gaia Milky Way catalog is "
        "built locally; a cached catalog skips it entirely.",
    ),
    "cellxgene_census": DependencySpec("cellxgene-census>=1.0.0", "demos"),
    "gdown": DependencySpec(
        "gdown",
        "",  # intentionally not in any extra — Google-Drive fetches only
        "Not part of any Luxar extra: it is only needed for the Google-Drive "
        "download path.",
    ),
    "h5py": DependencySpec("h5py>=3.0.0", "demos"),
    "kaggle": DependencySpec(
        "kaggle>=2.2,<3",
        "",  # intentionally not in any extra — authenticated competition fetches only
        "Not part of any Luxar extra: only the Biohub cell-tracking demo needs "
        "it, to reach an authenticated Kaggle COMPETITION endpoint (the arXiv "
        "Kaggle demo reads a public dataset URL and needs nothing). >=2.2 is "
        "where `~/.kaggle/access_token` / $KAGGLE_API_TOKEN are honoured, which "
        "is the only token form Kaggle issues now.",
    ),
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
    "metpy": DependencySpec(
        "metpy>=1.6.3,<2.0",
        "demos",
        "Pure-Python NEXRAD Level II decoder (metpy.io.Level2File). Needed only "
        "to re-decode and re-fit the radar volumes; the precomputed Git LFS "
        "bundle skips it entirely. The >=1.6.3 floor is load-bearing: metpy "
        "declares only numpy>=1.20.0, and 1.6.3 is the first release with "
        "NumPy 2.0 support, so below it the resolver pairs metpy with Luxar's "
        "numpy>=2.0 core pin and it breaks at runtime.",
    ),
    "moderngl": DependencySpec(
        "moderngl>=5.8",
        "demos",
        "GPU offscreen renderer for the esm3_protein_stories PDB turntables; the "
        "demo builds without turntables when it is absent.",
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
        "pyarrow>=13.0.0",
        "demos",
        "Version 13 added the `zero_copy_only` keyword to "
        "`ChunkedArray.to_numpy`, which the biodiversity and protein-universe "
        "demos use while loading their point tables.",
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
    """One row of :func:`survey`: a spec, whether it imports, and whether the
    installed version meets the spec's bound."""

    module: str
    spec: DependencySpec
    installed: bool
    #: Importable AND the installed version satisfies ``spec``'s bound.
    #: ``False`` for a missing module OR an out-of-date one (the CLI renders the
    #: latter as OUTDATED). Best-effort: ``True`` whenever the version cannot be
    #: decided — see :func:`_version_satisfied`.
    satisfied: bool


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


#: Modules the substitutive-LOD write path imports at module load — the
#: coarsening kernels in ``luxar.gsplats.lod.substitutive`` need ``torch``, and
#: importing that package pulls in its additive sibling, which does
#: ``from scipy import sparse``. Neither is a core dependency.
SUBSTITUTIVE_LOD_MODULES = ("torch", "scipy")


def substitutive_lod_or_flat(spec: Any, *, geometry: str = "Points") -> Any:
    """Return ``spec`` when substitutive LOD can be built here, else ``None``.

    Demos cache their expensive artifacts, so a warm cache is supposed to build
    a scene on any machine. But requesting ``substitutive_lod=`` imports
    :mod:`luxar.gsplats.lod` (see :data:`SUBSTITUTIVE_LOD_MODULES`), so a
    cache-complete machine without torch/scipy died with a ``ModuleNotFoundError``
    mid-build instead. Route every demo's ``substitutive_lod=`` argument through
    here: with both modules present the spec passes through unchanged; without
    them the caller writes a flat leaf, which is fully viewable — it just loses
    the coarse levels that replace it when zoomed out.

    Unlike :func:`require_module` this DEGRADES rather than raising, so it prints
    the one notice explaining what the scene lost and how to get it back.

    Args:
        spec: The ``substitutive_lod`` argument the demo would pass.
        geometry: ``Points`` or ``Lines`` — names the geometry in the notice.

    Returns:
        ``spec`` unchanged, or ``None`` when a required module is missing.
    """
    missing = [m for m in SUBSTITUTIVE_LOD_MODULES if not is_installed(m)]
    if not missing:
        return spec

    from arbol import aprint

    aprint(
        f"⚠️ {' and '.join(missing)} not installed — skipping {geometry} LOD "
        f"coarsening and building flat {geometry} instead. The scene is fully "
        "viewable; run `pip install 'luxar[gsplats]'` (the extra that carries "
        "torch and scipy at their pinned bounds) to rebuild with level-of-detail."
    )
    return None


def _version_satisfied(spec: str) -> bool:
    """Whether the INSTALLED distribution's version meets ``spec``'s bound.

    :func:`is_installed` only proves a module imports; a demo can still fail if
    the installed version sits below the pinned floor. Concretely, the ``demos``
    extra floors ``scipy>=1.15`` for ``demo_quantum_orbitals`` (it needs
    ``scipy.special.sph_harm_y``, added in 1.15) while the ``gsplats`` extra only
    floors it at ``1.9`` — on an env that satisfies just the gsplats floor the
    demo dies with an ``AttributeError`` the point-of-use gate cannot intercept.
    Checking the version here surfaces that as an OUTDATED row instead.

    Best-effort by design — a survey must never crash or invent an OUTDATED row
    it cannot substantiate — so this returns ``True`` (benefit of the doubt)
    whenever the version cannot be decided: ``packaging`` is not importable, the
    spec carries no version bound, or the distribution exposes no metadata.
    """
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as installed_version

        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.version import Version
    except ImportError:
        return True
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return True
    if not req.specifier:
        return True
    try:
        current = installed_version(req.name)
        if current is None:
            # METADATA with no `Version:` field — nothing to compare.
            return True
        # Parse explicitly: a bad version makes some `packaging` versions raise
        # (InvalidVersion) and others silently return False from `contains`.
        # prereleases=True: a legitimately installed pre-release still counts.
        return req.specifier.contains(Version(current), prereleases=True)
    except (PackageNotFoundError, ValueError, OSError):
        # Can't judge the installed version — not present as a distribution, or
        # absent/non-UTF-8/non-PEP440 `Version:` metadata (InvalidVersion and
        # UnicodeDecodeError are both ValueError). Give the benefit of the doubt
        # rather than crash the report or invent an OUTDATED row.
        return True


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
        module name so the CLI's output order is stable. Each row records both
        whether the module imports and whether its installed version meets the
        spec's bound (see :attr:`DependencyStatus.satisfied`).
    """
    rows = []
    for module, spec in INSTALL_SPECS.items():
        if extra is not None and spec.extra != extra:
            continue
        installed = is_installed(module)
        satisfied = installed and _version_satisfied(spec.spec)
        rows.append(DependencyStatus(module, spec, installed, satisfied))
    return sorted(rows, key=lambda r: r.module.lower())


def extras_for(rows: list[DependencyStatus]) -> list[str]:
    """The distinct Luxar extras that would install ``rows``, sorted.

    Specs outside every extra (``extra == ""``) contribute nothing — they cannot
    be installed via ``luxar[...]`` and must be named individually.
    """
    return sorted({r.spec.extra for r in rows if r.spec.extra})
