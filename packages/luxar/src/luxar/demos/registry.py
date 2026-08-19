"""AST-based demo registry: metadata for all demo_*.py without importing them.

Every demo script declares a ``DEMO_META`` dict literal right after its
module docstring. This registry extracts those literals with ``ast`` —
never by importing the modules, which is deliberately unsupported: many
demos parse ``sys.argv`` or create cache directories at import time, and
several carry heavy top-level imports (pandas, zarr, scipy).

This module is import-light by design so ``luxar demo list`` can render its
table in well under a second: stdlib only, plus ``luxar.core.citation``, which
is itself stdlib-only and shared with the scene writer so a demo and a scene
agree on what a citation is.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from ..core.citation import validate_citation

_DEMOS_DIR = Path(__file__).parent

# Mirror of luxar.utils.demos._DEFAULT_CACHE_ROOT, duplicated so this module
# never imports luxar.utils.demos (which pulls numpy). A unit test pins the
# two constants equal.
DEMO_CACHE_ROOT = Path.home() / ".cache" / "luxar"

# Sub-directories of the cache root that are NOT caches. `luxar demo stop`'s
# pidfile registry lives in `running/` (luxar.utils.demo_runs.DEMO_RUNS_DIR):
# its entries name LIVE demo process groups, so listing it as an unclaimed
# cache would invite `demo cache clear --orphans` to delete the record of what
# is still running. A unit test pins the name against DEMO_RUNS_DIR.
NON_CACHE_DIRS = frozenset({"running"})

# Cache-root directories that hold a hand-placed demo INPUT: bytes the user (or
# a licence) had to put there by hand, with no download path to get them back.
# That is a third classification, alongside a cache (re-downloadable, clearable)
# and NON_CACHE_DIRS live state (not cached bytes at all). Unlike NON_CACHE_DIRS
# these ARE inventoried — the user should see the bytes they are holding — but
# they count as claimed by definition (never reported as an orphan) and `demo
# cache clear` never deletes them: not by key, not under `--all`, not under
# `--orphans`. `milky_way_gaia_3m` is the Gaia DR3 star catalog: CC BY-NC, so it
# is not shipped, and recovering a deleted copy means a ~90-minute ESA TAP
# re-query (see demo_gaia_milky_way_3m.CACHE_FILE; a unit test pins the name).
# Deliberately INDEPENDENT of DEMO_META: a demo declaring the name in its
# `caches` (which is how `luxar demo` reports it as cached) must not thereby
# make it clearable, and no later edit to a demo file can quietly disarm the
# guard.
PROTECTED_INPUT_DIRS = frozenset({"milky_way_gaia_3m"})

# Closed vocabularies for DEMO_META fields. The enforcement test in
# tests/test_demo_meta.py validates every demo file against these.
# category/geometry mirror scripts/gallery/manifest.json's vocabulary
# (cross-validated by test) plus values only non-gallery demos need
# (photogrammetry for the classical-splat interop demos).
CATEGORY_VALUES = frozenset(
    {
        "astronomy",
        "connectome",
        "embeddings",
        "genomics",
        "geoscience",
        "medical",
        "microscopy",
        "networks",
        "photogrammetry",
        "structural",
        "synthetic",
    }
)
GEOMETRY_VALUES = frozenset(
    {"points", "lines", "gsplats", "mesh", "mixed", "points+lines"}
)
COMPUTE_VALUES = frozenset({"light", "medium", "heavy"})
GPU_VALUES = frozenset({"none", "optional", "required"})
LOCAL_DATA_VALUES = frozenset({None, "git-lfs", "kaggle-auth", "manual-file"})

_META_KEYS = {
    "key",
    "title",
    "description",
    "category",
    "geometry",
    "requirements",
    "caches",
    "outputs",
}
# Keys a demo MAY declare. `citation` is optional only while the corpus is being
# populated (see the credit-the-authors issue): once every demo declares one it
# moves into _META_KEYS, and the enforcement test then makes an uncredited demo
# impossible to add. Keep this set small — an optional key is a schema hole.
_OPTIONAL_META_KEYS = {"citation"}
_REQUIREMENT_KEYS = {"download_mb", "compute", "gpu", "local_data"}


class DemoMetaError(ValueError):
    """A demo file has a missing or malformed ``DEMO_META`` block."""


@dataclass(frozen=True)
class DemoInfo:
    """Validated metadata for one demo script."""

    key: str
    index: int  # 1-based position in key-sorted order (stable per checkout)
    module: str  # e.g. "luxar.demos.demo_lorenz"
    path: Path
    title: str
    description: str
    category: str
    geometry: str
    requirements: dict[str, Any]
    caches: tuple[str, ...]
    outputs: tuple[str, ...]
    #: Dataset credit: ``{"short", "doi"?, "license"?, "url"?}``, or ``None`` for a
    #: procedurally generated demo that owes no credit. ``None`` is also what an
    #: as-yet-unpopulated demo reports, so consumers must not treat it as proof
    #: that a demo is synthetic until the key is required.
    citation: Optional[dict[str, Any]] = None

    @property
    def download_mb(self) -> int:
        """Approximate download size in megabytes, from ``requirements``."""
        return int(self.requirements["download_mb"])

    @property
    def compute(self) -> str:
        """Compute-cost tier (``light`` / ``medium`` / ``heavy``)."""
        return str(self.requirements["compute"])

    @property
    def gpu(self) -> str:
        """GPU requirement (``none`` / ``optional`` / ``required``)."""
        return str(self.requirements["gpu"])

    @property
    def local_data(self) -> Optional[str]:
        """Local-data provisioning mode (``git-lfs`` / ``kaggle-auth`` /
        ``manual-file``), or ``None`` when nothing has to be provisioned locally
        — the demo may still download at runtime (see ``download_mb``)."""
        return self.requirements["local_data"]


@dataclass(frozen=True)
class CacheEntry:
    """One directory under the demo cache root."""

    path: Path
    size_bytes: int
    demo_keys: tuple[str, ...]  # claimed by these demos' DEMO_META (empty = none)
    # A hand-placed demo input (:data:`PROTECTED_INPUT_DIRS`): claimed by
    # definition, so an empty ``demo_keys`` here is NOT an orphan, and never
    # deletable by `demo cache clear`.
    protected: bool = False


def validate_meta(meta: Any, path: Path) -> None:
    """Validate a raw ``DEMO_META`` literal; raise :class:`DemoMetaError`.

    Shared between :func:`extract_demo_meta` and the enforcement test so the
    schema has exactly one definition.
    """

    def fail(reason: str) -> None:
        raise DemoMetaError(f"{path.name}: {reason}")

    if not isinstance(meta, dict):
        fail("DEMO_META must be a dict literal")
    present = set(meta.keys())
    missing = _META_KEYS - present
    extra = present - _META_KEYS - _OPTIONAL_META_KEYS
    if missing or extra:
        fail(
            f"DEMO_META keys mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
        )

    key = meta["key"]
    if not isinstance(key, str) or not key:
        fail("key must be a non-empty string")
    # Underscores allowed so keys can equal gallery-manifest ids and dataset
    # stems verbatim (48 of 57 gallery ids use underscores).
    if not all(c.islower() or c.isdigit() or c in "-_" for c in key):
        fail(f"key {key!r} must be a lowercase slug ([a-z0-9_-])")

    for field in ("title", "description"):
        value = meta[field]
        if not isinstance(value, str) or not value.strip():
            fail(f"{field} must be a non-empty string")
        if "\n" in value:
            fail(f"{field} must be a single line")

    if meta["category"] not in CATEGORY_VALUES:
        fail(f"category {meta['category']!r} not in {sorted(CATEGORY_VALUES)}")
    if meta["geometry"] not in GEOMETRY_VALUES:
        fail(f"geometry {meta['geometry']!r} not in {sorted(GEOMETRY_VALUES)}")

    req = meta["requirements"]
    if not isinstance(req, dict) or set(req.keys()) != _REQUIREMENT_KEYS:
        fail(f"requirements must have exactly the keys {sorted(_REQUIREMENT_KEYS)}")
    if not isinstance(req["download_mb"], int) or req["download_mb"] < 0:
        fail("requirements.download_mb must be an int >= 0")
    if req["compute"] not in COMPUTE_VALUES:
        fail(f"requirements.compute {req['compute']!r} not in {sorted(COMPUTE_VALUES)}")
    if req["gpu"] not in GPU_VALUES:
        fail(f"requirements.gpu {req['gpu']!r} not in {sorted(GPU_VALUES)}")
    if req["local_data"] not in LOCAL_DATA_VALUES:
        fail(
            f"requirements.local_data {req['local_data']!r} not in "
            f"{sorted(v for v in LOCAL_DATA_VALUES if v is not None)} or None"
        )

    # `citation` credits whoever produced the underlying dataset. Explicit
    # `None` means "procedurally generated, nothing to credit" — a deliberate
    # statement rather than an omission, which is why None is spelled out rather
    # than allowing the key to be absent once it becomes required. The rules live
    # in luxar.core.citation so a demo and a scene agree on what a citation is.
    if "citation" in meta:
        try:
            validate_citation(meta["citation"])
        except ValueError as exc:
            fail(str(exc))

    for field in ("caches", "outputs"):
        seq = meta[field]
        if not isinstance(seq, list) or not all(
            isinstance(item, str) and item for item in seq
        ):
            fail(f"{field} must be a (possibly empty) list of non-empty strings")
        # Path containment: these names are joined onto the cache/output roots
        # and `demo cache clear` rmtree's the result — an absolute path or a
        # `..`/separator segment would escape the root (pathlib's `/` with an
        # absolute RHS REPLACES the base). Single path-safe segments only.
        for item in seq:
            if not all(c.isalnum() or c in "._-" for c in item) or item.startswith("."):
                fail(
                    f"{field} entry {item!r} must be a single path-safe segment "
                    "([A-Za-z0-9._-], not dot-leading — no separators, no '..')"
                )


def extract_demo_meta(path: Path) -> dict[str, Any]:
    """Extract and validate the ``DEMO_META`` literal from a demo file.

    Pure static analysis: the module is parsed, never imported.

    Raises:
        DemoMetaError: when the block is missing, not a literal, or invalid.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        raise DemoMetaError(f"{path.name}: cannot parse: {e}") from e

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEMO_META" for t in node.targets
        ):
            try:
                meta = ast.literal_eval(node.value)
            except ValueError as e:
                raise DemoMetaError(
                    f"{path.name}: DEMO_META must be a pure literal "
                    f"(no names, calls, or f-strings): {e}"
                ) from e
            validate_meta(meta, path)
            return meta

    raise DemoMetaError(f"{path.name}: no top-level DEMO_META assignment found")


def _iter_demo_paths() -> Iterator[Path]:
    """Yield the ``demo_*.py`` script paths in sorted (stable) order."""
    yield from sorted(_DEMOS_DIR.glob("demo_*.py"))


# (path, mtime_ns) -> DemoInfo-without-index; index depends on the full set,
# so it is assigned per iter_demos() call after sorting.
_meta_cache: dict[tuple[Path, int], dict[str, Any]] = {}


def iter_demos(*, refresh: bool = False) -> list[DemoInfo]:
    """All demos, key-sorted, with stable 1-based indices.

    Metadata is memoized per (path, mtime), so repeat calls are effectively
    free; pass ``refresh=True`` to drop the memo first.

    Raises:
        DemoMetaError: if any demo file has missing/invalid DEMO_META (a
            broken demo should be loud, not silently absent from the table).
    """
    if refresh:
        _meta_cache.clear()

    metas: list[tuple[Path, dict[str, Any]]] = []
    for path in _iter_demo_paths():
        cache_key = (path, path.stat().st_mtime_ns)
        meta = _meta_cache.get(cache_key)
        if meta is None:
            meta = extract_demo_meta(path)
            _meta_cache[cache_key] = meta
        metas.append((path, meta))

    keys = [meta["key"] for _, meta in metas]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise DemoMetaError(f"duplicate demo keys: {sorted(duplicates)}")

    metas.sort(key=lambda item: item[1]["key"])
    return [
        DemoInfo(
            key=meta["key"],
            index=i + 1,
            module=f"luxar.demos.{path.stem}",
            path=path,
            title=meta["title"],
            description=meta["description"],
            category=meta["category"],
            geometry=meta["geometry"],
            requirements=dict(meta["requirements"]),
            caches=tuple(meta["caches"]),
            outputs=tuple(meta["outputs"]),
            citation=validate_citation(meta.get("citation")),
        )
        for i, (path, meta) in enumerate(metas)
    ]


def get_demo(key_or_index: str) -> DemoInfo:
    """Resolve a demo by key or 1-based index (as shown by ``luxar demo``).

    Raises:
        KeyError: with close-match suggestions when nothing resolves.
    """
    demos = iter_demos()

    token = key_or_index.strip()
    # int() rather than isdigit-gating alone: a token like "--5" passes
    # lstrip("-").isdigit() but int() rejects it — treat that as a key
    # lookup (→ the KeyError path below), not an uncaught ValueError.
    try:
        index = int(token)
    except ValueError:
        index = None
    if index is not None:
        if 1 <= index <= len(demos):
            return demos[index - 1]
        raise KeyError(
            f"demo index {index} out of range (1..{len(demos)}); "
            "run `luxar demo list` to see indices"
        )

    by_key = {demo.key: demo for demo in demos}
    if token in by_key:
        return by_key[token]

    suggestions = difflib.get_close_matches(token, by_key.keys(), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise KeyError(f"unknown demo {token!r}.{hint} Run `luxar demo list`.")


def demo_output_paths(info: DemoInfo, demos_dir: Optional[Path] = None) -> list[Path]:
    """Resolve a demo's declared output stems to scene paths.

    Stems normally resolve to ``<demos_dir>/<stem>.luxar.zarr``; a stem that
    already carries a ``.zarr`` suffix is used verbatim.
    """
    if demos_dir is None:
        # Deferred: get_demos_output_dir needs the project root, which `luxar
        # demo list` should not require. create=False so merely resolving output
        # paths (for status/inventory) never creates datasets/ dirs.
        from luxar.utils.paths import get_demos_output_dir

        demos_dir = get_demos_output_dir(create=False)
    return [
        demos_dir / (stem if stem.endswith(".zarr") else f"{stem}.luxar.zarr")
        for stem in info.outputs
    ]


def demo_cache_dirs(info: DemoInfo, cache_root: Optional[Path] = None) -> list[Path]:
    """The ``~/.cache/luxar/<name>/`` directories a demo reads/writes."""
    root = cache_root if cache_root is not None else DEMO_CACHE_ROOT
    return [root / name for name in info.caches]


def dir_size_bytes(path: Path) -> int:
    """Total size of all files under ``path`` (0 if it doesn't exist)."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def inventory_caches(cache_root: Optional[Path] = None) -> list[CacheEntry]:
    """Every cache directory under the cache root, mapped to the demos claiming it.

    Directories claimed by no demo's ``caches`` list are reported with an
    empty ``demo_keys`` tuple (orphans — e.g. leftovers from renamed demos),
    except a :data:`PROTECTED_INPUT_DIRS` directory, which is flagged
    ``protected`` and is claimed by definition however empty its ``demo_keys``.
    :data:`NON_CACHE_DIRS` is skipped entirely: those hold live state, not
    cached bytes.
    """
    root = cache_root if cache_root is not None else DEMO_CACHE_ROOT
    if not root.exists():
        return []

    claims: dict[str, list[str]] = {}
    for demo in iter_demos():
        for name in demo.caches:
            claims.setdefault(name, []).append(demo.key)

    entries = [
        CacheEntry(
            path=subdir,
            size_bytes=dir_size_bytes(subdir),
            demo_keys=tuple(sorted(claims.get(subdir.name, ()))),
            protected=subdir.name in PROTECTED_INPUT_DIRS,
        )
        for subdir in sorted(root.iterdir())
        if subdir.is_dir() and subdir.name not in NON_CACHE_DIRS
    ]
    return entries
