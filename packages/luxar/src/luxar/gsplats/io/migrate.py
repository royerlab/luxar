"""Migrate legacy .gsplats.zarr layouts to the current node-tree format.

Five input shapes are auto-detected:

* **v1.0** ``.gsplats.zarr`` — single flat splat set under ``/splats``.
* **v1.1** ``.gsplats.zarr`` — multi-LOD additive with ``/splats/lod_<i>/``.
* **v2.0** ``.gsplats.zarr`` — the 2-D ``substitutive_<s>/additive_<a>`` matrix.
* **Substitutive directory** — a directory of ``level_<i>.gsplats.zarr`` files
  + ``manifest.json`` (the pre-v2.0 ``lod substitutive`` output).
* **v3.0 / v3.1 with legacy lod selector attrs** — a node-tree store whose
  ``kind=lod`` groups still carry the pre-v3.2 ``selector='pixel_size'`` /
  per-child ``min_pixel_size`` attrs (renamed in v3.2 to ``selector='coverage'``
  / ``coverage_fraction``). Re-written through the current node-tree
  reader/writer, which derives fresh ``coverage_fraction`` thresholds from the
  per-level splat counts and stamps the current format (v3.3).

Each legacy decoder is *frozen* here (the v1.x and v2.0 decode loops were removed
from the live ``load_gsplats`` path at the v3.0 cutover) and the result is
re-written via the unified node-tree writer (``write_gsplats_tree``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import zarr

from luxar.encoding import ArrayDecoder, EncodingMode
from luxar.gsplats.gsplat_data import (
    AdditiveSubLOD,
    GSplatData,
    SubstitutiveLevel,
)
from luxar.gsplats.io._archive import extract_compressed_zarr

__all__ = ["migrate_format", "detect_legacy_format"]


def _zarr_tree_has_legacy_lod_attrs(group: zarr.Group) -> bool:
    """True when any ``kind=lod`` group in ``group``'s subtree still carries the
    pre-v3.2 selector attrs (``selector='pixel_size'`` on the group, or a
    ``child_<i>`` with ``min_pixel_size`` but no ``coverage_fraction``)."""
    if group.attrs.get("kind") == "lod":
        if group.attrs.get("selector") == "pixel_size":
            return True
        for name, child in group.groups():
            if (
                str(name).startswith("child_")
                and "min_pixel_size" in child.attrs
                and "coverage_fraction" not in child.attrs
            ):
                return True
    for _, child in group.groups():
        if _zarr_tree_has_legacy_lod_attrs(child):
            return True
    return False


def detect_legacy_format(input_path: Path) -> str:
    """Return one of ``"v1.0"``, ``"v1.1"``, ``"substitutive_dir"``, ``"v2.0"``,
    ``"v3.0-lod-pixel-size"``, or ``"v3.1-lod-pixel-size"``.

    The two ``v3.x-lod-pixel-size`` results identify current node-tree stores
    whose ``kind=lod`` groups still carry the pre-v3.2 ``pixel_size`` selector
    attrs; a v3.x store without them is *already current* and raises.

    Raises:
        ValueError: If the input is unrecognised, or already current.
    """
    input_path = Path(input_path)
    # Substitutive directory: has manifest.json + level_<i>.gsplats.zarr files
    if input_path.is_dir() and (input_path / "manifest.json").exists():
        manifest = json.loads((input_path / "manifest.json").read_text())
        if manifest.get("lod_kind") == "substitutive":
            return "substitutive_dir"
    # Otherwise treat as a .gsplats.zarr or compressed archive; sniff the
    # root .zattrs to read format_version
    if input_path.is_dir() or str(input_path).endswith((".zip", ".tar.gz")):
        zarr_path = input_path
        cleanup_temp = None
        try:
            if input_path.is_file():
                zarr_path = extract_compressed_zarr(input_path)
                cleanup_temp = zarr_path.parent
            try:
                root = zarr.open_group(str(zarr_path), mode="r")
            except Exception:
                # Not a zarr group at all — fall through to the catch-all raise
                root = None
            if root is not None:
                fv = root.attrs.get("format_version")
                if fv in ("1.0", "1.1", "2.0"):
                    return f"v{fv}"
                if fv in ("3.0", "3.1"):
                    # A v3.0/v3.1 node tree is current UNLESS its kind=lod
                    # groups still carry the pre-v3.2 'pixel_size' selector
                    # attrs (renamed to 'coverage' / coverage_fraction in
                    # v3.2) — those need a rewrite for correct viewer LOD.
                    if _zarr_tree_has_legacy_lod_attrs(root):
                        return f"v{fv}-lod-pixel-size"
                    raise ValueError(
                        f"Input {input_path} is already format v{fv} "
                        f"(a current node-tree format); no migration needed."
                    )
                if fv in ("3.2", "3.3"):
                    # v3.3 only adds the optional luxar_delta_v1 filter on
                    # quantized arrays — nothing to migrate in either version.
                    raise ValueError(
                        f"Input {input_path} is already format v{fv} "
                        f"(a current node-tree format); no migration needed."
                    )
        finally:
            if cleanup_temp is not None and cleanup_temp.exists():
                shutil.rmtree(cleanup_temp, ignore_errors=True)
    raise ValueError(
        f"Unrecognised input layout: {input_path}. Expected a "
        f".gsplats.zarr directory/archive or a substitutive directory "
        f"with manifest.json."
    )


def _read_v1_x_root(
    root: zarr.Group, include_stats: bool = True
) -> tuple[GSplatData, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Read a v1.0 or v1.1 zarr root and return (data, fitting_info,
    fitting_config, provenance_info).

    The returned :class:`GSplatData` always has shape ``[1, M]`` where
    ``M`` is the additive sub-LOD count (1 for v1.0, ``n_lods`` for v1.1).
    """
    format_version = root.attrs.get("format_version")
    if format_version not in ("1.0", "1.1"):
        raise ValueError(
            f"_read_v1_x_root expected format_version 1.0 or 1.1, got {format_version!r}"
        )

    decoder = ArrayDecoder()
    splats_group = root["splats"]
    truncation_radius = float(splats_group.attrs.get("truncation_radius", 3.0))

    # Build per-AdditiveSubLOD list
    sublods: List[AdditiveSubLOD] = []
    if format_version == "1.1":
        n_lods = int(splats_group.attrs.get("n_lods", 1))
        for i in range(n_lods):
            lod_group = splats_group[f"lod_{i}"]
            lod_stats: Dict[str, Any] = {}
            ls_raw = lod_group.attrs.get("lod_stats", {})
            if isinstance(ls_raw, dict):
                lod_stats = dict(ls_raw)
            lod_stats["n_splats"] = lod_group.attrs.get("n_splats")
            lod_stats["ndim"] = lod_group.attrs.get("ndim")
            lod_stats["ordering"] = lod_group.attrs.get("ordering", "none")
            sublods.append(
                AdditiveSubLOD(
                    centers=decoder.decode(lod_group["centers"], root),
                    amplitudes=decoder.decode(lod_group["amplitudes"], root),
                    cholesky_factors=decoder.decode(
                        lod_group["cholesky_factors"], root
                    ),
                    colors=decoder.decode(lod_group["colors"], root)
                    if "colors" in lod_group
                    else None,
                    stats=lod_stats,
                    truncation_radius=truncation_radius,
                )
            )
    else:
        # v1.0: single flat AdditiveSubLOD at /splats
        lod_stats = {
            "n_splats": splats_group.attrs.get("n_splats"),
            "ndim": splats_group.attrs.get("ndim"),
            "ordering": splats_group.attrs.get("ordering", "none"),
        }
        sublods.append(
            AdditiveSubLOD(
                centers=decoder.decode(splats_group["centers"], root),
                amplitudes=decoder.decode(splats_group["amplitudes"], root),
                cholesky_factors=decoder.decode(splats_group["cholesky_factors"], root),
                colors=decoder.decode(splats_group["colors"], root)
                if "colors" in splats_group
                else None,
                stats=lod_stats,
                truncation_radius=truncation_radius,
            )
        )

    # Build top-level stats + carry-along groups
    stats: Dict[str, Any] = {}
    fitting_info: Dict[str, Any] = {}
    fitting_config: Dict[str, Any] = {}
    provenance_info: Dict[str, Any] = {}
    if include_stats:
        if "fitting" in root:
            fitting_info = dict(root["fitting"].attrs)
            if "config" in root["fitting"]:
                fitting_config = dict(root["fitting"]["config"].attrs)
        if "provenance" in root:
            provenance_info = dict(root["provenance"].attrs)
        stats["format_version_legacy"] = format_version
        stats["timestamp_legacy"] = root.attrs.get("timestamp")

    data = GSplatData(additive_sublods=sublods, stats=stats)
    return data, fitting_info, fitting_config, provenance_info


def _read_v2_0_root(
    root: zarr.Group, include_stats: bool = True
) -> tuple[GSplatData, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Read a v2.0 ``substitutive_<s>/additive_<a>`` matrix root.

    Frozen copy of the v2.0 decode loop that lived in ``load_gsplats`` before the
    v3.0 cutover. Returns ``(data, fitting_info, fitting_config, provenance_info)``.
    """
    if root.attrs.get("format_version") != "2.0":
        raise ValueError(
            f"_read_v2_0_root expected format_version 2.0, "
            f"got {root.attrs.get('format_version')!r}"
        )

    decoder = ArrayDecoder()
    splats_group = root["splats"]
    truncation_radius = float(splats_group.attrs.get("truncation_radius", 3.0))
    n_substitutive = int(
        root.attrs.get("n_substitutive", splats_group.attrs.get("n_substitutive", 1))
    )
    # A legacy file's `default_substitutive` is intentionally NOT carried: the
    # v3.0 data model fixes the default at the finest level (index 0), and the
    # on-disk default_level is the viewer's separate coarsest-first render hint
    # (stamped by the serializer). The substitutive order (finest at index 0) is
    # preserved below, which is what actually matters.

    substitutive_levels: List[SubstitutiveLevel] = []
    for s in range(n_substitutive):
        sub_group = splats_group[f"substitutive_{s}"]
        n_additive_sublods = int(sub_group.attrs.get("n_additive_sublods", 1))
        compression_factor = int(sub_group.attrs.get("compression_factor", 1))
        parent_method_raw = sub_group.attrs.get("parent_method", "")
        parent_method = (
            None if parent_method_raw in ("", None) else str(parent_method_raw)
        )
        level_index = int(sub_group.attrs.get("level_index", s))
        ls_raw = sub_group.attrs.get("level_stats", {})
        level_stats: Dict[str, Any] = dict(ls_raw) if isinstance(ls_raw, dict) else {}

        additive_sublods: List[AdditiveSubLOD] = []
        for a in range(n_additive_sublods):
            add_group = sub_group[f"additive_{a}"]
            lod_stats: Dict[str, Any] = {}
            ls2 = add_group.attrs.get("lod_stats", {})
            if isinstance(ls2, dict):
                lod_stats = dict(ls2)
            additive_sublods.append(
                AdditiveSubLOD(
                    centers=decoder.decode(add_group["centers"], root),
                    amplitudes=decoder.decode(add_group["amplitudes"], root),
                    cholesky_factors=decoder.decode(
                        add_group["cholesky_factors"], root
                    ),
                    colors=decoder.decode(add_group["colors"], root)
                    if "colors" in add_group
                    else None,
                    stats=lod_stats,
                    truncation_radius=float(
                        add_group.attrs.get("truncation_radius", truncation_radius)
                    ),
                )
            )
        substitutive_levels.append(
            SubstitutiveLevel(
                additive_sublods=additive_sublods,
                compression_factor=compression_factor,
                parent_method=parent_method,
                level_index=level_index,
                stats=level_stats,
            )
        )

    fitting_info: Dict[str, Any] = {}
    fitting_config: Dict[str, Any] = {}
    provenance_info: Dict[str, Any] = {}
    if include_stats:
        if "fitting" in root:
            fitting_info = dict(root["fitting"].attrs)
            if "config" in root["fitting"]:
                fitting_config = dict(root["fitting"]["config"].attrs)
        if "provenance" in root:
            provenance_info = dict(root["provenance"].attrs)

    data = GSplatData(
        substitutive_levels=substitutive_levels,
    )
    return data, fitting_info, fitting_config, provenance_info


def _read_substitutive_directory(input_path: Path) -> GSplatData:
    """Read a directory of ``level_<i>.gsplats.zarr`` + ``manifest.json``.

    Each per-level file is a v1.0 single-LOD .gsplats.zarr. The combined
    output is shape ``[N, 1]`` — one substitutive level per file, each
    with a single additive sub-LOD.
    """
    manifest = json.loads((input_path / "manifest.json").read_text())
    if manifest.get("lod_kind") != "substitutive":
        raise ValueError(
            f"manifest.json lod_kind = {manifest.get('lod_kind')!r}; expected 'substitutive'"
        )
    compression_factor = int(manifest.get("compression_factor", 4))
    method = manifest.get("method", "kmeans_lloyd")
    # Sort by the declared level index, NOT manifest list order: the v2.0
    # invariant is substitutive_levels[0] == finest (default_substitutive=0).
    # A manifest that lists levels in any other order would otherwise land the
    # coarsest level at index 0, so the default view returns the wrong splats.
    levels_data = sorted(manifest.get("levels_data", []), key=lambda e: int(e["level"]))
    declared_levels = [int(e["level"]) for e in levels_data]
    if declared_levels != list(range(len(levels_data))):
        raise ValueError(
            "substitutive manifest levels must form a contiguous 0..N-1 range "
            f"(finest=0); got {declared_levels}"
        )

    substitutive_levels: List[SubstitutiveLevel] = []
    for level_entry in levels_data:
        level_idx = int(level_entry["level"])
        file_name = level_entry["file"]
        n_splats_meta = level_entry.get("n_splats")
        level_path = input_path / file_name
        if not level_path.exists():
            raise FileNotFoundError(
                f"manifest references {file_name} but it doesn't exist under {input_path}"
            )
        root = zarr.open_group(str(level_path), mode="r")
        data_one, _, _, _ = _read_v1_x_root(root, include_stats=False)
        # data_one always has n_substitutive == 1; take its single additive sub-LOD
        only_sublod = data_one.substitutive_levels[0].additive_sublods[0]
        # Compute compression_factor for this level: K^level_idx
        K_level = compression_factor**level_idx
        substitutive_levels.append(
            SubstitutiveLevel(
                additive_sublods=[only_sublod],
                compression_factor=K_level,
                parent_method=None if level_idx == 0 else method,
                level_index=level_idx,
                stats={"n_splats_total": n_splats_meta} if n_splats_meta else {},
            )
        )

    return GSplatData(substitutive_levels=substitutive_levels)


def _read_v3_root(
    root: zarr.Group,
) -> tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Read a v3.0 / v3.1 node-tree root via the **live** reader.

    Used for the ``v3.x-lod-pixel-size`` migration: the arrays and topology are
    already current, only the ``kind=lod`` selector attrs are stale. The live
    reader ignores the legacy ``min_pixel_size`` / ``selector='pixel_size'``
    attrs, and the live writer re-derives fresh ``coverage_fraction``
    thresholds (``sqrt(N_i/N_finest)``) from the per-level splat counts on the
    subsequent :func:`write_gsplats_tree` — so read→write *is* the migration.

    Returns ``(node, fitting_info, fitting_config, provenance_info,
    pipeline_info)``. Note the whole tree is materialized in memory (matching
    the other migrate readers).
    """
    from luxar.io._compiler.gsplat_tree import read_gsplat_node

    node = read_gsplat_node(root, root)
    fitting_info: Dict[str, Any] = {}
    fitting_config: Dict[str, Any] = {}
    provenance_info: Dict[str, Any] = {}
    pipeline_info: Dict[str, Any] = {}
    if "fitting" in root:
        fitting_info = dict(root["fitting"].attrs)
        if "config" in root["fitting"]:
            fitting_config = dict(root["fitting"]["config"].attrs)
    if "provenance" in root:
        provenance_info = dict(root["provenance"].attrs)
    if "pipeline" in root:
        pipeline_info = dict(root["pipeline"].attrs)
    return node, fitting_info, fitting_config, provenance_info, pipeline_info


def migrate_format(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    zip_deflate: bool = False,
    encoding_mode: EncodingMode = EncodingMode.AUTO,
) -> str:
    """Convert a legacy .gsplats.zarr (v1.0 / v1.1 / v2.0), a substitutive
    directory, or a v3.0/v3.1 store with pre-v3.2 lod selector attrs to the
    current node-tree format.

    The **container format is preserved from the output extension**: an
    ``output_path`` ending in ``.zip`` / ``.tar.gz`` is written as a compressed
    archive (so migrating a compressed legacy file in place stays compressed),
    while a plain path is written as a ``.gsplats.zarr`` directory. ``zip_deflate``
    selects DEFLATE vs the default STORED for ``.zip`` outputs.

    **Encoding is not a pure rewrap.** The output uses the current node-tree
    encoding policy (``encoding_mode``, default :data:`EncodingMode.AUTO`), so legacy
    *float32* Cholesky factors are re-encoded as the split diagonal /
    off-diagonal arrays with per-column quantization — uint8 under AUTO and
    MEMORY (AUTO carries an encode-time covariance certificate and escalates
    to uint16 only when the measured Σ error demands it). Pass
    ``encoding_mode=EncodingMode.PRECISION`` for a lossless float32 migration of
    archival data. Other arrays (centers, etc.) follow the same per-array policy.

    Returns the detected legacy format identifier (``"v1.0"``, ``"v1.1"``,
    ``"v2.0"``, ``"substitutive_dir"``, ``"v3.0-lod-pixel-size"``, or
    ``"v3.1-lod-pixel-size"``).

    Raises:
        ValueError: If ``output_path`` exists and ``overwrite`` is False, the
            input is already in the current node-tree format (a v3.x store with
            no pre-v3.2 lod selector attrs left to upgrade), or the layout is
            unrecognised.
        FileNotFoundError: If ``input_path`` doesn't exist.
    """
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"Output {output_path} exists; pass overwrite=True to replace it."
        )

    detected = detect_legacy_format(input_path)

    fitting_info: Dict[str, Any] = {}
    fitting_config: Dict[str, Any] = {}
    provenance_info: Dict[str, Any] = {}
    pipeline_info: Dict[str, Any] = {}

    if detected == "substitutive_dir":
        node: Any = _read_substitutive_directory(input_path).tree
    else:
        # v1.0 / v1.1 / v2.0 / v3.x-lod-pixel-size .gsplats.zarr (or compressed)
        zarr_path = input_path
        cleanup_temp = None
        try:
            if input_path.is_file():
                zarr_path = extract_compressed_zarr(input_path)
                cleanup_temp = zarr_path.parent
            root = zarr.open_group(str(zarr_path), mode="r")
            if detected.endswith("-lod-pixel-size"):
                # v3.0/v3.1 node tree whose kind=lod groups still carry the
                # pre-v3.2 'pixel_size' selector attrs. The live read→write
                # round-trip IS the migration: the reader ignores the stale
                # attrs and the writer re-derives coverage_fraction /
                # selector='coverage' from the per-level splat counts.
                (
                    node,
                    fitting_info,
                    fitting_config,
                    provenance_info,
                    pipeline_info,
                ) = _read_v3_root(root)
            else:
                reader = _read_v2_0_root if detected == "v2.0" else _read_v1_x_root
                data, fitting_info, fitting_config, provenance_info = reader(
                    root, include_stats=True
                )
                node = data.tree
        finally:
            if cleanup_temp is not None and cleanup_temp.exists():
                shutil.rmtree(cleanup_temp, ignore_errors=True)

    if output_path.exists() and overwrite:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    # Preserve the container format implied by the output extension: a .zip /
    # .tar.gz output is written compressed (so an in-place migration of a
    # compressed legacy file stays compressed), not a bare directory.
    out_name = output_path.name
    compress: Optional[Literal["zip", "tar.gz"]] = None
    if out_name.endswith(".zip"):
        compress = "zip"
    elif out_name.endswith(".tar.gz"):
        compress = "tar.gz"

    # Write via the single current node-tree writer. ordering="none" preserves
    # the source element order (no Morton/Hilbert re-sort); encoding still
    # follows the current policy (encoding_mode) — so float32 Cholesky is
    # re-encoded to the split + per-column quantization unless PRECISION is
    # requested (see the docstring). Fitting / provenance / pipeline flow
    # through as first-class write inputs.
    write_gsplats_tree(
        output_path,
        node,
        ordering="none",
        encoding_mode=encoding_mode,
        fitting_info=fitting_info or None,
        fitting_config=fitting_config or None,
        provenance_info=provenance_info or None,
        pipeline_info=pipeline_info or None,
        description=f"Migrated from legacy format {detected}",
        compress=compress,
        zip_deflate=zip_deflate,
    )

    return detected
