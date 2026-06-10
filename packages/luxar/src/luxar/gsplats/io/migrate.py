"""Migrate legacy .gsplats.zarr layouts to the current v3.0 node-tree format.

Four input shapes are auto-detected:

* **v1.0** ``.gsplats.zarr`` — single flat splat set under ``/splats``.
* **v1.1** ``.gsplats.zarr`` — multi-LOD additive with ``/splats/lod_<i>/``.
* **v2.0** ``.gsplats.zarr`` — the 2-D ``substitutive_<s>/additive_<a>`` matrix.
* **Substitutive directory** — a directory of ``level_<i>.gsplats.zarr`` files
  + ``manifest.json`` (the pre-v2.0 ``lod substitutive`` output).

Each legacy decoder is *frozen* here (the v1.x and v2.0 decode loops were removed
from the live ``load_gsplats`` path at the v3.0 cutover) and the result is
re-written via the unified v3.0 node-tree writer (``write_gsplats_tree``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import zarr

from luxar.encoding import ArrayDecoder
from luxar.gsplats.gsplat_data import (
    AdditiveSubLOD,
    GSplatData,
    SubstitutiveLevel,
)

__all__ = ["migrate_format", "detect_legacy_format"]


def _extract_compressed_zarr(compressed_path: Path) -> Path:
    """Extract compressed zarr archive to temporary directory.

    Mirrors the helper in ``load_gsplats``; duplicated here so this
    module has no inbound dependency on the live loader.
    """
    import tarfile
    import zipfile

    temp_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_migrate_"))
    suffix = compressed_path.suffix
    if suffix == ".zip" or str(compressed_path).endswith(".gsplats.zarr.zip"):
        with zipfile.ZipFile(compressed_path, "r") as zip_ref:
            temp_dir_resolved = Path(temp_dir).resolve()
            for zip_member in zip_ref.namelist():
                if "\\" in zip_member or zip_member.startswith("/"):
                    raise ValueError(
                        f"Zip member '{zip_member}' has unsafe path separator"
                    )
                zip_member_path = (Path(temp_dir) / zip_member).resolve()
                try:
                    zip_member_path.relative_to(temp_dir_resolved)
                except ValueError as exc:
                    raise ValueError(
                        f"Zip member '{zip_member}' would escape extraction directory"
                    ) from exc
            zip_ref.extractall(temp_dir)
    elif suffix == ".gz" or str(compressed_path).endswith(
        (".tar.gz", ".gsplats.zarr.tar.gz")
    ):
        with tarfile.open(compressed_path, "r:gz") as tar_ref:
            for member in tar_ref.getmembers():
                member_path = Path(temp_dir) / member.name
                if not member_path.resolve().is_relative_to(Path(temp_dir).resolve()):
                    raise ValueError(
                        f"Tar member '{member.name}' would escape extraction directory"
                    )
            tar_ref.extractall(temp_dir)
    else:
        raise ValueError(f"Unsupported compression format: {compressed_path}")

    for d in temp_dir.iterdir():
        if d.is_dir() and d.name.endswith(".gsplats.zarr"):
            return d
    children = list(temp_dir.iterdir())
    if children and children[0].is_dir():
        return children[0]
    raise ValueError(f"No .gsplats.zarr directory found in {compressed_path}")


def detect_legacy_format(input_path: Path) -> str:
    """Return one of ``"v1.0"``, ``"v1.1"``, ``"substitutive_dir"``, or ``"v2.0"``.

    Raises:
        ValueError: If the input is unrecognised.
    """
    input_path = Path(input_path)
    # Substitutive directory: has manifest.json + level_<i>.gsplats.zarr files
    if input_path.is_dir() and (input_path / "manifest.json").exists():
        manifest = json.loads((input_path / "manifest.json").read_text())
        if manifest.get("lod_kind") == "substitutive":
            return "substitutive_dir"
    # Otherwise treat as a .gsplats.zarr or compressed archive; sniff the
    # root .zattrs to read format_version
    if (
        input_path.is_dir()
        or input_path.suffix in (".zip", ".gz")
        or str(input_path).endswith((".gsplats.zarr.zip", ".gsplats.zarr.tar.gz"))
    ):
        zarr_path = input_path
        cleanup_temp = None
        try:
            if input_path.is_file():
                zarr_path = _extract_compressed_zarr(input_path)
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
                if fv == "3.0":
                    raise ValueError(
                        f"Input {input_path} is already format v3.0 "
                        f"(the current node-tree format); no migration needed."
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
    default_substitutive = int(
        root.attrs.get(
            "default_substitutive", splats_group.attrs.get("default_substitutive", 0)
        )
    )

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
        default_substitutive=default_substitutive,
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


def migrate_format(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    zip_deflate: bool = False,
) -> str:
    """Convert a legacy .gsplats.zarr (v1.0 / v1.1 / v2.0) or substitutive
    directory to the current v3.0 node-tree format.

    The **container format is preserved from the output extension**: an
    ``output_path`` ending in ``.zip`` / ``.tar.gz`` is written as a compressed
    archive (so migrating a compressed legacy file in place stays compressed),
    while a plain path is written as a ``.gsplats.zarr`` directory. ``zip_deflate``
    selects DEFLATE vs the default STORED for ``.zip`` outputs.

    Returns the detected legacy format identifier (``"v1.0"``, ``"v1.1"``,
    ``"v2.0"``, or ``"substitutive_dir"``).

    Raises:
        ValueError: If ``output_path`` exists and ``overwrite`` is False, the
            input is already v3.0, or the layout is unrecognised.
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

    if detected == "substitutive_dir":
        data = _read_substitutive_directory(input_path)
    else:
        # v1.0 / v1.1 / v2.0 .gsplats.zarr (or compressed)
        zarr_path = input_path
        cleanup_temp = None
        try:
            if input_path.is_file():
                zarr_path = _extract_compressed_zarr(input_path)
                cleanup_temp = zarr_path.parent
            root = zarr.open_group(str(zarr_path), mode="r")
            reader = _read_v2_0_root if detected == "v2.0" else _read_v1_x_root
            data, fitting_info, fitting_config, provenance_info = reader(
                root, include_stats=True
            )
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

    # Write via the single v3.0 node-tree writer. ordering="none" keeps the
    # migrated arrays byte-equivalent to the source (a pure rewrap); fitting /
    # provenance flow through as first-class write inputs (no post-hoc splice).
    write_gsplats_tree(
        output_path,
        data.tree,
        ordering="none",
        fitting_info=fitting_info or None,
        fitting_config=fitting_config or None,
        provenance_info=provenance_info or None,
        description=f"Migrated from legacy format {detected}",
        compress=compress,
        zip_deflate=zip_deflate,
    )

    return detected
