"""Migrate legacy .gsplats.zarr layouts to format v2.0.

Three input shapes are auto-detected:

* **v1.0** ``.gsplats.zarr`` — single flat splat set under ``/splats``.
  Migrates to v2.0 with shape ``[1, 1]``.
* **v1.1** ``.gsplats.zarr`` — multi-LOD additive with
  ``/splats/lod_<i>/`` subgroups. Migrates to v2.0 with shape ``[1, M]``.
* **Substitutive directory** — a directory of
  ``level_<i>.gsplats.zarr`` files + ``manifest.json``, as produced by
  the pre-v2.0 ``luxar gsplat lod substitutive`` command. Migrates to
  v2.0 with shape ``[N, 1]``.

All variants are read with the v1.x decoder logic preserved here (it
was removed from the live ``load_gsplats`` path) and re-written via
the unified v2.0 writer.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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
) -> str:
    """Convert a legacy .gsplats.zarr or substitutive-directory layout to v2.0.

    Returns the detected legacy format identifier (``"v1.0"``,
    ``"v1.1"``, or ``"substitutive_dir"``).

    Raises:
        ValueError: If ``output_path`` exists and ``overwrite`` is False,
            or the input is already v2.0, or the layout is unrecognised.
        FileNotFoundError: If ``input_path`` doesn't exist.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"Output {output_path} exists; pass overwrite=True to replace it."
        )

    detected = detect_legacy_format(input_path)
    if detected == "v2.0":
        raise ValueError(
            f"Input {input_path} is already format v2.0; no migration needed."
        )

    if detected == "substitutive_dir":
        data = _read_substitutive_directory(input_path)
        fitting_info: Dict[str, Any] = {}
        fitting_config: Dict[str, Any] = {}
        provenance_info: Dict[str, Any] = {}
    else:
        # v1.0 or v1.1 .gsplats.zarr (or compressed)
        zarr_path = input_path
        cleanup_temp = None
        try:
            if input_path.is_file():
                zarr_path = _extract_compressed_zarr(input_path)
                cleanup_temp = zarr_path.parent
            root = zarr.open_group(str(zarr_path), mode="r")
            (
                data,
                fitting_info,
                fitting_config,
                provenance_info,
            ) = _read_v1_x_root(root, include_stats=True)
        finally:
            if cleanup_temp is not None and cleanup_temp.exists():
                shutil.rmtree(cleanup_temp, ignore_errors=True)

    # Write via the v2.0 writer
    if output_path.exists() and overwrite:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    # Use GSplatData.save which already routes to v2.0 layout. We pass
    # ordering="none" so the migrated arrays are byte-equivalent to the
    # source — the source already had its own ordering (or lack thereof)
    # and the migration is a pure rewrap.
    data.save(
        output_path,
        ordering="none",
        description=f"Migrated from legacy format {detected}",
    )

    # Re-attach the v1.x fitting / provenance groups via direct zarr writes
    # (GSplatData.save's plumbing only writes them when stats hold them in
    # a specific shape; the cleanest approach is to splice the raw attrs back)
    if fitting_info or provenance_info:
        root = zarr.open_group(str(output_path), mode="a")
        if fitting_info:
            fitting_group = (
                root["fitting"] if "fitting" in root else root.create_group("fitting")
            )
            fitting_group.attrs.update(fitting_info)
            if fitting_config:
                config_group = (
                    fitting_group["config"]
                    if "config" in fitting_group
                    else fitting_group.create_group("config")
                )
                config_group.attrs.update(fitting_config)
        if provenance_info:
            prov_group = (
                root["provenance"]
                if "provenance" in root
                else root.create_group("provenance")
            )
            prov_group.attrs.update(provenance_info)
        zarr.consolidate_metadata(root.store)

    return detected
