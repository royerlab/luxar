"""`gsplat import` / `gsplat export` — classical (photogrammetric) splat interchange.

``import`` reads the common classical Gaussian-splat dialects (INRIA
``point_cloud.ply``, antimatter15 ``.splat``, Niantic/Scaniverse ``.spz``,
SuperSplat compressed ``.ply``, and PlayCanvas ``SOG`` bundles) into a
:class:`~luxar.gsplats.gsplat_data.GSplatData` and writes a current-format
``.gsplats.zarr``. ``export`` is the inverse: a ``.gsplats.zarr`` becomes an
INRIA PLY that classical viewers load directly. The heavy lifting lives in
:mod:`luxar.gsplats.interop.classical_splats` /
:mod:`luxar.gsplats.interop.inria_export`; this module is the thin Typer
surface. (Distinct from top-level ``luxar export``, which packages a whole
scene + viewer into an offline folder.)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection


def import_command(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Classical splat file: INRIA point_cloud.ply, antimatter15 .splat, "
        "Niantic .spz, SuperSplat compressed .ply, or a PlayCanvas SOG bundle "
        "(directory with meta.json, that meta.json, or a .sog ZIP).",
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr (current node-tree format)."
    ),
    format: Literal["auto", "inria", "splat", "spz", "supersplat", "sog"] = (
        typer.Option(
            "auto",
            "--format",
            "-f",
            help="Source dialect; 'auto' sniffs the extension/PLY header, or a "
            "directory/meta.json/.sog as SOG.",
        )
    ),
    reorient: Optional[bool] = typer.Option(
        None,
        "--reorient/--no-reorient",
        help="Apply the canonical COLMAP → Y-up orientation fix (180° rotation "
        "about X). Default: per-dialect — on for the Y-down dialects "
        "(INRIA PLY / .splat / SuperSplat / SOG), off for Y-up SPZ.",
    ),
    flip: str = typer.Option(
        "",
        "--flip",
        help="Additional axes to mirror after reorientation, e.g. 'x' or 'xz'.",
    ),
    encoding: Literal["auto", "precision", "memory"] = typer.Option(
        "auto",
        "--encoding",
        "-e",
        help="Output encoding: auto=quantized with covariance certificate, "
        "precision=float32 (archival), memory=uint8 (smallest).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
) -> None:
    """Convert a classical Gaussian-splat file to Luxar .gsplats.zarr.

    Spherical harmonics are reduced to the DC band (baked to per-splat RGB);
    opacity maps to amplitudes. The result is a standard single-leaf
    .gsplats.zarr — pipe it through `gsplat lod` for streaming ladders/tiles,
    `gsplat convert` for a viewer scene, or graft it into a scene from Python.

    \b
    Examples:
      luxar gsplat import garden.splat garden.gsplats.zarr
      luxar gsplat import point_cloud.ply scene.gsplats.zarr --no-reorient
      luxar gsplat import capture.spz capture.gsplats.zarr -e precision
      luxar gsplat import sog_bundle/ city.gsplats.zarr   # PlayCanvas SOG dir
    """
    try:
        run_import(
            input_path=input_path,
            output_path=output_path,
            format=format,
            reorient=reorient,
            flip=flip,
            encoding=encoding,
            overwrite=overwrite,
        )
    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except FileNotFoundError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except (ValueError, NotImplementedError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        aprint(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def run_import(
    *,
    input_path: Path,
    output_path: Path,
    format: str,
    reorient: Optional[bool],
    flip: str,
    encoding: str,
    overwrite: bool,
) -> None:
    """Implementation of `gsplat import` (kept separate from the Typer surface)."""
    from luxar.cli.gsplat_ops.encoding import _resolve_encoding_mode
    from luxar.gsplats.interop.classical_splats import (
        detect_classical_format,
        import_gsplats,
    )
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.io.save_gsplats import FORMAT_VERSION
    from luxar.gsplats.tree import total_splats

    if output_path.exists():
        if not overwrite:
            raise typer.BadParameter(
                f"Output already exists: {output_path} (pass --overwrite)"
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    fmt = detect_classical_format(input_path) if format == "auto" else format

    with asection(f"Importing {input_path.name} ({fmt}) → v{FORMAT_VERSION}"):
        data = import_gsplats(input_path, format=fmt, rotate_x180=reorient, flip=flip)
        interop = data.stats.get("interop", {})
        aprint(
            f"Decoded {data.n_splats:,} splats "
            f"(source SH degree {interop.get('source_sh_degree', 0)}, "
            f"DC color baked per splat)"
        )

        data.save(output_path, encoding_mode=_resolve_encoding_mode(encoding))

        # Post-write read-back: confirm the output is a loadable current-format
        # file rather than reporting success blind (same contract as
        # `gsplat migrate-format`).
        import zarr

        verify_node, _ = load_gsplat_node(output_path, include_stats=False)
        out_attrs = dict(zarr.open_group(str(output_path), mode="r").attrs)
        written_version = out_attrs.get("format_version")
        if written_version != FORMAT_VERSION:
            aprint(
                f"❌ Import produced format_version={written_version!r}, "
                f"expected {FORMAT_VERSION!r}"
            )
            raise typer.Exit(1)
        aprint(
            f"✓ Verified v{FORMAT_VERSION} output: "
            f"{total_splats(verify_node):,} splats → {output_path}"
        )


def export_command(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Input .gsplats.zarr (flat / additive ladder / lod-matrix shapes; "
        "flatten partitions first with `gsplat flatten`).",
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .ply (INRIA 3DGS point_cloud format)."
    ),
    format: Literal["inria"] = typer.Option(
        "inria",
        "--format",
        "-f",
        help="Output dialect (only 'inria' in v1; more may follow).",
    ),
    opacity: Literal["normalized", "amplitude", "constant"] = typer.Option(
        "normalized",
        "--opacity",
        help="Amplitude → opacity mapping: normalized=robust rescale into "
        "(0,1) (default; honest for unbounded emission weights), "
        "amplitude=clip raw values (lossless for imported data), "
        "constant=fixed value.",
    ),
    constant_opacity: float = typer.Option(
        1.0, "--constant-opacity", help="Opacity used with --opacity constant."
    ),
    color: Literal["auto", "colors", "colormap", "white"] = typer.Option(
        "auto",
        "--color",
        help="Color source: auto=per-splat colors if present, else --colormap "
        "if given, else white.",
    ),
    colormap: Optional[str] = typer.Option(
        None,
        "--colormap",
        help="Colormap name for baking scalar amplitudes to RGB "
        "(builtin/matplotlib/colorcet).",
    ),
    sh_degree: int = typer.Option(
        0,
        "--sh-degree",
        min=0,
        max=3,
        help="0 (default) writes only the DC band; higher degrees emit "
        "zero-filled f_rest bands for viewers that insist on them.",
    ),
    timepoint: Optional[int] = typer.Option(
        None,
        "--timepoint",
        help="For >3D data: slice the LAST dimension (where `gsplat merge "
        "--as-dimension` stacks time) at this index.",
    ),
    slice_dim: Optional[int] = typer.Option(
        None, "--slice-dim", help="For >3D data: dimension to slice away."
    ),
    slice_index: Optional[int] = typer.Option(
        None, "--slice-index", help="For >3D data: index along --slice-dim."
    ),
    keep_orientation: bool = typer.Option(
        False,
        "--keep-orientation",
        help="Do NOT invert the orientation recorded at import time "
        "(by default import → export round-trips in the source frame).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
) -> None:
    """Export a .gsplats.zarr to a classical INRIA 3DGS PLY file.

    Cholesky factors are eigendecomposed back to log-scales + rotation
    quaternions; amplitudes map to opacity logits; per-splat colors (or a
    baked colormap) become the SH DC band. The result opens in SuperSplat,
    PlayCanvas, gsplat.js and other classical viewers.

    \b
    Examples:
      luxar gsplat export fit.gsplats.zarr fit.ply --colormap viridis
      luxar gsplat export imported.gsplats.zarr back.ply --opacity amplitude
      luxar gsplat export timelapse.gsplats.zarr t42.ply --timepoint 42
    """
    try:
        run_export(
            input_path=input_path,
            output_path=output_path,
            format=format,
            opacity=opacity,
            constant_opacity=constant_opacity,
            color=color,
            colormap=colormap,
            sh_degree=sh_degree,
            timepoint=timepoint,
            slice_dim=slice_dim,
            slice_index=slice_index,
            keep_orientation=keep_orientation,
            overwrite=overwrite,
        )
    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except FileNotFoundError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except ValueError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        aprint(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def run_export(
    *,
    input_path: Path,
    output_path: Path,
    format: str,
    opacity: str,
    constant_opacity: float,
    color: str,
    colormap: Optional[str],
    sh_degree: int,
    timepoint: Optional[int],
    slice_dim: Optional[int],
    slice_index: Optional[int],
    keep_orientation: bool,
    overwrite: bool,
) -> None:
    """Implementation of `gsplat export` (kept separate from the Typer surface)."""
    from luxar.gsplats.interop.classical_splats import read_inria_ply
    from luxar.gsplats.interop.inria_export import export_inria_ply

    if format != "inria":
        raise typer.BadParameter(f"Unknown export format {format!r} (v1: inria)")
    if output_path.exists() and not overwrite:
        raise typer.BadParameter(
            f"Output already exists: {output_path} (pass --overwrite)"
        )
    if colormap is not None:
        from luxar.colormaps import resolve_colormap

        try:
            resolve_colormap(colormap)
        except Exception as exc:
            raise typer.BadParameter(f"Unknown colormap {colormap!r}: {exc}")

    with asection(f"Exporting {input_path.name} → INRIA PLY"):
        n = export_inria_ply(
            input_path,
            output_path,
            opacity_policy=opacity,
            constant_opacity=constant_opacity,
            color_source=color,
            colormap=colormap,
            sh_degree=sh_degree,
            undo_orientation=not keep_orientation,
            timepoint=timepoint,
            slice_dim=slice_dim,
            slice_index=slice_index,
        )
        # Post-write read-back: the exported PLY must parse with our own
        # INRIA reader (same contract as `gsplat import`'s verification).
        verified = read_inria_ply(output_path)
        if verified.n_splats != n:
            aprint(
                f"❌ Export wrote {n:,} splats but read-back found "
                f"{verified.n_splats:,}"
            )
            raise typer.Exit(1)
        size_mb = output_path.stat().st_size / 1e6
        aprint(f"✓ Verified INRIA PLY: {n:,} splats, {size_mb:.1f} MB → {output_path}")


def register_interchange_commands(app: typer.Typer) -> None:
    """Attach the classical-format interchange commands to the gsplat CLI."""
    app.command("import")(import_command)
    app.command("export")(export_command)
