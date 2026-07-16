"""`gsplat import` — bring classical (photogrammetric) splat files into Luxar.

Reads the four common classical Gaussian-splat dialects (INRIA ``point_cloud.ply``,
antimatter15 ``.splat``, Niantic/Scaniverse ``.spz``, SuperSplat compressed
``.ply``) into a :class:`~luxar.gsplats.gsplat_data.GSplatData` and writes a
current-format ``.gsplats.zarr``. The heavy lifting lives in
:mod:`luxar.gsplats.interop.classical_splats`; this module is the thin Typer
surface (`gsplat export` — the PLY writer — joins it in a follow-up).
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
        "Niantic .spz, or SuperSplat compressed .ply.",
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr (current node-tree format)."
    ),
    format: Literal["auto", "inria", "splat", "spz", "supersplat"] = typer.Option(
        "auto",
        "--format",
        "-f",
        help="Source dialect; 'auto' sniffs the extension and PLY header.",
    ),
    reorient: Optional[bool] = typer.Option(
        None,
        "--reorient/--no-reorient",
        help="Apply the canonical COLMAP → Y-up orientation fix (180° rotation "
        "about X). Default: per-dialect — on for the Y-down dialects "
        "(INRIA PLY / .splat / SuperSplat), off for Y-up SPZ.",
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


def register_interchange_commands(app: typer.Typer) -> None:
    """Attach the classical-format interchange commands to the gsplat CLI."""
    app.command("import")(import_command)
