"""``luxar mesh import`` — classical mesh files → a Luxar scene.

Structured like ``cli/gsplat_ops/interchange_commands.py``: the Typer function is a thin
error funnel around one ``run_import`` call, so the option surface and the work stay
separable and the command body stays under ruff's complexity ceiling.

One shape difference from ``gsplat import``, forced by the data model. That command
writes a ``.gsplats.zarr``, a standalone store ``GSplatData`` knows how to save. **There
is no standalone mesh store** — the only sink for a mesh is ``Scene.add_mesh`` — so this
writes a ``.luxar.zarr`` SCENE, and the read-back verification checks the mesh node's
attrs rather than a format version.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import typer
from arbol import aprint, asection

from ...mesh.interop import MESH_FORMATS, TriangleMesh, import_mesh

# Typer renders a Literal as a choice list; spelled out rather than built from
# MESH_FORMATS because Literal needs literal members.
_FormatOption = str


def _axis_names(ndim: int) -> list[str]:
    """Names for a 3D import. Mesh files are always 3D, so this is x/y/z."""
    return ["x", "y", "z"][:ndim]


def run_import(
    *,
    input_path: Path,
    output_path: Path,
    format: str,
    name: str,
    unit: str,
    center: bool,
    scale: float,
    weld: bool,
    keep_normals: bool,
    overwrite: bool,
) -> TriangleMesh:
    """Read a mesh file and write it as a single-node Luxar scene."""
    from luxar import Dimension, Dimensions, LuxarZarrCompiler
    from luxar.core.viewer_config import ViewerConfig

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

    with asection(f"Importing {input_path.name}"):
        mesh = import_mesh(input_path, format=format, weld=weld)
        aprint(
            f"Read {mesh.n_vertices:,} vertices / {mesh.n_faces:,} faces "
            f"({mesh.source_format})"
        )

        vertices = mesh.vertices.astype(np.float32, copy=True)
        if scale != 1.0:
            vertices *= scale
        if center:
            # Centre on the BOUNDING-BOX midpoint, not the vertex mean: the mean is
            # pulled toward wherever the mesh happens to be finely tessellated, which
            # for a scan with a dense region puts the model off-centre in the viewer.
            midpoint = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
            vertices -= midpoint
            aprint(f"Centred on the bounding-box midpoint {midpoint.round(4).tolist()}")

        normals = mesh.normals if keep_normals else None
        if mesh.normals is not None and not keep_normals:
            aprint(
                "Discarding stored normals (--no-keep-normals): shading falls back "
                "to the derivative flat normal"
            )

        dims = Dimensions(
            [Dimension(n, unit=unit, display=True) for n in _axis_names(3)]
        )
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                # ACES stated explicitly — the house default, and saying so keeps the
                # compiler's "nothing was chosen" tone-mapping notice quiet.
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )
            scene.add_mesh(
                name,
                vertices,
                mesh.faces,
                normals=normals,
                # Required whenever normals are passed: in nD there is no implicit
                # "first three dimensions", and an imported mesh is always plain 3D.
                normal_dims=[0, 1, 2] if normals is not None else None,
                colors=mesh.colors,
            )

        _verify(output_path, name, mesh)
        aprint(f"✓ Wrote {output_path}")
    return mesh


def _verify(output_path: Path, name: str, mesh: TriangleMesh) -> None:
    """Read the node back and confirm the counts survived the write.

    The mesh analogue of `gsplat import`'s format-version check. A write that silently
    dropped faces would otherwise only surface in the viewer.
    """
    from luxar.io.reader import LuxarScene

    node = LuxarScene.load(output_path).get_mesh(name)
    got_v, got_f = int(node.vertices.shape[0]), int(node.faces.shape[0])
    if (got_v, got_f) != (mesh.n_vertices, mesh.n_faces):
        raise RuntimeError(
            f"Verification failed: wrote {mesh.n_vertices}/{mesh.n_faces} "
            f"vertices/faces, read back {got_v}/{got_f}"
        )
    # Checked because it is the one attribute with no default: normals without
    # `normal_dims` cannot be oriented, and the pair is easy to break silently.
    if (node.normals is None) != (mesh.normals is None):
        raise RuntimeError(
            f"Verification failed: normals {'lost' if mesh.normals is not None else 'invented'} "
            "on the round trip"
        )
    aprint(f"✓ Verified {got_v:,} vertices / {got_f:,} faces on disk")


def import_command(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Classical mesh file: .ply (ascii or binary), .obj, .stl (ascii or "
        "binary), .gltf or .glb.",
    ),
    output_path: Path = typer.Argument(..., help="Output .luxar.zarr scene."),
    format: str = typer.Option(
        "auto",
        "--format",
        "-f",
        help=f"Source dialect; 'auto' sniffs the extension (one of {', '.join(MESH_FORMATS)}).",
    ),
    name: str = typer.Option("mesh", "--name", help="Node name inside the scene."),
    unit: str = typer.Option("um", "--unit", help="Physical unit for the three axes."),
    center: bool = typer.Option(
        True,
        "--center/--no-center",
        help="Recentre on the bounding-box midpoint. Most mesh files sit far from the "
        "origin, which fights the viewer's default framing.",
    ),
    scale: float = typer.Option(
        1.0, "--scale", help="Uniform scale applied to vertices."
    ),
    weld: bool = typer.Option(
        True,
        "--weld/--no-weld",
        help="Merge duplicate vertex positions. On by default because STL (always) and "
        "index-free glTF arrive as triangle soups, which defeat per-vertex normals and "
        "give picking a different id per corner per triangle.",
    ),
    keep_normals: bool = typer.Option(
        True,
        "--keep-normals/--no-keep-normals",
        help="Keep stored per-vertex normals. Drop them to force the shader's "
        "derivative flat-normal path (a faceted look).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing output."
    ),
) -> None:
    """Convert a classical mesh file into a Luxar scene.

    Reads PLY / OBJ / STL / glTF with no extra dependencies, welds duplicate vertices,
    fan-triangulates polygons, and writes a single-node `.luxar.zarr` you can serve
    directly with `luxar serve`.

    \b
    Examples:
      luxar mesh import bunny.ply bunny.luxar.zarr
      luxar mesh import scan.stl scan.luxar.zarr --unit mm --name Skull
      luxar mesh import model.glb model.luxar.zarr --no-center
      luxar mesh import surface.obj surface.luxar.zarr --scale 0.001 --unit m
    """
    try:
        run_import(
            input_path=input_path,
            output_path=output_path,
            format=format,
            name=name,
            unit=unit,
            center=center,
            scale=scale,
            weld=weld,
            keep_normals=keep_normals,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1) from exc


def register_import_commands(app: typer.Typer) -> None:
    """Attach the mesh interchange commands to the mesh CLI."""
    # `import` is a Python keyword, so the function is `import_command` and the
    # user-facing name is supplied here — same as `gsplat import`.
    app.command("import")(import_command)
