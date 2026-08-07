"""``luxar mesh lod`` — a mesh scene → a substitutive LOD ladder.

The mesh peer of ``luxar gsplat lod``, and shaped differently for one structural
reason: ``gsplat lod`` reads and writes ``.gsplats.zarr``, a standalone store, so
it is a store-to-store transform. **There is no standalone mesh store** — the only
sink for a mesh is ``Scene.add_mesh`` — so this reads a ``.luxar.zarr`` SCENE,
takes one mesh node out of it, and writes a new scene whose node is a ``kind=lod``
group. Same asymmetry ``luxar mesh import`` already carries, for the same reason.

The ladder itself is built by ``add_mesh(substitutive_lod=…)``; this command is an
option surface plus a read/write shell around it, so the CLI and the Python API
cannot disagree about what a level is.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from arbol import aprint, asection

from ...core.group.lod.group import MESH_SUBSTITUTIVE_METHODS


def _pick_mesh(scene: Any, input_path: Path, node_name: Optional[str]) -> str:
    """Which mesh node to coarsen.

    Naming it is optional when the scene has exactly one mesh — which is what
    ``luxar mesh import`` produces, and therefore the common input. With several,
    the ambiguity is refused rather than guessed at: picking "the first" would
    silently coarsen a different surface than the user meant, on a scene whose
    node order they never see.
    """
    meshes = scene.list_meshes()
    if not meshes:
        raise ValueError(
            f"{input_path} contains no mesh node. `luxar mesh lod` coarsens a "
            "surface; use `luxar gsplat lod` for splats, or `luxar mesh import` "
            "to bring a mesh file in first."
        )
    if node_name is not None:
        if node_name not in meshes:
            raise ValueError(
                f"No mesh node named {node_name!r} in {input_path}. Available: "
                f"{', '.join(sorted(meshes))}"
            )
        return node_name
    if len(meshes) > 1:
        raise ValueError(
            f"{input_path} has {len(meshes)} mesh nodes and none was named. Pass "
            f"--node to choose one: {', '.join(sorted(meshes))}"
        )
    return str(meshes[0])


def run_lod(
    *,
    input_path: Path,
    output_path: Path,
    node_name: Optional[str],
    levels: int,
    compression_factor: int,
    method: str,
    overwrite: bool,
) -> List[int]:
    """Write ``input_path``'s mesh as a substitutive ladder. Returns the counts."""
    from luxar import LuxarZarrCompiler

    # Before the deletion, for the reason `mesh import` documents: `--overwrite`
    # removes the output first, so an output that IS the input would delete the
    # source and only then discover there is nothing to read.
    if output_path.resolve() == input_path.resolve():
        raise ValueError(
            f"--output must differ from the input ({input_path}); this command "
            "writes a new scene rather than editing one in place."
        )
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

    from ...io.reader import LuxarScene

    with asection(f"Building a mesh LOD ladder from {input_path}"):
        source = LuxarScene.load(input_path)
        # A scene with no dimensions block cannot be rewritten: every node's
        # coordinates are interpreted against it, so inventing a default here
        # would silently relabel the axes of whatever we wrote out.
        if source.dimensions is None:
            raise ValueError(
                f"{input_path} declares no scene dimensions, so its mesh cannot be "
                "rewritten into a new scene without inventing axes for it."
            )
        node_path = _pick_mesh(source, input_path, node_name)
        data = source.get_mesh(node_path)
        attrs = source.get_node_metadata(node_path)
        aprint(
            f"Source {node_path!r}: {data.vertices.shape[0]:,} vertices, "
            f"{data.faces.shape[0]:,} faces"
        )

        spec: Dict[str, Any] = {
            "levels": levels,
            "compression_factor": compression_factor,
            "method": method,
        }
        # The LEAF name, not the full path: the ladder is written at the scene
        # root, so a nested source node keeps its own name rather than inventing
        # a group hierarchy the user did not ask for.
        with LuxarZarrCompiler(str(output_path)) as compiler:
            scene = compiler.create_scene(dimensions=source.dimensions)
            scene.add_mesh(
                node_path.rsplit("/", 1)[-1],
                data.vertices,
                data.faces,
                normals=data.normals,
                normal_dims=data.normal_dims,
                colors=data.colors,
                shading=attrs.get("shading"),
                double_sided=bool(attrs.get("double_sided", True)),
                substitutive_lod=spec,
            )

    # Read back rather than trusting the write: the ladder's whole value is that
    # the viewer can select among its levels, and a store that came out with one
    # level (or with duplicate counts) still "succeeds" at the API boundary.
    written = LuxarScene.load(output_path)
    counts = sorted(
        int(written.get_node_metadata(m)["n_vertices"]) for m in written.list_meshes()
    )
    if len(counts) < 2:
        aprint(
            "Note: the surface was too coarse to reduce, so a plain mesh leaf was "
            "written instead of a ladder."
        )
    else:
        aprint(f"✓ {len(counts)} levels on disk, vertex counts {counts}")
    return counts


def lod_command(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .luxar.zarr scene containing a mesh node."
    ),
    output_path: Path = typer.Argument(..., help="Output .luxar.zarr scene."),
    node: Optional[str] = typer.Option(
        None,
        "--node",
        help="Which mesh node to coarsen. Optional when the scene has exactly one.",
    ),
    levels: int = typer.Option(
        3, "--levels", "-L", help="Number of coarse levels below the original."
    ),
    compression_factor: int = typer.Option(
        4,
        "--compression-factor",
        "-K",
        help="Vertex-count reduction per level: level i targets V / K**i.",
    ),
    method: str = typer.Option(
        "auto",
        "--method",
        "-m",
        help=(
            "Decimation method, one of "
            f"{', '.join(sorted(MESH_SUBSTITUTIVE_METHODS))}. 'auto' resolves to "
            "'cluster' today. NOTE these are NOT the `gsplat lod` methods: those "
            "reduce a Gaussian mixture, which a surface is not."
        ),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing output."
    ),
) -> None:
    """Build a substitutive LOD ladder for a mesh scene.

    Writes a `kind=lod` group whose coarse children are progressively decimated
    copies of the surface and whose finest child is the original. The viewer shows
    exactly one at a time, chosen by how much of the screen the object covers.

    Levels that cannot reduce the surface are dropped, so a small mesh may come
    back with fewer than `--levels` — or, if it cannot be reduced at all, as a
    plain leaf.

    \b
    Examples:
      luxar mesh lod bunny.luxar.zarr bunny_lod.luxar.zarr
      luxar mesh lod scan.luxar.zarr scan_lod.luxar.zarr -L 4 -K 3
      luxar mesh lod multi.luxar.zarr out.luxar.zarr --node surfaces/skull
    """
    try:
        run_lod(
            input_path=input_path,
            output_path=output_path,
            node_name=node,
            levels=levels,
            compression_factor=compression_factor,
            method=method,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1) from exc


def register_lod_commands(app: typer.Typer) -> None:
    """Attach the mesh LOD commands to the mesh CLI."""
    app.command("lod")(lod_command)
