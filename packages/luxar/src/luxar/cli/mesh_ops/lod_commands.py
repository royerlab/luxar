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
    """Write ``input_path``'s mesh as a substitutive ladder. Returns the counts.

    ``output_path`` is normalized to ``<stem>.luxar.zarr`` — the store the
    compiler actually writes — before any guard or deletion looks at it.
    """
    from luxar import LuxarZarrCompiler

    from ...core.group.lod.mesh import resolve_substitutive_axis_mesh
    from ...core.viewer_config import ViewerConfig
    from ...io._compiler.node_common import KNOWN_RENDER_ATTRS
    from ...io.reader import LuxarScene
    from ...utils.paths import normalize_zarr_path

    # NORMALIZE FIRST, and guard the normalized path only. `LuxarZarrCompiler`
    # applies exactly this normalization to whatever it is handed, so the store
    # it writes is `<stem>.luxar.zarr` — guarding the raw argument guards a path
    # nothing ever writes to. Three real data-loss cases came of that:
    # `--output scene` next to `scene.luxar.zarr` slipped the same-path check and
    # rewrote the INPUT; an existing `out.luxar.zarr` slipped the exists check and
    # was replaced without `--overwrite`; and an output normalizing onto the
    # directory that holds the input rmtree'd the source. Each reported itself
    # only afterwards, as "Scene not found" from the read-back below.
    output_path = normalize_zarr_path(output_path, ".luxar.zarr")

    # Before the deletion, for the reason `mesh import` documents: `--overwrite`
    # removes the output first, so an output that IS the input — or a directory
    # CONTAINING it, where `rmtree` takes the whole tree — would delete the
    # source and only then discover there is nothing to read. The relation is
    # checked BOTH ways, because the message promises a path outside the input's
    # tree: a destination INSIDE the source store survived the one-directional
    # check and wrote a whole nested scene into it.
    source_resolved = input_path.resolve()
    destination = output_path.resolve()
    if (
        destination == source_resolved
        or destination in source_resolved.parents
        or source_resolved in destination.parents
    ):
        raise ValueError(
            f"Output {output_path} is the input scene itself, or a directory "
            "containing it, or a path inside it. Writing there would destroy or "
            "corrupt the source; choose an output path outside the input's tree."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

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
        aprint(
            f"Source {node_path!r}: {data.vertices.shape[0]:,} vertices, "
            f"{data.faces.shape[0]:,} faces"
        )

        spec: Dict[str, Any] = {
            "levels": levels,
            "compression_factor": compression_factor,
            "method": method,
        }
        # Validated HERE rather than inside `add_mesh`, because the deletion
        # below is irreversible: `--method qem` used to remove an existing output
        # and only then discover the method does not exist. This is the very
        # validator `add_mesh` runs, so the two cannot disagree about what is
        # accepted.
        resolve_substitutive_axis_mesh(spec)

        # The authored ATTRS of the source node: the placement (transform /
        # nd_transform), the nD visibility broadcast, and the render attrs.
        # Forwarding only `shading`/`double_sided` dropped all of it silently —
        # most sharply the transform, which put the coarsened surface somewhere
        # else in the scene with nothing saying so.
        #
        # What still does NOT come across is the per-vertex LABEL channels
        # (`labels` / `image_labels`): `MeshData` has no field for them, so the
        # reader never surfaces them and this round trip cannot carry what it
        # cannot read. A labelled source therefore comes back unlabelled rather
        # than half-labelled, which is at least uniform across every level.
        #
        # An allow-list rather than "everything outside MESH_RESERVED_ATTRS":
        # that set is the keys the writer refuses FROM A CALLER, and the store
        # carries stamps that never pass through that gate — the derived
        # `scalar_data_range` / `color_data_range` the array writers put straight
        # onto the group, and the `content_hash` finalize adds to every node.
        # Each of those is rejected as an unknown attribute on the way back in,
        # so an exclusion list would break this command every time a new stamp
        # lands. `shading` / `double_sided` stay explicit arguments below.
        #
        # `data.metadata`, NOT `get_node_metadata`: `get_mesh` has already turned
        # the stored column-major 16-list back into a 4x4 matrix, which is the
        # convention `add_mesh` expects — handing back the raw list would
        # transpose the transform on the round trip.
        #
        # The compositing keys among these (transform, opacity, blending_mode, …)
        # land on the `kind=lod` WRAPPER group rather than the children: the
        # adder splits `COMPOSITING_ATTRS` onto the wrapper, which is where a
        # per-layer setting belongs and where the viewer inherits it from.
        forwardable = KNOWN_RENDER_ATTRS | {
            "transform",
            "nd_transform",
            "extend_to_all",
        }
        forwarded = {k: v for k, v in data.metadata.items() if k in forwardable}

        # `colormap='custom'` is a SENTINEL, not a name: the writer resolves any
        # non-builtin colormap — an ndarray LUT, but also a plain matplotlib or
        # colorcet name like 'magma' — to a `colormap_lut` dataset plus that
        # word. Forwarding the word alone reaches the resolver as a name and
        # raises "Unknown colormap 'custom'", so a `magma` mesh could not be
        # laddered at all; accepting it would have been worse, silently
        # substituting the default LUT. Handing back the ARRAY is lossless: the
        # writer re-resolves it to the same LUT plus the same sentinel on every
        # child.
        if forwarded.get("colormap") == "custom":
            lut = source.get_colormap_lut(node_path)
            if lut is None:
                raise ValueError(
                    f"Mesh node {node_path!r} declares colormap='custom' but has no "
                    "'colormap_lut' dataset, so its colors cannot be reproduced. "
                    "The store is inconsistent; re-write the source scene."
                )
            forwarded["colormap"] = lut

        # Everything above this line reads or validates; the destination is
        # destroyed only once the write is certain to be attempted.
        if output_path.exists():
            shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

        # The LEAF name, not the full path: the ladder is written at the scene
        # root, so a nested source node keeps its own name rather than inventing
        # a group hierarchy the user did not ask for.
        # The scene-level viewer config travels with the scene, and dropping it
        # undoes the colormap fidelity above on exactly the scenes that have a
        # custom LUT: with no `tone_mapping` the viewer applies its ACES default,
        # which intentionally shifts hues, and the compiler re-emits the notice
        # saying so. `luxar mesh import` states ACES explicitly, so the
        # documented import → lod pipeline lost it too. Same fallback and same
        # reasoning as that command: ACES is the house default, and saying so
        # keeps the "nothing was chosen" notice quiet.
        viewer_config = source.viewer_config or ViewerConfig(tone_mapping="ACES")

        with LuxarZarrCompiler(str(output_path)) as compiler:
            scene = compiler.create_scene(
                dimensions=source.dimensions, viewer_config=viewer_config
            )
            scene.add_mesh(
                node_path.rsplit("/", 1)[-1],
                data.vertices,
                data.faces,
                normals=data.normals,
                normal_dims=data.normal_dims,
                colors=data.colors,
                scalars=data.scalars,
                shading=data.metadata.get("shading"),
                double_sided=bool(data.metadata.get("double_sided", True)),
                substitutive_lod=spec,
                **forwarded,
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
