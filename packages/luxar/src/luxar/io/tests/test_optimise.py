"""``luxar optimise`` — re-chunking must change the grid and nothing else.

Every assertion here is about something that fails SILENTLY if it regresses: a
dropped attr decodes to wrong values rather than raising, a lost consolidated
index loads as an empty scene, a chunk that is not a multiple of the spatial
atom makes a row-range read straddle a boundary, and an unchanged
``content_hash`` lets a warm viewer cache serve chunks whose keys have moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from luxar._zarr_compat import (
    consolidate,
    create_array,
    is_consolidated,
    open_group,
    read_consolidated_attrs,
    set_zarr_format,
    zarr_format,
)
from luxar.core.dimensions import Dimensions
from luxar.io import LuxarZarrCompiler
from luxar.io.optimise import (
    _INDEX_ARRAYS,
    CHUNK_PROFILES,
    optimise_store,
    plan_optimisation,
    resolve_target_bytes,
    summarise_chunk_layout,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _walk(group: Any, path: str = ""):
    """Yield ``(path, node, is_array)`` for every node, root included."""
    yield path or "/", group, False
    for name in sorted(group.array_keys()):
        yield (f"{path}/{name}" if path else name), group[name], True
    for name in sorted(group.group_keys()):
        yield from _walk(group[name], f"{path}/{name}" if path else name)


def _arrays(group: Any) -> dict[str, Any]:
    return {p: n for p, n, is_arr in _walk(group) if is_arr}


def _groups(group: Any) -> dict[str, Any]:
    return {p: n for p, n, is_arr in _walk(group) if not is_arr}


def _declared_atom(attrs: dict, array_name: str) -> tuple[int | None, str | None]:
    """Read the node's atom straight out of attrs — deliberately NOT via the
    production helper, so the alignment test cannot pass by agreeing with the
    code it is checking."""
    if array_name == "segments":
        nested = attrs.get("segment_ordering")
        if isinstance(nested, dict) and "chunk_size" in nested:
            return int(nested["chunk_size"]), "segment_chunk_bounds"
        return None, None
    nested = attrs.get("vertex_ordering")
    if isinstance(nested, dict) and "chunk_size" in nested:
        return int(nested["chunk_size"]), "vertex_chunk_bounds"
    if "chunk_size" in attrs:
        return int(attrs["chunk_size"]), "chunk_bounds"
    return None, None


def assert_values_identical(src_path: Path, dst_path: Path) -> int:
    """Every array present, same shape/dtype, byte-for-byte equal."""
    src = open_group(src_path, mode="r")
    dst = open_group(dst_path, mode="r")
    src_arrays, dst_arrays = _arrays(src), _arrays(dst)
    assert set(src_arrays) == set(dst_arrays)
    for path, a in src_arrays.items():
        b = dst_arrays[path]
        assert tuple(a.shape) == tuple(b.shape), path
        assert np.dtype(a.dtype) == np.dtype(b.dtype), path
        assert np.asarray(a[...]).tobytes() == np.asarray(b[...]).tobytes(), path
    return len(src_arrays)


def assert_attrs_identical(src_path: Path, dst_path: Path) -> None:
    """Group and array attrs match exactly, except the two the optimiser is
    contractually required to move (``content_hash``) or add
    (``chunk_layout``)."""
    src = open_group(src_path, mode="r")
    dst = open_group(dst_path, mode="r")
    for path, node, _is_arr in _walk(src):
        mine = dict(node.attrs)
        theirs = dict(dict(_walk_lookup(dst, path)).get("attrs", {}))
        mine.pop("content_hash", None)
        theirs.pop("content_hash", None)
        theirs.pop("chunk_layout", None)
        assert mine == theirs, path


def _walk_lookup(root: Any, path: str) -> dict:
    node = root if path == "/" else root[path]
    return {"attrs": dict(node.attrs)}


def assert_atom_aligned(store_path: Path) -> int:
    """Every chunk on a REAL spatial-index node is a whole multiple of that
    node's atom (or the array is a single chunk).

    Scoped to nodes carrying the matching bounds array, because that — not a
    bare ``chunk_size`` attr — is what makes the atom a partition grid the
    viewer resolves row ranges against.
    """
    root = open_group(store_path, mode="r")
    checked = 0
    for _gpath, group in _groups(root).items():
        attrs = dict(group.attrs)
        names = set(group.array_keys())
        for name in sorted(names):
            if name in _INDEX_ARRAYS:
                continue
            atom, proof = _declared_atom(attrs, name)
            if atom is None or proof not in names:
                continue
            arr = group[name]
            rows, n_rows = int(arr.chunks[0]), int(arr.shape[0])
            if n_rows <= 1:
                continue
            assert rows % atom == 0 or rows == n_rows, (
                f"{name}: chunk rows {rows} is neither a multiple of the atom "
                f"{atom} nor the whole array ({n_rows})"
            )
            checked += 1
    return checked


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_scene(path: Path, **kwargs: Any) -> Path:
    """A scene with every structural shape the optimiser has to handle."""
    n = 20_000
    rng = _rng(7)
    with LuxarZarrCompiler(str(path), **kwargs) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            rng.random((n, 3)).astype(np.float32),
            colors=rng.random((n, 3)).astype(np.float32),
            radii=0.01,
        )
        nv = 12_000
        scene.add_lines(
            "curve",
            rng.random((nv, 3)).astype(np.float32),
            widths=np.full(nv, 0.02, dtype=np.float32),
            scalars=rng.random(nv).astype(np.float32),
            colormap="viridis",
        )
        scene.add_gsplats(
            "splats",
            centers=rng.random((n, 3)).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
            ),
        )
        # Mesh has no spatial index at all — no atom, pure byte target.
        nm = 6000
        scene.add_mesh(
            "surface",
            vertices=rng.random((nm, 3)).astype(np.float32),
            faces=np.arange(nm, dtype=np.uint32).reshape(-1, 3),
        )
    return path


# --------------------------------------------------------------------------
# Round-trip: values, attrs, index, consolidation
# --------------------------------------------------------------------------


class TestRoundTrip:
    def test_flat_leaves_are_bit_identical(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(src, dst, verify=True)
        assert plan.n_rechunked > 0, "nothing was re-chunked — test is vacuous"
        assert plan.target_n_chunks < plan.source_n_chunks
        assert assert_values_identical(src, dst) > 0

    def test_lines_with_scalars_keeps_its_nested_ordering_attrs(
        self, tmp_path: Path
    ) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        before = dict(open_group(src, mode="r")["curve"].attrs)
        after = dict(open_group(dst, mode="r")["curve"].attrs)
        # Lines nests its atom, unlike Points/GSplats — the exact shape that a
        # shallow attr copy would flatten or drop.
        assert before["vertex_ordering"] == after["vertex_ordering"]
        assert before["segment_ordering"] == after["segment_ordering"]
        assert "scalars" in open_group(dst, mode="r")["curve"]

    def test_every_attr_survives(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        assert_attrs_identical(src, dst)

    def test_additive_ladder(self, tmp_path: Path) -> None:
        src = tmp_path / "ladder.luxar.zarr"
        with LuxarZarrCompiler(str(src)) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                _rng(3).random((20_000, 3)).astype(np.float32),
                additive_lod={"n_lods": 4},
            )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        root = open_group(dst, mode="r")
        assert int(root["pts"].attrs["n_additive_sublods"]) == 4
        assert "additive_0" in root["pts"]
        assert_values_identical(src, dst)
        assert_atom_aligned(dst)

    def test_substitutive_lod_group(self, tmp_path: Path) -> None:
        src = tmp_path / "lod.luxar.zarr"
        with LuxarZarrCompiler(str(src)) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                _rng(4).random((20_000, 3)).astype(np.float32),
                substitutive_lod=True,
            )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        assert dict(open_group(dst, mode="r")["pts"].attrs).get("kind") == "lod"
        assert_values_identical(src, dst)
        assert_attrs_identical(src, dst)
        assert_atom_aligned(dst)

    def test_partition(self, tmp_path: Path) -> None:
        src = tmp_path / "part.luxar.zarr"
        with LuxarZarrCompiler(str(src)) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                _rng(5).random((20_000, 3)).astype(np.float32),
                partition={"max_elements": 5000},
            )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        root = open_group(dst, mode="r")
        assert dict(root["pts"].attrs).get("kind") == "partition"
        assert "part_0" in root["pts"]
        assert_values_identical(src, dst)
        assert_atom_aligned(dst)

    def test_consolidated_metadata_is_present_and_readable(
        self, tmp_path: Path
    ) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        # Load-bearing: the viewer builds the whole scene graph from this index
        # and has no directory-walk fallback.
        assert is_consolidated(dst)
        index = read_consolidated_attrs(dst)
        assert "/" in index
        assert {"cloud", "curve", "splats", "surface"} <= set(index)
        # Exactly one index — a nested one would serve pre-edit attributes.
        assert index["cloud"]["type"] == "points"


# --------------------------------------------------------------------------
# The alignment invariant (the mutation-check target)
# --------------------------------------------------------------------------


class TestAlignment:
    def test_every_emitted_chunk_is_a_multiple_of_its_atom(
        self, tmp_path: Path
    ) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(src, dst)
        assert assert_atom_aligned(dst) > 0, "no indexed arrays checked"
        # Non-vacuity: at least one array must land on a PROPER multiple of its
        # atom rather than trivially on one whole-array chunk, or the invariant
        # above would hold for a pass that did no rounding at all.
        strict = [
            a
            for a in plan.arrays
            if a.rechunked
            and a.atom
            and a.target_chunks[0] < a.shape[0]
            and a.target_chunks[0] % a.atom == 0
        ]
        assert strict, "every indexed array collapsed to a single chunk"

    @pytest.mark.parametrize("profile", sorted(CHUNK_PROFILES))
    def test_alignment_holds_at_every_profile(
        self, tmp_path: Path, profile: str
    ) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / f"out_{profile}.luxar.zarr"
        optimise_store(src, dst, target_bytes=CHUNK_PROFILES[profile], profile=profile)
        assert assert_atom_aligned(dst) > 0
        assert_values_identical(src, dst)

    def test_a_chunk_is_never_smaller_than_one_atom(self, tmp_path: Path) -> None:
        """A tiny byte target must not round an indexed chunk down to zero
        atoms — that would put a partition's rows across two zarr chunks."""
        src = build_scene(tmp_path / "src.luxar.zarr")
        root = open_group(src, mode="r")
        plan = plan_optimisation(root, target_bytes=64)
        for a in plan.arrays:
            if a.atom is not None and a.rechunked:
                assert a.target_chunks[0] >= min(a.atom, a.shape[0])


# --------------------------------------------------------------------------
# Format preservation
# --------------------------------------------------------------------------


@pytest.fixture
def restore_zarr_format():
    original = zarr_format()
    yield
    set_zarr_format(original)


class TestFormatPreservation:
    @pytest.mark.parametrize("fmt", [2, 3])
    def test_zarr_format_is_preserved(
        self, tmp_path: Path, fmt: int, restore_zarr_format: None
    ) -> None:
        set_zarr_format(fmt)
        src = build_scene(tmp_path / "src.luxar.zarr")
        assert int(open_group(src, mode="r").metadata.zarr_format) == fmt
        # Flip the process default so an output that merely follows it fails.
        set_zarr_format(2 if fmt == 3 else 3)
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        out = open_group(dst, mode="r")
        assert int(out.metadata.zarr_format) == fmt
        for path, arr in _arrays(out).items():
            assert int(arr.metadata.zarr_format) == fmt, path
        assert is_consolidated(dst)
        assert_values_identical(src, dst)

    @pytest.mark.parametrize("fmt", [2, 3])
    def test_codecs_are_reused_not_re_derived(
        self, tmp_path: Path, fmt: int, restore_zarr_format: None
    ) -> None:
        """An omitted compressor is not "no compressor" — zarr's ``"auto"`` is
        Blosc/lz4 at format 2 and zstd at format 3, and Luxar has arrays that
        must stay RAW."""
        set_zarr_format(fmt)
        src = tmp_path / "codecs.zarr"
        root = open_group(src, mode="w")
        create_array(
            root,
            "raw",
            data=np.arange(50_000, dtype=np.uint8),
            chunks=(1000,),
            compressor=None,
        )
        consolidate(root)
        dst = tmp_path / "out.zarr"
        optimise_store(src, dst, generic=True, verify=True)
        out = open_group(dst, mode="r")["raw"]
        assert tuple(out.compressors) == (), "a RAW array picked up a compressor"
        assert out.chunks[0] > 1000


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


class TestDegenerate:
    def test_single_element_node(self, tmp_path: Path) -> None:
        src = tmp_path / "one.luxar.zarr"
        with LuxarZarrCompiler(str(src)) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", np.zeros((1, 3), dtype=np.float32))
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        assert assert_values_identical(src, dst) > 0

    def test_array_ref_and_broadcast_shapes_are_left_alone(
        self, tmp_path: Path
    ) -> None:
        """``(0, D)`` refs and ``(1,)``/``(1, k)`` broadcasts carry no rows to
        re-chunk; a naive first pass emitted a 0-row chunk for the first."""
        src = tmp_path / "special.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        ref = create_array(
            root,
            "ref",
            data=np.zeros((0, 3), dtype=np.float32),
            chunks=(1, 3),
            compressor=None,
        )
        ref.attrs["encoding"] = {
            "name": "array_ref",
            "target": "other/positions",
            "hash": "deadbeef",
            "original_shape": [1000, 3],
            "original_dtype": "float32",
        }
        bcast = create_array(
            root,
            "radii",
            data=np.array([0.5], dtype=np.float32),
            chunks=(1,),
            compressor=None,
        )
        bcast.attrs["encoding"] = {"name": "broadcasted", "n_elements": 1000}
        bcast_2d = create_array(
            root,
            "uniform",
            data=np.zeros((1, 6), dtype=np.float32),
            chunks=(1, 6),
            compressor=None,
        )
        bcast_2d.attrs["encoding"] = {"name": "broadcasted", "n_elements": 1000}
        consolidate(root)

        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, verify=True)
        reasons = {a.path: a.skip_reason for a in plan.arrays}
        assert reasons == {
            "ref": "array_ref",
            "radii": "broadcast",
            "uniform": "broadcast",
        }
        out = open_group(dst, mode="r")
        assert tuple(out["ref"].shape) == (0, 3)
        assert tuple(out["ref"].chunks) == (1, 3)
        assert tuple(out["radii"].chunks) == (1,)
        assert dict(out["ref"].attrs)["encoding"]["target"] == "other/positions"

    def test_bounds_arrays_are_never_rechunked(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        root = open_group(src, mode="r")
        plan = plan_optimisation(root, target_bytes=CHUNK_PROFILES["archive"])
        bounds = [a for a in plan.arrays if a.path.rsplit("/", 1)[-1] in _INDEX_ARRAYS]
        assert bounds, "the fixture grew no spatial index"
        for a in bounds:
            assert a.skip_reason == "spatial index"
            assert a.target_chunks == a.source_chunks

    def test_a_store_with_no_spatial_index_falls_back_to_bytes(
        self, tmp_path: Path
    ) -> None:
        src = build_scene(tmp_path / "noidx.luxar.zarr", enable_spatial_index=False)
        assert not any(
            "chunk_bounds" in p for p in _arrays(open_group(src, mode="r"))
        ), "the fixture grew a spatial index"
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(src, dst, verify=True)
        assert plan.n_rechunked > 0
        # The Points node carries no `chunk_size` at all, so there is nothing to
        # align to and the byte target is used raw. (A GSplats node written with
        # `ordering="none"` still gets a vestigial `chunk_size` stamped — that
        # case is covered by the test below.)
        assert all(a.atom is None for a in plan.arrays if a.path.startswith("cloud/"))
        assert plan.target_n_chunks < plan.source_n_chunks
        assert_values_identical(src, dst)

    def test_a_chunk_size_attr_the_arrays_ignore_is_not_treated_as_an_atom(
        self, tmp_path: Path
    ) -> None:
        """A gsplat leaf written with ``ordering="none"`` still gets a default
        ``chunk_size`` stamped, and its arrays are NOT on that grid. Trusting it
        would inflate every chunk to a boundary that indexes nothing."""
        src = tmp_path / "vestigial.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        # 997 is prime, so nothing here can accidentally be a multiple of it.
        root.attrs["chunk_size"] = 997
        create_array(
            root,
            "amplitudes",
            data=np.arange(50_000, dtype=np.float32),
            chunks=(1000,),
            compressor=None,
        )
        consolidate(root)
        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, verify=True)
        (amps,) = [a for a in plan.arrays if a.path == "amplitudes"]
        assert amps.atom is None
        assert amps.target_chunks == (16_384,)  # 64 KB / 4 bytes, no rounding

    def test_a_zipped_store_round_trips_both_ways(self, tmp_path: Path) -> None:
        """A ``.zarr.zip`` owns a file handle and only writes a valid archive on
        close — without one the output re-read as "File is not a zip file"."""
        import shutil as _shutil

        src_dir = build_scene(tmp_path / "src.luxar.zarr")
        _shutil.make_archive(str(tmp_path / "src.luxar.zarr"), "zip", str(src_dir))
        src_zip = tmp_path / "src.luxar.zarr.zip"
        assert src_zip.is_file()

        from_zip = optimise_store(
            src_zip, tmp_path / "from_zip.luxar.zarr", verify=True
        )
        assert from_zip.target_n_chunks < from_zip.source_n_chunks

        to_zip = optimise_store(src_dir, tmp_path / "out.luxar.zarr.zip", verify=True)
        assert to_zip.target_n_chunks < to_zip.source_n_chunks
        assert_values_identical(src_dir, tmp_path / "out.luxar.zarr.zip")

    @pytest.mark.parametrize("n_rows", [4095, 4096, 4097])
    def test_rows_around_the_atom_boundary(self, tmp_path: Path, n_rows: int) -> None:
        """±1 around one atom. The array must never end up chunked at anything
        between one atom and the whole array."""
        atom = 4096
        src = tmp_path / f"edge_{n_rows}.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        root.attrs["chunk_size"] = atom
        create_array(
            root,
            "chunk_bounds",
            data=np.zeros((max(1, -(-n_rows // atom)), 3, 2), dtype=np.float32),
            chunks=(max(1, -(-n_rows // atom)), 3, 2),
            compressor=None,
        )
        create_array(
            root,
            "values",
            data=np.arange(n_rows, dtype=np.uint8),
            chunks=(1024,),
            compressor=None,
        )
        consolidate(root)
        dst = tmp_path / f"out_{n_rows}.zarr"
        plan = optimise_store(src, dst, verify=True)
        (vals,) = [a for a in plan.arrays if a.path == "values"]
        assert vals.atom == atom
        rows = vals.target_chunks[0]
        assert rows % atom == 0 or rows == n_rows, rows
        assert assert_values_identical(src, dst) == 2


# --------------------------------------------------------------------------
# Never shrink
# --------------------------------------------------------------------------


class TestNeverShrink:
    def test_an_already_large_chunk_is_left_alone(self, tmp_path: Path) -> None:
        src = tmp_path / "big.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "values",
            data=np.arange(1_000_000, dtype=np.float32),
            chunks=(250_000,),
            compressor=None,  # 1 MB chunks
        )
        consolidate(root)
        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, target_bytes=64 * 1024, verify=True)
        (vals,) = plan.arrays
        assert vals.skip_reason == "already at or above target"
        assert tuple(open_group(dst, mode="r")["values"].chunks) == (250_000,)

    def test_optimising_twice_is_a_fixed_point(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        once = tmp_path / "once.luxar.zarr"
        twice = tmp_path / "twice.luxar.zarr"
        optimise_store(src, once)
        second = optimise_store(once, twice)
        assert second.n_rechunked == 0
        assert second.source_n_chunks == second.target_n_chunks

    def test_no_array_ever_gains_chunks(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        root = open_group(src, mode="r")
        for target in (1024, 64 * 1024, 1024 * 1024):
            plan = plan_optimisation(root, target_bytes=target)
            for a in plan.arrays:
                assert a.target_n_chunks <= a.source_n_chunks, (a.path, target)


# --------------------------------------------------------------------------
# Cache invalidation
# --------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_the_scene_content_hash_changes(self, tmp_path: Path) -> None:
        """The bug this prevents is silent: the viewer's cache validates on
        ``content_hash``, so an unchanged hash means a warm client keeps
        serving chunks whose keys now cover different rows."""
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        before = dict(open_group(src, mode="r").attrs)["content_hash"]
        after = dict(open_group(dst, mode="r").attrs)["content_hash"]
        assert before != after

    def test_the_hash_changes_even_when_nothing_is_rechunked(
        self, tmp_path: Path
    ) -> None:
        """The conservative half of the guard. Values are identical and no
        chunk moved, but the store is still a NEW artifact under a new hash —
        which is right, because the alternative is a hash that can be stale."""
        src = build_scene(tmp_path / "src.luxar.zarr")
        once = tmp_path / "once.luxar.zarr"
        twice = tmp_path / "twice.luxar.zarr"
        optimise_store(src, once)
        plan = optimise_store(once, twice)
        assert plan.n_rechunked == 0
        h1 = dict(open_group(once, mode="r").attrs)["content_hash"]
        h2 = dict(open_group(twice, mode="r").attrs)["content_hash"]
        assert h1 != h2

    def test_chunk_layout_is_stamped_on_the_root(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(
            src, dst, target_bytes=CHUNK_PROFILES["hosting"], profile="hosting"
        )
        layout = dict(open_group(dst, mode="r").attrs)["chunk_layout"]
        assert layout["profile"] == "hosting"
        assert layout["target_bytes"] == CHUNK_PROFILES["hosting"]
        assert layout["chunks_after"] == plan.target_n_chunks
        assert layout["chunks_before"] == plan.source_n_chunks
        # It has to reach the consolidated index too, or an HTTP reader that
        # only fetches the root document never sees it.
        assert "chunk_layout" in read_consolidated_attrs(dst)["/"]


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------


class TestGuards:
    def test_in_place_is_refused(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        with pytest.raises(ValueError, match="in place"):
            optimise_store(src, src, overwrite=True)

    def test_an_existing_output_needs_overwrite(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        with pytest.raises(FileExistsError, match="--overwrite"):
            optimise_store(src, dst)
        optimise_store(src, dst, overwrite=True)
        assert_values_identical(src, dst)

    def test_a_plain_zarr_store_needs_generic(self, tmp_path: Path) -> None:
        src = tmp_path / "plain.zarr"
        root = open_group(src, mode="w")
        create_array(
            root,
            "x",
            data=np.arange(10_000, dtype=np.float32),
            chunks=(100,),
            compressor=None,
        )
        consolidate(root)
        with pytest.raises(ValueError, match="--generic"):
            optimise_store(src, tmp_path / "a.zarr")
        plan = optimise_store(src, tmp_path / "b.zarr", generic=True, verify=True)
        assert plan.n_rechunked == 1
        # `--generic` must not stamp scene semantics onto a foreign store.
        assert "content_hash" not in dict(
            open_group(tmp_path / "b.zarr", mode="r").attrs
        )

    def test_verify_reads_every_array(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        # A failure here raises; the point is that verify is not a no-op.
        optimise_store(src, dst, verify=True)
        assert assert_values_identical(src, dst) >= 10

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"target_kb": 0}, "positive"),
            ({"target_bytes": -1}, "positive"),
            ({"profile": "nope"}, "unknown profile"),
            ({"target_kb": 32, "profile": "local"}, "alternatives"),
        ],
    )
    def test_target_resolution_errors(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            resolve_target_bytes(**kwargs)

    def test_target_resolution_defaults_and_profiles(self) -> None:
        from luxar.typing_utils.constants import TARGET_CHUNK_BYTES

        assert resolve_target_bytes() == TARGET_CHUNK_BYTES
        assert resolve_target_bytes(target_kb=128) == 128 * 1024
        assert resolve_target_bytes(profile="hosting") == CHUNK_PROFILES["hosting"]


# --------------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------------


class TestSummary:
    def test_the_summary_improves_after_optimising(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        before = summarise_chunk_layout(open_group(src, mode="r"))
        after = summarise_chunk_layout(open_group(dst, mode="r"))
        assert after.n_chunks < before.n_chunks
        assert after.mean_chunk_bytes > before.mean_chunk_bytes
        assert after.n_arrays == before.n_arrays
        assert 0.0 <= after.share_under_floor <= 1.0

    def test_projected_requests_match_the_plan(self, tmp_path: Path) -> None:
        src = build_scene(tmp_path / "src.luxar.zarr")
        root = open_group(src, mode="r")
        assert summarise_chunk_layout(root).n_chunks == (
            plan_optimisation(root).source_n_chunks
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCli:
    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from luxar.cli.main import app

        src = build_scene(tmp_path / "src.luxar.zarr")
        before = sorted(p.name for p in tmp_path.iterdir())
        result = CliRunner().invoke(app, ["optimise", str(src), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Nothing was written" in result.output
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_dry_run_rejects_an_output_argument(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from luxar.cli.main import app

        src = build_scene(tmp_path / "src.luxar.zarr")
        result = CliRunner().invoke(
            app, ["optimise", str(src), str(tmp_path / "o.zarr"), "--dry-run"]
        )
        assert result.exit_code == 1

    def test_end_to_end(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from luxar.cli.main import app

        src = build_scene(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        result = CliRunner().invoke(
            app,
            ["optimise", str(src), str(dst), "--profile", "hosting", "--verify"],
        )
        assert result.exit_code == 0, result.output
        assert_values_identical(src, dst)
        assert assert_atom_aligned(dst) > 0

    def test_info_stats_reports_the_chunk_layout(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from luxar.cli.main import app

        src = build_scene(tmp_path / "src.luxar.zarr")
        result = CliRunner().invoke(app, ["info", str(src), "--stats", "--no-tree"])
        assert result.exit_code == 0, result.output
        assert "Chunk Layout" in result.output
        assert "Projected requests" in result.output
