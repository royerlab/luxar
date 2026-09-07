"""``luxar optimise`` — re-chunking must change the grid and nothing else.

Every assertion here is about something that fails SILENTLY if it regresses: a
dropped attr decodes to wrong values rather than raising, a lost consolidated
index loads as an empty scene, a chunk that is not a multiple of the spatial
atom makes a row-range read straddle a boundary, an unchanged ``content_hash``
lets a warm viewer cache serve chunks whose keys have moved, and a destination
guard that only tests equality deletes the directory holding the source.

The compiled fixtures are SESSION-scoped. A full ``build_scene`` compile is ~2 s
and nothing here mutates its source, so building it once per session rather than
once per test is what keeps this file out of the ``make test-fast`` inner loop.
"""

from __future__ import annotations

import os
import shutil
import zipfile
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
    write_raw_bytes,
    zarr_format,
)
from luxar.core.dimensions import Dimensions
from luxar.io import LuxarZarrCompiler
from luxar.io import optimise as optimise_mod
from luxar.io.optimise import (
    _INDEX_ARRAYS,
    CHUNK_PROFILES,
    _compute_content_hashes_streaming,
    _verify,
    optimise_store,
    plan_optimisation,
    resolve_target_bytes,
    summarise_chunk_layout,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

#: Names zarr uses for METADATA documents, in either format. Everything else a
#: store holds is a chunk object — i.e. one request.
_META_DOCS = frozenset({"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"})


def _is_case_insensitive_fs(where: Path) -> bool:
    """Does ``where``'s filesystem fold case (macOS, Windows)?

    A case-differing payload name like ``Zarr.json`` is an ordinary distinct
    file on Linux and the group's own ``zarr.json`` here, so the fixture that
    needs real bytes under that name is unbuildable on one side of the divide.
    """
    probe = where / "CaseProbe"
    probe.write_text("x")
    try:
        return (where / "caseprobe").exists()
    finally:
        probe.unlink()


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


def _chunk_files(store: Path) -> int:
    """Chunk objects actually on disk — every file that is not zarr metadata."""
    return sum(1 for p in store.rglob("*") if p.is_file() and p.name not in _META_DOCS)


def _codec_configs(array: Any) -> list[dict]:
    """The array's compressor configuration AS STORED, at either zarr format.

    v2 keeps numcodecs objects (``get_config()``); v3 keeps ``zarr.codecs``
    ones (``to_dict()``), whose Blosc configuration additionally carries the
    dtype-derived ``typesize`` the byte shuffle depends on.
    """
    return [
        c.get_config() if hasattr(c, "get_config") else c.to_dict()
        for c in array.compressors
    ]


def _tree_snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """Every path under ``root`` with its size and mtime — a "nothing moved" proof."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime)
        for p in sorted(root.rglob("*"))
    }


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
        mine.pop("chunk_layout", None)
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
        # 20k vertices, not 12k: below that `curve/scalars` lands in a single
        # chunk and the Lines round-trip silently exercises only vertices and
        # segments.
        nv = 20_000
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


#: A minimal valid PNG (1x1 red pixel) — an overlay payload without needing PIL.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)

#: Minimal WebM-signature payload accepted by ``validate_video_input``.
_TINY_WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 60


def build_scene_with_an_overlay_image(path: Path) -> Path:
    """A small scene carrying a plain payload file — an image overlay.

    Compiled rather than assembled by hand, so the store really is the shape
    ``add_image`` produces: an ``overlays/logo`` group whose ``image_file`` attr
    names a file written straight into that group's own directory, reachable
    through no array or group API.
    """
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", _rng(13).random((20_000, 3)).astype(np.float32))
        scene.add_image(_TINY_PNG, position=(0.9, 0.05), name="logo")
    return path


def build_scene_with_an_overlay_video(path: Path) -> Path:
    """A small scene carrying video and poster plain payload files."""
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", _rng(14).random((20_000, 3)).astype(np.float32))
        scene.add_video(
            _TINY_WEBM,
            position=(0.1, 0.1),
            name="clip",
            poster=_TINY_PNG,
        )
    return path


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The full four-geometry scene. Read-only for every test that takes it."""
    return build_scene(tmp_path_factory.mktemp("scene") / "src.luxar.zarr")


@pytest.fixture(scope="session")
def scene_without_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_scene(
        tmp_path_factory.mktemp("noidx") / "noidx.luxar.zarr",
        enable_spatial_index=False,
    )


@pytest.fixture(scope="session")
def ladder_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ladder") / "ladder.luxar.zarr"
    with LuxarZarrCompiler(str(path)) as compiler:
        s = compiler.create_scene(dimensions=Dimensions.default_3d())
        s.add_points(
            "pts",
            _rng(3).random((20_000, 3)).astype(np.float32),
            additive_lod={"n_lods": 4},
        )
    return path


@pytest.fixture(scope="session")
def lod_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("lod") / "lod.luxar.zarr"
    with LuxarZarrCompiler(str(path)) as compiler:
        s = compiler.create_scene(dimensions=Dimensions.default_3d())
        s.add_points(
            "pts",
            _rng(4).random((20_000, 3)).astype(np.float32),
            substitutive_lod=True,
        )
    return path


@pytest.fixture(scope="session")
def partition_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("part") / "part.luxar.zarr"
    with LuxarZarrCompiler(str(path)) as compiler:
        s = compiler.create_scene(dimensions=Dimensions.default_3d())
        s.add_points(
            "pts",
            _rng(5).random((20_000, 3)).astype(np.float32),
            partition={"max_elements": 5000},
        )
    return path


@pytest.fixture
def restore_zarr_format():
    original = zarr_format()
    yield
    set_zarr_format(original)


@pytest.fixture(scope="session", params=[2, 3])
def formatted_scene(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[int, Path]:
    """A scene compiled at a pinned zarr format, plus the format it used."""
    fmt = int(request.param)
    original = zarr_format()
    set_zarr_format(fmt)
    try:
        path = build_scene(tmp_path_factory.mktemp(f"fmt{fmt}") / "src.luxar.zarr")
    finally:
        set_zarr_format(original)
    return fmt, path


# --------------------------------------------------------------------------
# Round-trip: values, attrs, index, consolidation
# --------------------------------------------------------------------------


class TestRoundTrip:
    def test_flat_leaves_are_bit_identical(self, scene: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(scene, dst, verify=True)
        assert plan.n_rechunked > 0, "nothing was re-chunked — test is vacuous"
        assert plan.target_n_chunks < plan.source_n_chunks
        assert assert_values_identical(scene, dst) > 0

    def test_lines_with_scalars_keeps_its_nested_ordering_attrs(
        self, scene: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(scene, dst)
        before = dict(open_group(scene, mode="r")["curve"].attrs)
        after = dict(open_group(dst, mode="r")["curve"].attrs)
        # Lines nests its atom, unlike Points/GSplats — the exact shape that a
        # shallow attr copy would flatten or drop.
        assert before["vertex_ordering"] == after["vertex_ordering"]
        assert before["segment_ordering"] == after["segment_ordering"]
        assert "scalars" in open_group(dst, mode="r")["curve"]
        # Non-vacuity: `scalars` must actually be re-chunked, or this test says
        # nothing about the per-vertex arrays beyond `vertices`/`segments`.
        (scalars,) = [a for a in plan.arrays if a.path == "curve/scalars"]
        assert scalars.rechunked, scalars.skip_reason
        assert scalars.target_chunks != scalars.source_chunks

    def test_every_attr_survives(self, scene: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(scene, dst)
        assert_attrs_identical(scene, dst)

    def test_additive_ladder(self, ladder_scene: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(ladder_scene, dst, verify=True)
        root = open_group(dst, mode="r")
        assert int(root["pts"].attrs["n_additive_sublods"]) == 4
        assert "additive_0" in root["pts"]
        assert_values_identical(ladder_scene, dst)
        assert_atom_aligned(dst)

    def test_substitutive_lod_group(self, lod_scene: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(lod_scene, dst, verify=True)
        assert dict(open_group(dst, mode="r")["pts"].attrs).get("kind") == "lod"
        assert_values_identical(lod_scene, dst)
        assert_attrs_identical(lod_scene, dst)
        assert_atom_aligned(dst)

    def test_partition(self, partition_scene: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(partition_scene, dst, verify=True)
        root = open_group(dst, mode="r")
        assert dict(root["pts"].attrs).get("kind") == "partition"
        assert "part_0" in root["pts"]
        assert_values_identical(partition_scene, dst)
        assert_atom_aligned(dst)

    def test_an_overlay_image_survives_the_copy(self, tmp_path: Path) -> None:
        """An overlay PNG is neither an array nor a subgroup, so a mirror that
        walks only those leaves it behind — and the output then carries an
        ``image_file`` attr naming a file that is not there, which the viewer
        shows as a vanished overlay rather than an error."""
        src = build_scene_with_an_overlay_image(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        filename = str(
            dict(open_group(dst, mode="r")["overlays/logo"].attrs)["image_file"]
        )
        assert (src / "overlays" / "logo" / filename).read_bytes() == _TINY_PNG
        copied = dst / "overlays" / "logo" / filename
        assert copied.is_file(), "the overlay image was dropped from the output"
        assert copied.read_bytes() == _TINY_PNG

    def test_an_overlay_video_and_poster_survive_the_copy(self, tmp_path: Path) -> None:
        src = build_scene_with_an_overlay_video(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)

        clip = open_group(dst, mode="r")["overlays/clip"]
        attrs = dict(clip.attrs)
        expected = {
            str(attrs["video_file"]): _TINY_WEBM,
            str(attrs["poster_file"]): _TINY_PNG,
        }
        for filename, payload in expected.items():
            copied = dst / "overlays" / "clip" / filename
            assert copied.is_file(), f"the video payload {filename!r} was dropped"
            assert copied.read_bytes() == payload

    def test_overlay_video_bytes_change_the_content_hash(self, tmp_path: Path) -> None:
        src = build_scene_with_an_overlay_video(tmp_path / "src.luxar.zarr")
        root = open_group(src, mode="r+")
        before = _compute_content_hashes_streaming(root)

        video = src / "overlays" / "clip" / "video.webm"
        video.write_bytes(_TINY_WEBM[:-1] + b"\x01")

        assert _compute_content_hashes_streaming(root) != before

    def test_consolidated_metadata_is_present_and_readable(
        self, scene: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(scene, dst)
        # Load-bearing: the viewer builds the whole scene graph from this index
        # and has no directory-walk fallback.
        assert is_consolidated(dst)
        index = read_consolidated_attrs(dst)
        assert "/" in index
        assert {"cloud", "curve", "splats", "surface"} <= set(index)
        # Exactly one index — a nested one would serve pre-edit attributes.
        assert index["cloud"]["type"] == "points"

    def test_a_standalone_gsplats_tree_is_restamped(self, tmp_path: Path) -> None:
        """The ``.gsplats.zarr`` branch of the restamp: no ``type=scene``, so it
        takes the metadata-only stamp rather than the value walk."""
        from luxar.gsplats.io.save_gsplats import save_gsplats

        src = tmp_path / "fit.gsplats.zarr"
        n = 20_000
        rng = _rng(11)
        save_gsplats(
            src,
            centers=rng.random((n, 3)).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32) + 0.1,
            cholesky_factors=np.tile(
                np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
            ),
        )
        before = dict(open_group(src, mode="r").attrs)
        assert dict(before).get("type") != "scene"
        assert "content_hash" in before

        dst = tmp_path / "out.gsplats.zarr"
        plan = optimise_store(src, dst, verify=True)
        assert plan.n_rechunked > 0
        after = dict(open_group(dst, mode="r").attrs)
        assert after["content_hash"] != before["content_hash"]
        assert_values_identical(src, dst)
        assert_atom_aligned(dst)


# --------------------------------------------------------------------------
# Payload files: the bytes no array or group API reaches
# --------------------------------------------------------------------------


def _store_with_a_payload_attr(
    path: Path, filename: str, payload: bytes | None = _TINY_PNG
) -> Path:
    """A tiny scene whose ``overlays/logo`` group names a payload file.

    Hand-built rather than compiled, because ``add_image`` cannot produce the
    names these tests are about — a name that is not a plain file name, one that
    collides with a zarr metadata document, one naming nothing at all — and the
    copy reads the attr, not the writer that stamped it.
    """
    root = open_group(path, mode="w")
    root.attrs["type"] = "scene"
    create_array(
        root,
        "x",
        data=np.arange(5_000, dtype=np.float32),
        chunks=(100,),
        compressor=None,
    )
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["type"] = "overlay_image"
    logo.attrs["image_file"] = filename
    if payload is not None:
        write_raw_bytes(logo, filename, payload)
    consolidate(root)
    return path


class TestPayloadFiles:
    def test_a_payload_name_that_is_not_a_plain_file_name_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """A name addressing a key outside the group's own directory must be
        refused by the copy, not resolved: the hasher folds such a name in
        without ever reading it, so a copy that wrote it would put bytes at a
        path the digest says nothing about — and one directory up from where the
        attr claims they are."""
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", "../evil.png", payload=None
        )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        attrs = dict(open_group(dst, mode="r")["overlays/logo"].attrs)
        assert attrs["image_file"] == "../evil.png"
        assert not any(p.name == "evil.png" for p in dst.rglob("*"))

    def test_a_payload_named_exactly_like_a_metadata_document_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """The OTHER class ``_is_safe_payload_name`` refuses, and the one the
        skip notice used to mis-describe as "not a plain file name". Reading such
        a name would fold in the very ``content_hash`` the walk is stamping, and
        writing it would replace the document the node is read through — so the
        copy must skip it and leave the output's own document intact, which is
        what reading the group's attrs back proves."""
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", ".zattrs", payload=None
        )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        attrs = dict(open_group(dst, mode="r")["overlays/logo"].attrs)
        assert attrs["image_file"] == ".zattrs"
        assert attrs["type"] == "overlay_image"

    def test_a_payload_named_like_a_metadata_document_is_refused(
        self, tmp_path: Path
    ) -> None:
        """``_is_safe_payload_name`` compares zarr's document names exactly,
        which is sound for a read gate and a clobber primitive for a write one:
        on a case-insensitive filesystem ``Zarr.json`` IS the group's own
        ``zarr.json``, so copying a payload under that name overwrites the
        document the whole store is read through. On a case-SENSITIVE one it is
        an ordinary distinct file the source really holds, and skipping it
        shipped — under exit code 0 — the very state the copy exists to prevent:
        an ``image_file`` attr naming a file the output does not have."""
        if _is_case_insensitive_fs(tmp_path):
            pytest.skip("a case-insensitive filesystem cannot hold the fixture")
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", "Zarr.json", payload=None
        )
        (src / "overlays" / "logo" / "Zarr.json").write_bytes(_TINY_PNG)
        dst = tmp_path / "out.luxar.zarr"
        with pytest.raises(ValueError, match=r"Zarr\.json"):
            optimise_store(src, dst, verify=True)
        assert not dst.exists()
        # The source's own node document survived — asserted by READING the group
        # back rather than by naming the document, which only exists as
        # `zarr.json` at format 3 (the v2 pair is `.zgroup`/`.zattrs`, so the
        # literal made this the one format-blind test in the file).
        assert dict(open_group(src, mode="r")["overlays/logo"].attrs) == {
            "type": "overlay_image",
            "image_file": "Zarr.json",
        }

    def test_a_dangling_attr_naming_a_metadata_document_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """Refusing is about BYTES the copy cannot carry faithfully. With no
        file behind the attr there are none, so a metadata-like name must not
        strand a store that is merely missing its overlay — the output is
        exactly as complete as its input. ``.ZMetadata`` is absent from a
        subgroup in both zarr formats, so this branch stays covered on every
        filesystem."""
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", ".ZMetadata", payload=None
        )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        attrs = dict(open_group(dst, mode="r")["overlays/logo"].attrs)
        assert attrs["image_file"] == ".ZMetadata"
        assert attrs["type"] == "overlay_image"
        assert not any(p.name == ".ZMetadata" for p in dst.rglob("*"))

    def test_a_dangling_metadata_document_name_is_skipped_under_a_case_fold(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_zarr_format: None,
    ) -> None:
        """The same dangling attr as above, on a case-INSENSITIVE filesystem —
        the platforms this whole branch exists for, and the ones CI never runs
        on. There, an open-by-name for ``Zarr.json`` is resolved by the OS onto
        the group's own ``zarr.json``, so a read can never report "no such key"
        for precisely the names whose absence decides skip-vs-refuse: the branch
        was dead on macOS/Windows, every store of this shape was refused, and the
        refusal quoted the metadata document's byte count as the payload's.

        The fold is simulated at the narrowest seam that produces it — the copy's
        own byte reader, made to resolve a name case-blind exactly as the
        filesystem would. The fix decides the branch from a directory LISTING
        instead, which no filesystem folds, so the skip must survive this patch.
        """
        # Pinned to format 3, the one whose node document IS named `zarr.json` —
        # at format 2 the pair is `.zgroup`/`.zattrs` and there is nothing for
        # this name to fold onto, so the fixture would not pose the question.
        set_zarr_format(3)
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", "Zarr.json", payload=None
        )
        exact_read = optimise_mod.read_raw_bytes

        def case_insensitive_read(group: Any, filename: str) -> bytes | None:
            """What macOS does: fall back to a case-blind match on the key."""
            found = exact_read(group, filename)
            if found is not None:
                return found
            for key in optimise_mod.list_raw_keys(group):
                if key.lower() == filename.lower():
                    return exact_read(group, key)
            return None

        monkeypatch.setattr(optimise_mod, "read_raw_bytes", case_insensitive_read)
        # The patch really does fold, or the test proves nothing.
        logo = open_group(src, mode="r")["overlays/logo"]
        assert case_insensitive_read(logo, "Zarr.json") is not None

        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        attrs = dict(open_group(dst, mode="r")["overlays/logo"].attrs)
        assert attrs["image_file"] == "Zarr.json"
        assert attrs["type"] == "overlay_image"
        assert not any(p.name == "Zarr.json" for p in dst.rglob("*"))

    def test_a_metadata_document_name_held_by_a_directory_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """:func:`list_raw_keys` reports every immediate child key — subgroups,
        arrays and plain subdirectories, not only files. So a case-shifted
        metadata-document name that is a DIRECTORY lists like a payload while
        holding no bytes at all, and refusing over it would advise renaming a
        file that does not exist to save bytes that do not exist. The read the
        listing gates is what tells the two apart: it comes back empty-handed for
        a directory, and the pass takes the same skip as for a dangling attr.
        Case-sensitive filesystems only, as above."""
        probe = tmp_path / "CaseProbe"
        probe.write_text("x")
        if (tmp_path / "caseprobe").exists():
            pytest.skip("a case-insensitive filesystem cannot hold the fixture")
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", "Zarr.json", payload=None
        )
        (src / "overlays" / "logo" / "Zarr.json").mkdir()
        # The fixture really does pose the question: the name lists, so only the
        # read outcome can separate it from the refusable shape.
        logo = open_group(src, mode="r")["overlays/logo"]
        assert "Zarr.json" in optimise_mod.list_raw_keys(logo)

        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        attrs = dict(open_group(dst, mode="r")["overlays/logo"].attrs)
        assert attrs["image_file"] == "Zarr.json"
        assert attrs["type"] == "overlay_image"
        assert not any(p.name == "Zarr.json" for p in dst.rglob("*"))

    @pytest.mark.parametrize(
        "payload",
        [b"\x89PNG\r\n\x1a\n" + b"p" * 4242, b""],
        ids=["with-bytes", "zero-byte"],
    )
    def test_the_refusal_reports_the_payload_file_s_own_byte_count(
        self, tmp_path: Path, payload: bytes
    ) -> None:
        """The count in the refusal is the user's evidence that something real is
        at stake, so the branch has to still fire for a genuine distinct file and
        quote THAT file's own length — not the node document's, which would
        advise renaming a file of a size that exists nowhere. What it pins is the
        pairing: the listing must find the payload's exact spelling (an inverted
        or empty or lowercasing :func:`list_raw_keys` all fail here) and the
        length must be the one read back under it. Case-sensitive filesystems
        only: elsewhere the fixture's two keys are one key.

        The zero-byte case additionally pins that the copy's skip test is ``is
        None`` and not truthiness: an empty file reads back as ``b""``, so under
        ``if not held`` a real, distinct, EMPTY ``Zarr.json`` ships under exit
        code 0 — and on macOS writing it truncates the node document to nothing,
        which is the whole failure class this branch exists to prevent."""
        probe = tmp_path / "CaseProbe"
        probe.write_text("x")
        if (tmp_path / "caseprobe").exists():
            pytest.skip("a case-insensitive filesystem cannot hold the fixture")
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", "Zarr.json", payload=None
        )
        logo_dir = src / "overlays" / "logo"
        (logo_dir / "Zarr.json").write_bytes(payload)
        # Named by SET rather than by literal, so the assertion holds at either
        # on-disk format (v3 has `zarr.json`, v2 the `.zgroup`/`.zattrs` pair).
        # The count the refusal quotes is the one read back under the payload's
        # own name, and no node document here shares that length.
        documents = [p for p in logo_dir.iterdir() if p.name in _META_DOCS]
        assert documents and all(p.stat().st_size != len(payload) for p in documents)

        dst = tmp_path / "out.luxar.zarr"
        with pytest.raises(ValueError, match=rf"dropping the {len(payload)} bytes"):
            optimise_store(src, dst, verify=True)
        assert not dst.exists()

    def test_a_payload_the_source_does_not_have_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """An attr naming a file that is already gone must not strand the store:
        the source has no bytes to hand over, so refusing would leave a store
        nobody can ever re-chunk, and the output is as complete as its input."""
        src = _store_with_a_payload_attr(
            tmp_path / "src.luxar.zarr", "gone.png", payload=None
        )
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        attrs = dict(open_group(dst, mode="r")["overlays/logo"].attrs)
        assert attrs["image_file"] == "gone.png"
        assert not (dst / "overlays" / "logo" / "gone.png").exists()

    @pytest.mark.skipif(
        getattr(os, "geteuid", lambda: 1)() == 0,
        reason="root ignores the mode bits this test relies on",
    )
    def test_a_payload_that_cannot_be_read_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        """The hasher degrades an unreadable payload to a deterministic term, so
        the very store that compiles cleanly can reach the copy — which refuses,
        because shipping a re-chunked store whose overlay silently vanished is
        the failure this pass exists to prevent. The refusal has to name the
        file and the way out; a bare ``PermissionError`` traceback names neither
        and leaves the user with no next move."""
        src = _store_with_a_payload_attr(tmp_path / "src.luxar.zarr", "logo.png")
        (src / "overlays" / "logo" / "logo.png").chmod(0o000)
        dst = tmp_path / "out.luxar.zarr"
        with pytest.raises(ValueError, match=r"logo\.png"):
            optimise_store(src, dst)
        assert not dst.exists()
        assert not any(p.name.startswith(".out.luxar.zarr") for p in tmp_path.iterdir())

    def test_a_payload_survives_a_zipped_destination(self, tmp_path: Path) -> None:
        """A ``.zarr.zip`` output is compressed out of the staging directory, so
        a payload written anywhere but there — or after the archive is sealed —
        is silently absent from the published store, and ``--verify`` has to
        read it back through a ``ZipStore`` rather than a directory."""
        src = _store_with_a_payload_attr(tmp_path / "src.luxar.zarr", "logo.png")
        dst = tmp_path / "out.luxar.zarr.zip"
        optimise_store(src, dst, verify=True)
        with zipfile.ZipFile(dst) as archive:
            assert "overlays/logo/logo.png" in archive.namelist()
            assert archive.read("overlays/logo/logo.png") == _TINY_PNG

    def test_verify_counts_the_payload_files_it_compared(self, tmp_path: Path) -> None:
        """The count is what the run reports, and it used to be a blanket claim
        about "every payload file" — printed identically for the vast majority
        of stores that have none at all, and for one whose only payload the
        verify walk skipped by construction."""
        src = _store_with_a_payload_attr(tmp_path / "src.luxar.zarr", "logo.png")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, verify=True)
        counts = _verify(open_group(src, mode="r"), open_group(dst, mode="r"))
        assert counts == (1, 1)

    def test_verify_catches_a_truncated_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--verify`` walked arrays only, so the one thing the payload copy
        writes sat outside the byte-for-byte promise the run prints. Every real
        path through the copy hands the source bytes over verbatim, so the
        damage is provoked by monkeypatching the narrowest thing that can
        produce a short write — the copy's own byte writer."""
        src = _store_with_a_payload_attr(tmp_path / "src.luxar.zarr", "logo.png")
        write = optimise_mod.write_raw_bytes
        monkeypatch.setattr(
            optimise_mod,
            "write_raw_bytes",
            lambda group, key, payload: write(group, key, payload[:-1]),
        )
        dst = tmp_path / "out.luxar.zarr"
        with pytest.raises(ValueError, match=r"payload 'logo\.png'.*differs"):
            optimise_store(src, dst, verify=True)
        assert not dst.exists()

    def test_verify_catches_a_payload_missing_from_the_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dropped-overlay case, which is what the copy exists to prevent
        and therefore the one ``--verify`` must not take on trust. Monkeypatched
        at the same single seam as the truncation above."""
        src = _store_with_a_payload_attr(tmp_path / "src.luxar.zarr", "logo.png")
        monkeypatch.setattr(
            optimise_mod, "write_raw_bytes", lambda group, key, payload: None
        )
        dst = tmp_path / "out.luxar.zarr"
        with pytest.raises(ValueError, match=r"payload 'logo\.png'.*is missing"):
            optimise_store(src, dst, verify=True)
        assert not dst.exists()


# --------------------------------------------------------------------------
# The alignment invariant (the mutation-check target)
# --------------------------------------------------------------------------


class TestAlignment:
    def test_every_emitted_chunk_is_a_multiple_of_its_atom(
        self, scene: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(scene, dst)
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
        self, scene: Path, tmp_path: Path, profile: str
    ) -> None:
        dst = tmp_path / f"out_{profile}.luxar.zarr"
        optimise_store(
            scene, dst, target_bytes=CHUNK_PROFILES[profile], profile=profile
        )
        assert assert_atom_aligned(dst) > 0
        assert_values_identical(scene, dst)

    def test_a_chunk_is_never_smaller_than_one_atom(self, tmp_path: Path) -> None:
        """A tiny byte target must not round an indexed chunk down to zero
        atoms — that would put a partition's rows across two zarr chunks.

        The floor is only OBSERVABLE on a source chunked BELOW one atom. Every
        indexed array in a compiled scene already holds at least one atom, so
        the earlier version of this test — which walked ``scene`` at
        ``target_bytes=64`` and asserted inside ``if a.atom and a.rechunked`` —
        never executed its own assertion: measured, all four arrays came back
        ``spatial index`` / ``already at or above target`` / ``broadcast``, zero
        reached the loop body, and deleting the ``max(1, ideal // atom)`` floor
        in ``_atom_aligned_rows`` left the whole file green.
        """
        atom = 1000
        n = 10_000
        src = tmp_path / "sub_atom.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        root.attrs["chunk_size"] = atom
        # The proof array — without it the `chunk_size` attr is vestigial and
        # `_resolve_atom` correctly refuses to treat it as a grid.
        create_array(
            root,
            "chunk_bounds",
            data=np.zeros((n // atom, 3, 2), dtype=np.float32),
            chunks=(n // atom, 3, 2),
            compressor=None,
        )
        create_array(
            root,
            "positions",
            data=_rng(3).random((n, 3)).astype(np.float32),
            chunks=(100, 3),  # a tenth of one atom
            compressor=None,
        )
        consolidate(root)

        plan = plan_optimisation(open_group(src, mode="r"), target_bytes=64)
        (pos,) = [a for a in plan.arrays if a.path == "positions"]
        assert pos.atom == atom
        # Without the floor the 64-byte budget asks for 5 rows, which rounds to
        # zero atoms; the array would then be left alone rather than raised.
        assert pos.rechunked, "the sub-atom chunk was not raised to one atom"
        assert pos.target_chunks == (atom, 3), "the atom floor did not hold"

        dst = tmp_path / "out.zarr"
        optimise_store(src, dst, target_bytes=64, verify=True)
        assert assert_atom_aligned(dst) > 0


# --------------------------------------------------------------------------
# Format preservation
# --------------------------------------------------------------------------


class TestFormatPreservation:
    def test_zarr_format_is_preserved(
        self,
        formatted_scene: tuple[int, Path],
        tmp_path: Path,
        restore_zarr_format: None,
    ) -> None:
        fmt, src = formatted_scene
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

    @pytest.mark.parametrize("fmt", [2, 3])
    def test_a_compressed_arrays_codec_config_survives(
        self, tmp_path: Path, fmt: int, restore_zarr_format: None
    ) -> None:
        """The RAW case above is only half of it, and the cheaper half.

        Real Luxar arrays carry a MEASURED Blosc zstd-9 policy with a per-dtype
        ``typesize`` and shuffle, and zarr's ``"auto"`` silently replaces it —
        with Blosc/lz4/clevel-5 at format 2 and a plain zstd level 0 at format 3.
        ``--verify`` compares values, which survive either way, so nothing else
        here would notice: swapping ``compressor=source.compressors`` for
        ``"auto"`` left all 80 tests green. Hence an assertion on the STORED
        codec configuration.
        """
        from numcodecs import Blosc

        set_zarr_format(fmt)
        src = tmp_path / "zstd.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "z",
            data=_rng(21).random((50_000, 4)).astype(np.float32),
            chunks=(1000, 4),
            compressor=Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE),
        )
        consolidate(root)
        source = open_group(src, mode="r")["z"]

        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, verify=True)
        assert plan.n_rechunked == 1
        out = open_group(dst, mode="r")["z"]
        assert _codec_configs(out) == _codec_configs(source)
        # Stated outright, so the test still says what it protects if the
        # source's own policy ever changes underneath it.
        (config,) = _codec_configs(out)
        config = config.get("configuration", config)
        assert config["cname"] == "zstd"
        assert config["clevel"] == 9
        assert_values_identical(src, dst)

    def test_filters_fill_value_and_memory_order_survive(
        self, tmp_path: Path, restore_zarr_format: None
    ) -> None:
        """Named by the changelog and both READMEs, and all three are silent on
        loss: a dropped filter writes codes nobody can invert, a changed
        ``fill_value`` invents data for an unwritten chunk, and a flipped memory
        ``order`` reads a transposed array."""
        from numcodecs import Delta

        set_zarr_format(2)  # `order` is stored metadata at v2, not a runtime knob
        src = tmp_path / "codecs2.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "coded",
            data=np.arange(200_000, dtype=np.int32).reshape(50_000, 4),
            chunks=(1000, 4),
            compressor=None,
            filters=[Delta(dtype="int32")],
            fill_value=7,
            order="F",
        )
        consolidate(root)

        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, verify=True)
        assert plan.n_rechunked == 1
        out = open_group(dst, mode="r")["coded"]
        assert [f.get_config()["id"] for f in out.filters] == ["delta"]
        assert int(out.fill_value) == 7
        assert out.order == "F"
        assert_values_identical(src, dst)

    def test_the_chunk_key_layout_is_preserved(
        self, tmp_path: Path, restore_zarr_format: None
    ) -> None:
        """A v2 store written with the NESTED ``/`` separator must not come out
        flat: that layout was chosen for exactly the per-directory pressure this
        pass exists to relieve."""
        from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

        set_zarr_format(2)
        src = tmp_path / "nested.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        arr = root.create_array(
            "y",
            shape=(50_000, 2),
            dtype="f4",
            chunks=(1000, 2),
            chunk_key_encoding=V2ChunkKeyEncoding(separator="/"),
        )
        arr[:] = _rng(2).random((50_000, 2)).astype(np.float32)
        consolidate(root)
        assert (src / "y" / "0" / "0").is_file(), "the fixture is not nested"

        dst = tmp_path / "out.zarr"
        optimise_store(src, dst, verify=True)
        out = open_group(dst, mode="r")["y"]
        assert out.metadata.dimension_separator == "/"
        assert (dst / "y" / "0" / "0").is_file()
        assert_values_identical(src, dst)

    def test_the_v3_chunk_key_layout_is_preserved(
        self, tmp_path: Path, restore_zarr_format: None
    ) -> None:
        """The v3 half of the same contract, and the untested one: deleting the
        ``chunk_key_encoding`` forwarding left all 80 tests green while
        demonstrably renaming every object on disk (``c.0.0`` → ``c/0/0``) — a
        client that knows the old keys 404s on every chunk."""
        from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

        set_zarr_format(3)
        src = tmp_path / "keys.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        arr = root.create_array(
            "y",
            shape=(50_000, 2),
            dtype="f4",
            chunks=(1000, 2),
            chunk_key_encoding=DefaultChunkKeyEncoding(separator="."),
        )
        arr[:] = _rng(8).random((50_000, 2)).astype(np.float32)
        consolidate(root)
        assert (src / "y" / "c.0.0").is_file(), "the fixture is not dot-separated"

        dst = tmp_path / "out.zarr"
        optimise_store(src, dst, verify=True)
        out = open_group(dst, mode="r")["y"]
        assert out.metadata.chunk_key_encoding.separator == "."
        assert (dst / "y" / "c.0.0").is_file()
        assert not (dst / "y" / "c").is_dir(), "the objects were renamed"
        assert_values_identical(src, dst)

    def test_dimension_names_are_forwarded(
        self, tmp_path: Path, restore_zarr_format: None
    ) -> None:
        set_zarr_format(3)
        src = tmp_path / "named.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        arr = root.create_array(
            "v",
            shape=(50_000, 3),
            dtype="f4",
            chunks=(1000, 3),
            dimension_names=("element", "axis"),
        )
        arr[:] = _rng(6).random((50_000, 3)).astype(np.float32)
        consolidate(root)
        dst = tmp_path / "out.zarr"
        optimise_store(src, dst, verify=True)
        out = open_group(dst, mode="r")["v"]
        assert tuple(out.metadata.dimension_names) == ("element", "axis")


# --------------------------------------------------------------------------
# Sharding — a v3 array whose objects are shards, not chunks
# --------------------------------------------------------------------------


class TestSharding:
    def _sharded_store(self, path: Path) -> Path:
        root = open_group(path, mode="w", zarr_format=3)
        root.attrs["kind"] = "leaf"
        arr = root.create_array(
            "x",
            shape=(200_000, 4),
            dtype="f4",
            chunks=(1000, 4),
            shards=(50_000, 4),
        )
        arr[:] = _rng(1).random((200_000, 4)).astype(np.float32)
        consolidate(root)
        return path

    def test_a_sharded_array_keeps_its_shard_grid(self, tmp_path: Path) -> None:
        """Recreating a sharded array from ``chunks`` alone promotes its INNER
        chunk shape to the top level: 5 files became 201, a 40x INCREASE in
        round trips, while the plan reported "unchanged"."""
        src = self._sharded_store(tmp_path / "sharded.zarr")
        before_files = _chunk_files(src)
        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, verify=True)

        (x,) = plan.arrays
        assert x.skip_reason == "sharded"
        out = open_group(dst, mode="r")["x"]
        assert tuple(out.shards) == (50_000, 4), "the shard grid was dropped"
        assert tuple(out.chunks) == (1000, 4)
        assert _chunk_files(dst) == before_files
        assert_values_identical(src, dst)

    def test_the_plan_counts_shards_not_inner_chunks(self, tmp_path: Path) -> None:
        src = self._sharded_store(tmp_path / "sharded.zarr")
        root = open_group(src, mode="r")
        (x,) = plan_optimisation(root).arrays
        # 200000/50000 = 4 objects, not 200000/1000 = 200 nominal grid cells.
        assert x.source_n_chunks == 4
        assert x.target_n_chunks == 4
        assert summarise_chunk_layout(root).n_chunks == 4


# --------------------------------------------------------------------------
# Slab walking — the bounded-memory copy, actually iterating
#
# Mutation-proved gap: truncating `_copy_array`'s slab loop to its first
# iteration left all 80 tests green, and so did the same cut in
# `_verify_values`. Nothing anywhere fed either one an array larger than the
# 64 MB budget, so the budget's entire purpose was untested at both ends. These
# shrink the budget instead of building a 64 MB fixture.
# --------------------------------------------------------------------------


class TestSlabWalking:
    def test_the_copy_walks_every_slab(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A copy that stops after one slab writes a store whose tail is
        ``fill_value`` — silently, because the shapes and the metadata are all
        correct."""
        monkeypatch.setattr(optimise_mod, "_SLAB_BYTES", 4096)
        n = 100_000
        src = tmp_path / "slabs.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        # Nowhere zero, so an unwritten tail cannot pass as fill_value.
        create_array(
            root,
            "v",
            data=_rng(17).integers(1, 255, n, dtype=np.uint8),
            chunks=(512,),
            compressor=None,
        )
        consolidate(root)

        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, target_bytes=1024, verify=True)
        (v,) = plan.arrays
        assert v.target_chunks == (1024,)
        # Slabs are chunk-aligned: 1024 rows x (4096 // 1024) = 4096 per slab.
        slab_rows = 1024 * (4096 // 1024)
        assert n // slab_rows >= 20, "one slab covers the array — test is vacuous"
        assert assert_values_identical(src, dst) == 1

    def test_verify_walks_every_slab(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``--verify`` that stops after one slab is a pass with no coverage of
        the other 95% of the array."""
        n = 20_000
        src, dst = _pair(tmp_path, a=np.arange(n, dtype=np.float32))
        dst["a"][n - 1] = -1.0
        monkeypatch.setattr(optimise_mod, "_SLAB_BYTES", 4096)  # 1024 rows
        # The damage is in the LAST slab, so only a loop that reaches it fails.
        with pytest.raises(ValueError, match="differs at rows 19456:20000"):
            _verify(src, dst)


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

    @pytest.mark.parametrize("flavour", ["structured", "vlen_string"])
    def test_a_dtype_that_str_cannot_round_trip_is_still_planned(
        self, tmp_path: Path, flavour: str, restore_zarr_format: None
    ) -> None:
        """``ArrayPlan`` recorded its dtype as ``str(np.dtype(...))`` and re-parsed
        it to reach the itemsize — which ``np.dtype`` REJECTS for a structured
        dtype (``"[('a', '<i4'), ('b', '<f8')]"``) and for numpy's variable-width
        ``StringDType`` (``"StringDType()"``).

        Both are ordinary in the third-party stores ``--generic`` exists for (an
        AnnData/cellxgene ``.zarr``, an OME-Zarr label table), and the ``TypeError``
        surfaced on the two READ-ONLY paths only: ``luxar info --stats`` exited 1
        on a store its own ``--format json`` path printed fine, and ``optimise
        --dry-run`` died with a traceback while the real copy of the same store
        succeeded — a dry run strictly less capable than the run it predicts.
        """
        set_zarr_format(3)
        src = tmp_path / f"{flavour}.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        if flavour == "structured":
            create_array(
                root,
                "table",
                data=np.zeros(50_000, dtype=[("a", "<i4"), ("b", "<f8")]),
                chunks=(1000,),
                compressor=None,
            )
        else:
            labels = root.create_array(
                "table", shape=(50_000,), dtype=str, chunks=(1000,)
            )
            labels[:] = np.array(["label"] * 50_000)
        create_array(
            root,
            "plain",
            data=np.arange(50_000, dtype=np.float32),
            chunks=(1000,),
            compressor=None,
        )
        consolidate(root)

        # The plan's byte properties are what `--dry-run` and `--stats` read.
        plan = plan_optimisation(open_group(src, mode="r"), target_bytes=65_536)
        for a in plan.arrays:
            assert a.itemsize > 0, a.path
            assert a.target_chunk_bytes > 0, a.path
            assert a.source_file_bytes > 0, a.path
        assert summarise_chunk_layout(open_group(src, mode="r")).n_arrays == 2

        assert _run(str(src), "--dry-run").exit_code == 0
        assert _info(str(src), "--stats").exit_code == 0

        dst = tmp_path / "out.zarr"
        optimise_store(src, dst, verify=True)
        assert_values_identical(src, dst)

    def test_a_skip_reason_does_not_claim_more_than_it_knows(
        self, tmp_path: Path
    ) -> None:
        """``--dry-run`` groups its "Left alone" tally by ``skip_reason``, so a
        reason is a claim about the user's data.

        Two were wrong for a foreign store. ``(1, 1, Z, Y, X)`` — the canonical
        OME-Zarr level shape, 512 chunk files here — was reported as
        ``broadcast``, which is a statement about a Luxar ENCODING, not a shape;
        and ``single chunk`` was reported for an array chunked on trailing axes,
        which is 64 files. This pass merges rows and nothing else, so what both
        establish is only that axis 0 is already one chunk.
        """
        src = tmp_path / "trailing.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "level",
            data=np.zeros((1, 1, 8, 64, 64), dtype=np.uint16),
            chunks=(1, 1, 8, 32, 32),
            compressor=None,
        )
        create_array(
            root,
            "tiles",
            data=np.zeros((8, 128, 128), dtype=np.uint16),
            chunks=(8, 32, 32),
            compressor=None,
        )
        consolidate(root)

        plan = plan_optimisation(open_group(src, mode="r"))
        reasons = {a.path: a.skip_reason for a in plan.arrays}
        assert reasons == {
            "level": "rows already in one chunk",
            "tiles": "rows already in one chunk",
        }
        # Both really do hold more than one chunk file, which is why neither may
        # be called "single chunk".
        assert {a.path: a.source_n_chunks for a in plan.arrays} == {
            "level": 4,
            "tiles": 16,
        }

    def test_scalar_and_empty_arrays_round_trip(self, tmp_path: Path) -> None:
        """The two skip reasons an ``array_ref`` shadows in a compiled scene: a
        0-d array has no rows to slab, and a ``(0, D)`` one writes no chunk at
        all — which the request count must report as zero files, not one."""
        src = tmp_path / "degenerate.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "scalar",
            data=np.array(3.5, dtype=np.float32),
            chunks=(),
            compressor=None,
        )
        create_array(
            root,
            "empty",
            data=np.zeros((0, 3), dtype=np.float32),
            chunks=(1, 3),
            compressor=None,
        )
        consolidate(root)

        dst = tmp_path / "out.zarr"
        plan = optimise_store(src, dst, verify=True)
        assert {a.path: a.skip_reason for a in plan.arrays} == {
            "scalar": "scalar array",
            "empty": "empty",
        }
        assert {a.path: a.source_n_chunks for a in plan.arrays} == {
            "scalar": 1,
            "empty": 0,
        }
        out = open_group(dst, mode="r")
        assert float(out["scalar"][...]) == 3.5
        assert tuple(out["empty"].shape) == (0, 3)

    def test_a_bounds_array_reports_no_atom(self, scene: Path) -> None:
        """A bounds array is indexed by CHUNK, so it is on no element grid.

        ``_atom_candidate`` special-cased only the name ``segments``, so
        ``segment_chunk_bounds`` fell through to the per-vertex branch and
        ``ArrayPlan.atom`` reported the node's VERTEX atom (3276 where the
        segment grid is 4096) — wrong on a public field, and inert today only
        because the structural skip fires before the atom is used.
        """
        plan = plan_optimisation(open_group(scene, mode="r"))
        bounds = [a for a in plan.arrays if a.path.rsplit("/", 1)[-1] in _INDEX_ARRAYS]
        assert bounds, "the fixture carries no spatial index"
        assert {a.path.rsplit("/", 1)[-1] for a in bounds} >= {
            "vertex_chunk_bounds",
            "segment_chunk_bounds",
        }, "the Lines node's two bounds arrays are what made this wrong"
        for a in bounds:
            assert a.atom is None, a.path
            assert a.skip_reason == "spatial index"

    def test_bounds_arrays_are_never_rechunked(self, scene: Path) -> None:
        root = open_group(scene, mode="r")
        plan = plan_optimisation(root, target_bytes=CHUNK_PROFILES["archive"])
        bounds = [a for a in plan.arrays if a.path.rsplit("/", 1)[-1] in _INDEX_ARRAYS]
        assert bounds, "the fixture grew no spatial index"
        for a in bounds:
            assert a.skip_reason == "spatial index"
            assert a.target_chunks == a.source_chunks

    def test_a_store_with_no_spatial_index_falls_back_to_bytes(
        self, scene_without_index: Path, tmp_path: Path
    ) -> None:
        assert not any(
            "chunk_bounds" in p
            for p in _arrays(open_group(scene_without_index, mode="r"))
        ), "the fixture grew a spatial index"
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(scene_without_index, dst, verify=True)
        assert plan.n_rechunked > 0
        # No bounds array anywhere, so no node has a trustworthy atom — the
        # vestigial `chunk_size` a gsplat leaf gets under `ordering="none"`
        # included.
        assert all(a.atom is None for a in plan.arrays)
        assert plan.target_n_chunks < plan.source_n_chunks
        assert_values_identical(scene_without_index, dst)

    def test_a_chunk_size_attr_the_arrays_ignore_is_not_treated_as_an_atom(
        self, tmp_path: Path
    ) -> None:
        """A gsplat leaf written with ``ordering="none"`` still gets a default
        ``chunk_size`` stamped, and its arrays are NOT on that grid. Trusting it
        would inflate every chunk to a boundary that indexes nothing.

        The value is the one the writer actually emits — ``min(1024, max(64,
        n_splats))``, a power of two — over a source chunk that is a whole
        multiple of it, which is exactly the coincidence an "or the array
        already follows the grid" fallback mistook for proof. Only the missing
        ``chunk_bounds`` array distinguishes the two, and it is the only thing
        that may.
        """
        src = tmp_path / "vestigial.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        root.attrs["chunk_size"] = 1024
        create_array(
            root,
            "amplitudes",
            data=np.arange(500_000, dtype=np.uint8),
            chunks=(2048,),  # a whole multiple of the vestigial "atom"
            compressor=None,
        )
        consolidate(root)
        dst = tmp_path / "out.zarr"
        # 100_000 bytes is deliberately NOT a multiple of 1024: an atom-aligned
        # chunk would round down to 99_328 rows, so the two answers differ.
        plan = optimise_store(src, dst, target_bytes=100_000, verify=True)
        (amps,) = [a for a in plan.arrays if a.path == "amplitudes"]
        assert amps.atom is None
        assert amps.target_chunks == (100_000,)

    def test_a_zipped_store_round_trips_both_ways(
        self, scene: Path, tmp_path: Path
    ) -> None:
        """A ``.zarr.zip`` owns a file handle and only writes a valid archive on
        close — without one the output re-read as "File is not a zip file"."""
        shutil.make_archive(str(tmp_path / "src.luxar.zarr"), "zip", str(scene))
        src_zip = tmp_path / "src.luxar.zarr.zip"
        assert src_zip.is_file()

        from_zip = optimise_store(
            src_zip, tmp_path / "from_zip.luxar.zarr", verify=True
        )
        assert from_zip.target_n_chunks < from_zip.source_n_chunks

        to_zip = optimise_store(scene, tmp_path / "out.luxar.zarr.zip", verify=True)
        assert to_zip.target_n_chunks < to_zip.source_n_chunks
        assert_values_identical(scene, tmp_path / "out.luxar.zarr.zip")

    def test_a_zipped_output_holds_each_member_once(
        self, scene: Path, tmp_path: Path
    ) -> None:
        """A ``ZipStore`` APPENDS, and zarr re-serializes a group document on
        every attr write and child creation. Writing the store directly into one
        produced 211 members for 50 unique names (26 copies of
        ``cloud/zarr.json``), 50 ``UserWarning: Duplicate name:`` — a hard
        failure under ``-W error`` — and left the FIRST copy of every group
        document as the pre-attrs stub, which a first-match unzipper reads as an
        empty scene.
        """
        dst = tmp_path / "out.luxar.zarr.zip"
        optimise_store(scene, dst)
        with zipfile.ZipFile(dst) as archive:
            names = archive.namelist()
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, f"{len(names)} members for {len(set(names))} names"
        assert any(n.endswith("zarr.json") or n.endswith(".zgroup") for n in names)
        assert_values_identical(scene, dst)

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

    def test_optimising_twice_is_a_fixed_point(
        self, scene: Path, tmp_path: Path
    ) -> None:
        once = tmp_path / "once.luxar.zarr"
        twice = tmp_path / "twice.luxar.zarr"
        optimise_store(scene, once)
        second = optimise_store(once, twice)
        assert second.n_rechunked == 0
        assert second.source_n_chunks == second.target_n_chunks

    def test_no_array_ever_gains_chunks(self, scene: Path) -> None:
        root = open_group(scene, mode="r")
        for target in (1024, 64 * 1024, 1024 * 1024):
            plan = plan_optimisation(root, target_bytes=target)
            for a in plan.arrays:
                assert a.target_n_chunks <= a.source_n_chunks, (a.path, target)


# --------------------------------------------------------------------------
# Cache invalidation
# --------------------------------------------------------------------------


#: Why a hash-equivalence failure below is never "just a flaky assert".
_HASH_DRIFT = (
    "the finalize-time content hash has CHANGED. "
    "`_compute_content_hashes_streaming` in luxar/io/optimise.py is a "
    "deliberate slab-wise copy of "
    "luxar/io/_compiler/finalize/hashing.py::compute_content_hashes (it exists "
    "only to avoid that one's whole-array materialisation), and it must be "
    "updated to match the new digest — term for term, including any shard or "
    "chunk-shape terms. Until it is, `luxar optimise` writes stores whose "
    "content_hash disagrees with the compiler's, and the viewer's cache "
    "validation compares exactly that field."
)


class TestCacheInvalidation:
    def test_the_scene_content_hash_changes(self, scene: Path, tmp_path: Path) -> None:
        """The bug this prevents is silent: the viewer's cache validates on
        ``content_hash``, so an unchanged hash means a warm client keeps
        serving chunks whose keys now cover different rows."""
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(scene, dst)
        before = dict(open_group(scene, mode="r").attrs)["content_hash"]
        after = dict(open_group(dst, mode="r").attrs)["content_hash"]
        assert before != after

    def test_generic_does_not_disable_the_guard_on_a_luxar_scene(
        self, scene: Path, tmp_path: Path
    ) -> None:
        """``--generic`` describes the INPUT, not the output's cache safety.
        Gating the restamp on it shipped a re-chunked scene under the SOURCE's
        ``content_hash`` — and the viewer's validation queue answers
        ``mode: 'content-hash'`` as soon as one exists and never falls back to a
        byte digest, so a warm OPFS cache serves chunks whose keys moved."""
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(scene, dst, generic=True)
        assert plan.n_rechunked > 0, "nothing moved — the test cannot fail"
        before = dict(open_group(scene, mode="r").attrs)["content_hash"]
        after = dict(open_group(dst, mode="r").attrs)["content_hash"]
        assert before != after

    def test_the_hash_moves_exactly_while_the_layout_does(
        self, scene: Path, tmp_path: Path
    ) -> None:
        """Run 2 re-chunks nothing and yet still moves the hash, because the
        ``chunk_layout`` attr records counts that changed (22 → 13 becomes
        13 → 13). Run 3 records the SAME layout over the same values, so it is a
        genuine fixed point — asserted rather than glossed, because "the hash
        always moves" is false from here on."""
        once = tmp_path / "once.luxar.zarr"
        twice = tmp_path / "twice.luxar.zarr"
        thrice = tmp_path / "thrice.luxar.zarr"
        optimise_store(scene, once)
        second = optimise_store(once, twice)
        assert second.n_rechunked == 0
        optimise_store(twice, thrice)

        def h(p: Path) -> str:
            return str(dict(open_group(p, mode="r").attrs)["content_hash"])

        assert h(once) != h(twice)
        assert h(twice) == h(thrice)

    def test_the_streaming_hash_is_byte_identical(
        self, scene: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The optimiser recomputes the scene hash slab-wise instead of calling
        ``compute_content_hashes``, which does ``dataset[:].tobytes()`` and peaks
        at twice an array's size. That is only sound if the digest is IDENTICAL
        — a divergent hash is a silent format break — so it is pinned here,
        every node, with the slab budget shrunk so multi-slab arrays are the
        common case rather than the exception."""
        from luxar.io._compiler.finalize.hashing import compute_content_hashes

        work = tmp_path / "hash.luxar.zarr"
        shutil.copytree(scene, work)
        root = open_group(work, mode="r+")

        monkeypatch.setattr(optimise_mod, "_SLAB_BYTES", 4096)
        streamed = _compute_content_hashes_streaming(root)
        per_node_streamed = {
            p: dict(n.attrs)["content_hash"] for p, n, a in _walk(root) if not a
        }
        reference = compute_content_hashes(root)
        per_node_reference = {
            p: dict(n.attrs)["content_hash"] for p, n, a in _walk(root) if not a
        }
        assert streamed == reference, _HASH_DRIFT
        assert per_node_streamed == per_node_reference, _HASH_DRIFT
        assert len(per_node_reference) > 1, "single-node tree — recursion untested"

    def test_the_streaming_hash_matches_on_a_sharded_array(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_zarr_format: None
    ) -> None:
        """Sharding is the term the compiled fixtures never exercise, and the
        one #1719 is in flight to fold into the finalize digest — so without an
        actual sharded array here, a shard-term divergence between the two
        implementations would land green."""
        from luxar.io._compiler.finalize.hashing import compute_content_hashes

        set_zarr_format(3)
        src = tmp_path / "sharded_hash.zarr"
        root = open_group(src, mode="w")
        root.attrs["type"] = "scene"
        rng = _rng(23)
        sharded = root.create_array(
            "sharded",
            shape=(20_000, 4),
            dtype="f4",
            chunks=(500, 4),
            shards=(5_000, 4),
        )
        sharded[:] = rng.random((20_000, 4)).astype(np.float32)
        create_array(
            root,
            "plain",
            data=rng.random((3_000, 2)).astype(np.float32),
            chunks=(300, 2),
            compressor=None,
        )
        child = root.create_group("nested")
        inner = child.create_array(
            "also_sharded",
            shape=(8_000, 3),
            dtype="f4",
            chunks=(200, 3),
            shards=(2_000, 3),
        )
        inner[:] = rng.random((8_000, 3)).astype(np.float32)
        consolidate(root)
        assert open_group(src, mode="r")["sharded"].shards == (5_000, 4)

        monkeypatch.setattr(optimise_mod, "_SLAB_BYTES", 4096)
        assert _compute_content_hashes_streaming(root) == compute_content_hashes(
            root
        ), _HASH_DRIFT

    def test_the_streaming_hash_matches_on_an_f_order_array(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_zarr_format: None
    ) -> None:
        """``ndarray.tobytes()`` is C-order whatever the memory order, which is
        what makes slab concatenation equal the whole-array bytes. An F-order
        array is the case that would expose it if that were not true."""
        from luxar.io._compiler.finalize.hashing import compute_content_hashes

        set_zarr_format(2)
        src = tmp_path / "orders.zarr"
        root = open_group(src, mode="w")
        root.attrs["type"] = "scene"
        rng = _rng(9)
        for name, order in (("c_order", "C"), ("f_order", "F")):
            create_array(
                root,
                name,
                data=rng.random((5_000, 4)).astype(np.float32),
                chunks=(500, 4),
                compressor=None,
                order=order,
            )
        child = root.create_group("nested")
        create_array(
            child,
            "vals",
            data=rng.random((3_000, 2)).astype(np.float32),
            chunks=(300, 2),
            compressor=None,
            order="F",
        )
        consolidate(root)

        monkeypatch.setattr(optimise_mod, "_SLAB_BYTES", 512)
        assert _compute_content_hashes_streaming(root) == compute_content_hashes(
            root
        ), _HASH_DRIFT

    def test_the_streaming_hash_matches_on_a_payload_file(self, tmp_path: Path) -> None:
        """A group's overlay image reaches the digest through neither
        ``array_keys()`` nor ``group_keys()`` — only the attrs-driven payload
        step folds its bytes. Without a payload file in the tree, a divergence
        in that term between the two walks lands green."""
        from luxar.io._compiler.finalize.hashing import compute_content_hashes

        root = open_group(tmp_path / "payload.luxar.zarr", mode="w")
        root.attrs["type"] = "scene"
        create_array(
            root,
            "vals",
            data=_rng(31).random((2_000, 3)).astype(np.float32),
            chunks=(200, 3),
            compressor=None,
        )
        logo = root.create_group("overlays").create_group("logo")
        logo.attrs["type"] = "overlay_image"
        logo.attrs["image_file"] = "image.png"
        write_raw_bytes(logo, "image.png", _TINY_PNG)

        streamed = _compute_content_hashes_streaming(root)
        assert streamed == compute_content_hashes(root), _HASH_DRIFT
        # Non-vacuity: the two must agree because both fold the bytes, not
        # because both ignore them. Same-length edit, so it is the bytes that
        # move the digest rather than a length prefix.
        write_raw_bytes(logo, "image.png", _TINY_PNG[:-1] + b"\x83")
        assert _compute_content_hashes_streaming(root) != streamed

    def test_the_streaming_hash_uses_case_exact_payload_presence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The re-chunk hasher must not reintroduce the case-folding read that
        the finalize-time hasher rejects for a dangling ``Zarr.json``."""
        from luxar.io._compiler.finalize import hashing

        root = open_group(tmp_path / "payload.luxar.zarr", mode="w")
        root.attrs["type"] = "scene"
        logo = root.create_group("overlays").create_group("logo")
        logo.attrs["type"] = "overlay_image"
        logo.attrs["image_file"] = "Zarr.json"
        expected = hashing.compute_content_hashes(root)

        original_read = hashing.read_raw_bytes
        folded_reads = 0

        def _case_folding_read(group: Any, key: str) -> bytes | None:
            nonlocal folded_reads
            if group.path == logo.path and key == "Zarr.json":
                folded_reads += 1
                return original_read(group, "zarr.json")
            return original_read(group, key)

        monkeypatch.setattr(hashing, "read_raw_bytes", _case_folding_read)

        assert _compute_content_hashes_streaming(root) == expected
        assert _compute_content_hashes_streaming(root) == expected
        assert folded_reads == 0

    def test_chunk_layout_is_stamped_on_the_root(
        self, scene: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.luxar.zarr"
        plan = optimise_store(
            scene, dst, target_bytes=CHUNK_PROFILES["hosting"], profile="hosting"
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
# Guard rails: the destination is someone's filesystem
# --------------------------------------------------------------------------


def _tiny_store(path: Path, *, luxar: bool = True) -> Path:
    """A minimal store — enough to be opened and planned, cheap to build."""
    root = open_group(path, mode="w")
    if luxar:
        root.attrs["kind"] = "leaf"
    create_array(
        root,
        "x",
        data=np.arange(10_000, dtype=np.float32),
        chunks=(100,),
        compressor=None,
    )
    consolidate(root)
    return path


class TestDestinationGuards:
    def test_in_place_is_refused(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        with pytest.raises(ValueError, match="in place"):
            optimise_store(src, src, overwrite=True)
        assert (src / "x").exists()

    def test_a_destination_containing_the_source_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The measured footgun: ``luxar optimise mydata/s.luxar.zarr mydata
        --overwrite`` deleted all of ``mydata/`` — the source and an unrelated
        file beside it — and only then raised ``FileNotFoundError``."""
        holder = tmp_path / "mydata"
        holder.mkdir()
        precious = holder / "IRREPLACEABLE.txt"
        precious.write_text("not a zarr store")
        src = _tiny_store(holder / "s.luxar.zarr")

        with pytest.raises(ValueError, match="CONTAINS the source"):
            optimise_store(src, holder, overwrite=True)
        assert precious.read_text() == "not a zarr store"
        assert (src / "x").exists()

    def test_a_destination_inside_the_source_is_refused(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        with pytest.raises(ValueError, match="inside the source"):
            optimise_store(src, src / "nested", overwrite=True)
        assert (src / "x").exists()

    def test_overwrite_refuses_a_destination_that_is_not_a_store(
        self, tmp_path: Path
    ) -> None:
        """``--overwrite`` means "replace an existing OUTPUT store", not "delete
        whatever is at this path"."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        occupied = tmp_path / "not_a_store"
        occupied.mkdir()
        (occupied / "thesis.tex").write_text("years of work")

        with pytest.raises(ValueError, match="neither a zarr store"):
            optimise_store(src, occupied, overwrite=True)
        assert (occupied / "thesis.tex").read_text() == "years of work"

    def test_the_source_is_validated_before_the_destination_is_touched(
        self, tmp_path: Path
    ) -> None:
        """``luxar optimise plain.zarr known-good.zarr --overwrite`` used to
        delete ``known-good.zarr`` and THEN refuse to work for want of
        ``--generic``."""
        plain = _tiny_store(tmp_path / "plain.zarr", luxar=False)
        known_good = _tiny_store(tmp_path / "known-good.luxar.zarr")
        before = sorted(p.name for p in known_good.rglob("*"))

        with pytest.raises(ValueError, match="--generic"):
            optimise_store(plain, known_good, overwrite=True)
        assert sorted(p.name for p in known_good.rglob("*")) == before

    def test_an_empty_directory_destination_is_accepted(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        dst.mkdir()
        optimise_store(src, dst, overwrite=True, verify=True)
        assert_values_identical(src, dst)

    def test_an_existing_output_needs_overwrite(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst)
        with pytest.raises(FileExistsError, match="--overwrite"):
            optimise_store(src, dst)
        optimise_store(src, dst, overwrite=True)
        assert_values_identical(src, dst)

    def test_an_existing_zipped_output_is_replaced(self, tmp_path: Path) -> None:
        """The FILE destination branch — plausible for a ``.zarr.zip`` output,
        and the one place the replace is an unlink rather than an rmtree."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr.zip"
        optimise_store(src, dst)
        assert dst.is_file()
        with pytest.raises(FileExistsError, match="--overwrite"):
            optimise_store(src, dst)
        optimise_store(src, dst, overwrite=True, verify=True)
        assert dst.is_file()
        assert_values_identical(src, dst)

    def test_a_plain_zarr_store_needs_generic(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "plain.zarr", luxar=False)
        with pytest.raises(ValueError, match="--generic"):
            optimise_store(src, tmp_path / "a.zarr")
        assert not (tmp_path / "a.zarr").exists()
        plan = optimise_store(src, tmp_path / "b.zarr", generic=True, verify=True)
        assert plan.n_rechunked == 1
        # `--generic` must not stamp scene semantics onto a foreign store.
        assert "content_hash" not in dict(
            open_group(tmp_path / "b.zarr", mode="r").attrs
        )

    @pytest.mark.parametrize(
        "attrs",
        [
            # Generic enough for a foreign store to carry, and neither names a
            # Luxar type/format/kind. `content_hash` was accepted on PRESENCE
            # alone, and so was any `kind` whatsoever.
            {"content_hash": "deadbeef"},
            {"kind": "dataset"},
            {"type": "AnnData"},
            {"format_type": "ome-zarr"},
            # Unhashable values must answer "not a marker", not raise.
            {"kind": ["leaf"]},
            {"type": {"name": "scene"}},
        ],
    )
    def test_a_foreign_marker_does_not_bypass_generic(
        self, tmp_path: Path, attrs: dict
    ) -> None:
        """The ``--generic`` gate tests recognized VALUES, not attr presence.

        Slipping past it is not a no-op: the output gets a ``chunk_layout`` attr
        and — for anything carrying a ``content_hash`` — that hash REPLACED under
        Luxar's hashing semantics, which is exactly the consequence the flag
        exists to make the caller ask for.
        """
        src = _tiny_store(tmp_path / "foreign.zarr", luxar=False)
        root = open_group(src, mode="a")
        root.attrs.update(attrs)
        consolidate(root)

        with pytest.raises(ValueError, match="--generic"):
            optimise_store(src, tmp_path / "a.zarr")
        assert not (tmp_path / "a.zarr").exists()
        # And the real markers still let a Luxar store through untouched.
        assert optimise_store(src, tmp_path / "b.zarr", generic=True).n_rechunked == 1

    @pytest.mark.parametrize(
        "attrs",
        [
            {"type": "scene"},
            {"type": "points"},
            {"format_type": "gsplats_zarr"},
            {"kind": "leaf"},
            {"kind": "lod"},
            {"kind": "partition"},
        ],
    )
    def test_every_real_luxar_marker_is_still_accepted(
        self, tmp_path: Path, attrs: dict
    ) -> None:
        """Tightening the gate must not start demanding ``--generic`` for a store
        Luxar itself wrote: a scene or scene-subtree root, a standalone
        ``.gsplats.zarr`` header, or any gsplat node kind."""
        src = _tiny_store(tmp_path / "luxarish.zarr", luxar=False)
        root = open_group(src, mode="a")
        root.attrs.update(attrs)
        consolidate(root)
        assert optimise_store(src, tmp_path / "out.zarr").n_rechunked == 1

    def test_a_failed_run_leaves_no_output_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection, not mocking: the behaviour under test is the
        CLEANUP. Before the staging rename, a mid-copy failure left a
        valid-looking zarr root with no arrays and no consolidated index at the
        user's chosen path — which the viewer renders as an empty scene, and
        which a retry then refuses to touch without ``--overwrite``."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        real_copy = optimise_mod._copy_array

        def explode(*args: Any, **kwargs: Any) -> Any:
            result = real_copy(*args, **kwargs)
            raise OSError("disk full")
            return result  # pragma: no cover - unreachable, documents intent

        monkeypatch.setattr(optimise_mod, "_copy_array", explode)
        with pytest.raises(OSError, match="disk full"):
            optimise_store(src, dst)
        assert not dst.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["src.luxar.zarr"]

    def test_a_failure_while_packaging_leaves_no_partial_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one window the all-or-nothing promise did not cover. The artifact
        path used to be bound from ``_package``'s RETURN, so an archive write
        that failed partway (ENOSPC on the last member) left a hidden
        ``.<name>.optimise-<pid>-<uuid>.zip`` beside the destination that the
        ``finally`` never saw."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr.zip"
        real_package = optimise_mod._package

        def explode(staging: Path, artifact: Path) -> None:
            real_package(staging, artifact)  # the archive now exists on disk
            raise OSError("disk full")

        monkeypatch.setattr(optimise_mod, "_package", explode)
        with pytest.raises(OSError, match="disk full"):
            optimise_store(src, dst)
        assert not dst.exists()
        # Hidden files included: the leak was a dotfile.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["src.luxar.zarr"]

    def test_a_failed_replace_keeps_the_previous_store_and_discards_nothing_else(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_replace`` used to ``rmtree`` the destination and only THEN rename.
        An rmtree that fails partway (a read-only child, EBUSY on a mount point,
        an interrupt) propagated before the artifact was marked consumed, so the
        ``finally`` deleted the new store too — and for a directory destination
        the artifact IS the staging tree. One ``--overwrite`` onto a good store
        could cost both copies, which is exactly what ``optimise_store``'s
        docstring promises cannot happen."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, target_bytes=8192)
        good = tuple(open_group(dst, mode="r")["x"].chunks)
        real_replace = os.replace
        failed: list[Any] = []

        def flaky(a: Any, b: Any) -> Any:
            # Only the swap of the NEW store into place — the restore that
            # follows it has to be allowed to work.
            if Path(b) == dst and not failed:
                failed.append(b)
                raise OSError("EBUSY: device or resource busy")
            return real_replace(a, b)

        monkeypatch.setattr(optimise_mod.os, "replace", flaky)
        with pytest.raises(OSError, match="EBUSY"):
            optimise_store(src, dst, overwrite=True, target_bytes=32_768)

        # The previous store is still there, still readable, still ITS grid.
        assert tuple(open_group(dst, mode="r")["x"].chunks) == good
        assert assert_values_identical(src, dst) == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "out.luxar.zarr",
            "src.luxar.zarr",
        ]

    def test_a_symlinked_destination_is_refused_before_any_work(
        self, tmp_path: Path
    ) -> None:
        """``exists()`` and ``is_dir()`` follow a symlink; every mutator here
        acts on the link. A link to a real store passed every guard, ran the
        whole copy and ``--verify``, and then died in ``shutil.rmtree``
        ("Cannot call rmtree on a symbolic link") — after which the ``finally``
        discarded the finished output. Symlinked output paths are ordinary
        (small home, big ``/mnt``), so the refusal has to be actionable."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        real = _tiny_store(tmp_path / "real_out.luxar.zarr")
        link = tmp_path / "out.luxar.zarr"
        link.symlink_to(real, target_is_directory=True)

        with pytest.raises(ValueError, match="symlink"):
            optimise_store(src, link, overwrite=True)
        assert link.is_symlink()
        assert (real / "x").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "out.luxar.zarr",
            "real_out.luxar.zarr",
            "src.luxar.zarr",
        ]

    def test_a_dangling_symlink_destination_is_refused(self, tmp_path: Path) -> None:
        """``Path.exists()`` is False for a dangling link, so the ``--overwrite``
        requirement was skipped outright and ``os.replace`` then raised
        ``NotADirectoryError`` — again with the whole copy already done."""
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        link = tmp_path / "out.luxar.zarr"
        link.symlink_to(tmp_path / "gone", target_is_directory=True)

        with pytest.raises(ValueError, match="symlink"):
            optimise_store(src, link)  # no --overwrite: the link is not a store
        assert link.is_symlink()
        assert not (tmp_path / "gone").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "out.luxar.zarr",
            "src.luxar.zarr",
        ]

    def test_a_failed_verify_leaves_the_previous_output_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(src, dst, target_bytes=8192)
        good = tuple(open_group(dst, mode="r")["x"].chunks)

        def refuse(*args: Any, **kwargs: Any) -> int:
            raise ValueError("verify: 'x' differs")

        monkeypatch.setattr(optimise_mod, "_verify", refuse)
        with pytest.raises(ValueError, match="verify"):
            optimise_store(src, dst, overwrite=True, verify=True)
        assert tuple(open_group(dst, mode="r")["x"].chunks) == good
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "out.luxar.zarr",
            "src.luxar.zarr",
        ]

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
# --verify actually discriminates
#
# Mutation-proved gap: replacing `_verify`'s body with `return 0` left every
# verify-touching test green, because none of them ever fed it a mismatched
# pair. Each test below perturbs one thing and asserts the matching branch.
# --------------------------------------------------------------------------


def _pair(tmp_path: Path, **arrays: np.ndarray) -> tuple[Any, Any]:
    """Two identical stores, opened writable — a source and a copy to damage."""
    src = tmp_path / "v_src.zarr"
    dst = tmp_path / "v_dst.zarr"
    for path in (src, dst):
        root = open_group(path, mode="w")
        root.attrs["kind"] = "leaf"
        for name, data in arrays.items():
            create_array(root, name, data=data, chunks=(64,), compressor=None)
        consolidate(root)
    return open_group(src, mode="r+"), open_group(dst, mode="r+")


class TestVerifyDiscriminates:
    def test_a_matching_pair_passes(self, tmp_path: Path) -> None:
        src, dst = _pair(tmp_path, a=np.arange(500, dtype=np.float32))
        assert _verify(src, dst) == (1, 0)

    def test_a_single_flipped_value_is_caught(self, tmp_path: Path) -> None:
        src, dst = _pair(tmp_path, a=np.arange(500, dtype=np.float32))
        dst["a"][123] = -1.0
        with pytest.raises(ValueError, match="differs at rows"):
            _verify(src, dst)

    def test_a_missing_array_is_caught(self, tmp_path: Path) -> None:
        src, dst = _pair(
            tmp_path,
            a=np.arange(500, dtype=np.float32),
            b=np.arange(500, dtype=np.float32),
        )
        del dst["b"]
        with pytest.raises(ValueError, match="is missing from the output store"):
            _verify(src, dst)

    def test_a_missing_group_is_caught(self, tmp_path: Path) -> None:
        """A group holding no arrays of its own — an overlay group is exactly
        that shape, its payload file reachable through no array API — is
        invisible to a walk that only compares arrays."""
        src, dst = _pair(tmp_path, a=np.arange(500, dtype=np.float32))
        src.create_group("overlays")
        with pytest.raises(ValueError, match="group 'overlays' is missing"):
            _verify(src, dst)

    def test_a_shape_mismatch_is_caught(self, tmp_path: Path) -> None:
        src, dst = _pair(tmp_path, a=np.arange(500, dtype=np.float32))
        dst["a"].resize((400,))
        with pytest.raises(ValueError, match="shape"):
            _verify(src, dst)

    def test_a_dtype_mismatch_is_caught(self, tmp_path: Path) -> None:
        src, dst = _pair(tmp_path, a=np.arange(500, dtype=np.float32))
        root = open_group(Path(dst.store.root), mode="r+")
        create_array(
            root,
            "a",
            data=np.arange(500, dtype=np.float64),
            chunks=(64,),
            compressor=None,
        )
        with pytest.raises(ValueError, match="dtype"):
            _verify(src, open_group(Path(dst.store.root), mode="r"))

    def test_an_array_attr_mismatch_is_caught(self, tmp_path: Path) -> None:
        src, dst = _pair(tmp_path, a=np.arange(500, dtype=np.float32))
        src["a"].attrs["encoding"] = {"name": "raw"}
        with pytest.raises(ValueError, match="attrs differ"):
            _verify(src, dst)

    def test_a_variable_width_string_array_is_really_compared(
        self, tmp_path: Path, restore_zarr_format: None
    ) -> None:
        """``tobytes()`` is the wrong comparison for a variable-width dtype.

        A zarr-v3 ``dtype=str`` array decodes to ``numpy.dtypes.StringDType``,
        whose buffer holds descriptors into a character ARENA — so
        ``['x'*60, 'x'*60]`` and ``['x'*60, 'y'*60]`` compare IDENTICAL byte for
        byte. Reachable through ``--generic`` on an OME-Zarr label table or an
        AnnData/cellxgene ``.zarr``, and ``--verify`` is the only mechanism
        asserting the headline bit-identity claim, so a vacuous pass there is
        the whole safety net.
        """
        set_zarr_format(3)
        paths = []
        for name, second in (("s_src.zarr", "x" * 60), ("s_dst.zarr", "y" * 60)):
            path = tmp_path / name
            root = open_group(path, mode="w")
            root.attrs["kind"] = "leaf"
            arr = root.create_array("labels", shape=(2,), dtype=str, chunks=(2,))
            arr[:] = np.array(["x" * 60, second])
            consolidate(root)
            paths.append(path)
        src, dst = (open_group(p, mode="r") for p in paths)

        raw_src = np.ascontiguousarray(src["labels"][:]).tobytes()
        raw_dst = np.ascontiguousarray(dst["labels"][:]).tobytes()
        assert raw_src == raw_dst, "the arena-pointer trap no longer reproduces"
        with pytest.raises(ValueError, match="differs"):
            _verify(src, dst)

        # And no false positive: a real copy of the same store still passes.
        out = tmp_path / "s_out.zarr"
        optimise_store(paths[0], out, verify=True)
        assert _verify(src, open_group(out, mode="r")) == (1, 0)

    @pytest.mark.parametrize(
        "left, right",
        [
            # `np.array_equal` calls both of these equal; the bytes differ, and
            # the contract is bit-identity on disk.
            (np.float32(0.0), np.float32(-0.0)),
            (
                np.frombuffer(np.uint32(0x7FC00001).tobytes(), dtype=np.float32)[0],
                np.frombuffer(np.uint32(0x7FC00002).tobytes(), dtype=np.float32)[0],
            ),
        ],
    )
    def test_bytes_beat_numeric_equality(
        self, tmp_path: Path, left: Any, right: Any
    ) -> None:
        data = np.zeros(500, dtype=np.float32)
        src, dst = _pair(tmp_path, a=data)
        src["a"][7] = left
        dst["a"][7] = right
        a, b = np.asarray(src["a"][:]), np.asarray(dst["a"][:])
        assert a.tobytes() != b.tobytes(), "the fixture is not actually different"
        with pytest.raises(ValueError, match="differs at rows"):
            _verify(src, dst)


# --------------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------------


class TestSummary:
    def test_the_summary_improves_after_optimising(
        self, scene: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(scene, dst)
        before = summarise_chunk_layout(open_group(scene, mode="r"))
        after = summarise_chunk_layout(open_group(dst, mode="r"))
        assert after.n_chunks < before.n_chunks
        assert after.mean_chunk_bytes > before.mean_chunk_bytes
        assert after.n_arrays == before.n_arrays
        # The floor share must MOVE, not merely stay in [0, 1] (which it cannot
        # leave); re-chunking is exactly what lifts arrays over the 16 KB floor.
        assert after.n_arrays_under_floor < before.n_arrays_under_floor
        assert after.share_under_floor < before.share_under_floor

    def test_arrays_that_fetch_nothing_are_left_out_of_the_floor_share(
        self, tmp_path: Path
    ) -> None:
        """An ``array_ref`` placeholder is physically ``(0, D)``: it writes no
        chunk and costs no request, so booking it "under the 16 KB floor" only
        dilutes the diagnostic. Measured on a 6-node scene with deduplicated
        positions, ``info --stats`` printed "23/24 arrays (96%)" where the
        honest answer is 13/14 (93%)."""
        src = tmp_path / "refs.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "real",
            data=np.zeros(256, dtype=np.float32),
            chunks=(256,),
            compressor=None,
        )
        for i in range(3):
            create_array(
                root,
                f"ref{i}",
                data=np.zeros((0, 3), dtype=np.float32),
                chunks=(1, 3),
                compressor=None,
            )
        consolidate(root)

        summary = summarise_chunk_layout(open_group(src, mode="r"))
        assert summary.n_chunks == 1
        assert summary.n_arrays == 1, "a zero-file array was counted"
        assert summary.n_arrays_under_floor == 1
        assert summary.share_under_floor == 1.0

    def test_the_projected_request_count_is_the_file_count_on_disk(
        self, tmp_path: Path, restore_zarr_format: None
    ) -> None:
        """The diagnostic's whole job is predicting requests, so it is checked
        against the objects zarr actually wrote — not against the plan, which
        derives its count from the same walk and so cannot disagree."""
        set_zarr_format(3)
        src = tmp_path / "counted.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        rng = _rng(13)
        create_array(
            root,
            "plain",
            data=rng.integers(0, 255, 50_000, dtype=np.uint8),
            chunks=(1000,),
            compressor=None,
        )  # 50 files
        sharded = root.create_array(
            "sharded",
            shape=(200_000, 4),
            dtype="f4",
            chunks=(1000, 4),
            shards=(50_000, 4),
        )  # 4 files, NOT 200
        sharded[:] = rng.random((200_000, 4)).astype(np.float32)
        create_array(
            root,
            "empty",
            data=np.zeros((0, 3), dtype=np.float32),
            chunks=(1, 3),
            compressor=None,
        )  # 0 files, NOT 1
        consolidate(root)

        on_disk = _chunk_files(src)
        assert on_disk == 54, on_disk
        assert summarise_chunk_layout(open_group(src, mode="r")).n_chunks == on_disk


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run(*args: str):
    from typer.testing import CliRunner

    from luxar.cli.main import app

    return CliRunner().invoke(app, ["optimise", *args])


def _info(*args: str):
    """``luxar info`` — the other consumer of the plan walk (``--stats``)."""
    from typer.testing import CliRunner

    from luxar.cli.main import app

    return CliRunner().invoke(app, ["info", *args])


class TestCli:
    def test_dry_run_writes_nothing(self, scene: Path) -> None:
        """Snapshotted over the directory that actually HOLDS the source, paths
        and sizes and mtimes, recursively. Watching an unrelated empty
        ``tmp_path`` made this vacuous: a ``--dry-run`` mutated to write a file
        next to the source kept all four dry-run tests green."""
        home = scene.parent
        before = _tree_snapshot(home)
        assert before, "the snapshot covers nothing"
        result = _run(str(scene), "--dry-run")
        assert result.exit_code == 0, result.output
        assert "Would re-chunk" in result.output
        assert "Nothing was written" in result.output
        assert _tree_snapshot(home) == before

    def test_dry_run_takes_the_generic_gate(self, tmp_path: Path) -> None:
        """A dry run that plans a store the identical real run refuses is a plan
        nobody can act on — and it also leaked the source's store handle, which
        keeps a ``.zarr.zip``'s ``ZipStore`` open."""
        src = _tiny_store(tmp_path / "plain.zarr", luxar=False)
        refused = _run(str(src), "--dry-run")
        assert refused.exit_code == 1
        assert "--generic" in refused.output
        assert "Would re-chunk" not in refused.output

        allowed = _run(str(src), "--dry-run", "--generic")
        assert allowed.exit_code == 0, allowed.output
        assert "Would re-chunk" in allowed.output

    def test_dry_run_on_an_optimised_store_says_so(
        self, scene: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.luxar.zarr"
        optimise_store(scene, dst)
        result = _run(str(dst), "--dry-run")
        assert result.exit_code == 0, result.output
        assert "Nothing to re-chunk" in result.output

    def test_dry_run_rejects_an_output_argument(
        self, scene: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "o.zarr"
        result = _run(str(scene), str(out), "--dry-run")
        assert result.exit_code == 1
        assert "--dry-run writes nothing" in result.output
        assert not out.exists()

    def test_an_output_is_required_without_dry_run(self, scene: Path) -> None:
        result = _run(str(scene))
        assert result.exit_code == 1
        assert "An output path is required" in result.output

    def test_a_missing_source_is_reported(self, tmp_path: Path) -> None:
        result = _run(str(tmp_path / "nope.zarr"), str(tmp_path / "out.zarr"))
        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert not (tmp_path / "out.zarr").exists()

    def test_a_bad_budget_is_reported(self, scene: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.zarr"
        result = _run(str(scene), str(out), "--target-kb", "0")
        assert result.exit_code == 1
        assert "positive" in result.output
        assert not out.exists()

    def test_target_kb(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "src.luxar.zarr")
        dst = tmp_path / "out.luxar.zarr"
        result = _run(str(src), str(dst), "--target-kb", "8")
        assert result.exit_code == 0, result.output
        # 8 KB / 4 bytes = 2048 rows, no atom to round to.
        assert tuple(open_group(dst, mode="r")["x"].chunks) == (2048,)

    def test_overwrite_and_generic(self, tmp_path: Path) -> None:
        src = _tiny_store(tmp_path / "plain.zarr", luxar=False)
        dst = tmp_path / "out.zarr"
        refused = _run(str(src), str(dst))
        assert refused.exit_code == 1
        assert "--generic" in refused.output

        first = _run(str(src), str(dst), "--generic")
        assert first.exit_code == 0, first.output
        clash = _run(str(src), str(dst), "--generic")
        assert clash.exit_code == 1
        assert "--overwrite" in clash.output
        again = _run(str(src), str(dst), "--generic", "--overwrite")
        assert again.exit_code == 0, again.output
        assert_values_identical(src, dst)

    def test_end_to_end(self, scene: Path, tmp_path: Path) -> None:
        dst = tmp_path / "out.luxar.zarr"
        result = _run(str(scene), str(dst), "--profile", "hosting", "--verify")
        assert result.exit_code == 0, result.output
        assert "Verified" in result.output
        assert_values_identical(scene, dst)
        assert assert_atom_aligned(dst) > 0

    def test_info_stats_reports_the_chunk_layout(self, scene: Path) -> None:
        from typer.testing import CliRunner

        from luxar.cli.main import app

        result = CliRunner().invoke(app, ["info", str(scene), "--stats", "--no-tree"])
        assert result.exit_code == 0, result.output
        assert "Chunk Layout" in result.output
        assert "Projected requests" in result.output

    def test_info_stats_recommends_optimise_when_the_plan_is_real(
        self, tmp_path: Path
    ) -> None:
        src = _tiny_store(tmp_path / "small.luxar.zarr")  # 400-byte chunks
        assert plan_optimisation(open_group(src, mode="r")).n_rechunked == 1

        from typer.testing import CliRunner

        from luxar.cli.main import app

        result = CliRunner().invoke(app, ["info", str(src), "--stats", "--no-tree"])
        assert result.exit_code == 0, result.output
        assert "would cut this" in result.output

    def test_info_stats_warns_when_chunks_are_rtt_bound_on_http1(
        self, tmp_path: Path
    ) -> None:
        """A 400-byte-chunk store is round-trip bound on an HTTP/1.1 host (which
        ``luxar serve`` is); a 128 KB single-chunk array is not. The warning is
        gated on the mean chunk size alone — unlike the optimise hint, it also
        applies to a store the optimiser could not re-chunk, because the round
        trips are paid either way."""
        from typer.testing import CliRunner

        from luxar.cli.main import app

        small = _tiny_store(tmp_path / "small.luxar.zarr")  # 400-byte chunks
        result = CliRunner().invoke(app, ["info", str(small), "--stats", "--no-tree"])
        assert result.exit_code == 0, result.output
        assert "HTTP/1.1" in result.output
        assert "luxar optimise --profile hosting" in result.output

        big = tmp_path / "big.luxar.zarr"
        root = open_group(big, mode="w")
        root.attrs["kind"] = "leaf"
        create_array(
            root,
            "a",
            data=np.zeros(32_768, dtype=np.float32),  # one 128 KB chunk
            chunks=(32_768,),
            compressor=None,
        )
        consolidate(root)
        result = CliRunner().invoke(app, ["info", str(big), "--stats", "--no-tree"])
        assert result.exit_code == 0, result.output
        assert "Chunk Layout" in result.output
        assert "HTTP/1.1" not in result.output

    def test_info_stats_does_not_recommend_a_no_op(self, tmp_path: Path) -> None:
        """Ten 1 KB single-chunk arrays sit under the floor and yet have nothing
        to re-chunk: the hint has to be gated on a real plan, not on the
        average."""
        from typer.testing import CliRunner

        from luxar.cli.main import app

        src = tmp_path / "tiny.luxar.zarr"
        root = open_group(src, mode="w")
        root.attrs["kind"] = "leaf"
        for i in range(10):
            create_array(
                root,
                f"a{i}",
                data=np.zeros(256, dtype=np.float32),
                chunks=(256,),
                compressor=None,
            )
        consolidate(root)
        assert plan_optimisation(open_group(src, mode="r")).n_rechunked == 0

        result = CliRunner().invoke(app, ["info", str(src), "--stats", "--no-tree"])
        assert result.exit_code == 0, result.output
        assert "Under the 16 KB floor: 10/10" in result.output
        assert "would cut this" not in result.output
