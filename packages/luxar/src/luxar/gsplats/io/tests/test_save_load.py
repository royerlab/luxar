"""Tests for v3.0 save and load (node-tree ``.gsplats.zarr``).

A single splat set is a leaf node at the file root (arrays directly under root);
an additive ladder writes ``additive_<i>/`` subgroups under the root leaf. Color
SDR/HDR is auto-detected (no explicit ``color_mode`` knob).
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import read_consolidated_attrs
from luxar.conftest import array_compressor, confine_temp_dirs
from luxar.encoding import EncodingMode
from luxar.gsplats import GSplatData
from luxar.gsplats.io import (
    format_gsplats_info,
    inspect_gsplats_zarr,
    load_gsplats,
    save_gsplats,
)
from luxar.typing_utils._format_contract import GSPLATS_FORMAT_VERSION
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


def _positive_diag(chol: np.ndarray, d: int = 3) -> np.ndarray:
    """Force the packed-Cholesky diagonal strictly positive, in place.

    A real Cholesky factor has ``L[i, i] > 0``; unconstrained random vectors
    give ~50% negative diagonals that the writer's positive-diagonal gate
    legitimately rejects. Off-diagonals are left untouched (kept signed) so the
    fixture still exercises negative off-diagonals through the round-trip.
    """
    diag_idx = np.cumsum(np.arange(1, d + 1)) - 1
    chol[..., diag_idx] = np.abs(chol[..., diag_idx]) + 0.1
    return chol


def create_test_splats_3d(n_splats: int = 100) -> dict:
    """Create test 3D Gaussian splats."""
    rng = np.random.default_rng(0)
    return {
        "centers": rng.random((n_splats, 3)).astype(np.float32) * 10,
        "amplitudes": rng.random(n_splats).astype(np.float32) * 2,
        "cholesky_factors": rng.random((n_splats, 6)).astype(np.float32),
    }


def _wide_amplitude_splats(n_splats: int = 1000) -> dict:
    splats = create_test_splats_3d(n_splats)
    splats["amplitudes"] = np.geomspace(1.0, 1000.0, n_splats).astype(np.float32)
    return splats


def _zip_store_flat(store: Path, archive: Path) -> Path:
    """Zip a store's files with the store root AT THE ARCHIVE ROOT.

    The "flat" archive shape: ``.zgroup``/``zarr.json`` and ``centers/…`` sit at
    depth 0, as ``zip -r x.gsplats.zarr.zip .`` from inside a store produces.
    """
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_ref:
        for f in sorted(store.rglob("*")):
            if f.is_file():
                zip_ref.write(f, arcname=str(f.relative_to(store)))
    return archive


def _targz_store_flat(store: Path, archive: Path) -> Path:
    """Tar a store's files with the store root AT THE ARCHIVE ROOT.

    The tar sibling of :func:`_zip_store_flat` (``tar czf x.gsplats.zarr.tar.gz
    .`` from inside a store). zarr has no tar store, so extraction is the only
    route into this container — which makes it the container where the flat shape
    can ONLY work through the extractor's store-root resolution.
    """
    import tarfile

    with tarfile.open(archive, "w:gz") as tar_ref:
        for f in sorted(store.rglob("*")):
            if f.is_file():
                tar_ref.add(f, arcname=str(f.relative_to(store)))
    return archive


def _luxar_temp_dirs() -> set:
    """The extractor's temp directories currently present, as a set of paths."""
    return set(Path(tempfile.gettempdir()).glob("luxar_gsplat_archive_*"))


def _uncompressed_model_bytes(n_splats: int, ndim: int = 3) -> int:
    """The inspector's own uncompressed-size model, in exact bytes.

    Deliberately recomputed from the fixture's shape rather than from the info
    dict's ROUNDED ``uncompressed_mb``: dividing the rounded value back out makes
    the expectation circular AND fixture-size-dependent (at 1000-2000 splats the
    rounding error alone is ~5%).
    """
    chol = ndim * (ndim + 1) // 2
    return n_splats * (ndim * 4 + 4 + chol * 4)


class TestSaveGsplats:
    @pytest.mark.parametrize(
        ("source_dtype", "expected_dtype", "expected_encoding"),
        [
            ("uint8", np.uint8, "geolog_scalar_uint8"),
            ("uint16", np.uint16, "bounded_scalar_uint16"),
            ("float32", np.uint16, "bounded_scalar_uint16"),
            ("not-a-dtype", np.uint16, "bounded_scalar_uint16"),
            (None, np.uint16, "bounded_scalar_uint16"),
        ],
    )
    def test_amplitude_bits_auto_follows_source_dtype(
        self,
        tmp_path: Path,
        source_dtype: str | None,
        expected_dtype: np.dtype,
        expected_encoding: str,
    ) -> None:
        data = GSplatData(**_wide_amplitude_splats())
        data.stats["source_dtype"] = source_dtype

        path = tmp_path / f"{source_dtype}.gsplats.zarr"
        data.save(path, ordering="none", amplitude_bits="auto")

        root = zarr.open_group(str(path), mode="r")
        assert root["amplitudes"].dtype == expected_dtype
        assert root["amplitudes"].attrs["encoding"]["name"] == expected_encoding
        assert (
            root["amplitudes"].nbytes
            == len(data.amplitudes) * np.dtype(expected_dtype).itemsize
        )

    def test_amplitude_bits_auto_round_trips_within_uint8_error_bound(
        self, tmp_path: Path
    ) -> None:
        data = GSplatData(**_wide_amplitude_splats())
        data.stats["source_dtype"] = "uint8"
        path = tmp_path / "round-trip.gsplats.zarr"

        data.save(path, ordering="none", amplitude_bits="auto")
        decoded = GSplatData.load(path)

        relative_error = np.abs(decoded.amplitudes - data.amplitudes) / data.amplitudes
        assert float(relative_error.max()) < 4e-2

    def test_amplitude_bits_default_stays_uint16_for_uint8_source(
        self, tmp_path: Path
    ) -> None:
        data = GSplatData(**_wide_amplitude_splats())
        data.stats["source_dtype"] = "uint8"

        path = tmp_path / "default.gsplats.zarr"
        data.save(path, ordering="none")

        root = zarr.open_group(str(path), mode="r")
        assert root["amplitudes"].dtype == np.uint16
        assert root["amplitudes"].attrs["encoding"]["name"] == "bounded_scalar_uint16"

    def test_amplitude_bits_rejects_unknown_tier_before_writing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "invalid.gsplats.zarr"
        data = GSplatData(**_wide_amplitude_splats())

        with pytest.raises(ValueError, match="amplitude_bits must be"):
            data.save(path, amplitude_bits=12)  # type: ignore[arg-type]

        assert not path.exists()

    def test_cli_fit_save_uses_source_matched_amplitude_bits(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.fitting.fit_utils import save_fit_output

        data = GSplatData(**_wide_amplitude_splats())
        data.stats["source_dtype"] = "uint8"
        path = tmp_path / "fit-output.gsplats.zarr"

        save_fit_output(data, path, compress=None, verbose=False)

        root = zarr.open_group(str(path), mode="r")
        assert root["amplitudes"].dtype == np.uint8
        assert root["amplitudes"].attrs["encoding"]["name"] == "geolog_scalar_uint8"

    def test_cli_fit_save_tree_uses_source_matched_amplitude_bits(
        self, tmp_path: Path
    ) -> None:
        from luxar.cli.gsplat_ops.fitting.fit_utils import save_fit_output
        from luxar.gsplats.tree import GSplatPartition

        data = GSplatData(**_wide_amplitude_splats())
        tree = GSplatPartition(
            children=[data.tree],
            meta={"fit_stats": {"source_dtype": "uint8"}},
        )
        path = tmp_path / "fit-tree-output.gsplats.zarr"

        save_fit_output(tree, path, compress=None, verbose=False)

        root = zarr.open_group(str(path), mode="r")
        assert root["part_0/amplitudes"].dtype == np.uint8
        assert (
            root["part_0/amplitudes"].attrs["encoding"]["name"] == "geolog_scalar_uint8"
        )

    def test_tree_auto_does_not_require_persisting_fit_metadata(
        self, tmp_path: Path
    ) -> None:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatPartition

        data = GSplatData(**_wide_amplitude_splats())
        tree = GSplatPartition(children=[data.tree])
        path = tmp_path / "tree-without-fitting-info.gsplats.zarr"

        write_gsplats_tree(
            path,
            tree,
            ordering="none",
            amplitude_bits="auto",
            source_dtype="uint8",
        )

        root = zarr.open_group(str(path), mode="r")
        assert "fitting" not in root
        assert root["part_0/amplitudes"].dtype == np.uint8

    def test_streaming_partition_auto_requires_dtype_for_callable_metadata(
        self, tmp_path: Path
    ) -> None:
        from luxar.gsplats.io.save_gsplats import write_partition_streaming

        data = GSplatData(**_wide_amplitude_splats())
        path = tmp_path / "streaming-missing-dtype.gsplats.zarr"

        with pytest.raises(ValueError, match="source_dtype is required"):
            write_partition_streaming(
                path,
                lambda: iter([data.tree]),
                amplitude_bits="auto",
                fitting_info=lambda: {"source_dtype": "uint8"},
            )

        assert not path.exists()

    def test_streaming_partition_auto_uses_explicit_source_dtype(
        self, tmp_path: Path
    ) -> None:
        from luxar.gsplats.io.save_gsplats import write_partition_streaming

        data = GSplatData(**_wide_amplitude_splats())
        path = tmp_path / "streaming-source-dtype.gsplats.zarr"

        write_partition_streaming(
            path,
            lambda: iter([data.tree]),
            amplitude_bits="auto",
            source_dtype="uint8",
            fitting_info=lambda: {"part_provenance": []},
        )

        root = zarr.open_group(str(path), mode="r")
        assert root["part_0/amplitudes"].dtype == np.uint8

    def test_amplitude_auto_does_not_require_persisting_fit_metadata(
        self, tmp_path: Path
    ) -> None:
        data = GSplatData(**_wide_amplitude_splats())
        data.stats["source_dtype"] = "uint8"
        path = tmp_path / "without-fitting-info.gsplats.zarr"

        data.save(
            path,
            ordering="none",
            amplitude_bits="auto",
            include_fitting_info=False,
        )

        root = zarr.open_group(str(path), mode="r")
        assert "fitting" not in root
        assert root["amplitudes"].dtype == np.uint8

    """Test save_gsplats function (v3.0 leaf root)."""

    def test_save_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")

            assert path.exists()
            root = zarr.open_group(str(path), mode="r")
            assert root.attrs["format_type"] == "gsplats_zarr"
            # v3.2 renames the lod selector attrs (coverage_fraction); leaf
            # layout is unchanged from v3.1.
            assert root.attrs["format_version"] == GSPLATS_FORMAT_VERSION
            # v3.1+: leaf at root, no "splats" group; Cholesky factors split.
            assert "splats" not in root
            assert "cholesky_factors_diag" in root
            assert "cholesky_factors" not in root
            assert root.attrs["type"] == "gsplats"
            assert root.attrs["n_splats"] == 100
            assert root.attrs["ndim"] == 3

    def test_save_stamps_content_hash(self) -> None:
        # The web viewer's persistent cache invalidates on the root
        # ``content_hash`` — without it, a regenerated file at the same URL
        # serves stale data indefinitely.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50), ordering="none")
            root = zarr.open_group(str(path), mode="r")
            content_hash = root.attrs["content_hash"]
            assert isinstance(content_hash, str) and len(content_hash) > 0
            # The hash must also land in consolidated metadata: the viewer
            # builds its scene graph from that index, so a hash present only on
            # the node itself would never be seen.
            assert read_consolidated_attrs(path)["/"]["content_hash"] == content_hash

    def test_resave_changes_content_hash(self) -> None:
        # Identical data re-saved must yield a DIFFERENT hash (the timestamp
        # attr folds in) so the viewer cache invalidates on regeneration.
        splats = create_test_splats_3d(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.gsplats.zarr"
            path_b = Path(tmpdir) / "b.gsplats.zarr"
            save_gsplats(path=path_a, **splats, ordering="none")
            save_gsplats(path=path_b, **splats, ordering="none")
            hash_a = zarr.open_group(str(path_a), mode="r").attrs["content_hash"]
            hash_b = zarr.open_group(str(path_b), mode="r").attrs["content_hash"]
            assert hash_a != hash_b

    def test_streaming_partition_stamps_content_hash(self) -> None:
        # The streaming-partition writer path must stamp too (it is the merge
        # path for tiled fits — the largest, most re-generated artifacts).
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_partition_streaming
        from luxar.gsplats.tree import GSplatLeaf

        def make_leaf(seed: int) -> GSplatLeaf:
            rng = np.random.default_rng(seed)
            chol = np.zeros((20, 6), dtype=np.float32)
            chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(20, 3))
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.uniform(0, 50, (20, 3)).astype(np.float32),
                        amplitudes=rng.uniform(0.1, 1, (20,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "part.gsplats.zarr"
            write_partition_streaming(
                path, lambda: iter([make_leaf(0), make_leaf(1)]), ordering="none"
            )
            root = zarr.open_group(str(path), mode="r")
            assert isinstance(root.attrs["content_hash"], str)

    def test_stamped_content_hash_changes_with_chunk_layout(self) -> None:
        # #1718: the metadata-only stamp must fold in CHUNK layout, not just
        # (name, shape, dtype). The viewer cache holds encoded chunks keyed by
        # chunk index, so a re-chunked store with identical values is not
        # interchangeable with its input and must not share its hash.
        from luxar._zarr_compat import create_array, memory_group
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        data = np.arange(36, dtype=np.float32).reshape(12, 3)

        def stamped(chunks: tuple[int, int]) -> str:
            root = memory_group()
            root.attrs["type"] = "gsplats"
            create_array(root, "centers", data=data, chunks=chunks, compressor=None)
            return _stamp_content_hash(root)

        assert stamped((12, 3)) != stamped((3, 3))

    # Both formats: the stamp derives codec ids format-agnostically (format 3
    # lists them under `codecs`, format 2 under `filters` + `compressor`), and the
    # hole was equally real at either vintage. Luxar writes format 2 on demand
    # (`LUXAR_ZARR_FORMAT=2`) and re-stamps legacy v2 stores in place.
    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_stamped_content_hash_changes_with_codec_ids(
        self, zarr_format: int
    ) -> None:
        # #1718: the stamp folds in the array's codec IDS too — raw vs blosc vs
        # gzip encode the bytes a viewer caches at a chunk key entirely
        # differently, and nothing else here differs (same values, shape, chunks,
        # dtype, attrs). Ids only, so a compression-LEVEL tweak stays invisible;
        # that trade-off is pinned on the compiler side.
        import numcodecs

        from luxar._zarr_compat import create_array
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        data = np.arange(36, dtype=np.float32).reshape(12, 3)

        def stamped(compressor: object) -> str:
            root = zarr.create_group(
                store=zarr.storage.MemoryStore(), zarr_format=zarr_format
            )
            root.attrs["type"] = "gsplats"
            create_array(
                root, "centers", data=data, chunks=(3, 3), compressor=compressor
            )
            return _stamp_content_hash(root)

        raw = stamped(None)
        blosc = stamped(numcodecs.Blosc(cname="zstd", clevel=9))
        gzip = stamped(numcodecs.GZip(level=5))
        assert len({raw, blosc, gzip}) == 3, (raw, blosc, gzip)

    def test_stamped_content_hash_changes_with_per_array_attrs(self) -> None:
        # #1718: an array's OWN attrs are where Luxar keeps the DEQUANTIZATION
        # parameters, so they decide what decoded value the stored ints stand for.
        # A changed `encoding.min` shifts every decoded amplitude while values,
        # layout, codecs and the group attrs all stay put — and used to move
        # neither digest.
        from luxar._zarr_compat import create_array, memory_group
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        data = np.arange(12, dtype=np.uint8)

        def stamped(minimum: float) -> str:
            root = memory_group()
            root.attrs["type"] = "gsplats"
            array = create_array(
                root, "amplitudes", data=data, chunks=(12,), compressor=None
            )
            array.attrs["encoding"] = {
                "name": "bounded_scalar_uint8",
                "min": minimum,
                "max": 1.0,
                "bits": 8,
            }
            return _stamp_content_hash(root)

        assert stamped(0.0) != stamped(0.25)

    def test_stamped_content_hash_changes_with_child_group_name(self) -> None:
        # A node's own digest does not carry its NAME, and the parent folded in
        # only its children's digests — so renaming a child part/level while
        # leaving its contents alone left the root hash exactly where it was. A
        # group name is a path segment, so every cached key under it moves while
        # the token that would invalidate them says nothing changed.
        from luxar._zarr_compat import create_array, memory_group
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        data = np.arange(36, dtype=np.float32).reshape(12, 3)

        def stamped(child: str) -> str:
            root = memory_group()
            root.attrs["type"] = "gsplats"
            leaf = root.create_group(child)
            create_array(leaf, "centers", data=data, chunks=(12, 3), compressor=None)
            return _stamp_content_hash(root)

        assert stamped("part_0") != stamped("part_1")

    def test_stamped_content_hash_changes_with_sharded_inner_codec(self) -> None:
        # A sharded array's top-level pipeline is exactly one `ShardingCodec`, so
        # the codec IDS read `["sharding_indexed"]` and say nothing about the
        # inner codecs or the shard index. Nothing in Luxar emits a sharded store
        # today, so this is what keeps the expansion honest — and what stops a
        # future in-place re-layout tool from re-stamping a store to its input's
        # digest, which is the whole reason the metadata-only variant folds
        # layout in at all.
        import numcodecs

        from luxar._zarr_compat import create_array
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        data = np.arange(36, dtype=np.float32).reshape(12, 3)

        def stamped(compressor: object) -> str:
            # Format 3 explicitly: sharding does not exist at format 2.
            root = zarr.create_group(store=zarr.storage.MemoryStore(), zarr_format=3)
            root.attrs["type"] = "gsplats"
            create_array(
                root,
                "centers",
                data=data,
                chunks=(3, 3),
                shards=(6, 3),
                compressor=compressor,
            )
            return _stamp_content_hash(root)

        assert stamped(None) != stamped(numcodecs.Blosc(cname="zstd", clevel=9))

    def test_stamped_content_hash_never_reads_chunk_data(self) -> None:
        # The reason this variant exists at all: a splat store can be multi-GB, so
        # the stamp must stay metadata-only. #1718 added four more metadata reads
        # (`chunks`, `shards`, the codec pipeline, the array's attrs) under that
        # contract, so pin it: a chunk read would surface as a store `get` for a
        # key that is not one of zarr's metadata documents.
        from luxar._zarr_compat import ZARR_FORMAT, create_array
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        requested: list[str] = []

        class RecordingStore(zarr.storage.MemoryStore):
            async def get(self, key: str, *args: object, **kwargs: object) -> object:
                requested.append(key)
                return await super().get(key, *args, **kwargs)  # type: ignore[arg-type]

        store = RecordingStore()
        root = zarr.create_group(store=store, zarr_format=ZARR_FORMAT)
        root.attrs["type"] = "gsplats"
        create_array(
            root,
            "centers",
            data=np.arange(36, dtype=np.float32).reshape(12, 3),
            chunks=(3, 3),
            compressor=None,
        )

        # Re-open so the stamp reads metadata through the store rather than off a
        # handle already in memory, then record only the stamp's own calls.
        group = zarr.open_group(store=store, mode="a")
        requested.clear()
        _stamp_content_hash(group)

        assert requested, "the store recorded no reads — the probe is inert"
        docs = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}
        assert all(key.rsplit("/", 1)[-1] in docs for key in requested), requested

    def test_save_with_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(1).random((50, 3)).astype(np.float32)
            save_gsplats(
                path=path, **create_test_splats_3d(50), colors=colors, ordering="none"
            )
            root = zarr.open_group(str(path), mode="r")
            assert "colors" in root
            assert root.attrs["has_colors"] is True

    def test_label_ids_round_trip_exactly_through_ordering(
        self, tmp_path: Path
    ) -> None:
        splats = create_test_splats_3d(118)
        splats["centers"][:, 0] = np.arange(118, dtype=np.float32)
        splats["centers"][:, 1:] = 0.0
        label_ids = np.arange(118, dtype=np.uint16)
        label_ids[::7] = 117
        vocabulary = {i: f"class-{i}" for i in range(118)}
        data = GSplatData(
            **splats,
            label_ids=label_ids,
            label_vocabulary=vocabulary,
        )

        path = tmp_path / "labels.gsplats.zarr"
        data.save(path, ordering="hilbert", encoding_mode=EncodingMode.AUTO)

        root = zarr.open_group(str(path), mode="r")
        assert root.attrs["has_label_ids"] is True
        assert root["label_ids"].attrs["encoding"]["name"] == "uint8"

        loaded = GSplatData.load(path)
        assert loaded.label_ids is not None
        assert set(loaded.label_ids.tolist()) == set(label_ids.tolist())
        expected_by_center = {
            tuple(center): int(label_id)
            for center, label_id in zip(splats["centers"], label_ids)
        }
        np.testing.assert_array_equal(
            loaded.label_ids,
            [expected_by_center[tuple(center)] for center in loaded.centers],
        )
        assert loaded.label_vocabulary == vocabulary

    def test_constant_label_ids_use_smallest_unsigned_dtype(
        self, tmp_path: Path
    ) -> None:
        label_ids = np.full(8, 300, dtype=np.int64)
        data = GSplatData(
            **create_test_splats_3d(8),
            label_ids=label_ids,
            label_vocabulary={300: "class-300"},
        )

        path = tmp_path / "constant-labels.gsplats.zarr"
        data.save(path, ordering="none")

        root = zarr.open_group(str(path), mode="r")
        assert root["label_ids"].dtype == np.uint16
        assert root["label_ids"].attrs["encoding"]["name"] == "broadcasted"
        loaded = GSplatData.load(path)
        np.testing.assert_array_equal(loaded.label_ids, label_ids)

    def test_load_rejects_label_ids_missing_from_vocabulary(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "labels.gsplats.zarr"
        GSplatData(
            **create_test_splats_3d(4),
            label_ids=np.array([0, 1, 0, 1], dtype=np.uint8),
            label_vocabulary={0: "zero", 1: "one"},
        ).save(path, ordering="none")

        root = zarr.open_group(str(path), mode="a")
        root.attrs["label_vocabulary"] = {"0": "zero"}

        with pytest.raises(ValueError, match="missing ids.*1"):
            GSplatData.load(path)

    def test_load_rejects_non_string_label_vocabulary_name(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "labels.gsplats.zarr"
        GSplatData(
            **create_test_splats_3d(2),
            label_ids=np.array([0, 1], dtype=np.uint8),
            label_vocabulary={0: "zero", 1: "one"},
        ).save(path, ordering="none")

        root = zarr.open_group(str(path), mode="a")
        root.attrs["label_vocabulary"] = {"0": "zero", "1": 1}

        with pytest.raises(TypeError, match="values must be strings"):
            GSplatData.load(path)

    def test_save_rejects_label_ids_missing_from_vocabulary(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="missing ids.*1"):
            save_gsplats(
                tmp_path / "invalid-labels.gsplats.zarr",
                **create_test_splats_3d(4),
                label_ids=np.array([0, 1, 0, 1], dtype=np.uint8),
                label_vocabulary={0: "zero"},
                ordering="none",
            )

    def test_streaming_save_rejects_label_ids_missing_from_vocabulary(
        self, tmp_path: Path
    ) -> None:
        from luxar.gsplats import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import (
            StreamingSplatSetMetadata,
            write_flat_leaf_streaming,
        )

        splats = create_test_splats_3d(4)
        sublod = AdditiveSubLOD(
            **splats,
            label_ids=np.array([0, 1, 0, 1], dtype=np.uint8),
            label_vocabulary={0: "zero"},
        )
        metadata = StreamingSplatSetMetadata(
            n_splats=4,
            ndim=3,
            truncation_radius=sublod.truncation_radius,
            label_dtype=np.dtype(np.uint8),
            label_vocabulary={0: "zero"},
        )

        with pytest.raises(ValueError, match="missing ids.*1"):
            write_flat_leaf_streaming(
                tmp_path / "invalid-stream.gsplats.zarr",
                lambda: iter([sublod]),
                splat_set_metadata=[metadata],
                barrier_dims=[],
            )

    @pytest.mark.parametrize("mode", [EncodingMode.PRECISION, EncodingMode.AUTO])
    def test_rgba_colors_round_trip(self, mode: EncodingMode) -> None:
        # RGBA colors (per-splat opacity in the 4th column) survive
        # write→read through both the SDR (rgb_uint8/AUTO) and lossless
        # (PRECISION/float32) encoders. Alpha ∈ [0, 1] must never trip HDR.
        rng = np.random.default_rng(3)
        colors = rng.random((50, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rgba.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                colors=colors,
                encoding_mode=mode,
                ordering="none",
            )
            root = zarr.open_group(str(path), mode="r")
            assert root.attrs["has_colors"] is True
            loaded = load_gsplats(path)
            assert loaded.colors is not None
            assert loaded.colors.shape == (50, 4)
            atol = 1e-6 if mode == EncodingMode.PRECISION else 2.0 / 255.0
            assert np.allclose(loaded.colors, colors, atol=atol)

    def test_rgba_hdr_rgb_keeps_alpha_bounded(self) -> None:
        # HDR RGB (values > 1) routes through geolog per-channel; the alpha
        # column rides along and must round-trip within [0, 1].
        rng = np.random.default_rng(4)
        colors = np.empty((40, 4), dtype=np.float32)
        colors[:, :3] = rng.random((40, 3)).astype(np.float32) * 8.0  # HDR RGB
        colors[:, 3] = rng.random(40).astype(np.float32)  # opacity in [0, 1]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rgba_hdr.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(40),
                colors=colors,
                encoding_mode=EncodingMode.PRECISION,
                ordering="none",
            )
            loaded = load_gsplats(path)
            assert loaded.colors is not None and loaded.colors.shape == (40, 4)
            assert np.allclose(loaded.colors, colors, atol=1e-5)

    def test_save_with_morton_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            attrs = zarr.open_group(str(path), mode="r").attrs
            assert attrs["ordering"] == "morton"
            assert "ordering_min" in attrs
            assert "ordering_max" in attrs
            assert "ordering_bits_per_dim" in attrs

    def test_save_with_hilbert_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            try:
                save_gsplats(
                    path=path, **create_test_splats_3d(100), ordering="hilbert"
                )
            except ImportError:
                pytest.skip("hilbertcurve package not installed")
            assert zarr.open_group(str(path), mode="r").attrs["ordering"] == "hilbert"

    def test_save_with_encoding_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            splats = create_test_splats_3d(100)
            p_prec = Path(tmpdir) / "precision.gsplats.zarr"
            save_gsplats(
                path=p_prec,
                **splats,
                encoding_mode=EncodingMode.PRECISION,
                ordering="none",
            )
            assert zarr.open_group(str(p_prec), mode="r")["centers"].dtype == np.float32

            # MEMORY mode stores COORDINATE centers as uint16 per-axis fixed-point
            # (linear_perchannel_u16). float16 is intentionally disabled (WebGL has no
            # native float16 and float16 on coordinates is a precision footgun);
            # coordinates never use uint8 (too coarse). Decodes back to float32.
            # It also depends on the sigma rail (io/_compiler/gsplat_assembly.py)
            # staying quiet: the rail escalates centers to float32 when >0.1% of
            # splats have a marginal sigma under half the u16 grid step, and at
            # N=100 one such splat is already 1%. The rng.random((n, 6)) Cholesky
            # above draws under the extent-10 half-step (7.6e-5) for ~0.76% of
            # seeds; this seed is pinned to one that does not, so a mismatch here
            # would be the rail firing rather than an encoder tier change.
            p_mem = Path(tmpdir) / "memory.gsplats.zarr"
            save_gsplats(
                path=p_mem, **splats, encoding_mode=EncodingMode.MEMORY, ordering="none"
            )
            enc = zarr.open_group(str(p_mem), mode="r")["centers"].attrs.get(
                "encoding", {}
            )
            assert enc["name"] == "linear_perchannel_u16"

    def test_save_with_fitting_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={
                    "time_seconds": 45.3,
                    "iterations": 850,
                    "converged": True,
                    "fitter_name": "t",
                },
                fitting_config={"n_iters": 1000, "lr": 0.05},
                ordering="none",
            )
            root = zarr.open_group(str(path), mode="r")
            assert root["fitting"].attrs["time_seconds"] == 45.3
            assert "fitting/config" in root
            assert root["fitting/config"].attrs["n_iters"] == 1000

    def test_save_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            # Mismatched amplitudes shape
            with pytest.raises(ValueError, match="[Aa]mplitudes shape"):
                save_gsplats(
                    path=path,
                    centers=splats["centers"],
                    amplitudes=np.random.rand(50).astype(np.float32),
                    cholesky_factors=splats["cholesky_factors"],
                )
            # Mismatched cholesky shape
            with pytest.raises(ValueError):
                save_gsplats(
                    path=path,
                    centers=splats["centers"],
                    amplitudes=splats["amplitudes"],
                    cholesky_factors=np.random.rand(100, 3).astype(
                        np.float32
                    ),  # wrong k
                )


class TestCholeskySplitRoundTrip:
    """v3.1 splits Cholesky factors into diag + offdiag on disk and recombines
    them on read. Verify the packed (N, k) form survives the round-trip across
    dimensionalities, encoding modes, and the uniform/broadcast case."""

    @staticmethod
    def _splats(n: int, d: int, rng: np.random.Generator) -> dict:
        k = d * (d + 1) // 2
        # Realistic Cholesky factors: POSITIVE diagonal (a real L has L[i,i] > 0),
        # signed off-diagonal. (Random unconstrained vectors would give negative
        # diagonals the log encoder legitimately clamps — not representative.)
        chol = rng.standard_normal((n, k)).astype(np.float32)
        diag_idx = np.cumsum(np.arange(1, d + 1)) - 1
        chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.5
        return {
            "centers": (rng.random((n, d)).astype(np.float32) * 10),
            "amplitudes": rng.random(n).astype(np.float32) * 2,
            "cholesky_factors": chol,
        }

    @staticmethod
    def _cov_relF_p95(ref: np.ndarray, got: np.ndarray, d: int) -> float:
        """p95 relative Frobenius error of Σ=LLᵀ between two packed sets."""
        from luxar.gsplats.utils.trils import unpack_tril

        Lr = unpack_tril(ref.astype(np.float64), d)
        Lg = unpack_tril(got.astype(np.float64), d)
        Sr = Lr @ Lr.transpose(0, 2, 1)
        Sg = Lg @ Lg.transpose(0, 2, 1)
        rel = np.linalg.norm(Sg - Sr, axis=(1, 2)) / (
            np.linalg.norm(Sr, axis=(1, 2)) + 1e-30
        )
        return float(np.percentile(rel, 95))

    # ndim=1 included: it is the degenerate case where the off-diagonal array is
    # intentionally omitted (k - d == 0), so it exercises a distinct write/read path.
    # Per-mode precision: PRECISION=float32 (exact); AUTO=uint8 with the encode-time
    # covariance certificate, whose escalation threshold (COV_CERT_RELF_P95_MAX)
    # makes the AUTO bound a hard invariant, not an observation; MEMORY=uint8
    # unconditionally (no certificate).
    _COV_P95_BOUND = {
        EncodingMode.PRECISION: 0.0,
        EncodingMode.AUTO: 0.05,
        EncodingMode.MEMORY: 0.1,
    }
    _DIAG_ENCODING = {
        EncodingMode.PRECISION: "float32",
        EncodingMode.AUTO: "log_perchannel_u8",
        EncodingMode.MEMORY: "log_perchannel_u8",
    }

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "mode", [EncodingMode.PRECISION, EncodingMode.AUTO, EncodingMode.MEMORY]
    )
    def test_roundtrip_dims_and_modes(self, ndim: int, mode: EncodingMode) -> None:
        rng = np.random.default_rng(ndim)
        splats = self._splats(64, ndim, rng)
        k = ndim * (ndim + 1) // 2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.gsplats.zarr"
            save_gsplats(path=path, **splats, encoding_mode=mode, ordering="none")

            # On disk: split arrays, no single packed array. The off-diagonal
            # array is present iff there are off-diagonal elements (d > 1).
            root = zarr.open_group(str(path), mode="r")
            assert "cholesky_factors_diag" in root
            assert "cholesky_factors" not in root
            assert root["cholesky_factors_diag"].shape[1] == ndim
            assert (
                root["cholesky_factors_diag"].attrs["encoding"]["name"]
                == self._DIAG_ENCODING[mode]
            )
            # AUTO carries the covariance certificate as provenance; MEMORY and
            # PRECISION are unconditional tiers and must not.
            diag_enc = dict(root["cholesky_factors_diag"].attrs["encoding"])
            if mode == EncodingMode.AUTO:
                cert = diag_enc["certificate"]
                assert cert["metric"] == "cov_relf_p95"
                assert cert["tier"] == "u8"
                assert cert["value"] <= cert["threshold"]
            else:
                assert "certificate" not in diag_enc
            if k - ndim > 0:
                assert "cholesky_factors_offdiag" in root
                assert root["cholesky_factors_offdiag"].shape[1] == k - ndim
            else:
                assert "cholesky_factors_offdiag" not in root  # d == 1

            # Recombined on read into the packed (N, k) form.
            result = load_gsplats(path)
            assert result.cholesky_factors.shape == (64, k)
            if mode == EncodingMode.PRECISION:
                np.testing.assert_array_equal(
                    result.cholesky_factors, splats["cholesky_factors"]
                )
            else:
                cov_p95 = self._cov_relF_p95(
                    splats["cholesky_factors"], result.cholesky_factors, ndim
                )
                assert cov_p95 <= self._COV_P95_BOUND[mode], (
                    f"{mode} cov relF p95 {cov_p95:.2e} exceeds "
                    f"{self._COV_P95_BOUND[mode]:.0e}"
                )

    def test_roundtrip_uniform_cholesky(self) -> None:
        """Broadcast/uniform Cholesky (shape (1, k)) splits and recombines."""
        rng = np.random.default_rng(7)
        n, d = 50, 3
        k = d * (d + 1) // 2
        uniform = _positive_diag(rng.standard_normal(k).astype(np.float32), d)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "u.gsplats.zarr"
            save_gsplats(
                path=path,
                centers=(rng.random((n, d)).astype(np.float32) * 10),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=np.tile(uniform, (n, 1)),
                ordering="none",
            )
            result = load_gsplats(path)
            for row in result.cholesky_factors:
                np.testing.assert_allclose(row, uniform, rtol=0, atol=1e-5)

    def test_roundtrip_2d_offdiag_single_column(self) -> None:
        """2D has exactly one off-diagonal element (k-d = 1)."""
        rng = np.random.default_rng(2)
        splats = self._splats(40, 2, rng)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t2.gsplats.zarr"
            # default mode = AUTO → uint8 (certified; escalation not triggered here)
            save_gsplats(path=path, **splats, ordering="none")
            root = zarr.open_group(str(path), mode="r")
            assert root["cholesky_factors_offdiag"].shape[1] == 1
            enc = dict(root["cholesky_factors_offdiag"].attrs["encoding"])
            assert enc["name"] == "signed_log_perchannel_u8"
            # the funnel routed through encode_cholesky_split → certificate present
            assert enc["certificate"]["tier"] == "u8"
            result = load_gsplats(path)
            assert (
                self._cov_relF_p95(
                    splats["cholesky_factors"], result.cholesky_factors, 2
                )
                <= 0.05
            )

    def test_corrupt_missing_offdiag_for_dgt1_raises(self) -> None:
        """A d>1 store with the diagonal but no off-diagonal array is corrupt;
        the reader must fail loud rather than silently drop off-diagonals."""
        import shutil

        rng = np.random.default_rng(3)
        splats = self._splats(32, 3, rng)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "c.gsplats.zarr"
            save_gsplats(path=path, **splats, ordering="none")
            # Simulate a partial write: delete the off-diagonal array.
            shutil.rmtree(path / "cholesky_factors_offdiag")
            with pytest.raises(ValueError, match="offdiag.*missing|missing.*offdiag"):
                load_gsplats(path)


class TestLoadGsplats:
    def test_load_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")
            result = load_gsplats(path)
            assert result.centers.shape == (100, 3)
            assert result.amplitudes.shape == (100,)
            assert result.cholesky_factors.shape == (100, 6)
            assert result.centers.dtype == np.float32
            assert result.amplitudes.dtype in (np.float32, np.float16)

    def test_load_with_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                encoding_mode=EncodingMode.MEMORY,
                ordering="none",
            )
            result = load_gsplats(path)
            assert result.centers.dtype in (np.float32, np.float16)
            assert result.amplitudes.dtype in (np.float32, np.float16)

    def test_load_with_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={"time_seconds": 45.3, "iterations": 850},
                description="Test dataset",
                ordering="none",
            )
            result = load_gsplats(path, include_stats=True)
            assert result.stats["time_seconds"] == 45.3
            assert result.stats["description"] == "Test dataset"

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_gsplats("/nonexistent/path.gsplats.zarr")

    def test_load_invalid_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            root = zarr.open_group(str(path), mode="w")
            root.attrs["format_type"] = "wrong_format"
            with pytest.raises(ValueError, match="Invalid format_type"):
                load_gsplats(path)

    def test_load_partition_file_raises(self) -> None:
        """A standalone partition/nested tree has no flat GSplatData equivalent;
        load_gsplats() must raise clearly rather than silently mislead (TC-3)."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

        def _leaf(n, seed):
            rng = np.random.default_rng(seed)
            chol = np.zeros((n, 6), dtype=np.float32)
            chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
                        amplitudes=rng.uniform(0.1, 1, (n,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "part.gsplats.zarr"
            write_gsplats_tree(
                path,
                GSplatPartition(children=[_leaf(20, 0), _leaf(20, 1)]),
                ordering="none",
            )
            with pytest.raises(ValueError, match="(?i)matrix|partition|tree"):
                load_gsplats(path)

    def test_load_rejects_legacy_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old.gsplats.zarr"
            root = zarr.open_group(str(path), mode="w")
            root.attrs["format_type"] = "gsplats_zarr"
            root.attrs["format_version"] = "2.0"
            with pytest.raises(ValueError, match="migrate-format"):
                load_gsplats(path)


class TestRoundTrip:
    def test_roundtrip_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            save_gsplats(
                path=path,
                **splats,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )
            result = load_gsplats(path)
            assert np.allclose(result.centers, splats["centers"])
            assert np.allclose(result.amplitudes, splats["amplitudes"])
            assert np.allclose(result.cholesky_factors, splats["cholesky_factors"])

    def test_roundtrip_with_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            save_gsplats(path=path, **splats, ordering="morton")
            result = load_gsplats(path)
            assert len(result.centers) == len(splats["centers"])

    def test_roundtrip_with_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            save_gsplats(
                path=path, **splats, encoding_mode=EncodingMode.MEMORY, ordering="none"
            )
            result = load_gsplats(path)
            assert np.allclose(result.centers, splats["centers"], atol=0.01)
            assert np.allclose(result.amplitudes, splats["amplitudes"], atol=0.05)


class TestGSplatDataMethods:
    def test_result_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)
            result = GSplatData(
                **splats, stats={"time_seconds": 10.5, "iterations": 500}
            )
            result.save(path, include_fitting_info=True)
            assert path.exists()
            root = zarr.open_group(str(path), mode="r")
            assert root["fitting"].attrs["time_seconds"] == 10.5

    def test_result_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50), ordering="none")
            result = GSplatData.load(path)
            assert result.centers.shape == (50, 3)
            assert isinstance(result, GSplatData)

    def test_result_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            original = GSplatData(**create_test_splats_3d(100), stats={})
            original.save(path, ordering="none")
            loaded = GSplatData.load(path)
            assert np.allclose(loaded.centers, original.centers, atol=0.01)
            assert np.allclose(loaded.amplitudes, original.amplitudes, atol=0.05)


class TestColorRoundtrip:
    def test_roundtrip_with_sdr_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(1).random((100, 3)).astype(np.float32)
            save_gsplats(
                path=path,
                **create_test_splats_3d(100),
                colors=colors,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )
            result = load_gsplats(path)
            assert result.colors is not None
            assert result.colors.shape == (100, 3)
            assert np.allclose(result.colors, colors, atol=0.01)

    def test_roundtrip_with_uint8_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(2).integers(
                0, 256, size=(50, 3), dtype=np.uint8
            )
            save_gsplats(
                path=path, **create_test_splats_3d(50), colors=colors, ordering="none"
            )
            result = load_gsplats(path)
            assert result.colors is not None
            assert result.colors.shape == (50, 3)
            assert result.colors.dtype == np.uint8
            assert np.array_equal(result.colors, colors)

    def test_roundtrip_with_hdr_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(3).random((75, 3)).astype(np.float32) * 5.0
            save_gsplats(
                path=path,
                **create_test_splats_3d(75),
                colors=colors,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )
            result = load_gsplats(path)
            assert result.colors is not None
            assert np.allclose(result.colors, colors, atol=0.05)

    def test_roundtrip_without_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")
            assert load_gsplats(path).colors is None

    def test_result_method_roundtrip_with_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            colors = np.random.default_rng(4).random((100, 3)).astype(np.float32)
            original = GSplatData(
                **create_test_splats_3d(100), colors=colors, stats={"test": "value"}
            )
            original.save(path, ordering="none")
            loaded = GSplatData.load(path)
            assert loaded.colors is not None
            assert loaded.colors.shape == colors.shape
            assert np.allclose(loaded.colors, colors, atol=0.01)


class TestInspectGsplats:
    def test_inspect_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="morton")
            info = inspect_gsplats_zarr(path)
            assert info["n_splats"] == 100
            assert info["ndim"] == 3
            assert info["ordering"] == "morton"
            assert info["has_colors"] is False

    def test_inspect_label_vocabulary_uses_integer_ids(self, tmp_path: Path) -> None:
        data = GSplatData(
            **create_test_splats_3d(4),
            label_ids=np.array([0, 2, 2, 0], dtype=np.uint8),
            label_vocabulary={0: "zero", 2: "two"},
        )
        path = tmp_path / "labels.gsplats.zarr"
        data.save(path, ordering="none")

        info = inspect_gsplats_zarr(path)
        assert info["label_vocabulary"] == {0: "zero", 2: "two"}

    def test_inspect_with_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(
                path=path,
                **create_test_splats_3d(50),
                fitting_info={"time_seconds": 45.3, "iterations": 850},
                ordering="none",
            )
            info = inspect_gsplats_zarr(path)
            assert info["fitting"]["time_seconds"] == 45.3

    def test_inspect_multi_level_pyramid_reports_finest_stats(self) -> None:
        # For a kind=lod pyramid, inspect must descend into the FINEST leaf so
        # its headline stats match the data-model default (finest, index 0) and
        # the `gsplat info` CLI (which loads via GSplatData). On disk child_0 is
        # the COARSEST, so reading default_level would report the wrong (coarsest)
        # count — the data-default vs viewer-hint conflation.
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        def _level(n: int, seed: int) -> SubstitutiveLevel:
            rng = np.random.RandomState(seed)
            chol = np.zeros((n, 6), dtype=np.float32)
            # Packed lower-tri diagonal slots for d=3 are [0, 2, 5].
            chol[:, [0, 2, 5]] = 1.0
            return SubstitutiveLevel(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.rand(n, 3).astype(np.float32) * 50,
                        amplitudes=np.ones(n, dtype=np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pyr.gsplats.zarr"
            # finest=100 at index 0, coarsest=10 at index 2
            data = GSplatData.from_substitutive_levels(
                [_level(100, 0), _level(30, 1), _level(10, 2)]
            )
            data.save(path, ordering="none")

            info = inspect_gsplats_zarr(path)
            assert info["kind"] == "lod"
            assert info["n_substitutive"] == 3
            assert info["default_lod_level"] == 0  # on-disk viewer hint = coarsest
            # Headline n_splats is the FINEST level, matching GSplatData.load.
            assert info["n_splats"] == 100
            assert info["n_splats"] == GSplatData.load(path).n_splats

    def test_inspect_format_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            try:
                save_gsplats(
                    path=path,
                    **create_test_splats_3d(100),
                    fitting_info={
                        "time_seconds": 45.3,
                        "iterations": 850,
                        "converged": True,
                    },
                    ordering="hilbert",
                )
            except ImportError:
                pytest.skip("hilbertcurve package not installed")
            formatted = format_gsplats_info(inspect_gsplats_zarr(path))
            assert "100" in formatted
            assert "3D" in formatted
            assert "hilbert" in formatted
            assert "45.3" in formatted

    @pytest.mark.parametrize(
        "compress,suffix", [("zip", ".zip"), ("tar.gz", ".tar.gz")]
    )
    def test_inspect_archive_matches_directory(
        self, compress: str, suffix: str
    ) -> None:
        # An archive nests its store one directory deep, so handing the archive
        # itself to open_group used to raise GroupNotFoundError (#1625).
        with tempfile.TemporaryDirectory() as tmpdir:
            splats = create_test_splats_3d(100)
            directory = Path(tmpdir) / "d.gsplats.zarr"
            archive = Path(tmpdir) / f"a.gsplats.zarr{suffix}"
            save_gsplats(path=directory, **splats, ordering="none")
            save_gsplats(path=archive, **splats, ordering="none", compress=compress)

            info = inspect_gsplats_zarr(archive)
            reference = inspect_gsplats_zarr(directory)
            assert info["n_splats"] == reference["n_splats"] == 100
            assert info["ndim"] == reference["ndim"] == 3
            # Compare the WHOLE dict, not just the headline pair: resolving to the
            # wrong node inside the archive (a child group, an array subdirectory)
            # would keep n_splats plausible while everything else drifted. Only the
            # size/measurement keys legitimately differ — the archive is measured as
            # one compressed file, the directory as a file walk — and `timestamp`
            # comes from two separate save calls.
            measured = {"storage_bytes", "storage_mb", "compression_ratio", "timestamp"}
            assert {k: v for k, v in info.items() if k not in measured} == {
                k: v for k, v in reference.items() if k not in measured
            }

    @pytest.mark.parametrize("n_splats", [1000, 20000])
    def test_inspect_archive_size_is_the_archive_file(self, n_splats: int) -> None:
        # A single-file archive's own st_size IS its on-disk size; summing zero
        # there reported "0 bytes, 1.0x compression" (#1625).
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "a.gsplats.zarr.zip"
            save_gsplats(
                path=archive,
                **create_test_splats_3d(n_splats),
                ordering="none",
                compress="zip",
            )
            info = inspect_gsplats_zarr(archive)

            assert info["storage_bytes"] == archive.stat().st_size > 0
            # A real measurement, not the old `total_bytes > 0 else 1.0` guard.
            # Checked against the unrounded byte model (see
            # `_uncompressed_model_bytes`) so the expectation holds at ANY fixture
            # size; the field itself is rounded to 2 dp, hence `abs=`.
            expected = _uncompressed_model_bytes(n_splats) / info["storage_bytes"]
            assert info["compression_ratio"] == pytest.approx(expected, abs=0.005)
            assert info["compression_ratio"] > 1.0

    @pytest.mark.parametrize("write_format", (2, 3), ids=("v2", "v3"))
    def test_inspect_flat_zip_reports_the_archive_size(
        self, tmp_path: Path, monkeypatch, write_format: int
    ) -> None:
        # The FLAT archive shape: the store sits at the archive ROOT, so the
        # store-root resolution step never bites and this used to inspect fine —
        # while reporting `storage_bytes: 0` and a fabricated `compression_ratio:
        # 1.0` (exactly the case #1625 used to demonstrate the size bug). It must
        # keep working AND now report a real size.
        #
        # Both on-disk formats: the flat-store classifier matches the group
        # document BY NAME (`.zgroup` at 2, `zarr.json` at 3), so pinning only
        # the ambient format leaves half of `NODE_GROUP_DOCS` untested.
        from luxar._zarr_compat import ZARR_FORMAT, set_zarr_format

        confine_temp_dirs(tmp_path, monkeypatch)
        n_splats = 20000
        original = ZARR_FORMAT
        set_zarr_format(write_format)
        try:
            store = tmp_path / "d.gsplats.zarr"
            save_gsplats(path=store, **create_test_splats_3d(n_splats), ordering="none")
        finally:
            set_zarr_format(original)
        archive = _zip_store_flat(store, tmp_path / "flat.gsplats.zarr.zip")

        before = _luxar_temp_dirs()
        info = inspect_gsplats_zarr(archive)
        # A flat zip is opened in place as a ZipStore — nothing is extracted.
        assert _luxar_temp_dirs() - before == set()

        assert info["n_splats"] == n_splats
        assert info["ndim"] == 3
        assert info["storage_bytes"] == archive.stat().st_size > 0
        expected = _uncompressed_model_bytes(n_splats) / info["storage_bytes"]
        assert info["compression_ratio"] == pytest.approx(expected, abs=0.005)
        assert info["compression_ratio"] != 1.0

    def test_inspect_nested_archive_removes_its_temp_dir(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Cleanup was unverified: neutering the `shutil.rmtree` left the whole
        # suite green. A nested archive IS extracted, so its temp directory must be
        # gone afterwards.
        confine_temp_dirs(tmp_path, monkeypatch)
        archive = tmp_path / "a.gsplats.zarr.zip"
        save_gsplats(
            path=archive,
            **create_test_splats_3d(100),
            ordering="none",
            compress="zip",
        )
        before = _luxar_temp_dirs()
        assert inspect_gsplats_zarr(archive)["n_splats"] == 100
        assert _luxar_temp_dirs() - before == set()

    def test_inspect_removes_temp_dir_when_inspection_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The cleanup is in a `finally`, so it must also survive the raising path:
        # an archive of a plain zarr group is a resolvable store that is not a
        # gsplats store.
        from luxar._zarr_compat import open_group as zc_open_group

        confine_temp_dirs(tmp_path, monkeypatch)
        store = tmp_path / "bad.gsplats.zarr"
        zc_open_group(str(store), mode="w")
        archive = tmp_path / "bad.gsplats.zarr.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for f in sorted(store.rglob("*")):
                if f.is_file():
                    zip_ref.write(f, arcname=f"{store.name}/{f.relative_to(store)}")

        before = _luxar_temp_dirs()
        with pytest.raises(ValueError, match="format_type"):
            inspect_gsplats_zarr(archive)
        assert _luxar_temp_dirs() - before == set()

    def test_inspect_reports_an_unmeasurable_size_as_absent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Drive the inspector's OWN `except` arm — mutating it to bogus values
        # stayed green before this test, because the formatter test below only
        # pins the presentation, on a hand-built dict. A directory store whose
        # walk raises is the reachable case: unreadable permissions, a vanished
        # file, a stale mount. This does NOT also close the `if total_bytes > 0
        # else None` ternary: a real store always has bytes, so that `else` is
        # unreachable through inspect and no test here can drive it.
        real_rglob = Path.rglob
        store = tmp_path / "d.gsplats.zarr"
        save_gsplats(path=store, **create_test_splats_3d(100), ordering="none")

        def boom(self: Path, pattern, *args, **kwargs):
            if self == store:
                raise OSError("permission denied")
            return real_rglob(self, pattern, *args, **kwargs)

        monkeypatch.setattr(Path, "rglob", boom)
        info = inspect_gsplats_zarr(store)

        # The inspection itself still succeeds — only the size is unknown.
        assert info["n_splats"] == 100
        assert info["storage_bytes"] is None
        assert info["storage_mb"] is None
        # Not a fabricated 1.0: a ratio against an unmeasurable size is not a
        # measurement.
        assert info["compression_ratio"] is None
        # And with no size at all the formatter omits the whole line rather than
        # printing "None MB".
        assert "Size:" not in format_gsplats_info(info)

    def test_format_info_without_compression_ratio(self) -> None:
        # Inspect CAN produce `compression_ratio=None` (the test 30 lines above
        # does); what it cannot produce is this COMBINATION — a known
        # `storage_mb` with no ratio — because the only route to a `None` ratio
        # also loses the size, and then the formatter omits the line entirely.
        # So the branch is exercised directly, on a hand-built dict: it must
        # print the size line with no compression figure.
        line = format_gsplats_info(
            {
                "n_splats": 100,
                "ndim": 3,
                "ordering": "none",
                "has_colors": False,
                "storage_mb": 1.5,
                "uncompressed_mb": 3.0,
                "compression_ratio": None,
            }
        )
        assert "Size: 1.5 MB (3.0 MB uncompressed)" in line
        assert "compression" not in line

    def test_inspect_directory_size_walks_recursively(self) -> None:
        # Unchanged behaviour for a directory store: the recursive file walk.
        # Kept as an honest negative control — a directory was always walked, so
        # this cannot fail pre-fix; it guards the walk against a future
        # regression while the archive tests above carry the #1625 claim.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100), ordering="none")
            info = inspect_gsplats_zarr(path)
            expected = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            assert info["storage_bytes"] == expected > 0

    def test_inspect_format_output_archive_size_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "a.gsplats.zarr.zip"
            save_gsplats(
                path=archive,
                **create_test_splats_3d(20000),
                ordering="none",
                compress="zip",
            )
            info = inspect_gsplats_zarr(archive)
            formatted = format_gsplats_info(info)
            assert "0.0 MB (" not in formatted
            # Assert against what the formatter actually consumes. Recomputing
            # `st_size / MB` here and formatting it to 1 dp disagrees with the
            # code's `round(..., 2)`-then-`:.1f` for any value in a
            # [x.x45, x.x50) band, so it could fail on correct code after a
            # fixture or encoder change.
            assert f"Size: {info['storage_mb']:.1f} MB" in formatted
            assert info["storage_bytes"] == archive.stat().st_size > 0

    def test_inspect_rejects_non_archive_regular_file(self) -> None:
        # Same clear error the loader gives, instead of a zarr-level failure.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("not a zarr store")
            with pytest.raises(ValueError, match="regular file"):
                inspect_gsplats_zarr(path)


class TestFlatZipClassifier:
    """The two rules of ``_archive._zip_is_flat_store``, plus the opt-in gate.

    The rules are exercised on the private classifier directly. That is the
    honest unit under test: each rule decides between "open the archive in place"
    and "extract it", and the observable difference for a SYNTHETIC archive (no
    real store inside either candidate root) is an unrelated zarr error either
    way. Both rules survived deletion with the whole `gsplats/io` suite green
    before these tests existed.
    """

    @staticmethod
    def _write_zip(path: Path, members: list[tuple[str, str]]) -> None:
        """Write a zip with ``(name, text)`` members in exactly the given order."""
        with zipfile.ZipFile(path, "w") as zip_ref:
            for name, text in members:
                zip_ref.writestr(name, text)

    @pytest.mark.parametrize("doc", [".zgroup", "zarr.json"])
    def test_a_nested_store_beats_a_stray_root_group_doc(
        self, tmp_path: Path, doc: str
    ) -> None:
        """A ``*.gsplats.zarr/`` directory wins even with a group doc at depth 0.

        The extractor prefers that directory, so classifying this archive flat
        would open a DIFFERENT node than the extractor resolves — the two must
        not disagree. The stray is written FIRST so an order-dependent
        implementation fails here.
        """
        from luxar.gsplats.io._archive import _zip_is_flat_store

        archive = tmp_path / "both.gsplats.zarr.zip"
        self._write_zip(
            archive,
            [
                (doc, '{"zarr_format": 2, "node_type": "group"}'),
                (f"x.gsplats.zarr/{doc}", '{"zarr_format": 2, "node_type": "group"}'),
                ("x.gsplats.zarr/centers/.zarray", "{}"),
            ],
        )
        assert _zip_is_flat_store(archive) is False

    def test_a_root_zattrs_is_not_a_root_group_doc(self, tmp_path: Path) -> None:
        """A depth-0 ``.zattrs`` does NOT mark a store root (``NODE_GROUP_DOCS``).

        ``.zattrs`` also sits beside an ARRAY and may simply be a stray, so it
        says nothing about a group being here — what
        ``zip -r out.gsplats.zarr.zip mystore .zattrs`` produces is the nested
        (extract-me) shape, not a flat store.
        """
        from luxar.gsplats.io._archive import _zip_is_flat_store

        archive = tmp_path / "stray_attrs.gsplats.zarr.zip"
        self._write_zip(
            archive,
            [
                (".zattrs", '{"opacity": 0.5}'),
                ("mystore/.zgroup", '{"zarr_format": 2}'),
                ("mystore/.zattrs", '{"format_type": "gsplats_zarr"}'),
            ],
        )
        assert _zip_is_flat_store(archive) is False

    def test_a_bare_gsplats_zarr_directory_entry_still_counts(
        self, tmp_path: Path
    ) -> None:
        """An empty ``x.gsplats.zarr/`` ENTRY is the nested shape, not a flat one.

        ``zip -r`` emits a bare entry for a subdirectory with nothing under it,
        and ``extractall`` materializes it — so the extractor's first tier picks
        that directory. The classifier's other rule only sees members WITH a
        parent component, so without this the two disagreed: the inspector opened
        the flat root in place and reported a dataset, while the loader extracted
        and died on the empty directory.
        """
        from luxar.gsplats.io._archive import _zip_is_flat_store

        archive = tmp_path / "bare.gsplats.zarr.zip"
        self._write_zip(
            archive,
            [
                ("x.gsplats.zarr/", ""),
                (".zgroup", '{"zarr_format": 2}'),
                (".zattrs", '{"format_type": "gsplats_zarr"}'),
                ("centers/.zarray", "{}"),
            ],
        )
        assert _zip_is_flat_store(archive) is False

    def test_a_depth_zero_file_named_like_a_store_is_not_a_directory(
        self, tmp_path: Path
    ) -> None:
        """The bare-entry rule is guarded on ``is_dir()``, so a FILE cannot fire it.

        A depth-0 regular file called ``x.gsplats.zarr`` extracts as a file; no
        tier of the extractor picks it, and the flat store beside it is still the
        right answer.
        """
        from luxar.gsplats.io._archive import _zip_is_flat_store

        archive = tmp_path / "filenamed.gsplats.zarr.zip"
        self._write_zip(
            archive,
            [
                ("x.gsplats.zarr", "not a directory"),
                (".zgroup", '{"zarr_format": 2}'),
            ],
        )
        assert _zip_is_flat_store(archive) is True

    def test_opening_a_flat_zip_in_place_stays_opt_in(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """``resolve_store_path``'s default resolves a flat zip by EXTRACTION.

        Both routes work, so the opt-in is about COST, not capability: reading a
        flat zip in place as a ``ZipStore`` needs no temp space, which is the
        whole point for the metadata-only inspector, while extraction costs the
        full uncompressed size. The loader reads array data through a resolved
        directory store and takes the extraction path. Pinned at the seam itself
        because flipping the default to ``True`` otherwise leaves the whole
        `gsplats/io` suite green.
        """
        from luxar._zarr_compat import open_group as zc_open_group
        from luxar.gsplats.io._archive import resolve_store_path

        confine_temp_dirs(tmp_path, monkeypatch)
        store = tmp_path / "d.gsplats.zarr"
        save_gsplats(path=store, **create_test_splats_3d(50), ordering="none")
        archive = _zip_store_flat(store, tmp_path / "flat.gsplats.zarr.zip")

        # Opted in: the archive itself, and nothing to clean up.
        assert resolve_store_path(archive, flat_zip_in_place=True) == (archive, None)

        # Default: the extraction path instead, which now resolves the flat store
        # DETERMINISTICALLY (it used to land on whatever `iterdir()` yielded
        # first, or raise) — so the resolved path is asserted, not just the seam.
        resolved, temp_dir = resolve_store_path(archive)
        try:
            assert resolved != archive
            assert resolved.is_dir()
            root = zc_open_group(str(resolved), mode="r")
            assert root.attrs["format_type"] == "gsplats_zarr"
            # `temp_dir == resolved.parent` is a tautology (the resolver returns
            # exactly that pair), so the contract has to be asserted on the
            # LAYOUT: a flat tree is moved INSIDE a fresh extractor temp dir
            # rather than being one. Handing back the extraction dir as-is would
            # make every caller's `rmtree(temp_dir)` remove the system temp root.
            assert resolved.parent.name.startswith("luxar_gsplat_archive_")
            assert resolved.parent != Path(tempfile.gettempdir())
            assert resolved.name != resolved.parent.name
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


class TestFlatArchiveStoreRoot:
    """A FLAT archive is a readable store for every reader (issue #1628).

    "Flat" = the zarr store sits at the archive ROOT (``zarr.json`` / ``.zgroup``
    plus ``centers/…`` at depth 0), which is what ``zip -r x.gsplats.zarr.zip .``
    from inside a store, or a zarr-native ``ZipStore`` write, produces.
    ``save_gsplats(compress=…)`` never writes one, but they exist in the wild.

    Before #1628 only the metadata inspector handled the shape (it opens a flat
    zip in place as a ``ZipStore``): the extractor defined the store as a
    top-level DIRECTORY, so its fallback picked an arbitrary array
    sub-directory — the loader raised ``ContainsArrayError``, the appearance peek
    refused a depth-0 attrs document by design and answered ``{}``, and
    ``gsplat info`` reported one part of a partition as the whole dataset.

    Every case runs on BOTH on-disk formats (the store root's document is named
    ``.zgroup``/``.zattrs`` at 2 and a single ``zarr.json`` — group document AND
    attrs document at once — at 3, and both are matched by name) and BOTH
    containers (a ``.tar.gz`` has no in-place route at all).
    """

    CONTAINERS = ["zip", "tar.gz"]

    @staticmethod
    def _store(tmp_path: Path, write_format: int, **save_kwargs) -> Path:
        """A real single-leaf store written at ``write_format``."""
        from luxar._zarr_compat import ZARR_FORMAT, set_zarr_format

        original = ZARR_FORMAT
        set_zarr_format(write_format)
        try:
            store = tmp_path / "d.gsplats.zarr"
            GSplatData(**create_test_splats_3d(64)).save(
                store, ordering="none", **save_kwargs
            )
            return store
        finally:
            set_zarr_format(original)

    @classmethod
    def _flat(cls, store: Path, tmp_path: Path, container: str) -> Path:
        archive = tmp_path / f"flat.gsplats.zarr.{container}"
        if container == "zip":
            return _zip_store_flat(store, archive)
        return _targz_store_flat(store, archive)

    @staticmethod
    def _write_members(
        path: Path, container: str, members: list[tuple[str, str]]
    ) -> None:
        """Write a synthetic archive with ``(name, text)`` members, in order.

        A trailing ``/`` means a DIRECTORY member — a zip spells one as exactly
        that trailing-slash entry, a tar as an explicit ``DIRTYPE`` header.
        """
        import io
        import tarfile

        if container == "zip":
            with zipfile.ZipFile(path, "w") as zip_ref:
                for name, text in members:
                    zip_ref.writestr(name, text)
            return
        with tarfile.open(path, "w:gz") as tar_ref:
            for name, text in members:
                if name.endswith("/"):
                    dir_info = tarfile.TarInfo(name.rstrip("/"))
                    dir_info.type = tarfile.DIRTYPE
                    tar_ref.addfile(dir_info)
                    continue
                payload = text.encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar_ref.addfile(info, io.BytesIO(payload))

    @pytest.mark.parametrize("container", CONTAINERS)
    @pytest.mark.parametrize("write_format", (2, 3), ids=("v2", "v3"))
    def test_the_loader_reads_a_flat_archive(
        self, tmp_path: Path, monkeypatch, container: str, write_format: int
    ) -> None:
        """``load_gsplats`` / ``load_gsplat_node`` resolve the flat store root.

        This is the ``ContainsArrayError`` row of #1628: the extractor's fallback
        handed ``open_group`` an ARRAY sub-directory (``…/amplitudes``).
        """
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import node_ndim, total_splats

        confine_temp_dirs(tmp_path, monkeypatch)
        archive = self._flat(self._store(tmp_path, write_format), tmp_path, container)

        data = load_gsplats(archive)
        assert data.n_splats == 64
        assert data.ndim == 3

        node, _ = load_gsplat_node(archive)
        assert total_splats(node) == 64
        assert node_ndim(node) == 3

    @pytest.mark.parametrize("container", CONTAINERS)
    @pytest.mark.parametrize("write_format", (2, 3), ids=("v2", "v3"))
    def test_a_flat_archives_authored_appearance_survives(
        self, tmp_path: Path, container: str, write_format: int
    ) -> None:
        """The appearance peek reads a flat store's own root document.

        The measured symptom: ``gsplat lod flat.gsplats.zarr.zip out/`` wrote
        ``absorption 1.0, blending None, opacity 1.0`` where the source authored
        ``0.7 / volumetric / 0.5``, because the peek refused a depth-0 attrs
        document and ``{}`` is indistinguishable from "authored nothing".
        """
        from luxar.gsplats.io.load_gsplats import read_authored_appearance

        authored = {"absorption": 0.7, "blending_mode": "volumetric", "opacity": 0.5}
        store = self._store(tmp_path, write_format, root_attrs=authored)
        archive = self._flat(store, tmp_path, container)

        got = read_authored_appearance(archive)
        for key, value in authored.items():
            assert got.get(key) == value, f"{key}: {got!r}"
        # The archive must answer exactly what the same store answers as a
        # directory — the invariant, rather than a hand-listed key set.
        assert got == read_authored_appearance(store)

    @pytest.mark.parametrize("container", CONTAINERS)
    @pytest.mark.parametrize("write_format", (2, 3), ids=("v2", "v3"))
    def test_a_flat_archive_that_authored_nothing_carries_nothing(
        self, tmp_path: Path, container: str, write_format: int
    ) -> None:
        """A store that authored nothing answers its own root, not a sub-node's.

        A real store, so the assertion is about the shipped shape: its ARRAYS are
        the archive's only top-level directories, and ``centers/.zattrs`` (the
        tier below) must never stand in for the dataset's root. The writer stamps
        the identity appearance on every root, so "authored nothing" is not an
        empty answer — it is the same answer the directory store gives, which is
        the invariant worth pinning.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs
        from luxar.gsplats.io.load_gsplats import read_authored_appearance

        store = self._store(tmp_path, write_format)
        archive = self._flat(store, tmp_path, container)

        assert read_authored_appearance(archive) == read_authored_appearance(store)
        raw = read_archive_root_attrs(archive)
        # The structural key proves the ROOT document was read; `encoding` is an
        # array node's own attr and would betray a sub-directory standing in.
        assert raw["format_type"] == "gsplats_zarr"
        assert "encoding" not in raw

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_loading_a_flat_archive_leaves_no_temp_dir(
        self, tmp_path: Path, monkeypatch, container: str
    ) -> None:
        """Resolving a flat store re-parents the extraction; BOTH dirs must go.

        The store-root resolution moves the extracted tree one level down so the
        caller still gets a removable ``zarr_path.parent``, which means two temp
        directories are created and the caller only ever removes one. This is the
        test that catches the other one leaking.

        The leak probe alone would NOT catch the re-parenting being skipped: the
        caller's ``rmtree`` would then take the temp ROOT this probe globs, and
        "no new directories" holds vacuously over a wiped tree. Hence the second
        assertion — the fixtures live in that root, and they must survive a load.
        """
        confine_temp_dirs(tmp_path, monkeypatch)
        store = self._store(tmp_path, 3)
        archive = self._flat(store, tmp_path, container)

        before = _luxar_temp_dirs()
        assert load_gsplats(archive).n_splats == 64
        assert _luxar_temp_dirs() - before == set()
        assert archive.exists() and store.is_dir()

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_resolved_flat_store_sits_inside_its_temp_dir(
        self, tmp_path: Path, monkeypatch, container: str
    ) -> None:
        """The re-parenting contract, asserted as the layout it produces.

        Every caller treats ``zarr_path.parent`` as the removable temp directory,
        so for the flat shape — where the store IS the extraction directory — the
        tree has to be moved into a second one. Returned as-is, that parent is the
        SYSTEM temp root, and the first caller to clean up removes it. Asserting
        ``temp_dir == resolved.parent`` cannot catch that (the resolver returns
        exactly that pair whatever it resolved), so the directory names are.
        """
        from luxar.gsplats.io._archive import resolve_store_path

        confine_temp_dirs(tmp_path, monkeypatch)
        archive = self._flat(self._store(tmp_path, 3), tmp_path, container)

        resolved, temp_dir = resolve_store_path(archive)
        try:
            assert temp_dir == resolved.parent
            assert resolved.parent.name.startswith("luxar_gsplat_archive_")
            assert resolved.parent != Path(tempfile.gettempdir())
            assert resolved.name != resolved.parent.name
            assert resolved.name == "flat.gsplats.zarr"
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.parametrize(
        "archive_name",
        [
            "x.gsplats.zarr.zip",
            "x.gsplats.zarr.tar.gz",
            "x.zip",
            "x.tar.gz",
            ".zip",
            "..zip",
            "...zip",
            "..tar.gz",
            ".tar.gz",
        ],
    )
    def test_the_reparented_directory_name_is_always_a_fresh_component(
        self, archive_name: str
    ) -> None:
        """``_flat_store_dir_name`` answers one fresh, non-hidden component.

        Freshness is not what the ``("", ".", "..")`` fallback buys. It sits
        BEFORE the unconditional suffix append, so with it removed the stripped
        forms measure as ``""`` → ``.gsplats.zarr``, ``"."`` → ``..gsplats.zarr``
        and ``".."`` → ``...gsplats.zarr`` — each still a single new component,
        never ``outer`` itself nor its parent. What the fallback buys is the LAST
        assertion: ``.tar.gz`` strips to nothing and ``..zip`` strips to a dot,
        and both would name a HIDDEN directory for ``gsplat view`` to log as the
        store it is serving. (``.zip`` is carried for symmetry only and never
        reaches this function: ``Path(".zip").suffix`` is empty, so ``_is_zip``
        is False and ``extract_compressed_zarr`` refuses it as an unsupported
        format first.) Every fixture elsewhere in this file is already
        ``*.gsplats.zarr.<ext>``, exercising neither the append nor the dot
        cases, so they are pinned here directly. The ``parts`` assertion is a
        cheap structural guard rather than a load-bearing one — it holds for
        every input listed with or without the fallback.
        """
        from luxar.gsplats.io._archive import _flat_store_dir_name

        name = _flat_store_dir_name(archive_name)
        assert name.endswith(".gsplats.zarr")
        assert Path(name).parts == (name,)
        assert name not in ("", ".", "..")
        assert not name.startswith(".")

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_bare_gsplats_zarr_entry_stops_the_peek_calling_it_flat(
        self, tmp_path: Path, monkeypatch, container: str
    ) -> None:
        """The peek's flat rule sees a BARE ``x.gsplats.zarr/`` entry too.

        ``zip -r`` emits one for an empty subdirectory and a tar records it
        explicitly; ``extractall`` materializes it either way, so the extractor's
        first tier picks that directory. The peek used to derive "is there such a
        directory" from ``top_dirs``, which is built from member names with a
        parent component and so cannot see a bare entry — it answered the flat
        root's attrs while the extractor resolved the empty directory and the
        load died on it. Now both refuse the flat root.
        """
        from luxar.gsplats.io._archive import (
            read_archive_root_attrs,
            resolve_store_path,
        )

        confine_temp_dirs(tmp_path, monkeypatch)
        archive = tmp_path / f"bare.gsplats.zarr.{container}"
        self._write_members(
            archive,
            container,
            [
                ("x.gsplats.zarr/", ""),
                (".zgroup", '{"zarr_format": 2}'),
                (".zattrs", '{"whose": "flat-root"}'),
                ("centers/.zarray", "{}"),
            ],
        )

        assert read_archive_root_attrs(archive) == {}
        resolved, temp_dir = resolve_store_path(archive)
        try:
            assert temp_dir is not None
            assert temp_dir.name.startswith("luxar_gsplat_archive_")
            assert resolved.name == "x.gsplats.zarr"
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_depth_zero_file_named_like_a_store_does_not_stop_the_peek(
        self, tmp_path: Path, monkeypatch, container: str
    ) -> None:
        """The peek-side twin of the classifier's ``…_is_not_a_directory`` test.

        A depth-0 regular file called ``x.gsplats.zarr`` is not a store: nothing
        can be under it, ``extractall`` writes it out as a file, and no tier of
        the extractor picks it — the flat store beside it is still the right
        answer. So the peek's directory rule takes ``is_dir`` and refuses to fire
        on a name alone; drop that guard and the peek answers a CHILD's attrs (or
        ``{}``) while the classifier and the extractor still read the flat root,
        which is the gate/resolution disagreement #1628 exists to end. Both sides
        of the mirror are asserted here for the one archive.
        """
        from luxar.gsplats.io._archive import (
            read_archive_root_attrs,
            resolve_store_path,
        )

        confine_temp_dirs(tmp_path, monkeypatch)
        archive = tmp_path / f"filenamed.gsplats.zarr.{container}"
        self._write_members(
            archive,
            container,
            [
                ("x.gsplats.zarr", "not a directory"),
                (".zgroup", '{"zarr_format": 2}'),
                (".zattrs", '{"whose": "flat-root"}'),
                ("centers/.zarray", "{}"),
                ("centers/.zattrs", '{"whose": "child"}'),
            ],
        )

        assert read_archive_root_attrs(archive) == {"whose": "flat-root"}
        resolved, temp_dir = resolve_store_path(archive)
        try:
            assert resolved.name == "filenamed.gsplats.zarr"
            assert (resolved / "x.gsplats.zarr").is_file()
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("container", CONTAINERS)
    @pytest.mark.parametrize("doc", [".zgroup", "zarr.json"])
    def test_a_depth_zero_directory_named_like_a_group_doc_is_not_flat(
        self, tmp_path: Path, monkeypatch, container: str, doc: str
    ) -> None:
        """The flat tier is gated on the depth-0 entry being a FILE.

        A DIRECTORY called ``zarr.json`` / ``.zgroup`` is no group document —
        nothing reads attributes out of it — so the archive is not flat and its
        whole tree must not be re-parented as though the root were the store. The
        symlink half of the same guard is unreachable now that both extractors
        reject a link member before extraction, and stays as defence in depth;
        the ``is_file()`` half is reachable and was the suite's last surviving
        mutant.
        """
        from luxar.gsplats.io._archive import resolve_store_path

        confine_temp_dirs(tmp_path, monkeypatch)
        archive = tmp_path / f"flat.gsplats.zarr.{container}"
        self._write_members(archive, container, [(f"{doc}/junk", "not a document")])

        resolved, temp_dir = resolve_store_path(archive)
        try:
            # The directory tier, not the flat one — which would have re-parented
            # the extraction under a store-shaped name taken from the archive.
            assert resolved.name == doc
            assert resolved.name != "flat.gsplats.zarr"
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_stray_root_zattrs_still_loses_to_a_real_store(
        self, tmp_path: Path, container: str
    ) -> None:
        """A depth-0 ``.zattrs`` with no group document beside it is still a stray.

        The flat tier is gated on a depth-0 GROUP document precisely so this case
        keeps behaving: ``.zattrs`` also sits beside an ARRAY, so on its own it
        says nothing about a store root being here. The classifier already holds
        this line (``TestFlatZipClassifier``); the peek must hold it too.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"stray.gsplats.zarr.{container}"
        self._write_members(
            archive,
            container,
            [
                (".zattrs", '{"whose": "stray"}'),
                ("mystore/.zgroup", '{"zarr_format": 2}'),
                ("mystore/.zattrs", '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_flat_root_without_attrs_does_not_fall_back_to_a_child(
        self, tmp_path: Path, container: str
    ) -> None:
        """A flat store that authored no root attrs carries NOTHING, not a child's.

        At format 2 a store with no root attributes writes no depth-0 ``.zattrs``
        at all, so the flat tier has no candidate. Falling through to the "sole
        top-level directory" tier would then hand back an array sub-directory's
        attrs as the dataset's appearance; refusing is the safe answer.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"noattrs.gsplats.zarr.{container}"
        self._write_members(
            archive,
            container,
            [
                (".zgroup", '{"zarr_format": 2}'),
                ("centers/.zarray", "{}"),
                ("centers/.zattrs", '{"whose": "child"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {}

    @pytest.mark.parametrize("container", CONTAINERS)
    @pytest.mark.parametrize("doc", [".zgroup", "zarr.json"])
    def test_a_named_directory_beats_a_stray_root_group_doc(
        self, tmp_path: Path, monkeypatch, container: str, doc: str
    ) -> None:
        """A top-level ``*.gsplats.zarr/`` wins over a depth-0 group document.

        Both resolutions must agree on that (the flat-zip classifier already
        does), so extraction and the peek are pinned together here — an archive
        carrying both shapes is odd, and the nested one is what
        ``_compress_zarr`` writes.
        """
        from luxar.gsplats.io._archive import (
            read_archive_root_attrs,
            resolve_store_path,
        )

        confine_temp_dirs(tmp_path, monkeypatch)
        archive = tmp_path / f"both.gsplats.zarr.{container}"
        if doc == ".zgroup":
            members = [
                (".zgroup", '{"zarr_format": 2}'),
                (".zattrs", '{"whose": "stray"}'),
                ("x.gsplats.zarr/.zgroup", '{"zarr_format": 2}'),
                ("x.gsplats.zarr/.zattrs", '{"whose": "store"}'),
            ]
        else:
            # At format 3 one `zarr.json` IS both the group document and the
            # attrs document, so the stray is a single member here.
            v3 = '{"zarr_format": 3, "node_type": "group", "attributes": %s}'
            members = [
                ("zarr.json", v3 % '{"whose": "stray"}'),
                ("x.gsplats.zarr/zarr.json", v3 % '{"whose": "store"}'),
            ]
        self._write_members(archive, container, members)

        assert read_archive_root_attrs(archive)["whose"] == "store"
        resolved, temp_dir = resolve_store_path(archive)
        try:
            assert resolved.name == "x.gsplats.zarr"
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_stray_depth_zero_files_do_not_hide_the_store_directory(
        self, tmp_path: Path, monkeypatch, container: str
    ) -> None:
        """The last tier takes the first top-level DIRECTORY, not ``children[0]``.

        ``zip -r x.gsplats.zarr.zip mystore README.md``, or a macOS zip that
        picked up a ``.DS_Store``: one real store one level down, plus loose files
        at depth 0. Requiring the first ``iterdir()`` entry to be a directory made
        that a hard "No .gsplats.zarr directory found" on some filesystems and a
        clean load on others, while the peek resolved the store either way — the
        gate/resolution disagreement #1628 exists to end.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        confine_temp_dirs(tmp_path, monkeypatch)
        payload = tmp_path / "payload"
        payload.mkdir()
        shutil.move(str(self._store(tmp_path, 3)), str(payload / "mystore"))
        for stray in ("a.txt", "000", ".DS_Store", "README.md", "zzz.txt"):
            (payload / stray).write_text("stray")
        archive = self._flat(payload, tmp_path, container)

        assert load_gsplats(archive).n_splats == 64
        assert read_archive_root_attrs(archive)["format_type"] == "gsplats_zarr"

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_wrapper_group_around_a_store_resolves_to_the_wrapper(
        self, tmp_path: Path, monkeypatch, container: str
    ) -> None:
        """The flat tier outranks the directory tier — including where it costs.

        A parent zarr GROUP whose only child is the real store, archived flat
        (``zip -r x.gsplats.zarr.zip .`` from inside that parent), used to fall
        through to the sole-directory tier and load. It now resolves to the
        wrapper and fails loudly with ``Invalid format_type``. That reversal is
        accepted, not accidental: names alone cannot tell this apart from a store
        whose root has one child group, and gating the flat tier on "the sole
        child is not itself a group" would break the flat PARTITION that is
        #1628's own repro. Pinned so swapping the tiers back is a red build.
        """
        from luxar.gsplats.io._archive import (
            read_archive_root_attrs,
            resolve_store_path,
        )

        confine_temp_dirs(tmp_path, monkeypatch)
        wrapper = tmp_path / "wrapper"
        wrapper.mkdir()
        shutil.move(str(self._store(tmp_path, 3)), str(wrapper / "mystore"))
        (wrapper / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
        )
        archive = self._flat(wrapper, tmp_path, container)

        resolved, temp_dir = resolve_store_path(archive)
        try:
            # The wrapper won: the real store is a CHILD of what was resolved.
            assert (resolved / "mystore").is_dir()
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        with pytest.raises(ValueError, match="format_type"):
            load_gsplats(archive)
        assert read_archive_root_attrs(archive) == {}


class TestCompression:
    """Blosc compression + chunk sizing (v3.0 leaf-root paths)."""

    def test_default_compression_applied(self):

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(100))
            centers = zarr.open(str(path), mode="r")["centers"]
            comp = array_compressor(centers)
            assert comp is not None and comp.cname == "zstd"

    def test_compression_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, compressor=None, **create_test_splats_3d(100))
            assert array_compressor(zarr.open(str(path), mode="r")["centers"]) is None

    def test_chunk_capping_small_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, **create_test_splats_3d(50))
            root = zarr.open(str(path), mode="r")
            assert root["centers"].chunks[0] <= 50
            assert root["amplitudes"].chunks[0] <= 50
            # Both Cholesky halves share the same row-chunk size as centers.
            assert root["cholesky_factors_diag"].chunks[0] <= 50
            assert root["cholesky_factors_offdiag"].chunks[0] <= 50
            assert (
                root["cholesky_factors_diag"].chunks[0]
                == root["cholesky_factors_offdiag"].chunks[0]
            )

    def test_gsplatdata_save_default_compression(self):

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            GSplatData(**create_test_splats_3d(100)).save(path)
            # `array_compressor` is the bi-format spelling of "is this Blosc?":
            # it RAISES for a real non-blosc compressor and answers None only
            # for a genuinely raw array, so a non-None view IS the isinstance
            # check. Asking `.compressor` directly would raise at format 3.
            assert (
                array_compressor(zarr.open(str(path), mode="r")["centers"]) is not None
            )

    def test_multi_lod_compression(self):

        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.default_rng(0)
        lods = [
            AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
            )
            for n in (50, 80)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            GSplatData(additive_sublods=lods).save(path)
            root = zarr.open(str(path), mode="r")
            # Additive ladder → additive_<i>/ subgroups under the leaf root.
            assert array_compressor(root["additive_0/centers"]) is not None
            assert array_compressor(root["additive_1/centers"]) is not None
            assert root["additive_0/centers"].chunks[0] <= 50
            assert root["additive_1/centers"].chunks[0] <= 80

    def test_roundtrip_with_compression(self):
        rng = np.random.default_rng(42)
        splats = {
            "centers": rng.random((200, 3)).astype(np.float32) * 10,
            "amplitudes": rng.random(200).astype(np.float32) * 2,
            "cholesky_factors": rng.random((200, 6)).astype(np.float32),
        }
        g = GSplatData(**splats)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            np.testing.assert_allclose(g.centers, g2.centers, atol=1e-6)
            np.testing.assert_allclose(g.amplitudes, g2.amplitudes, atol=1e-6)
            np.testing.assert_allclose(
                g.cholesky_factors, g2.cholesky_factors, atol=1e-6
            )


class TestTruncationRadiusRoundtrip:
    def test_default_truncation_radius(self):
        g = GSplatData(**create_test_splats_3d(50))
        assert g.truncation_radius == DEFAULT_TRUNCATION_RADIUS
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            assert GSplatData.load(path).truncation_radius == DEFAULT_TRUNCATION_RADIUS

    def test_custom_truncation_radius_single_lod(self):
        g = GSplatData(**create_test_splats_3d(50), truncation_radius=2.75)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            assert GSplatData.load(path).truncation_radius == 2.75

    def test_custom_truncation_radius_multi_lod(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.default_rng(0)
        lods = [
            AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
                truncation_radius=2.5,
            )
            for n in (30, 50)
        ]
        g = GSplatData(additive_sublods=lods)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.truncation_radius == 2.5
            assert g2.additive_sublods[0].truncation_radius == 2.5
            assert g2.additive_sublods[1].truncation_radius == 2.5

    def test_per_level_truncation_radius_roundtrip(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.default_rng(0)
        lods = [
            AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
                truncation_radius=tr,
            )
            for n, tr in ((30, 2.5), (50, 4.0))
        ]
        g = GSplatData(additive_sublods=lods)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.additive_sublods[0].truncation_radius == 2.5
            assert g2.additive_sublods[1].truncation_radius == 4.0

    def test_level_stats_read_without_include_stats(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        rng = np.random.default_rng(0)

        def _lod(n: int) -> AdditiveSubLOD:
            return AdditiveSubLOD(
                centers=rng.standard_normal((n, 3)).astype(np.float32),
                amplitudes=rng.random(n).astype(np.float32),
                cholesky_factors=_positive_diag(
                    rng.standard_normal((n, 6)).astype(np.float32)
                ),
            )

        levels = [
            SubstitutiveLevel(
                additive_sublods=[_lod(8)],
                compression_factor=1,
                stats={"n_splats_total": 8},
            ),
            SubstitutiveLevel(
                additive_sublods=[_lod(2)],
                compression_factor=4,
                level_index=1,
                stats={"n_splats_total": 2},
            ),
        ]
        g = GSplatData.from_substitutive_levels(levels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)  # default: include_stats not set
            assert g2.substitutive_levels[1].stats.get("n_splats_total") == 2

    def test_truncation_radius_in_zarr_metadata(self):
        g = GSplatData(**create_test_splats_3d(50), truncation_radius=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            # v3.0: single leaf at root → truncation_radius on the root attrs.
            assert zarr.open(str(path), mode="r").attrs["truncation_radius"] == 2.0

    def test_backward_compat_missing_truncation_radius(self):
        g = GSplatData(**create_test_splats_3d(50))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            root = zarr.open(str(path), mode="r+")
            attrs = dict(root.attrs)
            del attrs["truncation_radius"]
            root.attrs.put(attrs)
            assert GSplatData.load(path).truncation_radius == DEFAULT_TRUNCATION_RADIUS


def test_save_explicit_none_compressor_disables_compression():
    """GSplatData.save(compressor=None) must write UNCOMPRESSED arrays so a
    cross-language (zarrita) reader can decode them. A plain None default used
    to be coerced to Blosc — making uncompressed output impossible and producing
    blosc-bitshuffle fixtures zarrita can't read (review full-suite finding)."""
    n = 16
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=np.random.rand(n, 3).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )
    with tempfile.TemporaryDirectory() as tmp:
        # Explicit None → no compression.
        raw = Path(tmp) / "raw.gsplats.zarr"
        data.save(raw, ordering="none", compressor=None)
        assert array_compressor(zarr.open_group(str(raw), mode="r")["centers"]) is None

        # Unspecified → default Blosc (compression still on by default).
        comp = Path(tmp) / "comp.gsplats.zarr"
        data.save(comp, ordering="none")
        assert (
            array_compressor(zarr.open_group(str(comp), mode="r")["centers"])
            is not None
        )


class TestCompressedLoadSecurity:
    """Path-safety of the shared compressed-archive extractor
    (``luxar.gsplats.io._archive.extract_compressed_zarr``), exercised through
    every entry point that consumes it (loader + format migrator)."""

    def _make_symlink_bomb(self, dest_dir: Path, outside: Path) -> Path:
        """Build a malicious .tar.gz: a symlink 'd' -> ``outside`` then a file
        'd/PWNED.txt' written through it (CVE-2007-4559-style symlink escape)."""
        import io
        import tarfile

        evil = dest_dir / "evil.gsplats.zarr.tar.gz"
        with tarfile.open(evil, "w:gz") as tar:
            link = tarfile.TarInfo("d")
            link.type = tarfile.SYMTYPE
            link.linkname = str(outside)
            tar.addfile(link)
            payload = b"arbitrary write outside extraction dir"
            f = tarfile.TarInfo("d/PWNED.txt")
            f.size = len(payload)
            tar.addfile(f, io.BytesIO(payload))
        return evil

    def test_targz_symlink_escape_rejected(self, tmp_path: Path) -> None:
        """The shared extractor must reject a symlink-escape tar.gz BEFORE any
        file is written, and leave no temp dir behind."""
        from luxar.gsplats.io._archive import extract_compressed_zarr

        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "PWNED.txt"

        evil = self._make_symlink_bomb(tmp_path, outside)
        with pytest.raises(ValueError, match="link|escape|device"):
            extract_compressed_zarr(evil)
        assert not target.exists(), "symlink escape wrote a file outside the temp dir"

    def test_targz_hardlink_rejected(self, tmp_path: Path) -> None:
        """Hardlink members are rejected too (not just symlinks)."""
        import io
        import tarfile

        from luxar.gsplats.io._archive import extract_compressed_zarr

        evil = tmp_path / "hard.gsplats.zarr.tar.gz"
        with tarfile.open(evil, "w:gz") as tar:
            payload = b"real"
            real = tarfile.TarInfo("real.txt")
            real.size = len(payload)
            tar.addfile(real, io.BytesIO(payload))
            link = tarfile.TarInfo("link.txt")
            link.type = tarfile.LNKTYPE
            link.linkname = "real.txt"
            tar.addfile(link)

        with pytest.raises(ValueError, match="link|device"):
            extract_compressed_zarr(evil)

    def test_zip_symlink_member_rejected(self, tmp_path: Path) -> None:
        """A zip symlink member is refused, as a tar's already was.

        Not an escape — ``extractall`` writes a symlink member out as a REGULAR
        file holding the target path, so nothing outside the temp dir is touched.
        It is a divergence: this archive's depth-0 ``.zgroup`` is a symlink, which
        the index-only resolvers (``_zip_is_flat_store``, the attrs peek) skip and
        the extractor would have counted, so the loader and the appearance peek
        would read different nodes out of one archive.
        """
        import stat as stat_mod

        from luxar.gsplats.io._archive import extract_compressed_zarr

        evil = tmp_path / "symlinked.gsplats.zarr.zip"
        with zipfile.ZipFile(evil, "w") as zip_ref:
            zip_ref.writestr("mystore/.zgroup", '{"zarr_format": 2}')
            zip_ref.writestr("mystore/.zattrs", '{"whose": "store"}')
            link = zipfile.ZipInfo(".zgroup")
            link.external_attr = (stat_mod.S_IFLNK | 0o777) << 16
            zip_ref.writestr(link, "/etc/passwd")

        with pytest.raises(ValueError, match="symlink"):
            extract_compressed_zarr(evil)

    def test_targz_member_count_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An archive with too many members is rejected (archive-bomb guard)."""
        import io
        import tarfile

        from luxar.gsplats.io import _archive

        monkeypatch.setattr(_archive, "_MAX_MEMBERS", 3)
        bomb = tmp_path / "bomb.gsplats.zarr.tar.gz"
        with tarfile.open(bomb, "w:gz") as tar:
            for i in range(5):
                info = tarfile.TarInfo(f"f{i}.txt")
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))

        with pytest.raises(ValueError, match="members"):
            _archive.extract_compressed_zarr(bomb)

    def test_migrate_path_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """The migration entry point uses the same safe extractor: a malicious
        archive passed to ``detect_legacy_format`` must not escape the temp dir."""
        from luxar.gsplats.io.migrate import detect_legacy_format

        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "PWNED.txt"

        evil = self._make_symlink_bomb(tmp_path, outside)
        with pytest.raises(ValueError, match="link|escape|device"):
            detect_legacy_format(evil)
        assert not target.exists(), "migrate path allowed a symlink escape"


class TestArchiveRootAttrsPeek:
    """Reading a compressed store's ROOT ``.zattrs`` without extracting it
    (``luxar.gsplats.io._archive.read_archive_root_attrs``).

    Regression for #1604: a ``.gsplats.zarr.zip`` / ``.tar.gz`` is a first-class
    input to the rebuild commands, so the authored-appearance carry has to be
    able to see inside one — and the old "peeking would extract GBs again"
    reasoning was simply wrong (it is one member). The peek must find the ROOT
    ``.zattrs`` *specifically*: a child group's attrs is a different object, and
    handing it to the writer would author an appearance nobody asked for.
    """

    @staticmethod
    def _write_zip(path: Path, members: list[tuple[str, str]]) -> None:
        """Write a zip with ``(name, text)`` members in exactly the given order."""
        import zipfile

        with zipfile.ZipFile(path, "w") as zip_ref:
            for name, text in members:
                zip_ref.writestr(name, text)

    @staticmethod
    def _write_targz(path: Path, members: list[tuple[str, str]]) -> None:
        """Write a tar.gz with ``(name, text)`` members in exactly the given order."""
        import io
        import tarfile

        with tarfile.open(path, "w:gz") as tar_ref:
            for name, text in members:
                payload = text.encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar_ref.addfile(info, io.BytesIO(payload))

    def _write(self, path: Path, fmt: str, members: list[tuple[str, str]]) -> None:
        if fmt == "zip":
            self._write_zip(path, members)
        else:
            self._write_targz(path, members)

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_reads_a_real_archived_store(self, tmp_path: Path, fmt: str) -> None:
        """The attrs a real ``save(compress=...)`` wrote come back off the archive.

        Covers the production nesting (``<name>.gsplats.zarr/.zattrs``) that
        ``_compress_zarr`` produces for both formats.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        data = GSplatData(**create_test_splats_3d(16))
        archive = tmp_path / f"peek.gsplats.zarr.{fmt}"
        data.save(archive, compress=fmt, root_attrs={"opacity": 0.75, "gamma": 1.3})

        attrs = read_archive_root_attrs(archive)
        assert attrs["format_type"] == "gsplats_zarr"
        assert attrs["opacity"] == 0.75
        assert attrs["gamma"] == 1.3

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    @pytest.mark.parametrize("write_format", (2, 3), ids=("v2", "v3"))
    def test_peek_reads_an_archive_of_either_on_disk_format(
        self, tmp_path: Path, fmt: str, write_format: int
    ) -> None:
        """The peek finds the root attrs whichever format wrote the archive.

        The document is named differently per format — ``.zattrs`` at 2, inside
        ``zarr.json`` at 3 — and the member is matched BY NAME, so recognising
        only one of them made every archive of the other format peek as ``{}``.
        That is indistinguishable from "this dataset authored no appearance", so
        `gsplat lod` silently dropped the whole authored appearance of an
        archived input instead of failing.

        The sibling test above uses the ambient write format, which pins only
        whichever one CI happens to run; this pins both explicitly.
        """
        from luxar._zarr_compat import ZARR_FORMAT, set_zarr_format
        from luxar.gsplats.io._archive import read_archive_root_attrs

        original = ZARR_FORMAT
        set_zarr_format(write_format)
        try:
            data = GSplatData(**create_test_splats_3d(16))
            archive = tmp_path / f"peek_v{write_format}.gsplats.zarr.{fmt}"
            data.save(archive, compress=fmt, root_attrs={"blending_mode": "volumetric"})
        finally:
            set_zarr_format(original)

        attrs = read_archive_root_attrs(archive)
        # Not vacuous: a peek that found nothing also returns a dict.
        assert attrs, f"format {write_format} {fmt} archive peeked as empty"
        assert attrs["blending_mode"] == "volumetric"
        # The structural key proves the ROOT was read, not some child node.
        assert attrs["format_type"] == "gsplats_zarr"

    #: The two store layouts ``extract_compressed_zarr`` accepts, as the archive
    #: member path of each one's ROOT ``.zattrs``: the ``*.gsplats.zarr``-named
    #: top-level directory ``_compress_zarr`` writes, and the extractor's fallback
    #: — the sole top-level directory, whatever it is called (what you get from
    #: ``tar czf x.gsplats.zarr.tar.gz mystore``).
    ROOT_LAYOUTS = ["x.gsplats.zarr/.zattrs", "mystore/.zattrs"]

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    @pytest.mark.parametrize("root", ROOT_LAYOUTS)
    def test_a_deeper_zattrs_never_wins(
        self, tmp_path: Path, fmt: str, root: str
    ) -> None:
        """A child group's ``.zattrs`` is not read as the root's, even listed first.

        Run for both store layouts the extractor accepts. The child member is
        deliberately written BEFORE the root so a first-match-wins implementation
        fails here.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        parent = root[: -len(".zattrs")]
        archive = tmp_path / f"nested.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                (f"{parent}lod_0/.zattrs", '{"whose": "child"}'),
                (root, '{"whose": "root"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "root"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    @pytest.mark.parametrize("root", ROOT_LAYOUTS)
    def test_the_stores_root_wins_over_a_shallower_stray(
        self, tmp_path: Path, fmt: str, root: str
    ) -> None:
        """A stray top-level ``.zattrs`` does not outrank the store's own root.

        ``extract_compressed_zarr`` defines the store as a top-level DIRECTORY, so
        that directory's ``.zattrs`` is the root even though a loose member sits
        one level shallower; reading the stray instead would author an appearance
        from something that is not the dataset. Both layouts matter, and the
        unnamed fallback is the sharp one: a ranking that merely preferred the
        ``*.gsplats.zarr`` name and then went shallowest-first hands back the
        stray's attrs there, while the real load succeeds with the store's. The
        stray is written FIRST so first-match-wins fails too.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"stray.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                (".zattrs", '{"whose": "stray"}'),
                (root, '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_depth_zero_only_archive_is_empty(self, tmp_path: Path, fmt: str) -> None:
        """An archive whose only ``.zattrs`` is at the root yields ``{}``.

        A depth-0 attrs document is a store root only when a depth-0 GROUP
        document is there with it (the flat tier's gate, #1628). With none, this
        is a stray ``.zattrs`` and nothing else — and treating a bare depth-0
        member as a store root is what would let a stray outrank a real store one
        directory down (see the test above), so it is refused outright.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"flat.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [(".zattrs", '{"whose": "stray"}')])
        assert read_archive_root_attrs(archive) == {}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_junk_sibling_directory_makes_the_unnamed_fallback_ambiguous(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """Two top-level directories and no ``*.gsplats.zarr`` name → ``{}``.

        The extractor's fallback is the SOLE top-level directory; with two of
        them it picks by ``iterdir()`` order, which no archive index can predict.
        A junk sibling listed first (``tar czf b.gsplats.zarr.tar.gz notes
        mystore``) would otherwise hand back ``notes``'s attrs while the load
        really read ``mystore`` — authoring an appearance from a node that is not
        the dataset. Carrying nothing is the status quo; guessing is not.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"siblings.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                ("notes/.zattrs", '{"whose": "junk"}'),
                ("mystore/.zattrs", '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {}

    @staticmethod
    def _write_with_empty_dir(
        path: Path, fmt: str, dir_name: str, files: list[tuple[str, str]]
    ) -> None:
        """Write an archive with an EMPTY explicit directory entry, plus files.

        A tar spells a directory as its own ``DIRTYPE`` header; a zip spells one
        as a member name ending in ``/``. Both are written here so the rule is
        exercised through each spelling.
        """
        import io
        import tarfile
        import zipfile

        bare = dir_name.rstrip("/")
        if fmt == "zip":
            with zipfile.ZipFile(path, "w") as zip_ref:
                zip_ref.writestr(zipfile.ZipInfo(f"{bare}/"), "")
                for name, text in files:
                    zip_ref.writestr(name, text)
        else:
            with tarfile.open(path, "w:gz") as tar_ref:
                dir_info = tarfile.TarInfo(bare)
                dir_info.type = tarfile.DIRTYPE
                tar_ref.addfile(dir_info)
                for name, text in files:
                    payload = text.encode("utf-8")
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    tar_ref.addfile(member, io.BytesIO(payload))

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_an_empty_stray_directory_does_not_suppress_the_carry(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """An EMPTY top-level directory beside the store is not a rival store.

        ``zip -r out.gsplats.zarr.zip mystore notes`` with an empty ``notes/`` has
        exactly one store in it. An empty directory can never be the node a
        SUCCESSFUL load read — ``zarr.open_group`` on one raises, so if the
        extractor lands there the whole load fails and no rebuild (hence no carry)
        happens at all. Counting it as a top-level directory could therefore never
        prevent a wrong carry, only manufacture false ambiguity and silently drop
        a correct one that the directory-store path keeps.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"empty_stray.gsplats.zarr.{fmt}"
        self._write_with_empty_dir(
            archive, fmt, "notes", [("mystore/.zattrs", '{"whose": "store"}')]
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_named_store_still_wins_over_a_junk_sibling(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """The uniqueness rule constrains the FALLBACK tier only.

        A top-level ``*.gsplats.zarr`` directory is the extractor's own first
        preference, so peek and extraction agree on it no matter what else the
        archive contains — the junk sibling must not suppress it.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"named_siblings.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                ("notes/.zattrs", '{"whose": "junk"}'),
                ("x.gsplats.zarr/.zattrs", '{"whose": "store"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "store"}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_flat_dump_of_a_tree_store_yields_its_root_not_a_child(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """An archive of a tree store's CONTENTS never yields a child's attrs.

        ``tar czf b.gsplats.zarr.tar.gz -C store .`` puts the store root's own
        ``.zattrs``/``.zgroup`` at depth 0 and its ``child_<i>/`` groups at depth
        1 — so every depth-1 candidate is a CHILD group, exactly what the ranking
        exists to keep out of the authored appearance. Since #1628 that depth-0
        pair IS a recognised store root (the flat tier), so the answer is the
        root's own attrs; before it, the archive resolved to nothing and the
        several-top-level-directories rule was what kept the children out. A real
        tree shape is used here for both reasons (a ``kind=lod`` /
        ``kind=partition`` group always has at least two children).
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        archive = tmp_path / f"contents.gsplats.zarr.{fmt}"
        self._write(
            archive,
            fmt,
            [
                (".zattrs", '{"whose": "root"}'),
                (".zgroup", '{"zarr_format": 2}'),
                ("child_0/.zattrs", '{"whose": "child_0"}'),
                ("child_1/.zattrs", '{"whose": "child_1"}'),
            ],
        )
        assert read_archive_root_attrs(archive) == {"whose": "root"}

    @staticmethod
    def _write_with_symlink(
        path: Path,
        fmt: str,
        link: tuple[str, str],
        extra: list[tuple[str, str]],
    ) -> None:
        """Write an archive whose ``link`` member is a real SYMLINK, plus files.

        Both formats spell a symlink differently — a unix mode in the zip's
        external attrs, a ``SYMTYPE`` header in the tar — so the guard has to be
        exercised through each spelling separately.
        """
        import io
        import stat
        import tarfile
        import zipfile

        name, target = link
        if fmt == "zip":
            with zipfile.ZipFile(path, "w") as zip_ref:
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zip_ref.writestr(info, target)
                for member_name, text in extra:
                    zip_ref.writestr(member_name, text)
        else:
            with tarfile.open(path, "w:gz") as tar_ref:
                info_tar = tarfile.TarInfo(name)
                info_tar.type = tarfile.SYMTYPE
                info_tar.linkname = target
                tar_ref.addfile(info_tar)
                for member_name, text in extra:
                    payload = text.encode("utf-8")
                    member = tarfile.TarInfo(member_name)
                    member.size = len(payload)
                    tar_ref.addfile(member, io.BytesIO(payload))

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_symlinked_zattrs_member_is_never_read(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """A ``.zattrs`` member that is a SYMLINK is skipped, not followed.

        The module's whole threat model is that no link is ever followed, for
        BOTH formats. An in-archive link is the sharp case: ``extractfile``
        happily resolves ``x.gsplats.zarr/.zattrs -> child/.zattrs`` inside the
        tar, so dropping the regular-file check silently promotes a CHILD group's
        attrs to the root's. An out-of-archive link is the other half: a symlink
        member's payload is its TARGET PATH, so following one leaks an arbitrary
        file's location into the attrs and hands the reader a path where a JSON
        object belongs. Third archive: a real ``*.gsplats.zarr`` root still wins
        even with a link listed before it.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        shadows_child = tmp_path / f"shadow.gsplats.zarr.{fmt}"
        self._write_with_symlink(
            shadows_child,
            fmt,
            ("x.gsplats.zarr/.zattrs", "child/.zattrs"),
            [("x.gsplats.zarr/child/.zattrs", '{"whose": "child"}')],
        )
        assert read_archive_root_attrs(shadows_child) == {}

        only_link = tmp_path / f"link.gsplats.zarr.{fmt}"
        self._write_with_symlink(
            only_link, fmt, ("x.gsplats.zarr/.zattrs", "/etc/passwd"), []
        )
        assert read_archive_root_attrs(only_link) == {}

        with_real = tmp_path / f"link_plus_real.gsplats.zarr.{fmt}"
        self._write_with_symlink(
            with_real,
            fmt,
            ("notes/.zattrs", "/etc/passwd"),
            [("x.gsplats.zarr/.zattrs", '{"whose": "store"}')],
        )
        assert read_archive_root_attrs(with_real) == {"whose": "store"}

    def test_non_archive_and_missing_paths_are_empty(self, tmp_path: Path) -> None:
        """A plain file, a directory and a missing path all yield ``{}``.

        ``read_authored_appearance`` funnels every non-directory input here, so
        "not an archive" has to be a quiet empty answer rather than a raise.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs
        from luxar.gsplats.io.load_gsplats import read_authored_appearance

        plain = tmp_path / "notes.txt"
        plain.write_text("not an archive")
        missing = tmp_path / "gone.gsplats.zarr.zip"

        assert read_archive_root_attrs(plain) == {}
        assert read_archive_root_attrs(tmp_path) == {}
        assert read_archive_root_attrs(missing) == {}
        assert read_authored_appearance(plain) == {}
        assert read_authored_appearance(missing) == {}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_oversized_attrs_member_refused(
        self, tmp_path: Path, fmt: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``.zattrs`` far too big for attrs is refused, not read into memory.

        The refusal WARNS: ``{}`` means "authored no appearance" to every caller,
        so a silent size refusal reaches the user only as a rebuild that reset
        the look (#1600 point 4). The message must also name the budget it
        crossed CORRECTLY — the label is derived from the document's name, so a
        ``.zattrs`` says "attributes" whatever the two constants happen to be.
        """
        from luxar.gsplats.io import _archive

        monkeypatch.setattr(_archive, "_MAX_ATTRS_BYTES", 8)
        archive = tmp_path / f"fat.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/.zattrs", '{"opacity": 0.75}')])
        with pytest.warns(
            UserWarning, match=r"metadata member \(declared size\) '.*\.zattrs'"
        ) as record:
            assert _archive.read_archive_root_attrs(archive) == {}
        assert "attributes budget of 8 bytes" in str(record[0].message)

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_oversized_node_document_refused(
        self, tmp_path: Path, fmt: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DOCUMENT budget's DECLARED-SIZE gate is a real gate.

        Symmetric with ``test_oversized_attrs_member_refused``, and the pin the
        new constant was missing: with none, setting it to 1 GiB left the whole
        peek suite green. The member here is a ``zarr.json``, so it is the raised
        budget being exercised and not the ``.zattrs`` one.

        The match is deliberately on the DECLARED-SIZE wording. That is the
        security-relevant half of the pair — it refuses before the payload is
        opened, so an archive-bomb member is never decompressed — and the two
        member-level refusals used to be worded identically, which meant deleting
        this gate merely handed the same input to the bytes-read one and this
        test stayed green. It also pins the budget LABEL: the patched constants
        are inverted here (64 < the 4 MiB attrs budget), so a label derived by
        comparing budget VALUES rather than reading the document's name calls a
        ``zarr.json`` "attributes".
        """
        from luxar.gsplats.io import _archive

        monkeypatch.setattr(_archive, "_MAX_NODE_DOC_BYTES", 64)
        raw = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"opacity": 0.75, "pad": "x" * 256},
            }
        )
        assert len(raw.encode("utf-8")) > 64
        archive = tmp_path / f"fat_doc.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/zarr.json", raw)])
        with pytest.warns(
            UserWarning, match=r"metadata member \(declared size\) '.*zarr\.json'"
        ) as record:
            assert _archive.read_archive_root_attrs(archive) == {}
        assert "document budget of 64 bytes" in str(record[0].message)

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_non_ascii_zattrs_under_the_budget_is_still_returned(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """A format-2 member read under the small budget is NOT measured twice.

        This pins the byte gate, and only that: a ``.zattrs`` IS the attributes
        mapping, was already bounded by :data:`_MAX_ATTRS_BYTES` as a member, and
        must reach the caller without a second measurement in a different
        currency. It does NOT pin what that second measurement would
        have said — the guard short-circuits before any re-serialization runs, so
        this document would survive the recap too (the shipped measure is
        compact: ~1,800,026 B against ~1,800,029 B on disk). The
        ``ensure_ascii=True`` inflation this document is built to demonstrate is
        pinned where it can actually bite, on the branch where the recap DOES
        run — see
        ``test_non_ascii_attributes_in_a_node_document_are_not_escape_inflated``.
        """
        from luxar.gsplats.io._archive import _MAX_ATTRS_BYTES, read_archive_root_attrs

        attrs = {"opacity": 0.75, "note": "é" * 900_000}
        raw = json.dumps(attrs, ensure_ascii=False)
        on_disk = len(raw.encode("utf-8"))
        escaped = len(json.dumps(attrs).encode("utf-8"))
        # Not vacuous: inside the budget as bytes, outside it once ASCII-escaped.
        assert on_disk < _MAX_ATTRS_BYTES < escaped

        archive = tmp_path / f"unicode.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/.zattrs", raw)])
        assert read_archive_root_attrs(archive) == attrs

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_zattrs_whose_reserialization_inflates_is_not_measured_twice(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """The byte gate from the FORMAT-2 side, on input the recap would refuse.

        Removing the ASCII escaping shrank the gap between "bytes on disk" and
        "bytes of a re-serialization" but did not close it: ``json.loads`` is not
        round-trip-preserving, and exponent notation is the plain case — the
        4-byte literal ``1e10`` comes back as the 13-byte ``10000000000.0``. A
        ``.zattrs`` of such values (a voxel size, an intensity scale, a physical
        extent — ordinary things to author) is therefore admitted as a member at
        2.76 MiB and would measure 4.47 MiB if it were re-checked, so the member
        budget's verdict and the re-check's verdict genuinely disagree.

        The recap runs only when the bytes READ exceeded
        :data:`_MAX_ATTRS_BYTES`, so this member is never re-measured and the
        disagreement cannot cost this dataset its appearance. For a ``.zattrs``
        that is arithmetic rather than a convention — its member budget IS that
        number, so the recap condition is unreachable on this branch whatever the
        document happens to contain; this test pins the rule from the format-2
        side, its ``zarr.json`` sibling
        (``test_a_node_document_inside_the_attrs_budget_is_not_re_capped``) pins
        it where the name-keyed rule and the byte-keyed rule actually differ.
        Without the gate the peek answers ``{}`` — "authored no appearance" — for
        a file that every earlier version of Luxar read fine.
        """
        from luxar.gsplats.io._archive import _MAX_ATTRS_BYTES, read_archive_root_attrs

        raw = "{" + ",".join(f'"k{i}":1e10' for i in range(200_000)) + "}"
        attrs = json.loads(raw)
        reserialized = len(
            json.dumps(attrs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        # Not vacuous: inside the member budget on disk, outside it re-serialized
        # even by the compact measure.
        assert len(raw.encode("utf-8")) < _MAX_ATTRS_BYTES < reserialized

        archive = tmp_path / f"inflating.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/.zattrs", raw)])
        assert read_archive_root_attrs(archive) == attrs

    @staticmethod
    def _fat_v3_root_doc(
        attributes: dict[str, object], min_bytes: int, *, ensure_ascii: bool = True
    ) -> str:
        """A format-3 root ``zarr.json`` padded past ``min_bytes`` HONESTLY.

        Real shape throughout — ``zarr_format: 3``, ``node_type: "group"``, the
        caller's real ``attributes`` — and the bulk is where a real one's bulk is:
        an inline ``consolidated_metadata`` holding one valid group record per
        child. That is what makes a format-3 root document grow with the tree
        while its attributes stay a handful of scalars, and it is why the one
        4 MiB budget that used to cover both was the wrong ruler (measured
        ~8-10 KB of root ``zarr.json`` per part of a real ``gsplat partition``,
        so ~400-500 parts crossed it).

        Synthesised rather than produced by a real ``gsplat partition``: the
        hundreds of parts needed to cross 4 MiB take far too long to fit for a
        unit test. The per-child record is padded to roughly the measured
        per-part size, and the entry count is derived arithmetically so this
        stays one ``json.dumps``.
        """
        entry = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"pad": "x" * 7000},
        }
        per_entry = len(json.dumps({"part_0": entry}))
        count = min_bytes // per_entry + 8
        return json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attributes,
                "consolidated_metadata": {
                    "kind": "inline",
                    "must_understand": False,
                    "metadata": {f"part_{i}": entry for i in range(count)},
                },
            },
            ensure_ascii=ensure_ascii,
        )

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_fat_consolidated_root_document_peeks_like_the_directory(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """A >4 MiB format-3 root document is peeked, not refused (#1600).

        The single 4 MiB budget was written for a format-2 ``.zattrs``, which
        literally IS the attributes mapping. A format-3 ``zarr.json`` at a
        CONSOLIDATED root — which every Luxar store is — carries the whole
        consolidated index beside attributes that are still a handful of scalars,
        so the document grows with the tree and crosses 4 MiB at roughly 400-500
        parts (routine for a ``batch-fit merge``). The peek then returned ``{}``,
        which reads as "this dataset authored no appearance", so every rewriting
        command (``gsplat lod``, ``additive``, ``flatten``, ``decimate``,
        ``reencode``, ``cull``, ``filter``, ``transform``, ``partition``,
        ``migrate-format``) silently wrote its own defaults over the authored
        appearance of a zipped/tarred large partition.

        Deliberately run with the REAL, unpatched constants: a monkeypatched
        budget is exactly what the pre-existing oversize test used, and it passed
        just as happily with this bug present.

        The assertion is AGREEMENT with the same tree opened as a directory
        store, not a hard-coded dict — that disagreement between the two input
        shapes is the actual defect, so the test tracks the real contract.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs
        from luxar.gsplats.io.load_gsplats import read_authored_appearance

        authored = {
            "format_type": "gsplats_zarr",
            "blending_mode": "additive",
            "opacity": 0.9,
            "gamma": 1.4,
        }
        raw = self._fat_v3_root_doc(authored, 4 * 1024**2)
        # Not vacuous: the document really is over the attrs budget.
        assert len(raw.encode("utf-8")) > 4 * 1024**2

        directory = tmp_path / "x.gsplats.zarr"
        directory.mkdir()
        (directory / "zarr.json").write_text(raw)

        archive = tmp_path / f"fat_root.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/zarr.json", raw)])

        peeked = read_archive_root_attrs(archive)
        assert peeked == authored
        assert read_authored_appearance(archive) == read_authored_appearance(directory)
        # And the carry is not the empty agreement of two broken paths.
        assert read_authored_appearance(archive) == {
            "blending_mode": "additive",
            "gamma": 1.4,
            "opacity": 0.9,
        }

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_an_oversized_zattrs_is_still_refused_at_the_small_budget(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """The old guard is intact for format 2, at the real 4 MiB constant.

        Nothing about a ``.zattrs`` changed: it IS the attributes mapping, so the
        larger document budget must not reach it. Written without monkeypatching
        so a future refactor that collapsed the two budgets back into one would
        be caught here rather than hidden by a patched-down number.
        """
        from luxar.gsplats.io._archive import read_archive_root_attrs

        raw = json.dumps({"opacity": 0.75, "pad": "x" * (4 * 1024**2 + 4096)})
        assert len(raw.encode("utf-8")) > 4 * 1024**2
        archive = tmp_path / f"fat_zattrs.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/.zattrs", raw)])
        with pytest.warns(UserWarning, match="Appearance peek refused"):
            assert read_archive_root_attrs(archive) == {}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_oversized_attributes_inside_a_node_document_are_refused(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """A fat ``attributes`` mapping is refused even in a within-budget document.

        The document budget had to grow for the consolidated index; that must not
        become licence to hand a caller an arbitrarily large attrs mapping. The
        post-parse cap in ``_parse_attrs`` is what keeps the two independent, so
        this document must sit STRICTLY BETWEEN the two real constants —
        admitted by the document budget, refused on the size of what it unwraps
        to. Asserting only "below 128 MiB" would pass with the post-parse cap
        deleted, for the wrong reason.

        Being over :data:`_MAX_ATTRS_BYTES` is also what puts this member on the
        re-cap branch at all, since the re-cap fires on the bytes READ rather
        than on the document's name — so the same assertion pins both halves.
        """
        from luxar.gsplats.io._archive import (
            _MAX_ATTRS_BYTES,
            _MAX_NODE_DOC_BYTES,
            read_archive_root_attrs,
        )

        raw = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "format_type": "gsplats_zarr",
                    "pad": "x" * (4 * 1024**2 + 4096),
                },
            }
        )
        assert _MAX_ATTRS_BYTES < len(raw.encode("utf-8")) < _MAX_NODE_DOC_BYTES
        archive = tmp_path / f"fat_attrs.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/zarr.json", raw)])
        with pytest.warns(UserWarning, match="attributes unwrapped from"):
            assert read_archive_root_attrs(archive) == {}

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_non_ascii_attributes_in_a_node_document_are_not_escape_inflated(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """The post-parse re-cap measures COMPACT bytes, not ASCII-escaped ones.

        The counterpart of ``test_oversized_attributes_inside_a_node_document_
        are_refused``: that one pins that the re-cap fires, this one pins the
        CURRENCY it fires in. A format-3 root document is the only branch where
        the re-cap runs at all, so this is the only place the choice can be
        pinned — and it is a live hazard rather than a theoretical one, because a
        node document's ``attributes`` may legitimately carry non-ASCII text
        (a channel name, a label, a note). "Only branch" is a fact about the
        bytes, not the name: the re-cap fires when a member's own bytes exceeded
        :data:`_MAX_ATTRS_BYTES`, which only a ``zarr.json`` read under the
        raised budget ever does.

        ``json.dumps`` defaults to ``ensure_ascii=True``, which turns one ``é``
        into six ASCII bytes: the attributes here are ~1.7 MiB and comfortably
        inside the 4 MiB attrs budget, but ~5.1 MiB once escaped. Measuring the
        escaped form refuses them and hands the caller ``{}`` — "this dataset
        authored no appearance" — which is exactly the silent appearance loss
        #1600 is about, now reintroduced by the fix's own re-cap. The document is
        deliberately over 4 MiB (a real fat consolidated root) so there is no
        doubt it took the raised-budget branch.
        """
        from luxar.gsplats.io._archive import (
            _MAX_ATTRS_BYTES,
            _MAX_NODE_DOC_BYTES,
            read_archive_root_attrs,
        )

        authored: dict[str, object] = {
            "format_type": "gsplats_zarr",
            "blending_mode": "additive",
            "opacity": 0.9,
            "note": "é" * 900_000,
        }
        compact = len(
            json.dumps(authored, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        escaped = len(json.dumps(authored).encode("utf-8"))
        # Not vacuous: the attributes fit the attrs budget as bytes, but not once
        # ASCII-escaped — so the two measures disagree about this document.
        assert compact < _MAX_ATTRS_BYTES < escaped

        raw = self._fat_v3_root_doc(authored, 4 * 1024**2, ensure_ascii=False)
        # On the raised-budget branch, and admitted by it.
        assert _MAX_ATTRS_BYTES < len(raw.encode("utf-8")) < _MAX_NODE_DOC_BYTES

        archive = tmp_path / f"unicode_doc.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/zarr.json", raw)])
        assert read_archive_root_attrs(archive) == authored

    @pytest.mark.parametrize("fmt", ["zip", "tar.gz"])
    def test_a_node_document_inside_the_attrs_budget_is_not_re_capped(
        self, tmp_path: Path, fmt: str
    ) -> None:
        """A ``zarr.json`` the OLD single 4 MiB budget read is not newly refused.

        The re-cap is keyed on the bytes actually READ (``len(raw) >
        _MAX_ATTRS_BYTES``), not on the document's NAME, and this input is what
        separates the two rules. It is a format-3 node document, so a name-keyed
        re-cap runs on it — but it never used the raised document budget at all,
        so re-capping it can only take away an answer every earlier version of
        Luxar gave.

        Reachable without any ASCII trickery, because ``json.loads`` is not
        round-trip-length-preserving: the 4-byte literal ``1e10`` comes back as
        the 13-byte ``10000000000.0``, so ~190,000 such values are 2.61 MiB as
        written and measure 4.24 MiB re-serialized (compactly — the measure the
        recap uses). A store authoring that many numeric scalars is a foreign
        producer's, since Python's own ``json.dumps`` never emits ``1e10``, but
        the peek's whole job is to read what is on disk.

        Under the name-keyed rule this returned ``{}`` plus a ``UserWarning`` —
        a silent-appearance-loss regression of exactly the class #1600 exists to
        remove.
        """
        from luxar.gsplats.io._archive import _MAX_ATTRS_BYTES, read_archive_root_attrs

        attrs_src = "{" + ",".join(f'"k{i}":1e10' for i in range(190_000)) + "}"
        raw = '{"zarr_format":3,"node_type":"group","attributes":' + attrs_src + "}"
        attrs = json.loads(attrs_src)
        measured = len(
            json.dumps(attrs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        # Not vacuous: the document as read is INSIDE the small attrs budget
        # (so the raised budget was never used), while its attributes are
        # OUTSIDE it once re-serialized (so a re-cap here would fire).
        assert len(raw.encode("utf-8")) < _MAX_ATTRS_BYTES < measured

        archive = tmp_path / f"inside_attrs_budget.gsplats.zarr.{fmt}"
        self._write(archive, fmt, [("x.gsplats.zarr/zarr.json", raw)])
        assert read_archive_root_attrs(archive) == attrs


def test_write_gsplats_tree_stamps_child_index_on_children() -> None:
    """Bare-root .gsplats.zarr trees stamp ``child_index`` (insertion order) on
    every ``child_<i>`` / ``part_<i>`` so the viewer restores napari-style order
    instead of zarr's alphabetical enumeration. Pre-fix the children carried no
    ``child_index``; a >=10-child group then reordered (child_10 before child_2)
    in the viewer's sibling sort.
    """
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    def _leaf(n: int, seed: int) -> GSplatLeaf:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
                    amplitudes=rng.uniform(0.1, 1, (n,)).astype(np.float32),
                    cholesky_factors=chol,
                )
            ]
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # kind=partition with 3 parts.
        ppath = Path(tmpdir) / "part.gsplats.zarr"
        write_gsplats_tree(
            ppath,
            GSplatPartition(children=[_leaf(20, 0), _leaf(20, 1), _leaf(20, 2)]),
            ordering="none",
        )
        proot = zarr.open_group(str(ppath), mode="r")
        for i in range(3):
            assert dict(proot[f"part_{i}"].attrs)["child_index"] == i

        # kind=lod: in-memory coarsest→finest, same as on disk (child_0 = coarsest),
        # and child_index must match the child_<i> numbering (0=coarsest..N=finest).
        lpath = Path(tmpdir) / "lod.gsplats.zarr"
        write_gsplats_tree(
            lpath,
            GSplatLodGroup(children=[_leaf(10, 4), _leaf(40, 3)]),
            ordering="none",
        )
        lroot = zarr.open_group(str(lpath), mode="r")
        for i in range(2):
            assert dict(lroot[f"child_{i}"].attrs)["child_index"] == i


def test_writer_derives_coverage_fractions_for_meta_less_lod_group() -> None:
    """#4: a meta-less (hand-built) kind=lod group gets viewport-relative
    ``coverage_fraction`` values from the writer fallback (screen-occupancy
    AREA halving: coarsest 0.0, finest at the half-screen-area anchor
    0.5). The builders normally stamp these into child meta — this
    exercises the fallback for a tree written without it."""
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    def _leaf(n: int, scale: float, seed: int) -> GSplatLeaf:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = scale  # isotropic; radius ∝ scale
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 100, (n, 3)).astype(np.float32),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol,
                )
            ]
        )

    # coarsest-first: 50 large (scale 4) then 800 small (scale 1). No authored meta.
    grp = GSplatLodGroup(children=[_leaf(50, 4.0, 0), _leaf(800, 1.0, 1)])
    assert "coverage_fraction" not in grp.children[0].meta
    assert "coverage_fraction" not in grp.children[1].meta

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "m.gsplats.zarr"
        write_gsplats_tree(path, grp, ordering="none")
        root = zarr.open_group(str(path), mode="r")
        assert root["child_0"].attrs["coverage_fraction"] == 0.0  # coarsest floor
        # Screen-occupancy AREA halving: any 2-level ladder → finest 0.5.
        assert root["child_1"].attrs["coverage_fraction"] == pytest.approx(
            WHOLE_OBJECT_FINEST_ANCHOR
        )


def test_writer_selector_threshold_consistency_gate() -> None:
    """A group's meta ``selector`` describes its AUTHORED thresholds, so the
    writer may preserve it only when EVERY child carries one. Three arms:

    * PARTIALLY-authored + ``selector="coverage"`` — without the gate, the
      writer's fallback derivation (screen-area units) filled the gaps under
      the legacy stamp: a mixed-units store. The gate scrubs the authored
      remnant, re-derives the whole ladder, and stamps ``screen-area`` so the
      written pair agrees.
    * FULLY authored + ``selector="coverage"`` — preserved verbatim (this is
      the legacy round-trip; the values must NOT be re-derived).
    * Unknown meta selector — refused before anything is written (the reader
      whitelists stale spellings away on LOAD; one arriving here is a
      hand-built tree that would otherwise write an out-of-vocabulary
      selector into a store claiming v3.4 compliance).
    """
    import tempfile
    from pathlib import Path

    import zarr

    from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    def _leaf(n: int, seed: int, cov: float | None) -> GSplatLeaf:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        meta = {} if cov is None else {"coverage_fraction": cov}
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 100, (n, 3)).astype(np.float32),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol,
                )
            ],
            meta=meta,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Arm 1: partially authored (legacy value on child_0 only) under a
        # "coverage" stamp → uniform re-derivation + screen-area stamp.
        mixed = GSplatLodGroup(
            children=[_leaf(50, 0, cov=0.0), _leaf(800, 1, cov=None)],
            meta={"selector": "coverage"},
        )
        p1 = Path(tmpdir) / "mixed.gsplats.zarr"
        write_gsplats_tree(p1, mixed, ordering="none")
        r1 = zarr.open_group(str(p1), mode="r")
        assert r1.attrs["selector"] == "screen-area"
        assert r1["child_0"].attrs["coverage_fraction"] == 0.0
        assert r1["child_1"].attrs["coverage_fraction"] == pytest.approx(
            WHOLE_OBJECT_FINEST_ANCHOR
        )

        # Arm 2: fully authored legacy ladder → selector AND values preserved.
        legacy = GSplatLodGroup(
            children=[_leaf(50, 2, cov=0.0), _leaf(800, 3, cov=2.0)],
            meta={"selector": "coverage"},
        )
        p2 = Path(tmpdir) / "legacy.gsplats.zarr"
        write_gsplats_tree(p2, legacy, ordering="none")
        r2 = zarr.open_group(str(p2), mode="r")
        assert r2.attrs["selector"] == "coverage"
        assert r2["child_1"].attrs["coverage_fraction"] == 2.0  # NOT re-derived

        # Arm 3: out-of-vocabulary selector → refused, nothing written.
        bogus = GSplatLodGroup(
            children=[_leaf(50, 4, cov=0.0), _leaf(800, 5, cov=1.0)],
            meta={"selector": "pixel_size"},
        )
        p3 = Path(tmpdir) / "bogus.gsplats.zarr"
        with pytest.raises(ValueError, match="must be one of"):
            write_gsplats_tree(p3, bogus, ordering="none")

        # Arm 4: fully authored but SELECTOR-LESS → stamped LEGACY "coverage",
        # values preserved. Authored values with no stated units are exactly
        # what the viewer's loader treats as legacy (its missing/unknown-
        # selector fallback), so relabeling them "screen-area" would silently
        # reinterpret them — authored = legacy is the library convention.
        selectorless = GSplatLodGroup(
            children=[_leaf(50, 6, cov=0.0), _leaf(800, 7, cov=2.0)],
        )
        p4 = Path(tmpdir) / "selectorless.gsplats.zarr"
        write_gsplats_tree(p4, selectorless, ordering="none")
        r4 = zarr.open_group(str(p4), mode="r")
        assert r4.attrs["selector"] == "coverage"
        assert r4["child_1"].attrs["coverage_fraction"] == 2.0

        # Arm 5: authored thresholds must honor the selector's contract. 2.0 is
        # legal legacy-diagonal (arm 2/4) but OUT OF RANGE for screen-area
        # (the clipped area metric tops out at 1.0), and a non-monotonic
        # ladder is invalid under either — both refused before writing.
        over_range = GSplatLodGroup(
            children=[_leaf(50, 8, cov=0.0), _leaf(800, 9, cov=2.0)],
            meta={"selector": "screen-area"},
        )
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            write_gsplats_tree(
                Path(tmpdir) / "over.gsplats.zarr", over_range, ordering="none"
            )
        non_monotonic = GSplatLodGroup(
            children=[
                _leaf(50, 10, cov=0.0),
                _leaf(200, 14, cov=0.5),
                _leaf(800, 11, cov=0.25),
            ],
            meta={"selector": "screen-area"},
        )
        with pytest.raises(ValueError, match="strictly greater"):
            write_gsplats_tree(
                Path(tmpdir) / "nonmono.gsplats.zarr", non_monotonic, ordering="none"
            )
        # The coarsest child must be exactly 0.0 (the always-eligible floor the
        # format requires) — [0.25, 0.5] is strictly ascending and in range but
        # leaves no eligible child below 0.25 occupancy.
        no_floor = GSplatLodGroup(
            children=[_leaf(50, 12, cov=0.25), _leaf(800, 13, cov=0.5)],
            meta={"selector": "screen-area"},
        )
        with pytest.raises(ValueError, match="must be exactly 0.0"):
            write_gsplats_tree(
                Path(tmpdir) / "nofloor.gsplats.zarr", no_floor, ordering="none"
            )


class TestBarrierAwareOrdering:
    """End-to-end: a 4D leaf with a time barrier writes single-timepoint chunks
    (the fix for per-timepoint viewer-load locality)."""

    @staticmethod
    def _make_4d_leaf(n: int, n_tps: int, seed: int) -> "GSplatData":
        """Random 4D splats spread over n_tps integer timepoints in column 3."""
        rng = np.random.default_rng(seed)
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100.0
        centers[:, 3] = rng.integers(0, n_tps, size=n).astype(np.float32)
        chol = np.zeros((n, 10), dtype=np.float32)
        diag_idx = [d * (d + 1) // 2 + d for d in range(4)]
        chol[:, diag_idx] = 2.0  # isotropic σ=2 in all 4 dims
        amps = (rng.random(n) + 0.5).astype(np.float32)
        return GSplatData(centers=centers, amplitudes=amps, cholesky_factors=chol)

    @staticmethod
    def _finest_chunk_bounds(path: Path) -> np.ndarray:
        """chunk_bounds of the (only) leaf's finest splat set."""
        root = zarr.open_group(str(path), mode="r")
        # single-leaf root: chunk_bounds directly under root
        if "chunk_bounds" in root:
            return np.asarray(root["chunk_bounds"])
        raise AssertionError("no chunk_bounds at leaf root")

    def test_explicit_barrier_yields_single_timepoint_chunks(self) -> None:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        # Many splats per timepoint (12k / 3 = 4k ≫ chunk_size ~1024) so a chunk
        # spans at most 2 adjacent timepoints (a boundary chunk), never all 3 —
        # this is the regime the real timelapse is in (~2.5M splats/timepoint).
        data = self._make_4d_leaf(12000, n_tps=3, seed=1)
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            bpath = Path(tmpdir) / "barrier.gsplats.zarr"
            npath = Path(tmpdir) / "nobarrier.gsplats.zarr"
            write_gsplats_tree(bpath, leaf, barrier_dims=[3])
            write_gsplats_tree(npath, leaf, barrier_dims=[])  # pure spatial

            bt = (
                self._finest_chunk_bounds(bpath)[:, 3, 1]
                - self._finest_chunk_bounds(bpath)[:, 3, 0]
            )
            nt = (
                self._finest_chunk_bounds(npath)[:, 3, 1]
                - self._finest_chunk_bounds(npath)[:, 3, 0]
            )

            # WITH barrier: most chunks single-timepoint. Since the write-side
            # padding shrank from ±0.5 step to the tiny float-boundary epsilon
            # (_BARRIER_BOUND_EPS = 1e-3), a single-timepoint chunk's extent is
            # ~2*eps (~0.002, no longer ~1.0), and a boundary chunk spanning 2
            # timepoints is ~1.0 + 2*eps. Never the whole-span-plus-sigma smear
            # the no-barrier ordering produces. The tight thresholds FAIL under
            # the legacy +/-0.5 padding — they pin the over-fetch fix's write side.
            assert np.median(bt) <= 0.1
            assert np.all(bt <= 1.0 + 0.1)
            # WITHOUT barrier: chunks smear across timepoints (σ-expanded too),
            # so the barrier version is decisively tighter — the fix's payoff.
            assert np.median(nt) > np.median(bt)
            assert nt.max() > bt.max()

            # ordering attrs advertise the barrier (mirrors Points/Lines).
            root = zarr.open_group(str(bpath), mode="r")
            assert list(root.attrs["slice_dims"]) == [3]
            assert list(root.attrs["ordering_dims"]) == [0, 1, 2]

    def test_barrier_derived_from_coarsen_dims_provenance(self) -> None:
        """When barrier_dims is not passed, it is derived from
        pipeline_info['coarsen_dims'] (barrier = complement)."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        data = self._make_4d_leaf(12000, n_tps=3, seed=2)
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prov.gsplats.zarr"
            # coarsen spatial dims 0,1,2 → barrier = [3] (time). No barrier_dims arg.
            write_gsplats_tree(path, leaf, pipeline_info={"coarsen_dims": [0, 1, 2]})
            bt = (
                self._finest_chunk_bounds(path)[:, 3, 1]
                - self._finest_chunk_bounds(path)[:, 3, 0]
            )
            assert np.median(bt) <= 1.5  # barrier honored via provenance
            root = zarr.open_group(str(path), mode="r")
            assert list(root.attrs["slice_dims"]) == [3]

    def test_explicit_full_coarsen_list_yields_no_barrier(self) -> None:
        """A caller passing an explicit full coarsen_dims list (barrier = empty
        complement) → pure spatial, NOT auto-detected [3].

        NOTE: this is _barrier_from_coarsen_dims's empty-complement branch, and
        since #1600 it is on the ORDINARY path rather than a direct-caller
        curiosity: `_normalise_coarsen_dims` still collapses coarsen-all to an
        internal `None`, but the three paths that WRITE the stamp
        (make_substitutive_lod, decimate's merge family, the batch-fit merge
        per-part record) now resolve that back to the full list through
        `resolved_merge_coarsen_dims` before writing it. (`lod --recipe
        adaptive` / `overview` and `fit --recipe levels` write no stamp at all,
        so they still land on the branch below — #1600.) The `None` spelling is
        indistinguishable from 'no provenance' and would land on auto-detect,
        re-imposing the barrier the reduction just blended away."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        data = self._make_4d_leaf(12000, n_tps=3, seed=4)  # integer time axis
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "coarsen_all.gsplats.zarr"
            write_gsplats_tree(path, leaf, pipeline_info={"coarsen_dims": [0, 1, 2, 3]})
            root = zarr.open_group(str(path), mode="r")
            assert list(root.attrs["slice_dims"]) == []
            assert list(root.attrs["ordering_dims"]) == [0, 1, 2, 3]

    def test_streaming_partition_explicit_barrier_on_sparse_time(self) -> None:
        """REGRESSION (deep-double-check, findings 1/6): the batch-merge streaming
        partition path passes an EXPLICIT barrier (the stacked-time axis) rather
        than relying on auto-detect, which false-negatives on sparse tiles (few
        splats per timepoint trips the n_unique*4<=n guard). Verify each part's
        finest chunk bounds are time-tight when barrier_dims is passed, and that
        auto-detect alone would NOT barrier this sparse part."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.io.save_gsplats import write_partition_streaming
        from luxar.gsplats.tree import GSplatLeaf
        from luxar.io.ordering import detect_barrier_dims

        def make_sparse_4d_leaf(seed: int) -> GSplatLeaf:
            rng = np.random.default_rng(seed)
            n = 40  # sparse: 40 splats over 14 timepoints (40 < 14*4=56)
            centers = np.empty((n, 4), dtype=np.float32)
            centers[:, :3] = rng.uniform(0, 50, (n, 3))
            centers[:, 3] = rng.integers(0, 14, size=n).astype(np.float32)
            chol = np.zeros((n, 10), dtype=np.float32)
            chol[:, [0, 2, 5, 9]] = 2.0
            return GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=centers,
                        amplitudes=rng.uniform(0.1, 1, (n,)).astype(np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )

        # Auto-detect MISSES the sparse time axis (the finding's failure mode),
        # so relying on it (barrier_dims=None default) leaves the axis σ-smeared.
        assert (
            detect_barrier_dims(make_sparse_4d_leaf(0).additive_sublods[0].centers)
            == []
        )

        def max_time_extent(path: Path, part: str) -> float:
            cb = np.asarray(zarr.open_group(str(path), mode="r")[part]["chunk_bounds"])
            return float((cb[:, 3, 1] - cb[:, 3, 0]).max())

        with tempfile.TemporaryDirectory() as tmpdir:
            bpath = Path(tmpdir) / "barrier.gsplats.zarr"
            npath = Path(tmpdir) / "auto.gsplats.zarr"
            # Explicit barrier=[3] — what _merge_partition now passes.
            write_partition_streaming(
                bpath,
                lambda: iter([make_sparse_4d_leaf(0), make_sparse_4d_leaf(1)]),
                barrier_dims=[3],
            )
            # Auto-detect fallback (the pre-fix batch behavior): no barrier found.
            write_partition_streaming(
                npath,
                lambda: iter([make_sparse_4d_leaf(0), make_sparse_4d_leaf(1)]),
            )
            for part in ("part_0", "part_1"):
                broot = zarr.open_group(str(bpath), mode="r")
                assert list(broot[part].attrs["slice_dims"]) == [3]
                assert (
                    list(
                        zarr.open_group(str(npath), mode="r")[part].attrs["slice_dims"]
                    )
                    == []
                )
                # Barrier removes the σ (coverage 3·2=6) expansion on the time
                # axis → strictly tighter time bounds than the auto-detect miss.
                assert max_time_extent(bpath, part) < max_time_extent(npath, part)

    def test_scene_barrier_from_dimension_discrete_beats_autodetect(self) -> None:
        """REGRESSION (deep-double-check, finding 5): a scene-embedded gsplat with
        a NON-INTEGER discrete axis (e.g. physical-time seconds {0.0,0.5,1.0})
        must get its barrier from the scene's authoritative Dimension.discrete
        metadata — value-based auto-detect would reject 0.5 as non-integer and
        leave the axis smeared across chunks. Mirrors the Points/Lines scene path."""
        import zarr

        from luxar import LuxarZarrCompiler
        from luxar.core.dimensions import Dimension, Dimensions

        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, discrete=True, range=(0.0, 1.0)),
            ]
        )
        rng = np.random.default_rng(12)
        n = 3000
        centers = np.empty((n, 4), dtype=np.float32)
        centers[:, :3] = rng.random((n, 3)) * 100
        centers[:, 3] = rng.choice([0.0, 0.5, 1.0], size=n)  # NON-integer time
        chol = np.zeros((n, 10), dtype=np.float32)
        chol[:, [0, 2, 5, 9]] = 2.0
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "scene.luxar.zarr"
            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_gsplats(
                    "gsplats", centers, amplitudes=1.0, cholesky_factors=chol
                )
            store = zarr.open_group(str(store_path), mode="r")
            attrs = dict(store["gsplats"].attrs)
            # Barrier came from Dimension.discrete (index 3), NOT auto-detect
            # (which would return [] because 0.5 is not near-integer).
            assert list(attrs["slice_dims"]) == [3]
            from luxar.io.ordering import detect_barrier_dims

            assert detect_barrier_dims(centers) == []  # proves auto-detect misses it

    def test_no_barrier_3d_unaffected(self) -> None:
        """A 3D leaf gets NO barrier (auto-detect returns [] for continuous
        floats), pure spatial ordering, σ-expanded bounds on every axis.

        Asserts the barrier machinery specifically (not just shape/count): would
        fail if detect_barrier_dims wrongly flagged a float axis (→ slice_dims
        non-empty and tight ±_BARRIER_BOUND_EPS ≈ ±1e-3 bounds — extent ~0.002
        — instead of σ-expanded)."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        rng = np.random.default_rng(3)
        centers = (rng.random((2000, 3)) * 100).astype(np.float32)
        chol = np.zeros((2000, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 2.0  # isotropic σ=2 in all 3 dims
        data = GSplatData(
            centers=centers,
            amplitudes=(rng.random(2000) + 0.5).astype(np.float32),
            cholesky_factors=chol,
        )
        leaf = GSplatLeaf(additive_sublods=list(data.flattened().additive_sublods))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat3d.gsplats.zarr"
            write_gsplats_tree(path, leaf)  # auto-detect → no barrier for floats
            root = zarr.open_group(str(path), mode="r")
            # No axis flagged as a barrier (the load-bearing assertion).
            assert list(root.attrs["slice_dims"]) == []
            assert list(root.attrs["ordering_dims"]) == [0, 1, 2]
            bounds = self._finest_chunk_bounds(path)
            assert bounds.shape[1] == 3
            # Every axis is σ-expanded — extents dwarf the ~2·eps (≈0.002,
            # _BARRIER_BOUND_EPS) a wrongly-flagged barrier axis would get
            # (proves NO axis received tight barrier bounds). σ=2,
            # coverage 3σ → ~6 extent, well over 1.5.
            extents = bounds[:, :, 1] - bounds[:, :, 0]
            assert np.all(extents.max(axis=0) > 1.5)
            got = GSplatData.load(path)
            assert got.n_splats == 2000


class TestRobustDisplayRange:
    """amplitude_data_range (the viewer's colormap display window) must use a
    robust upper (p99.9), not the raw max — and land on the SAME node as the
    'gray' colormap so a colormapped gsplat doesn't render near-black."""

    @staticmethod
    def _skewed_gsplat(n: int = 40000):
        rng = np.random.default_rng(0)
        amp = rng.exponential(0.003, n).astype(np.float32)  # heavy right skew
        amp[rng.integers(0, n, 40)] = rng.uniform(0.1, 0.3, 40)  # bright outliers
        c = (rng.random((n, 3)) * 100).astype(np.float32)
        chol = np.zeros((n, 6), np.float32)
        chol[:, [0, 2, 5]] = 2.0
        return GSplatData(centers=c, amplitudes=amp, cholesky_factors=chol), amp

    def test_leaf_display_range_is_robust_not_max(self) -> None:
        gd, amp = self._skewed_gsplat()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skew.gsplats.zarr"
            gd.save(path, ordering="none")
            root = zarr.open_group(str(path), mode="r")
            adr = root.attrs.get("amplitude_data_range")
            assert adr is not None, "leaf must carry a display range"
            hi = adr[1]
            # Robust: the upper is near p99.9, well below the outlier max.
            assert hi < float(amp.max()) * 0.5, (hi, float(amp.max()))
            assert abs(hi - float(np.percentile(amp, 99.9))) < 1e-4
            # Colocated with the default 'gray' colormap on the SAME node.
            assert root.attrs.get("colormap") == "gray"
            assert "amplitude_data_range" in dict(root.attrs)

    def test_lod_level_carries_display_range_with_colormap(self) -> None:
        """An additive-laddered LOD level (colormap on the level, arrays on its
        sub-LODs) must still carry a display range on the level node itself."""
        from luxar.gsplats.lod import make_substitutive_lod

        gd, _ = self._skewed_gsplat()
        pyr = make_substitutive_lod(
            gd,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pyr.gsplats.zarr"
            pyr.save(path, ordering="none")
            root = zarr.open_group(str(path), mode="r")
            # Every colormapped level node must ALSO have amplitude_data_range.
            for lvl in [k for k in root.group_keys() if k.startswith("child_")]:
                a = dict(root[lvl].attrs)
                if a.get("colormap"):
                    assert "amplitude_data_range" in a, f"{lvl} colormap without range"
                    assert a["amplitude_data_range"][1] > a["amplitude_data_range"][0]


class TestAtomicWrites:
    """Crash-safety: writers must never destroy a prior good store or leave a
    partial one — write-to-temp-sibling + atomic swap (see _atomic_finalize)."""

    @staticmethod
    def _leaf(seed: int):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
        from luxar.gsplats.tree import GSplatLeaf

        rng = np.random.default_rng(seed)
        chol = np.zeros((20, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(20, 3))
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 50, (20, 3)).astype(np.float32),
                    amplitudes=rng.uniform(0.1, 1, (20,)).astype(np.float32),
                    cholesky_factors=chol,
                )
            ]
        )

    @staticmethod
    def _no_tmp_siblings(directory: Path) -> bool:
        return not any(directory.glob(".*tmp-*"))

    def test_failed_tree_write_preserves_prior_store(self, monkeypatch) -> None:
        import importlib

        # NOT `import ... as sg`: the io package re-exports the FUNCTION
        # `save_gsplats`, which shadows the submodule on attribute lookup.
        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            # A good prior store.
            save_gsplats(path=path, **create_test_splats_3d(50), ordering="none")
            before = zarr.open_group(str(path), mode="r").attrs["content_hash"]

            # A rewrite that crashes mid-write (inside the node walker).
            def boom(*a, **k):
                raise RuntimeError("simulated mid-write crash")

            monkeypatch.setattr(sg, "write_gsplat_node", boom)
            with pytest.raises(RuntimeError, match="simulated mid-write crash"):
                sg.write_gsplats_tree(path, self._leaf(1), ordering="none")

            # Prior good store untouched; no temp sibling left behind.
            after = zarr.open_group(str(path), mode="r").attrs["content_hash"]
            assert after == before
            assert self._no_tmp_siblings(path.parent)

    def test_failed_streaming_write_preserves_prior_store(self, monkeypatch) -> None:
        from luxar.gsplats.io.save_gsplats import write_partition_streaming

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "part.gsplats.zarr"
            write_partition_streaming(
                path, lambda: iter([self._leaf(0), self._leaf(1)]), ordering="none"
            )
            before = zarr.open_group(str(path), mode="r").attrs["content_hash"]

            def parts_then_boom():
                yield self._leaf(2)
                raise RuntimeError("simulated producer crash")

            with pytest.raises(RuntimeError, match="simulated producer crash"):
                write_partition_streaming(path, parts_then_boom, ordering="none")

            after = zarr.open_group(str(path), mode="r").attrs["content_hash"]
            assert after == before
            assert self._no_tmp_siblings(path.parent)

    def test_streaming_zero_parts_leaves_nothing(self) -> None:
        # The n_written == 0 ValueError used to leave a partial root at the
        # destination; now neither the destination nor a temp sibling exists.
        from luxar.gsplats.io.save_gsplats import write_partition_streaming

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.gsplats.zarr"
            with pytest.raises(ValueError, match="no non-empty parts"):
                write_partition_streaming(path, lambda: iter([]), ordering="none")
            assert not path.exists()
            assert self._no_tmp_siblings(path.parent)

    def test_failed_compression_leaves_no_partial_archive(self, monkeypatch) -> None:
        import importlib

        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr.zip"

            def boom(*a, **k):
                raise RuntimeError("simulated compression crash")

            monkeypatch.setattr(sg, "_atomic_finalize", boom)
            with pytest.raises(RuntimeError, match="simulated compression crash"):
                save_gsplats(
                    path=path,
                    **create_test_splats_3d(30),
                    ordering="none",
                    compress="zip",
                )
            assert not path.exists()
            assert self._no_tmp_siblings(path.parent)

    def test_success_roundtrip_unchanged(self) -> None:
        # The atomic swap must not change what a successful save produces.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt.gsplats.zarr"
            splats = create_test_splats_3d(40)
            save_gsplats(path=path, **splats, ordering="none")
            data = load_gsplats(path)
            assert data.n_splats == 40
            # Spatial ordering off + AUTO encoding: match the tolerance the
            # existing roundtrip tests use for quantized amplitudes.
            assert np.allclose(
                np.sort(data.amplitudes), np.sort(splats["amplitudes"]), atol=0.05
            )
            assert self._no_tmp_siblings(path.parent)


class TestAtomicFinalizeTrashFirst:
    """The directory swap is trash-first: dest never absent while a prior
    good store existed, and a failed swap-in restores the original."""

    def test_failed_swap_in_restores_prior_store(self, tmp_path, monkeypatch) -> None:
        import importlib
        import os as _os

        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")
        dest = tmp_path / "store.gsplats.zarr"
        dest.mkdir()
        (dest / "old.txt").write_text("prior good store")
        tmp = tmp_path / ".store.tmp"
        tmp.mkdir()
        (tmp / "new.txt").write_text("new store")

        real_replace = _os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:  # the tmp→dest swap-in
                raise OSError("simulated swap-in failure")
            return real_replace(src, dst)

        monkeypatch.setattr(sg.os, "replace", flaky_replace)
        with pytest.raises(OSError, match="simulated swap-in failure"):
            sg._atomic_finalize(tmp, dest)
        # Prior store restored under its original name; tmp untouched.
        assert (dest / "old.txt").read_text() == "prior good store"
        assert (tmp / "new.txt").exists()

    def test_successful_swap_leaves_no_trash(self, tmp_path) -> None:
        import importlib

        sg = importlib.import_module("luxar.gsplats.io.save_gsplats")
        dest = tmp_path / "store.gsplats.zarr"
        dest.mkdir()
        (dest / "old.txt").write_text("x")
        tmp = tmp_path / ".store.tmp"
        tmp.mkdir()
        (tmp / "new.txt").write_text("y")
        sg._atomic_finalize(tmp, dest)
        assert (dest / "new.txt").read_text() == "y"
        assert not (dest / "old.txt").exists()
        assert not list(tmp_path.glob(".*trash-*"))


# ────────────────────────────────────────────────────────────────────────
# Topology-aware coverage_fraction FALLBACK (both writers)
#
# The stamped anchors are what recipes emit, but every writer also DERIVES a
# fallback for a node that carries no ``coverage_fraction`` in its ``meta`` —
# which is exactly the state ``luxar gsplat transform`` leaves the tree in after
# its scrub, and the state a legacy pre-v3.2 store loads in. The CHANGELOG's
# "scrub-and-re-derive stays a no-op" claim rests on that fallback picking the
# PARTITION-BOUND anchor for a partition-bound ladder, so pin it here for both
# ``write_gsplats_tree`` and its streaming sibling.
# ────────────────────────────────────────────────────────────────────────


def _strip_coverage(node):
    """The post-scrub state: no ``coverage_fraction`` anywhere in ``meta``."""
    from luxar.gsplats.tree import without_meta_key

    return without_meta_key(node, "coverage_fraction")


def _recipe_tree(recipe: str):
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

    rng = np.random.default_rng(0)
    n = 400
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
        cholesky_factors=chol,
    )
    params = RecipeParams(
        max_elements=120, compression_factor=4, levels=2, device="cpu", seed=0
    )
    return build_recipe(data, recipe, params)


@pytest.mark.parametrize("recipe", ["overview", "adaptive"])
def test_meta_less_partition_bound_tree_rederives_the_partitioned_anchor(
    recipe: str, tmp_path: Path
) -> None:
    """``write_gsplats_tree`` on a scrubbed tree must re-derive 0.0 → 1.0
    (``PARTITION_FINEST_AREA`` — the tile alone fills the screen)."""
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    node = _strip_coverage(_recipe_tree(recipe))
    out = tmp_path / f"{recipe}.gsplats.zarr"
    write_gsplats_tree(out, node)
    root = zarr.open_group(str(out), mode="r")
    # overview: the lod group IS the root. adaptive: one lod group per part.
    lod_group = root if recipe == "overview" else root["part_0"]
    covs = [
        float(lod_group[k].attrs["coverage_fraction"])
        for k in sorted(lod_group.group_keys(), key=lambda s: int(s.split("_")[1]))
    ]
    assert covs[0] == 0.0
    assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA), (
        f"{recipe}: meta-less re-derivation gave {covs}, expected the "
        "partition-bound anchor (finest = PARTITION_FINEST_AREA)"
    )
    assert all(covs[i] > covs[i - 1] for i in range(1, len(covs)))


def test_scrub_and_rederive_is_a_no_op_for_partition_bound_ladders(
    tmp_path: Path,
) -> None:
    """The exact ``gsplat transform`` round trip: stamped == re-derived."""
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    for recipe in ("overview", "adaptive"):
        node = _recipe_tree(recipe)
        stamped = tmp_path / f"{recipe}_stamped.gsplats.zarr"
        scrubbed = tmp_path / f"{recipe}_scrubbed.gsplats.zarr"
        write_gsplats_tree(stamped, node)
        write_gsplats_tree(scrubbed, _strip_coverage(node))

        def _covs(path):
            root = zarr.open_group(str(path), mode="r")
            g = root if recipe == "overview" else root["part_0"]
            return [
                float(g[k].attrs["coverage_fraction"])
                for k in sorted(g.group_keys(), key=lambda s: int(s.split("_")[1]))
            ]

        assert _covs(stamped) == pytest.approx(_covs(scrubbed)), recipe


def test_streaming_partition_writer_uses_the_partitioned_anchor(
    tmp_path: Path,
) -> None:
    """``write_partition_streaming``'s root IS a kind=partition, so every part is
    partition-bound by construction — its recursion must say so.

    Regression: the streaming writer started the walk at the default
    ``under_partition=False``, so a meta-less per-part ladder came out with the
    whole-object anchor (0.0/0.25/0.5) here while ``write_gsplats_tree`` on the
    same tree gave the tile anchor (0.0/0.5/1.0). Latent only because the batch
    merge stamps ``meta`` that wins over the fallback.
    """
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA
    from luxar.gsplats.io.save_gsplats import write_partition_streaming
    from luxar.gsplats.tree import GSplatPartition

    partitioned = _strip_coverage(_recipe_tree("adaptive"))
    assert isinstance(partitioned, GSplatPartition)
    parts = list(partitioned.children)

    out = tmp_path / "streamed.gsplats.zarr"
    write_partition_streaming(out, lambda: iter(parts), max_elements=120)

    root = zarr.open_group(str(out), mode="r")
    part0 = root["part_0"]
    covs = [
        float(part0[k].attrs["coverage_fraction"])
        for k in sorted(part0.group_keys(), key=lambda s: int(s.split("_")[1]))
    ]
    assert covs[0] == 0.0
    assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA), (
        f"streaming writer gave {covs}; every part_<i> is under a kind=partition "
        "root, so the fallback must use the partition-bound anchor"
    )


def test_meta_less_one_part_partition_rederives_the_whole_object_anchor(
    tmp_path: Path,
) -> None:
    """A ONE-part partition is not a tiling — its part covers the whole object,
    so the fallback must NOT hand the fills-screen anchor down.

    ``build_adaptive`` emits exactly this shape whenever the dataset fits
    ``max_elements``, and it stamps the whole-object anchor (0.5). If the
    writer's topology fallback disagreed, a ``gsplat transform``
    scrub-and-re-derive would silently re-coarsen the store back to the #1361
    behaviour.
    """
    from luxar.core.group.lod.group import WHOLE_OBJECT_FINEST_ANCHOR
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
    from luxar.gsplats.tree import GSplatPartition

    rng = np.random.default_rng(0)
    n = 120
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
        cholesky_factors=chol,
    )
    node = build_recipe(
        data,
        "adaptive",
        # max_elements unset → the default 1,000,000, so the BSP never splits.
        RecipeParams(compression_factor=4, levels=2, device="cpu", seed=0),
    )
    assert isinstance(node, GSplatPartition) and len(node.children) == 1

    stamped = tmp_path / "one_part_stamped.gsplats.zarr"
    scrubbed = tmp_path / "one_part_scrubbed.gsplats.zarr"
    write_gsplats_tree(stamped, node)
    write_gsplats_tree(scrubbed, _strip_coverage(node))

    def _covs(path: Path) -> list[float]:
        g = zarr.open_group(str(path), mode="r")["part_0"]
        return [
            float(g[k].attrs["coverage_fraction"])
            for k in sorted(g.group_keys(), key=lambda s: int(s.split("_")[1]))
        ]

    assert _covs(scrubbed)[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR), (
        f"one-part re-derivation gave {_covs(scrubbed)}, expected the "
        "whole-object half-screen-area anchor (finest = 0.5)"
    )
    # And the round trip is still a no-op, as it is for real tilings.
    assert _covs(stamped) == pytest.approx(_covs(scrubbed))


def test_one_part_partition_nested_in_a_tiling_keeps_the_tile_anchor(
    tmp_path: Path,
) -> None:
    """A one-part partition INSIDE a real tiling must not drop the outer binding.

    The one-part exclusion only ever ADDS a binding, never removes one: ``part_1``
    below is a lone-part wrapper, but it still sits inside ONE tile of a >=2-part
    partition, so the meta-less ladder underneath it keeps the fills-screen
    anchor. Overwriting the incoming flag instead of OR-ing it in would hand that
    ladder the whole-object 0.5 — and would put the writer out of step with
    ``graft_gsplat_node``, its scene-side mirror.
    """
    from luxar.core.group.lod.group import PARTITION_FINEST_AREA
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    def _leaf(n: int, seed: int) -> GSplatLeaf:
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
                    amplitudes=rng.uniform(0.1, 1, (n,)).astype(np.float32),
                    cholesky_factors=chol,
                )
            ]
        )

    # Outer = a genuine 2-part tiling. part_1 = a one-part wrapper holding a
    # meta-less coarse→fine ladder, so nothing authored beats the fallback.
    tree = GSplatPartition(
        children=[
            _leaf(40, 0),
            GSplatPartition(
                children=[GSplatLodGroup(children=[_leaf(25, 1), _leaf(100, 2)])]
            ),
        ]
    )

    out = tmp_path / "nested_one_part.gsplats.zarr"
    write_gsplats_tree(out, tree, ordering="none")

    ladder = zarr.open_group(str(out), mode="r")["part_1"]["part_0"]
    covs = [
        float(ladder[k].attrs["coverage_fraction"])
        for k in sorted(ladder.group_keys(), key=lambda s: int(s.split("_")[1]))
    ]
    assert covs[0] == 0.0
    assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA), (
        f"nested one-part wrapper gave {covs}; the outer >=2-part tiling still "
        "binds this ladder, so the finest must be PARTITION_FINEST_AREA"
    )


def _assert_strict_json(path: Path) -> None:
    """Parse every metadata document under *path* with NaN/Infinity FORBIDDEN.

    ``json.loads`` accepts the bare ``NaN`` / ``Infinity`` tokens Python's
    encoder emits; nothing else does. ``parse_constant`` is the hook that fires
    on exactly those three tokens, so raising from it reproduces what a strict
    reader (the viewer's ``JSON.parse``, jq, any non-Python zarr client) does
    with the same bytes.
    """

    def _reject(token: str) -> None:
        raise AssertionError(f"non-JSON token {token!r} in {path}")

    for doc in sorted(path.rglob("*")):
        if doc.is_file() and doc.name in ("zarr.json", ".zattrs", ".zmetadata"):
            json.loads(doc.read_text(), parse_constant=_reject)


def test_non_finite_fit_metrics_never_reach_the_store(tmp_path: Path) -> None:
    """A ``nan`` / ``inf`` metric must be dropped, not written.

    Both are reachable from ordinary fits: a constant or signal-free volume has
    no foreground, so ``foreground_psnr_db`` is ``nan``, and an exact
    reconstruction gives ``psnr_db == +inf``. zarr writes either as a bare
    ``NaN`` / ``Infinity`` token, and because the root document carries the
    consolidated index for the WHOLE tree, one such value makes the entire store
    unreadable to a strict parser rather than just losing that one number.
    """
    splats = create_test_splats_3d(20)
    data = GSplatData(
        **splats,
        stats={
            "fitter_name": "luxar.gsplats",
            "psnr_db": float("inf"),
            "foreground_psnr_db": float("nan"),
            "foreground_threshold": float("nan"),
            "foreground_fraction": 0.0,
            "ssim": 0.5,
        },
    )
    out = tmp_path / "nonfinite.gsplats.zarr"
    data.save(out)

    _assert_strict_json(out)

    attrs = zarr.open_group(str(out), mode="r")["fitting"].attrs
    assert "psnr_db" not in attrs
    assert "foreground_psnr_db" not in attrs
    assert "foreground_threshold" not in attrs
    # The finite companions survive — dropping the undefined dB value must not
    # take the numbers that explain WHY it is undefined with it.
    assert attrs["foreground_fraction"] == 0.0
    assert attrs["ssim"] == 0.5
    assert attrs["fitter_name"] == "luxar.gsplats"


def test_finite_fit_metrics_still_round_trip(tmp_path: Path) -> None:
    """Negative control for the guard above: a normal fit loses nothing."""
    from luxar.gsplats.io.save_gsplats import split_fitting_info

    fitting_info, _, _, _ = split_fitting_info(
        {
            "psnr_db": 31.5,
            "foreground_psnr_db": 18.25,
            "foreground_threshold": 0.125,
            "foreground_fraction": 0.0221,
            "source_shape": (24, 32, 32),
        }
    )
    assert fitting_info == {
        "psnr_db": 31.5,
        "foreground_psnr_db": 18.25,
        "foreground_threshold": 0.125,
        "foreground_fraction": 0.0221,
        "source_shape": [24, 32, 32],
    }


def test_fit_scale_diagnostics_save_in_fitting_group_and_round_trip(
    tmp_path: Path,
) -> None:
    """Fit diagnostics stay out of topology provenance and preserve nulls."""
    from luxar.gsplats.io.load_gsplats import load_gsplats
    from luxar.gsplats.io.save_gsplats import split_fitting_info

    diagnostics = {
        "configured_iterations": 20_000,
        "dynamic_ops_relocation_events": 17,
        "fit_init_sigma_vox": None,
        "fit_init_sigma_diag_vox": None,
        "fit_init_marginal_sigma_diag_vox_min": [0.5, 0.7, 0.9],
        "fit_init_marginal_sigma_diag_vox_median": [0.8, 1.0, 1.2],
        "fit_init_marginal_sigma_diag_vox_max": [1.1, 1.3, 1.5],
        "splats_near_fit_init_sigma_count": None,
        "splats_near_fit_init_sigma_fraction": None,
    }
    fitting_info, fitting_config, provenance, pipeline = split_fitting_info(diagnostics)
    assert fitting_info == diagnostics
    assert fitting_config is None
    assert provenance is None
    assert pipeline is None

    path = tmp_path / "diagnostics.gsplats.zarr"
    save_gsplats(
        path=path,
        **create_test_splats_3d(5),
        fitting_info=fitting_info,
        ordering="none",
    )

    root = zarr.open_group(str(path), mode="r")
    assert "pipeline" not in root
    assert dict(root["fitting"].attrs).items() >= diagnostics.items()
    loaded = load_gsplats(path, include_stats=True)
    for key, value in diagnostics.items():
        assert loaded.stats[key] == value
