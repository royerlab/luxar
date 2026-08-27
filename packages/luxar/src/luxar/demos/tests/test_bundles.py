"""Tests for precomputed demo bundle loading."""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from luxar.demos._support.datasets.lfs import _unshippable_reason


class TestCacheStaleness:
    """The demo cache must self-heal when packaged data is re-migrated
    (e.g. the gsplats v2.0 -> v3.0 cutover), not pin the first-seen copy."""

    def test_cache_is_stale_detects_refresh_conditions(self, tmp_path: Path) -> None:
        import os

        from luxar.demos._support.datasets.cache import _cache_is_stale

        cache = tmp_path / "cache.bin"
        source = tmp_path / "source.bin"

        # Missing cache → stale.
        source.write_bytes(b"x" * 100)
        assert _cache_is_stale(cache, source) is True

        # Identical size + same mtime → fresh.
        cache.write_bytes(b"x" * 100)
        os.utime(cache, (source.stat().st_atime, source.stat().st_mtime))
        assert _cache_is_stale(cache, source) is False

        # Source larger (re-migration changed content) → stale.
        source.write_bytes(b"x" * 250)
        os.utime(cache, (source.stat().st_atime, source.stat().st_mtime))
        assert _cache_is_stale(cache, source) is True

        # Same size but source newer (in-place rewrite) → stale.
        cache.write_bytes(b"y" * 250)
        old = source.stat().st_mtime - 100
        os.utime(cache, (old, old))
        assert _cache_is_stale(cache, source) is True

        # Source absent (unpulled LFS) → keep the cached copy.
        source.unlink()
        assert _cache_is_stale(cache, source) is False

    def test_load_precomputed_refreshes_a_stale_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A stale cached copy is replaced by the (changed) packaged source."""
        import os

        import luxar.demos._support.datasets.bundles as bundles
        from luxar.gsplats.gsplat_data import GSplatData

        data_root = tmp_path / "data"
        cache_root = tmp_path / "cache"
        monkeypatch.setattr(bundles, "_DEMOS_DATA_DIR", data_root)
        monkeypatch.setattr(bundles, "_DEFAULT_CACHE_ROOT", cache_root)

        n = 8
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        src_dir = data_root / "gsplats_x"
        src_dir.mkdir(parents=True)
        src = src_dir / "x.gsplats.zarr.zip"
        GSplatData(
            centers=np.random.rand(n, 3).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        ).save(src, ordering="none", compress="zip")

        # Seed the cache with a STALE, unreadable copy (old mtime).
        (cache_root / "gsplats_x").mkdir(parents=True)
        stale = cache_root / "gsplats_x" / "x.gsplats.zarr.zip"
        stale.write_bytes(b"stale-not-a-zarr")
        os.utime(stale, (0, 0))  # far in the past → source is newer

        out = bundles.load_precomputed_gsplats("gsplats_x", ["x.gsplats.zarr.zip"])
        # Refreshed from source and loaded (would raise on the stale bytes).
        assert out is not None and out[0].n_splats == n


class TestUnshippableData:
    """Data we may not redistribute is absent ON PURPOSE.

    Those datasets carry no in-repo copy, so the caller must be routed to its
    own rebuild path instead of being told to run ``git lfs pull`` for a file
    that does not exist in the repository and never will.
    """

    def test_local_compute_dataset_returns_none_instead_of_raising(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import luxar.demos._support.datasets.bundles as bundles

        # Neither an in-repo copy nor a cached one — the post-removal state of
        # every `local-compute` dataset on a fresh clone.
        monkeypatch.setattr(bundles, "_DEMOS_DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(bundles, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")

        out = bundles.load_precomputed_gsplats(
            "gsplats_tribolium", ["tribolium.gsplats.zarr.zip"]
        )
        assert out is None

    def test_shippable_dataset_still_raises_the_lfs_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A `zenodo` dataset that is merely unpulled must NOT be excused."""
        import luxar.demos._support.datasets.bundles as bundles

        monkeypatch.setattr(bundles, "_DEMOS_DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(bundles, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")

        with pytest.raises(FileNotFoundError):
            bundles.load_precomputed_gsplats("gsplats_dapi", ["dapi.gsplats.zarr.zip"])

    def test_unknown_dataset_is_treated_as_shippable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """No manifest entry → no excuse; the ordinary missing-file error wins."""
        import luxar.demos._support.datasets.bundles as bundles

        monkeypatch.setattr(bundles, "_DEMOS_DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(bundles, "_DEFAULT_CACHE_ROOT", tmp_path / "cache")

        assert _unshippable_reason("not_a_dataset") is None
        with pytest.raises(FileNotFoundError):
            bundles.load_precomputed_gsplats("not_a_dataset", ["x.gsplats.zarr.zip"])


class TestExtractBundleAndLoad:
    """The extraction body shared by both bundle loaders.

    Split out of ``load_precomputed_bundle`` so the manifest-driven
    ``load_dataset_bundle`` reuses it rather than growing a lookalike. The member
    matching is security-sensitive and the staleness stamp is what stops a
    re-migrated bundle serving stale frames, so both behaviours are pinned here.
    """

    @staticmethod
    def _bundle(path: Path, members: dict[str, bytes]) -> None:

        with zipfile.ZipFile(path, "w") as zf:
            for name, blob in members.items():
                zf.writestr(name, blob)

    @staticmethod
    def _load_stub(monkeypatch):
        """Return per-file payloads instead of parsing real gsplat stores."""
        from luxar.gsplats import gsplat_data as gd

        class _Stub:
            def __init__(self, blob: bytes) -> None:
                self.blob = blob
                self.amplitudes = blob  # the loader logs len(amplitudes)

            def __eq__(self, other: object) -> bool:
                return self.blob == other

        monkeypatch.setattr(
            gd.GSplatData,
            "load",
            classmethod(lambda cls, p, **kw: _Stub(p.read_bytes())),
        )

    def test_extracts_members_flattened_into_the_cache_dir(self, tmp_path, monkeypatch):
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        # A member nested in a directory must still land flat, by basename.
        self._bundle(b, {"inner/f0.zip": b"zero", "f1.zip": b"one"})
        cache = tmp_path / "cache"
        out = du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip", "f1.zip"], validate_lfs=False
        )
        assert out == [b"zero", b"one"]
        assert (cache / "f0.zip").read_bytes() == b"zero"
        assert not (cache / "inner").exists(), "member was not flattened"

    def test_second_call_reuses_the_extraction(self, tmp_path, monkeypatch):
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        cache = tmp_path / "cache"
        du._extract_bundle_and_load(b, "b.zip", cache, ["f0.zip"], validate_lfs=False)
        # Corrupt the extracted copy: an honoured stamp means it is NOT re-extracted.
        (cache / "f0.zip").write_bytes(b"stale-but-present")
        out = du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False
        )
        assert out == [b"stale-but-present"]

    def test_a_changed_bundle_forces_re_extraction(self, tmp_path, monkeypatch):
        """The self-healing property: a re-migrated bundle must refresh the cache.

        Without it a format re-migration leaves demos loading frames from the old
        bundle, which then fail against the newer reader.
        """
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        cache = tmp_path / "cache"
        self._bundle(b, {"f0.zip": b"v1"})
        du._extract_bundle_and_load(b, "b.zip", cache, ["f0.zip"], validate_lfs=False)
        self._bundle(b, {"f0.zip": b"v2-longer-payload"})  # new size => new stamp
        out = du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False
        )
        assert out == [b"v2-longer-payload"]

    def test_an_explicit_stamp_sees_a_swap_that_size_and_mtime_cannot(
        self, tmp_path, monkeypatch
    ):
        """The manifest-driven path keys staleness on the verified digest.

        A replacement bundle of the same byte length, written back with the
        previous mtime, is indistinguishable to the ``(size, mtime)`` fallback —
        it would keep serving the earlier extraction. The digest the manifest
        already carries settles it exactly.
        """
        import os

        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)

        def _swap_in_place(path: Path, payload: bytes) -> None:
            """Rewrite the bundle with an equal-length payload, mtime restored."""
            before = path.stat()
            self._bundle(path, {"f0.zip": payload})
            assert path.stat().st_size == before.st_size
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

        # Control: with the (size, mtime) key the swap is invisible.
        ctl, ctl_cache = tmp_path / "ctl.zip", tmp_path / "ctl-cache"
        self._bundle(ctl, {"f0.zip": b"v1"})
        du._extract_bundle_and_load(
            ctl, "ctl.zip", ctl_cache, ["f0.zip"], validate_lfs=False
        )
        _swap_in_place(ctl, b"v2")
        assert du._extract_bundle_and_load(
            ctl, "ctl.zip", ctl_cache, ["f0.zip"], validate_lfs=False
        ) == [b"v1"]

        # With the digest it is not.
        b, cache = tmp_path / "b.zip", tmp_path / "cache"
        self._bundle(b, {"f0.zip": b"v1"})
        du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False, stamp="sha256:aaa"
        )
        _swap_in_place(b, b"v2")
        assert du._extract_bundle_and_load(
            b, "b.zip", cache, ["f0.zip"], validate_lfs=False, stamp="sha256:bbb"
        ) == [b"v2"]

    def test_a_member_absent_from_the_bundle_raises(self, tmp_path, monkeypatch):
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        with pytest.raises(FileNotFoundError, match="not found in bundle"):
            du._extract_bundle_and_load(
                b, "b.zip", tmp_path / "cache", ["nope.zip"], validate_lfs=False
            )

    def test_a_missing_member_is_the_ROUTABLE_error_not_a_bare_one(
        self, tmp_path, monkeypatch
    ):
        """It must be distinguishable from the faults beside it (#1618).

        The bundle itself resolved and verified; only a per-frame name missed,
        and those names are derived from the caller's own flags (NEXRAD's
        ``--dbz-floor`` / ``--splats`` / ``--grid-m``), so recomputing is right.
        The demo therefore catches this SPECIFIC type — if it degrades back to a
        plain ``FileNotFoundError`` the demo either crashes on a legitimate
        non-default run, or has to widen its catch and start swallowing the
        checksum faults ``DatasetUnavailable`` exists to keep out.
        """
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        with pytest.raises(du.BundleMemberNotFound):
            du._extract_bundle_and_load(
                b, "b.zip", tmp_path / "cache", ["nope.zip"], validate_lfs=False
            )
        assert issubclass(du.BundleMemberNotFound, FileNotFoundError)

    def test_a_traversal_member_name_is_refused(self, tmp_path, monkeypatch):
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        self._bundle(b, {"f0.zip": b"zero"})
        with pytest.raises(ValueError, match="Unsafe zip path"):
            du._extract_bundle_and_load(
                b, "b.zip", tmp_path / "cache", ["../escape.zip"], validate_lfs=False
            )

    def test_a_member_whose_name_merely_contains_the_request_is_not_matched(
        self, tmp_path, monkeypatch
    ):
        """Members match by exact basename, not by substring.

        A bundle holding both ``decoy_f0.zip`` and ``f0.zip`` must yield the
        latter. Substring matching would take whichever the archive happens to
        list first and silently load the wrong frame -- and it passed every other
        test in this class, so it needs its own.
        """
        from luxar.demos._support.datasets import bundles as du

        self._load_stub(monkeypatch)
        b = tmp_path / "b.zip"
        # Decoy first: `matching[0]` picks it if the comparison is not exact.
        self._bundle(b, {"decoy_f0.zip": b"WRONG", "f0.zip": b"RIGHT"})
        out = du._extract_bundle_and_load(
            b, "b.zip", tmp_path / "cache", ["f0.zip"], validate_lfs=False
        )
        assert out == [b"RIGHT"]
