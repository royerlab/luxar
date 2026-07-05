"""Tests for the width-aware per-dtype compressor policy.

Measured in the manuscript's ``codec_selection`` supplementary: multi-byte
integer codes want byte-shuffled high-level zstd; uint8 codes and floats want
unshuffled high-level zstd (Blosc silently neutralises bit shuffle above
level 1 at 64 KiB chunks, and shuffles are no-ops/harmful for 1-byte data).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor


class TestResolveCompressor:
    @pytest.mark.parametrize(
        "dtype,shuffle",
        [
            (np.uint16, Blosc.SHUFFLE),
            (np.int32, Blosc.SHUFFLE),
            (np.uint64, Blosc.SHUFFLE),
            (np.uint8, Blosc.NOSHUFFLE),
            (np.int8, Blosc.NOSHUFFLE),
            (np.float32, Blosc.NOSHUFFLE),
            (np.float64, Blosc.NOSHUFFLE),
        ],
    )
    def test_width_policy(self, dtype, shuffle):
        c = resolve_compressor(WIDTH_AWARE_DEFAULT, dtype)
        assert c.cname == "zstd"
        assert c.clevel == 9
        assert c.shuffle == shuffle

    def test_explicit_compressor_passes_through(self):
        mine = Blosc(cname="lz4", clevel=1)
        assert resolve_compressor(mine, np.uint16) is mine

    def test_none_stays_uncompressed(self):
        assert resolve_compressor(None, np.uint16) is None


class TestStoreWideCompressorPolicy:
    """A saved gsplat store carries the per-dtype policy on every array."""

    def test_saved_store_compressors(self):
        from luxar.gsplats.io import save_gsplats

        rng = np.random.RandomState(0)
        n = 2000
        centers = (rng.rand(n, 3) * 1000).astype(np.float32)
        # wide-range amplitudes -> geolog u16
        amps = np.exp(rng.uniform(np.log(5e-4), np.log(2e4), n)).astype(np.float32)
        chol = rng.standard_normal((n, 6)).astype(np.float32)
        chol[:, [0, 2, 5]] = np.abs(chol[:, [0, 2, 5]]) + 0.5

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.gsplats.zarr"
            save_gsplats(
                path=p, centers=centers, amplitudes=amps, cholesky_factors=chol
            )
            g = zarr.open_group(str(p), mode="r")
            expectations = {
                "centers": ("uint16", Blosc.SHUFFLE),  # u16 fixed-point codes
                "amplitudes": ("uint16", Blosc.SHUFFLE),  # geolog u16 codes
                "cholesky_factors_diag": ("uint8", Blosc.NOSHUFFLE),
                "cholesky_factors_offdiag": ("uint8", Blosc.NOSHUFFLE),
                "chunk_bounds": ("float32", Blosc.NOSHUFFLE),
            }
            for name, (dtype, shuffle) in expectations.items():
                arr = g[name]
                assert str(arr.dtype) == dtype, (name, arr.dtype)
                c = arr.compressor
                assert c is not None and c.cname == "zstd", (name, c)
                assert c.clevel == 9, (name, c.clevel)
                assert c.shuffle == shuffle, (name, c.shuffle)

    def test_explicit_none_disables_compression(self):
        from luxar.gsplats.io import save_gsplats

        rng = np.random.RandomState(1)
        n = 200
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.gsplats.zarr"
            save_gsplats(
                path=p,
                centers=(rng.rand(n, 3) * 10).astype(np.float32),
                amplitudes=rng.rand(n).astype(np.float32),
                cholesky_factors=np.abs(rng.standard_normal((n, 6))).astype(np.float32)
                + 0.5,
                compressor=None,
                ordering="none",
            )
            g = zarr.open_group(str(p), mode="r")
            assert g["centers"].compressor is None
