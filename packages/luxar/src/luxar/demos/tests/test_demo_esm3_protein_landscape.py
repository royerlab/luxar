"""Tests for the ESM-3 demo's cache/dependency gate.

The demo's embeddings cache is quarantined to ``<name>.corrupt`` when it fails
validation. A quarantine left behind by an *earlier* run used to be invisible:
the ``.npy`` is already gone, so the validation block never fires and the demo
silently restarted a multi-gigabyte compute/download. The gate must name the
quarantined path, its size, and what to do about it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.demos.demo_esm3_protein_landscape import _compute_esm3_embeddings
from luxar.utils.download import QUARANTINE_SUFFIX

SEQUENCES = ["MKV", "MTL", "MGG"]


class TestQuarantineReporting:
    def test_pre_existing_quarantine_is_reported(self, tmp_path, capsys) -> None:
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CUDA available: the demo would compute instead of failing")

        corrupt = tmp_path / f"embeddings_esmc_300m.npy{QUARANTINE_SUFFIX}"
        corrupt.write_bytes(b"x" * 3072)

        with pytest.raises(RuntimeError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        message = str(excinfo.value)
        assert str(corrupt) in message
        assert "3.00 KB" in message
        assert "QUARANTINED" in message
        # Warned on the console too, before anything expensive was attempted.
        assert corrupt.name in capsys.readouterr().out

    def test_clean_cache_message_has_no_quarantine_noise(self, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CUDA available: the demo would compute instead of failing")

        with pytest.raises(RuntimeError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        assert "QUARANTINED" not in str(excinfo.value)

    def test_invalid_cache_is_quarantined_and_reported(self, tmp_path, capsys) -> None:
        """A wrong-shaped cache is renamed, then reported by the same gate."""
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("CUDA available: the demo would compute instead of failing")

        cache = tmp_path / "embeddings_esmc_300m.npy"
        np.save(cache, np.zeros((2, 7), dtype=np.float32))  # wrong shape

        with pytest.raises(RuntimeError) as excinfo:
            _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        quarantined = Path(f"{cache}{QUARANTINE_SUFFIX}")
        assert quarantined.is_file(), "invalid cache was not quarantined"
        assert not cache.exists()
        assert str(quarantined) in str(excinfo.value)
        assert "quarantined" in capsys.readouterr().out.lower()
