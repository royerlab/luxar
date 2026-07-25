"""Tests for the ESM-3 demo's cache/dependency gate.

The demo's embeddings cache is quarantined to ``<name>.corrupt`` when it fails
validation. A quarantine left behind by an *earlier* run used to be invisible:
the ``.npy`` is already gone, so the validation block never fires and the demo
silently restarted a multi-gigabyte compute/download. The gate must name the
quarantined path, its size, and what to do about it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from luxar.demos import demo_esm3_protein_landscape as demo
from luxar.demos.demo_esm3_protein_landscape import (
    _compute_esm3_embeddings,
    _require_module,
)
from luxar.utils.download import QUARANTINE_SUFFIX

SEQUENCES = ["MKV", "MTL", "MGG"]
HEAVY_DEPS = ("torch", "esm", "umap")


@pytest.fixture
def without_heavy_deps(monkeypatch):
    """Make torch / esm / umap-learn look uninstalled, machine-independently."""
    real = importlib.import_module

    def fake(name, package=None):
        if name.split(".")[0] in HEAVY_DEPS:
            raise ImportError(f"No module named {name!r}")
        return real(name, package)

    monkeypatch.setattr(demo.importlib, "import_module", fake)


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


class TestDependencyGatesAreDeferred:
    """The heavy deps must be demanded where used, never at the entry point.

    The demo tells the user that a complete cached embeddings file "skips the
    model entirely". An entry-point preflight that imports torch/esm/umap-learn
    and exits makes that advice a lie: a machine holding every artifact it needs
    would still be refused.
    """

    def test_complete_cache_needs_none_of_them(
        self, tmp_path, without_heavy_deps
    ) -> None:
        cache = tmp_path / "embeddings_esmc_300m.npy"
        np.save(cache, np.zeros((len(SEQUENCES), 960), dtype=np.float32))

        got = _compute_esm3_embeddings(SEQUENCES, tmp_path, model_name="esmc-300m")

        assert got.shape == (len(SEQUENCES), 960)

    def test_main_reaches_the_generator_without_them(
        self, tmp_path, monkeypatch, without_heavy_deps
    ) -> None:
        """main() must hand off to the generator, not sys.exit on a preflight."""
        called: list[Path] = []

        def fake_generate(output_path, **_kwargs):
            called.append(output_path)
            return 0  # 0 => main() returns without serving

        monkeypatch.setattr(demo, "generate_esm3_landscape", fake_generate)
        monkeypatch.setattr(demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(demo.sys, "argv", ["demo", "--no-serve"])

        demo.main()

        assert called, "main() exited before reaching generate_esm3_landscape"

    def test_missing_dep_message_is_actionable_at_point_of_use(
        self, without_heavy_deps
    ) -> None:
        with pytest.raises(ImportError) as excinfo:
            _require_module("esm")

        message = str(excinfo.value)
        assert "pip install 'esm>=3.0.0'" in message
        assert "luxar[demos]" in message
        # Names the cache escape hatch, which is the whole point of deferring.
        assert "cached embeddings" in message
