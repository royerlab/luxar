"""Tests for helpers in :mod:`luxar.demos._support._umap_utils`."""

from pathlib import Path

import numpy as np
import pytest

# The helpers import Pillow at module scope; Pillow ships in the ``demos``
# extra, not core.
pytest.importorskip("PIL")

from luxar.demos._support import _umap_utils  # noqa: E402


def test_generate_all_legends_writes_beside_demo_modules(
    monkeypatch, tmp_path: Path
) -> None:
    support_dir = tmp_path / "demos" / "_support"
    support_dir.mkdir(parents=True)
    monkeypatch.setattr(_umap_utils, "__file__", str(support_dir / "_umap_utils.py"))

    outputs: list[Path] = []
    monkeypatch.setattr(
        _umap_utils,
        "generate_legend_image",
        lambda _labels, _title, output_path, **_kwargs: outputs.append(output_path),
    )

    _umap_utils.generate_all_legends(
        {"celltype": np.array([0])},
        {"celltype": ["neuron"]},
        prefix="test",
        attr_display_names={"celltype": "Cell Type"},
    )

    assert outputs == [tmp_path / "demos" / "legends" / "legend_test_celltype.png"]
