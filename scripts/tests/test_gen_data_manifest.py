"""Focused tests for generated demo-manifest positional groups."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "gen_data_manifest.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_data_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_positional_pairs_stamp_every_member() -> None:
    generator = _load_generator()
    files = [{"name": "fit.zip"}, {"name": "colors.bin"}]

    result = generator._declare_positional_pairs(
        files, {"toy": ("fit.zip", "colors.bin")}
    )

    assert result is files
    assert [entry["positional_pair"] for entry in result] == ["toy", "toy"]


def test_positional_pairs_reject_a_partially_present_group() -> None:
    generator = _load_generator()

    with pytest.raises(ValueError, match="names missing files.*colors.bin"):
        generator._declare_positional_pairs(
            [{"name": "fit.zip"}], {"toy": ("fit.zip", "colors.bin")}
        )


def test_positional_pairs_reject_overlapping_groups() -> None:
    generator = _load_generator()
    files = [
        {"name": "fit.zip"},
        {"name": "colors.bin"},
        {"name": "labels.bin"},
    ]

    with pytest.raises(ValueError, match="more than one positional pair.*colors.bin"):
        generator._declare_positional_pairs(
            files,
            {
                "colors": ("fit.zip", "colors.bin"),
                "labels": ("colors.bin", "labels.bin"),
            },
        )


def test_positional_pairs_skip_a_wholly_absent_pruned_group() -> None:
    generator = _load_generator()
    files = [{"name": "unrelated.bin"}]

    assert generator._declare_positional_pairs(
        files, {"toy": ("fit.zip", "colors.bin")}
    ) == [{"name": "unrelated.bin"}]


def test_positional_pairs_reject_a_group_without_a_partner() -> None:
    generator = _load_generator()

    with pytest.raises(ValueError, match="must name at least two files"):
        generator._declare_positional_pairs(
            [{"name": "fit.zip"}], {"toy": ("fit.zip",)}
        )


def test_positional_pairs_reject_different_history_depths() -> None:
    generator = _load_generator()
    files = [
        {"name": "fit.zip", "superseded_sha256": ["old-fit"]},
        {
            "name": "colors.bin",
            "superseded_sha256": ["ancient-colors", "old-colors"],
        },
    ]

    with pytest.raises(ValueError, match="did not move atomically"):
        generator._declare_positional_pairs(files, {"toy": ("fit.zip", "colors.bin")})


def test_build_rejects_positional_pairs_on_variants(monkeypatch) -> None:
    generator = _load_generator()
    monkeypatch.setattr(
        generator,
        "DATASETS",
        {
            "toy": {
                "bucket": "zenodo",
                "license": "cc0-1.0",
                "variants": {},
                "positional_pairs": {"toy": ("fit.zip", "colors.bin")},
            }
        },
    )

    with pytest.raises(ValueError, match="variants.*positional_pairs"):
        generator.build()
