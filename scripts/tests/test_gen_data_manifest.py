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


def _manifest_entry(hosted: str | None, history: list[str] | None = None) -> dict:
    entry: dict[str, object] = {"name": "fit.zip"}
    if hosted is not None:
        entry["hosted_sha256"] = hosted
    if history is not None:
        entry["superseded_sha256"] = history
    return {"datasets": {"toy": {"files": [entry]}}}


def test_hosted_repin_accepts_outgoing_digest_and_existing_history() -> None:
    generator = _load_generator()
    outgoing = "a" * 64
    committed = _manifest_entry(outgoing, ["0" * 64])
    repinned = _manifest_entry("b" * 64, ["0" * 64, outgoing])

    generator._refuse_invalid_repin_history(repinned, committed)


def test_hosted_only_repin_requires_outgoing_digest_in_runtime_slot() -> None:
    generator = _load_generator()
    outgoing = "a" * 64
    committed = _manifest_entry(outgoing, ["0" * 64])
    repinned = _manifest_entry("b" * 64, ["0" * 64, outgoing, "f" * 64])

    with pytest.raises(ValueError, match="outgoing hosted digest last"):
        generator._refuse_invalid_repin_history(repinned, committed)


def test_combined_repin_keeps_outgoing_local_digest_in_runtime_slot() -> None:
    generator = _load_generator()
    committed_entry = {
        "name": "fit.zip",
        "sha256": "0" * 64,
        "hosted_sha256": "a" * 64,
    }
    authored_entry = {
        **committed_entry,
        "hosted_sha256": "b" * 64,
        "superseded_sha256": ["a" * 64],
    }
    (repinned_entry,) = generator._carry_hosted(
        [{"name": "fit.zip", "sha256": "1" * 64}], [authored_entry]
    )
    committed = {"datasets": {"toy": {"files": [committed_entry]}}}
    repinned = {"datasets": {"toy": {"files": [repinned_entry]}}}

    assert repinned_entry["superseded_sha256"] == ["a" * 64, "0" * 64]
    generator._refuse_invalid_repin_history(repinned, committed)

    repinned_entry["superseded_sha256"] = ["0" * 64, "a" * 64]
    with pytest.raises(ValueError, match="outgoing local digest last"):
        generator._refuse_invalid_repin_history(repinned, committed)


def test_hosted_repin_rejects_missing_outgoing_digest() -> None:
    generator = _load_generator()
    committed = _manifest_entry("a" * 64, ["0" * 64])
    repinned = _manifest_entry("b" * 64, ["0" * 64])

    with pytest.raises(ValueError, match="outgoing hosted digest"):
        generator._refuse_invalid_repin_history(repinned, committed)


def test_hosted_repin_rejects_wrong_appended_digest() -> None:
    generator = _load_generator()
    committed = _manifest_entry("a" * 64, ["0" * 64])
    repinned = _manifest_entry("b" * 64, ["0" * 64, "f" * 64])

    with pytest.raises(ValueError, match="outgoing hosted digest"):
        generator._refuse_invalid_repin_history(repinned, committed)


def test_hosted_repin_rejects_dropped_existing_history() -> None:
    generator = _load_generator()
    outgoing = "a" * 64
    committed = _manifest_entry(outgoing, ["0" * 64, "1" * 64])
    repinned = _manifest_entry("b" * 64, [outgoing])

    with pytest.raises(ValueError, match="drops 2 previously recorded"):
        generator._refuse_invalid_repin_history(repinned, committed)


def test_first_hosted_pin_needs_no_superseded_history() -> None:
    generator = _load_generator()
    committed = _manifest_entry(None)
    current = _manifest_entry("a" * 64)

    generator._refuse_invalid_repin_history(current, committed)


def test_unchanged_hosted_pin_may_remove_obsolete_history() -> None:
    generator = _load_generator()
    committed = _manifest_entry("a" * 64, ["0" * 64])
    current = _manifest_entry("a" * 64)

    generator._refuse_invalid_repin_history(current, committed)
