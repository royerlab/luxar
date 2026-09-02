"""Publication-state checks for the live Zenodo migration audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "zenodo_migration_audit.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("zenodo_migration_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deposition(*, submitted: bool) -> dict:
    return {
        "submitted": submitted,
        "files": [],
        "metadata": {
            "title": "Record",
            "creators": [{"name": "Example"}],
            "license": "cc-by-4.0",
            "upload_type": "dataset",
            "description": "<table><tr></tr></table>",
        },
    }


@pytest.mark.parametrize("published", [False, True])
def test_matching_publication_state_emits_no_warning(audit, published: bool) -> None:
    fails, warns = audit.check_deposition(
        "record",
        _deposition(submitted=published),
        {},
        {},
        {"published": published},
    )

    assert fails == []
    assert warns == []


@pytest.mark.parametrize(
    ("manifest_published", "live_published", "message"),
    [
        (True, False, "manifest marks published but live deposition is not"),
        (False, True, "live deposition is published but manifest is not"),
    ],
)
def test_publication_state_disagreement_is_reported(
    audit, manifest_published: bool, live_published: bool, message: str
) -> None:
    _fails, warns = audit.check_deposition(
        "record",
        _deposition(submitted=live_published),
        {},
        {},
        {"published": manifest_published},
    )

    assert warns == [f"[record] {message}"]
