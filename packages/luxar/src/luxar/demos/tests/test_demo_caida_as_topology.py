"""Regression tests for the CAIDA AS-organization parser.

CAIDA's real ``as-org2info`` snapshots spell their section headers as
``format:org_id`` and ``format:aut`` (without a space after the colon).  The
country view silently collapsed to the unknown bucket when the demo expected a
space and skipped both sections.  These tests pin the upstream spelling,
accepted whitespace variants, gzip input, and fail-loud empty-section behavior.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from luxar.demos.demo_caida_as_topology import parse_as_org  # noqa: E402


def _write_fixture(path: Path, text: str) -> None:
    """Write a plain-text or gzip-compressed CAIDA fixture."""
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize(
    ("org_header", "aut_header"),
    [
        ("# format:org_id", "# format:aut"),
        ("# format: org_id", "# format: aut"),
        ("  # format :  ORG_ID", "\t#format:\taut"),
    ],
)
def test_parse_as_org_accepts_real_and_whitespace_variant_headers(
    tmp_path: Path,
    compressed: bool,
    org_header: str,
    aut_header: str,
) -> None:
    suffix = ".txt.gz" if compressed else ".txt"
    path = tmp_path / f"snapshot.as-org2info{suffix}"
    _write_fixture(
        path,
        "\n".join(
            [
                f"{org_header}|changed|org_name|country|source",
                "ORG-EXAMPLE|20260801|Example Networks|US|caida",
                "ORG-NO-COUNTRY|20260801|No Country||caida",
                f"{aut_header}|changed|aut_name|org_id|opaque_id|source",
                "64500|20260801|Example ASN|ORG-EXAMPLE||caida",
                "64501|20260801|Countryless ASN|ORG-NO-COUNTRY||caida",
                "64502|20260801|Missing Org ASN|ORG-MISSING||caida",
                "",
            ]
        ),
    )

    frame = parse_as_org(path)

    assert frame.to_dict(orient="index") == {
        "64500": {"org_name": "Example Networks", "country": "US"},
        "64501": {"org_name": "No Country", "country": "??"},
        "64502": {"org_name": "unknown", "country": "??"},
    }


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        (
            "# format:org_id|changed|org_name|country|source\n"
            "ORG-EXAMPLE|20260801|Example Networks|US|caida\n",
            "ASN-to-organization records",
        ),
        (
            "# format:aut|changed|aut_name|org_id|opaque_id|source\n"
            "64500|20260801|Example ASN|ORG-EXAMPLE||caida\n",
            "organization records",
        ),
    ],
)
def test_parse_as_org_rejects_a_missing_required_section(
    tmp_path: Path,
    text: str,
    missing: str,
) -> None:
    path = tmp_path / "incomplete.as-org2info.txt"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        parse_as_org(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert missing in message
    assert "# format:org_id|..." in message
    assert "# format:aut|..." in message
