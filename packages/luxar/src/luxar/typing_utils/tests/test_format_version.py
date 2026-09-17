"""The on-disk format-version policy, pinned as ONE case table.

The same table — same ids, same inputs, same outcomes — lives in the viewer's
``src/tests/unit/data/format-version.test.ts``. Keep the two in lockstep: the
point of a shared policy is that a store the Python reader warns about is a
store the viewer warns about, and a store one refuses the other refuses.
"""

from __future__ import annotations

import warnings

import pytest

from luxar.typing_utils import _format_contract as fc
from luxar.typing_utils.format_version import (
    FormatVersionOutcome,
    UnsupportedFormatVersionError,
    check_format_version,
    enforce_gsplats_format_version,
    enforce_scene_format_version,
    parse_format_version,
    read_scene_format_version,
)

SCENE = ("scene", fc.SCENE_FORMAT_VERSION, fc.SUPPORTED_SCENE_VERSIONS)
GSPLATS = ("gsplats", fc.GSPLATS_FORMAT_VERSION, fc.SUPPORTED_GSPLATS_VERSIONS)

#: (id, kind-tuple, on-disk version, expected outcome). Mirrored by the TS test.
CASES = [
    pytest.param(
        SCENE, "0.1", FormatVersionOutcome.SUPPORTED, id="supported-scene-0.1"
    ),
    pytest.param(
        SCENE, "0.2", FormatVersionOutcome.SUPPORTED, id="supported-scene-0.2"
    ),
    pytest.param(
        GSPLATS, "3.0", FormatVersionOutcome.SUPPORTED, id="supported-gsplats-3.0"
    ),
    pytest.param(
        GSPLATS, "3.4", FormatVersionOutcome.SUPPORTED, id="supported-gsplats-3.4"
    ),
    pytest.param(
        SCENE, "0.3", FormatVersionOutcome.NEWER_MINOR, id="newer-minor-scene"
    ),
    pytest.param(
        GSPLATS, "3.5", FormatVersionOutcome.NEWER_MINOR, id="newer-minor-gsplats"
    ),
    pytest.param(
        SCENE, "0.0", FormatVersionOutcome.REFUSE, id="older-unsupported-scene"
    ),
    pytest.param(
        GSPLATS, "2.0", FormatVersionOutcome.REFUSE, id="older-unsupported-gsplats"
    ),
    pytest.param(SCENE, "9.9", FormatVersionOutcome.REFUSE, id="newer-major-scene"),
    pytest.param(GSPLATS, "9.9", FormatVersionOutcome.REFUSE, id="newer-major-gsplats"),
    pytest.param(SCENE, "abc", FormatVersionOutcome.REFUSE, id="unparsable"),
    pytest.param(SCENE, None, FormatVersionOutcome.REFUSE, id="unparsable-none"),
    pytest.param(SCENE, "0.2.1", FormatVersionOutcome.REFUSE, id="unparsable-patch"),
]


@pytest.mark.parametrize(("kind", "version", "expected"), CASES)
def test_check_format_version_case_table(kind, version, expected) -> None:
    name, current, supported = kind
    outcome, message = check_format_version(name, version, current, supported)
    assert outcome is expected
    if expected is FormatVersionOutcome.SUPPORTED:
        assert message == ""
    else:
        assert str(version) in message
        assert message


def test_gsplats_refuse_message_keeps_the_migrate_hint() -> None:
    """The ``luxar gsplat migrate-format <in> <out>`` remedy survives the move."""
    _, message = check_format_version("gsplats", "2.0", *GSPLATS[1:])
    assert "luxar gsplat migrate-format" in message
    assert "Unsupported format_version: '2.0'" in message


def test_scene_refuse_message_names_the_remedy() -> None:
    _, older = check_format_version("scene", "0.0", *SCENE[1:])
    _, newer = check_format_version("scene", "9.9", *SCENE[1:])
    assert "Rebuild the scene" in older
    assert "Upgrade Luxar" in newer


def test_newer_minor_message_says_loading_anyway() -> None:
    _, message = check_format_version("scene", "0.3", *SCENE[1:])
    assert "Loading anyway" in message
    assert fc.SCENE_FORMAT_VERSION in message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.2", (0, 2)),
        ("3.4", (3, 4)),
        ("10.12", (10, 12)),
        ("abc", None),
        ("0.2.1", None),
        ("v0.2", None),
        (" 0.2", None),
        (0.2, None),
        (None, None),
    ],
)
def test_parse_format_version(raw, expected) -> None:
    assert parse_format_version(raw) == expected


# --------------------------------------------------------------------------- #
# Scene root attrs → the enforcement arms
# --------------------------------------------------------------------------- #
def test_read_scene_format_version_prefers_format_version() -> None:
    assert read_scene_format_version({"format_version": "0.2"}) == "0.2"
    assert (
        read_scene_format_version({"format_version": "0.2", "luxar_version": "0.1"})
        == "0.2"
    )


def test_legacy_key_fallback() -> None:
    """``legacy-key-fallback``: a 0.1 root with only ``luxar_version`` loads."""
    attrs = {"type": "scene", fc.LEGACY_SCENE_VERSION_ATTR: "0.1"}
    assert read_scene_format_version(attrs) == "0.1"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert enforce_scene_format_version(attrs) is FormatVersionOutcome.SUPPORTED


def test_missing_without_format_type_is_tolerated() -> None:
    """``missing-without-format-type``: a pre-header store returns ``None``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert enforce_scene_format_version({"type": "scene"}) is None


def test_missing_with_format_type_is_refused() -> None:
    """``missing-with-format-type``: a stripped 0.2 header is corrupt, not legacy."""
    with pytest.raises(UnsupportedFormatVersionError, match="format_type"):
        enforce_scene_format_version(
            {"type": "scene", "format_type": fc.FORMAT_TYPE_SCENE}
        )


def test_enforce_scene_newer_minor_warns_and_returns() -> None:
    with pytest.warns(UserWarning, match="Loading anyway"):
        outcome = enforce_scene_format_version(
            {"type": "scene", "format_version": "0.3"}
        )
    assert outcome is FormatVersionOutcome.NEWER_MINOR


@pytest.mark.parametrize("version", ["0.0", "9.9", "abc"])
def test_enforce_scene_refuses(version: str) -> None:
    with pytest.raises(UnsupportedFormatVersionError) as exc:
        enforce_scene_format_version({"type": "scene", "format_version": version})
    assert version in str(exc.value)
    assert isinstance(exc.value, ValueError)


def test_enforce_scene_coerces_non_string_versions() -> None:
    """A float ``0.2`` (YAML-ish) still compares against the allowlist."""
    assert (
        enforce_scene_format_version({"type": "scene", "format_version": 0.2})
        is FormatVersionOutcome.SUPPORTED
    )


# --------------------------------------------------------------------------- #
# gsplats root attrs
# --------------------------------------------------------------------------- #
def test_enforce_gsplats_requires_format_type() -> None:
    with pytest.raises(UnsupportedFormatVersionError, match="Invalid format_type"):
        enforce_gsplats_format_version({"format_version": "3.4"})


def test_enforce_gsplats_supported_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outcome = enforce_gsplats_format_version(
            {"format_type": fc.FORMAT_TYPE_GSPLATS, "format_version": "3.4"}
        )
    assert outcome is FormatVersionOutcome.SUPPORTED


def test_enforce_gsplats_newer_minor_warns() -> None:
    with pytest.warns(UserWarning, match="3.5"):
        outcome = enforce_gsplats_format_version(
            {"format_type": fc.FORMAT_TYPE_GSPLATS, "format_version": "3.5"}
        )
    assert outcome is FormatVersionOutcome.NEWER_MINOR


def test_enforce_gsplats_refuses_with_migrate_hint() -> None:
    with pytest.raises(UnsupportedFormatVersionError, match="migrate-format"):
        enforce_gsplats_format_version(
            {"format_type": fc.FORMAT_TYPE_GSPLATS, "format_version": "2.0"}
        )
    with pytest.raises(UnsupportedFormatVersionError, match="None"):
        enforce_gsplats_format_version({"format_type": fc.FORMAT_TYPE_GSPLATS})
