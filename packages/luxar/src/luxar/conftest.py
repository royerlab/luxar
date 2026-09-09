"""
Root pytest configuration for the luxar package.

Adds an autouse fixture that seeds numpy's global RNG to a fixed value
before every test in the package. Tests using `np.random.*` (without an
explicit `default_rng(seed)`) inherit determinism for free. Tests that
already create their own seeded `default_rng` are unaffected.

Also pins zarr's ambient default format to whatever Luxar writes for the whole
session (see ``_zarr_format_follows_luxar``), clears the CLI traceback opt-in
before each test, and holds a few small shared test helpers:
``confine_temp_dirs`` isolates in-process temporary files,
``array_compressor`` reads an array's compressor without the caller knowing
which zarr format wrote it, and ``find_repo_relative_file`` / ``viewer_source`` /
``read_ts_number_const`` / ``read_ts_string_literals`` let the handful of
cross-language constant-lock tests read values straight out of a TypeScript
source rather than trust a prose comment to stay in sync.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import zarr


def confine_temp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point in-process ``tempfile`` creation and probes at this test's directory.

    Without this, cleanup probes glob the machine-global temp directory and can
    observe another concurrent process creating a matching directory. Both
    ``tempfile.mkdtemp()`` and ``tempfile.gettempdir()`` honour the cached
    ``tempfile.tempdir`` value, so setting it confines the producer and probe
    together. Unlike setting ``TMPDIR``, this does not propagate to subprocesses.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture(scope="session", autouse=True)
def _zarr_format_follows_luxar() -> Iterator[None]:
    """Make a bare ``zarr.group()`` write whatever format Luxar itself writes.

    ~700 test call sites construct a store directly rather than through Luxar's
    writers, and zarr's own ambient default is independent of Luxar's. Left
    alone, those fixtures would be a different format from production output and
    would fail in confusing, non-local ways — ``consolidate`` puts
    ``consolidated_metadata`` inside ``zarr.json`` at format 3 versus emitting a
    ``.zmetadata`` document at format 2, so a completion-sentinel check reports
    the wrong answer, and any assertion naming ``.zarray`` / ``.zattrs`` finds
    nothing.

    It tracks :data:`luxar._zarr_compat.ZARR_FORMAT` rather than naming a
    format, for two reasons. It cannot go stale: this fixture previously pinned
    2 with a docstring explaining that 2 was what Luxar wrote, and the day that
    changed the explanation silently became false. And it makes the override
    testable end to end — ``LUXAR_ZARR_FORMAT=2 hatch run test`` now exercises
    the whole format-2 path, writers and readers together, instead of only the
    handful of places that pass ``zarr_format=2`` by hand.

    Crucially this cannot paper over a production regression:
    :mod:`luxar._zarr_compat` passes ``zarr_format`` explicitly and never reads
    this config, and ``test_zarr_compat.py`` pins that independence by flipping
    the ambient default to the OTHER format and asserting the facade ignores it.

    KNOWN LIMIT — it covers this PROCESS only. Anything that writes a store from a
    subprocess, or from a script run outside pytest, gets zarr's own default
    instead. That is not hypothetical: it is exactly how
    ``packages/luxar-viewer/tests/fixtures/generate_test_data.py`` came to emit
    two format-3 fixtures back when Luxar wrote 2, since it is a standalone
    script rather than a test. Any such writer must go through
    :mod:`luxar._zarr_compat` or pass ``zarr_format=`` itself, which
    ``test_zarr_compat.py::test_no_writer_creates_a_store_without_pinning_the_format``
    enforces across the package, the scripts, the examples and the generators.
    """
    from luxar import _zarr_compat

    with zarr.config.set({"default_zarr_format": _zarr_compat.ZARR_FORMAT}):
        yield


@pytest.fixture(autouse=True)
def _seed_numpy_global_rng() -> None:
    """Seed numpy's legacy global RNG before each test (P7 determinism).

    Use `np.random.default_rng(seed)` in new tests for a per-test
    generator. This fixture only protects existing call sites that rely
    on the legacy module-level functions (`np.random.uniform`,
    `np.random.randn`, etc.).
    """
    np.random.seed(0xC0FFEE)


@pytest.fixture(autouse=True)
def _traceback_opt_in_is_test_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not let a developer's shell preference change CLI test semantics."""
    # Must match luxar.cli._traceback.TRACEBACK_ENV_VAR without importing the CLI.
    monkeypatch.delenv("LUXAR_TRACEBACK", raising=False)


class CompressorView(NamedTuple):
    """A zarr array's compressor, normalised across both formats.

    ``shuffle`` is the numcodecs integer (0 none / 1 byte / 2 bit), because that
    is what the compressor POLICY is stated in
    (:mod:`luxar.encoding.compression`) and what tests compare against.
    """

    cname: str
    clevel: int
    shuffle: int


class ValidationCase(NamedTuple):
    """One invalid-input case for an ordered validation-equivalence test."""

    name: str
    overrides: Mapping[str, Any]
    exception_type: type[Exception]
    message: str


def ordered_validation_pairs(
    cases: Sequence[ValidationCase],
) -> list[tuple[ValidationCase, ValidationCase]]:
    """Return every earlier/later pair from an ordered validation contract."""
    names = [case.name for case in cases]
    assert len(names) == len(set(names)), (
        f"validation case names must be unique: {names}"
    )
    pairs = [
        (earlier, later)
        for index, earlier in enumerate(cases)
        for later in cases[index + 1 :]
    ]
    for earlier, later in pairs:
        overlap = set(earlier.overrides) & set(later.overrides)
        assert not overlap, (
            f"validation cases {earlier.name!r} and {later.name!r} both override "
            f"{sorted(overlap)} and cannot form a two-violation input"
        )
    return pairs


def assert_validation_precedence(
    validate: Callable[[Mapping[str, Any]], Any],
    earlier: ValidationCase,
    later: ValidationCase,
) -> None:
    """Assert that ``earlier`` remains the exact failure for a two-error input."""
    overrides = {**later.overrides, **earlier.overrides}
    with pytest.raises(earlier.exception_type) as raised:
        validate(overrides)
    assert str(raised.value) == earlier.message


#: v3 spells blosc's shuffle as a NAME; numcodecs spells it as an int, and the
#: policy (:mod:`luxar.encoding.compression`) is stated in the ints.
_V3_SHUFFLE_INTS = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def array_compressor(array: Any) -> CompressorView | None:
    """The blosc compressor of ``array``, from a format-2 or format-3 store.

    ``None`` means stored RAW. A non-blosc compressor RAISES rather than
    answering, in both formats: every caller asserts against the blosc-shaped
    policy, and callers read ``None`` as "stored uncompressed", so quietly
    returning it for a zstd-compressed array would turn a compression
    regression into a passing test.

    Read through ``.compressors`` for BOTH formats. The singular
    ``.compressor`` is zarr-2-shaped and doubly unusable: it is deprecated (it
    warns on every format-2 read) and it RAISES on a format-3 array —
    ``TypeError: `compressor` is not available for Zarr format 3 arrays.`` —
    rather than returning ``None``, so even ``getattr(array, "compressor",
    None)`` does not absorb it, getattr's default covering only
    ``AttributeError``.

    What differs between the formats is only the SPELLING of what
    ``.compressors`` holds, and both are normalised here. Measured, for the
    same logical array:

    ========  ==================================  ==============================
    stored    format 2                            format 3
    ========  ==================================  ==============================
    raw       ``()``                              ``()``
    blosc     ``(Blosc(cname='zstd', ...),)``     ``(BloscCodec(cname=..., ...),)``
    zstd      ``(Zstd(level=9),)``                ``(ZstdCodec(level=9, ...),)``
    ========  ==================================  ==============================

    So ``()`` unambiguously means RAW in both — the mandatory ``bytes`` codec
    lives in ``.serializer``, never here — and an entry without a ``cname`` is
    always a real non-blosc compressor rather than structural noise to skip.
    ``cname``/``shuffle`` are plain values at format 2 and enums at format 3,
    hence the ``.value`` unwrapping.
    """
    codecs = tuple(getattr(array, "compressors", ()) or ())
    for codec in codecs:
        cname = getattr(codec, "cname", None)
        if cname is None:
            continue  # a real non-blosc compressor; reported below
        shuffle = getattr(codec, "shuffle", 0)
        shuffle = getattr(shuffle, "value", shuffle)  # v3 enum -> its name
        return CompressorView(
            cname=str(getattr(cname, "value", cname)),
            clevel=int(codec.clevel),
            shuffle=(
                _V3_SHUFFLE_INTS[shuffle] if isinstance(shuffle, str) else int(shuffle)
            ),
        )
    if codecs:
        raise TypeError(
            f"array is compressed by {codecs!r}, which is not blosc; "
            f"array_compressor only describes the blosc-shaped policy, and "
            f"returning None here would report it as stored RAW"
        )
    return None


def find_repo_relative_file(rel_path: Path, start: Path) -> Path | None:
    """Walk up from ``start``'s parents looking for ``parent / rel_path``.

    Used by cross-language constant-lock tests to locate a file elsewhere in
    the repo (e.g. a TypeScript source under ``packages/luxar-viewer``)
    without hard-coding a parent-directory depth — a package move would
    silently break a fixed ``../../..`` chain, whereas this keeps searching
    until it finds the file or runs out of ancestors.
    """
    return next((p / rel_path for p in start.parents if (p / rel_path).is_file()), None)


def viewer_source(rel_path: str) -> Path:
    """Return one viewer file, relative to ``packages/luxar-viewer``.

    Calls must pass a literal string: ``test_ci_diff_classifier.py`` scans them
    statically and checks that every consumed viewer source selects ``dom_py``.
    Keeping resolution here makes a new cross-language reader register its CI
    ownership at the same line that names the file.
    """
    relative = Path(rel_path)
    assert (
        relative.parts and not relative.is_absolute() and ".." not in relative.parts
    ), f"viewer_source() requires a path below packages/luxar-viewer, got {rel_path!r}"
    source = find_repo_relative_file(
        Path("packages/luxar-viewer") / relative, Path(__file__).resolve()
    )
    assert source is not None, (
        f"cannot locate packages/luxar-viewer/{rel_path}; if the viewer file moved, "
        "update the contract test"
    )
    return source


def read_ts_number_const(source: str, name: str) -> float:
    """Parse ``const NAME = <number>;`` out of TypeScript source text.

    Tolerates ``export const``, a plain ``const``, an optional ``: number``
    type annotation, and exponential literals (``1e-3``, ``2.5E+6``) — a
    float-safety epsilon is normally spelled that way, and matching only
    fixed-point notation would fail the parse rather than the comparison.
    Raises ``AssertionError`` (not a parse exception) so a failure reads as a
    normal, informative test failure.
    """
    m = re.search(
        rf"^\s*(?:export\s+)?const\s+{name}\s*(?::\s*number\s*)?=\s*"
        r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*;",
        source,
        re.MULTILINE,
    )
    assert m is not None, (
        f"no `const {name} = <number>;` declaration found in the given source. "
        "If it was renamed or computed, update the caller — do NOT delete it."
    )
    return float(m.group(1))


def read_ts_string_literals(source: str, name: str) -> frozenset[str]:
    """Parse a string vocabulary from a TypeScript type, property, or array.

    Accepts either ``type NAME = 'a' | 'b';`` (optionally exported) or an
    interface property spelled ``NAME: 'a' | 'b';``, plus a literal const array
    such as ``const NAME: readonly T[] = ['a', 'b'];``. Raises ``AssertionError``
    when the declaration is missing, ambiguous, computed, multiline, or repeats
    a member: source-lock tests should fail loudly when the TypeScript shape
    changes rather than silently compare an incomplete vocabulary.
    """
    source_without_comments = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    escaped_name = re.escape(name)
    union_matches = re.findall(
        rf"^\s*(?:(?:export\s+)?type\s+{escaped_name}\s*=|"
        rf"{escaped_name}\s*\??\s*:)\s*"
        r"((?:'[^'\r\n]+'\s*\|\s*)*'[^'\r\n]+')\s*;",
        source_without_comments,
        re.MULTILINE,
    )
    array_matches = re.findall(
        rf"^\s*(?:export\s+)?const\s+{escaped_name}(?:\s*:[^=]+)?=\s*\[\s*"
        r"((?:'[^'\r\n]+'\s*,\s*)*'[^'\r\n]+'\s*,?)\s*\]\s*;",
        source_without_comments,
        re.MULTILINE,
    )
    matches = union_matches + array_matches
    assert len(matches) == 1, (
        f"expected exactly one literal string declaration named {name!r}, found "
        f"{len(matches)}. If it was renamed, computed, or split across lines, "
        "update the caller or this parser — do NOT delete the cross-language lock."
    )
    members = re.findall(r"'([^']+)'", matches[0])
    assert len(members) == len(set(members)), (
        f"literal-string union {name!r} repeats a member: {members!r}"
    )
    return frozenset(members)
