"""
Root pytest configuration for the luxar package.

Adds an autouse fixture that seeds numpy's global RNG to a fixed value
before every test in the package. Tests using `np.random.*` (without an
explicit `default_rng(seed)`) inherit determinism for free. Tests that
already create their own seeded `default_rng` are unaffected.

Also pins zarr's ambient default format to whatever Luxar writes for the whole
session (see ``_zarr_format_follows_luxar``), and holds a few small shared test
helpers: ``array_compressor`` reads an array's compressor without the caller
knowing which zarr format wrote it, and ``find_repo_relative_file`` /
``read_ts_number_const`` let the handful of cross-language constant-lock tests
read a number straight out of a TypeScript source rather than trust a prose
comment to stay in sync.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import zarr


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


class CompressorView(NamedTuple):
    """A zarr array's compressor, normalised across both formats.

    ``shuffle`` is the numcodecs integer (0 none / 1 byte / 2 bit), because that
    is what the compressor POLICY is stated in
    (:mod:`luxar.encoding.compression`) and what tests compare against.
    """

    cname: str
    clevel: int
    shuffle: int


#: v3 spells blosc's shuffle as a name; the policy speaks numcodecs' integers.
_V3_SHUFFLE_INTS = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def array_compressor(array: Any) -> CompressorView | None:
    """The blosc compressor of ``array``, from a format-2 or format-3 store.

    ``None`` means stored RAW. Raises for a non-blosc compressor rather than
    guessing, since every caller asserts against the blosc-shaped policy.

    Formats differ in BOTH the accessor and the spelling: format 2 exposes a
    single numcodecs ``.compressor`` whose ``shuffle`` is an int; format 3
    exposes a ``.compressors`` tuple of ``zarr.codecs`` objects whose
    ``shuffle`` is a name.

    ``.compressor`` RAISES on a format-3 array — ``TypeError: `compressor` is
    not available for Zarr format 3 arrays.`` — rather than returning ``None``,
    so ``getattr(array, "compressor", None)`` does NOT absorb it: getattr's
    default only covers ``AttributeError``. That is the good outcome (loud, not
    silent), but it has to be caught here rather than at every call site.
    """
    try:
        legacy = array.compressor
    except (AttributeError, TypeError):
        legacy = None
    if legacy is not None:
        return CompressorView(
            cname=str(legacy.cname),
            clevel=int(legacy.clevel),
            shuffle=int(legacy.shuffle),
        )

    for codec in getattr(array, "compressors", ()) or ():
        name = getattr(codec, "cname", None)
        if name is None:
            continue  # not blosc (e.g. the mandatory `bytes` codec)
        shuffle = getattr(codec, "shuffle", None)
        shuffle_name = getattr(shuffle, "value", shuffle)
        return CompressorView(
            cname=str(getattr(name, "value", name)),
            clevel=int(codec.clevel),
            shuffle=_V3_SHUFFLE_INTS[str(shuffle_name)],
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


def read_ts_number_const(source: str, name: str) -> float:
    """Parse ``const NAME = <number>;`` out of TypeScript source text.

    Tolerates ``export const``, a plain ``const``, and an optional ``: number``
    type annotation. Raises ``AssertionError`` (not a parse exception) so a
    failure reads as a normal, informative test failure.
    """
    m = re.search(
        rf"^\s*(?:export\s+)?const\s+{name}\s*(?::\s*number\s*)?=\s*"
        r"([0-9]*\.?[0-9]+)\s*;",
        source,
        re.MULTILINE,
    )
    assert m is not None, (
        f"no `const {name} = <number>;` declaration found in the given source. "
        "If it was renamed or computed, update the caller — do NOT delete it."
    )
    return float(m.group(1))
