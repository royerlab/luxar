"""``content_hash`` must not depend on WHICH Luxar release wrote the scene.

Scene 0.2 stamps ``luxar_software_version`` (= ``luxar.__version__``) into the
root header for provenance. If that key folded into the digest, every Luxar
release would produce a different ``content_hash`` for byte-identical content,
and a ``luxar optimize`` restamp under a newer release would cold-start every
viewer's OPFS cache. Both hashers — the compile-time walk in
``_compiler/finalize/hashing.py`` and the streaming twin in ``io/optimize.py`` —
read one ``HASH_EXCLUDED_ATTRS``; these tests pin that each of them actually
honours it (mutating either hasher to hash the stamp fails a test here).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import luxar
from luxar import Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import open_group
from luxar.io._compiler.finalize.hashing import (
    HASH_EXCLUDED_ATTRS,
    compute_content_hashes,
)
from luxar.io.optimize import _compute_content_hashes_streaming
from luxar.typing_utils._format_contract import SOFTWARE_VERSION_ATTR


def _compile(path: Path) -> str:
    """Compile a small deterministic scene; return its root ``content_hash``."""
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((500, 3)).astype(np.float32)
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", positions)
    return str(dict(open_group(path, mode="r").attrs)["content_hash"])


def test_hash_excluded_attrs_names_the_software_stamp() -> None:
    assert HASH_EXCLUDED_ATTRS == frozenset({"content_hash", SOFTWARE_VERSION_ATTR})


def test_two_releases_compile_to_the_same_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same scene, two ``luxar.__version__`` values → same root digest."""
    monkeypatch.setattr(luxar, "__version__", "2026.01.01")
    hash_a = _compile(tmp_path / "a.luxar.zarr")
    monkeypatch.setattr(luxar, "__version__", "2027.12.31")
    hash_b = _compile(tmp_path / "b.luxar.zarr")

    # The stamp itself is live (so the exclusion is doing real work), and the
    # digest ignores it.
    attrs_a = dict(open_group(tmp_path / "a.luxar.zarr", mode="r").attrs)
    attrs_b = dict(open_group(tmp_path / "b.luxar.zarr", mode="r").attrs)
    assert attrs_a[SOFTWARE_VERSION_ATTR] == "2026.01.01"
    assert attrs_b[SOFTWARE_VERSION_ATTR] == "2027.12.31"
    assert hash_a == hash_b


def test_compile_time_hasher_ignores_a_rewritten_stamp(tmp_path: Path) -> None:
    """Rewriting the stamp on a finished store and re-hashing gives the same digest."""
    path = tmp_path / "scene.luxar.zarr"
    original = _compile(path)
    root = open_group(path, mode="r+")
    root.attrs[SOFTWARE_VERSION_ATTR] = "9999.99.99"
    assert compute_content_hashes(root) == original


def test_streaming_hasher_matches_the_compiler_and_ignores_the_stamp(
    tmp_path: Path,
) -> None:
    """The ``luxar optimize`` twin agrees with the compiler on a 0.2 store, with
    and without a different software stamp."""
    path = tmp_path / "scene.luxar.zarr"
    original = _compile(path)

    same = tmp_path / "same.luxar.zarr"
    shutil.copytree(path, same)
    assert _compute_content_hashes_streaming(open_group(same, mode="r+")) == original

    restamped = tmp_path / "restamped.luxar.zarr"
    shutil.copytree(path, restamped)
    root = open_group(restamped, mode="r+")
    root.attrs[SOFTWARE_VERSION_ATTR] = "9999.99.99"
    assert _compute_content_hashes_streaming(root) == original


def test_the_format_header_itself_is_hashed(tmp_path: Path) -> None:
    """Only the software stamp is excluded: ``format_version`` is content."""
    path = tmp_path / "scene.luxar.zarr"
    original = _compile(path)
    root = open_group(path, mode="r+")
    root.attrs["format_version"] = "0.1"
    assert compute_content_hashes(root) != original
