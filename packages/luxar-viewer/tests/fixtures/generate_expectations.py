#!/usr/bin/env python3
"""Generate Python decoder expectations for TypeScript round-trip tests.

The generated JSON is intentionally derived from Python's own ArrayDecoder,
then consumed by Vitest in Node.js. This gives us a fast, non-browser contract
check for Python encode -> zarr -> TypeScript decode consistency.

Run from the repository root:
    hatch run python packages/luxar-viewer/tests/fixtures/generate_expectations.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from arbol import aprint, asection

from luxar.encoding import ArrayDecoder

FIXTURES_DIR = Path(__file__).parent
EXPECTATIONS_PATH = FIXTURES_DIR / "roundtrip_expectations.json"

# Labels and image-label blobs use dedicated loaders rather than ArrayDecoder.
# Keep this test focused on numeric geometry/attribute arrays used by the viewer
# data path.
_EXCLUDED_PATH_PARTS = {
    "label_offsets",
    "label_bytes",
    "image_label_offsets",
    "image_label_bytes",
}


def _iter_arrays(group: zarr.Group, prefix: str = "") -> list[str]:
    """Return all array paths in a zarr group, sorted for determinism."""
    paths: list[str] = []
    for key in sorted(group.keys()):
        path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        if isinstance(item, zarr.core.Array):
            paths.append(path)
        elif isinstance(item, zarr.hierarchy.Group):
            paths.extend(_iter_arrays(item, path))
    return paths


def _should_include_array(path: str, array: zarr.Array) -> bool:
    """Whether an array belongs in the cross-language ArrayDecoder contract."""
    parts = set(path.split("/"))
    if parts & _EXCLUDED_PATH_PARTS:
        return False
    return bool(np.issubdtype(array.dtype, np.number))


def _encoding_name(array: zarr.Array) -> str:
    enc = array.attrs.get("encoding", None)
    if isinstance(enc, dict):
        name = enc.get("name")
        if isinstance(name, str):
            return name
    return "none"


def _float32_view(array: np.ndarray) -> np.ndarray:
    """Canonical flattened little-endian float32 view for JS comparison."""
    return np.ascontiguousarray(np.asarray(array, dtype="<f4").reshape(-1))


def _sha256_float32(array: np.ndarray) -> str:
    flat = _float32_view(array)
    return hashlib.sha256(flat.tobytes(order="C")).hexdigest()


def _samples(array: np.ndarray, max_samples: int = 16) -> list[dict[str, float | int]]:
    flat = _float32_view(array)
    if flat.size == 0:
        return []
    indices = np.unique(
        np.linspace(0, flat.size - 1, min(max_samples, flat.size), dtype=np.int64)
    )
    return [{"index": int(i), "value": float(flat[i])} for i in indices]


def _stats(array: np.ndarray) -> dict[str, float | None]:
    flat = _float32_view(array)
    if flat.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat, dtype=np.float64)),
    }


def _array_expectation(
    path: str, array: zarr.Array, root: zarr.Group
) -> dict[str, Any]:
    decoder = ArrayDecoder()
    decoded = decoder.decode(array, root)
    decoded = np.asarray(decoded)
    flat = _float32_view(decoded)

    expected_elements = int(decoded.shape[0]) if decoded.ndim > 0 else int(decoded.size)
    return {
        "path": path,
        "encoding": _encoding_name(array),
        "storage_shape": [int(v) for v in array.shape],
        "storage_dtype": str(array.dtype),
        "decoded_shape": [int(v) for v in decoded.shape],
        "decoded_dtype": str(decoded.dtype),
        "expected_elements": expected_elements,
        "flat_length": int(flat.size),
        "float32_sha256": _sha256_float32(decoded),
        "samples": _samples(decoded),
        "stats": _stats(decoded),
    }


def generate_expectations() -> dict[str, Any]:
    """Generate expectations for every ``test_*.zarr`` fixture."""
    fixtures: dict[str, Any] = {}

    for fixture_path in sorted(FIXTURES_DIR.glob("test_*.zarr")):
        if not fixture_path.is_dir():
            continue

        with asection(f"Expectations: {fixture_path.name}"):
            root = zarr.open_group(str(fixture_path), mode="r")
            arrays: dict[str, Any] = {}
            for array_path in _iter_arrays(root):
                array = root[array_path]
                if not _should_include_array(array_path, array):
                    continue
                arrays[array_path] = _array_expectation(array_path, array, root)
                enc = arrays[array_path]["encoding"]
                shape = arrays[array_path]["decoded_shape"]
                aprint(f"✓ {array_path} ({enc}) -> {shape}")

            fixtures[fixture_path.name] = {"arrays": arrays}

    return {
        "version": 1,
        "description": "Python ArrayDecoder expectations for TypeScript round-trip tests.",
        "fixtures": fixtures,
    }


def main() -> None:
    with asection("Generating TypeScript round-trip expectations"):
        expectations = generate_expectations()
        EXPECTATIONS_PATH.write_text(
            json.dumps(expectations, indent=2, sort_keys=True) + "\n"
        )
        n_fixtures = len(expectations["fixtures"])
        n_arrays = sum(
            len(item["arrays"]) for item in expectations["fixtures"].values()
        )
        aprint(f"✓ Wrote {EXPECTATIONS_PATH}")
        aprint(f"  Fixtures: {n_fixtures}")
        aprint(f"  Arrays: {n_arrays}")


if __name__ == "__main__":
    main()
