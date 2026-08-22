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
from collections import Counter
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

_LINEAR_KERNEL_ENCODINGS = {
    "rgb_uint8",
    "rgb_uint16",
    "bounded_scalar_uint8",
    "bounded_scalar_uint16",
}
_GEOLOG_KERNEL_ENCODINGS = {"geolog_scalar_uint8", "geolog_scalar_uint16"}


def _iter_arrays(group: zarr.Group, prefix: str = "") -> list[str]:
    """Return all array paths in a zarr group, sorted for determinism."""
    paths: list[str] = []
    for key in sorted(group.keys()):
        path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        # `zarr.Array` / `zarr.Group`, not `zarr.core.Array` / `zarr.hierarchy.Group`:
        # zarr 3 removed `zarr.hierarchy` entirely and repurposed `zarr.core`, so the
        # old spellings raise AttributeError. The top-level names work in both.
        if isinstance(item, zarr.Array):
            paths.append(path)
        elif isinstance(item, zarr.Group):
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


def _viewer_kernel_decode(array: zarr.Array, root: zarr.Group) -> np.ndarray | None:
    """Decode in the f32 order of ``src/wasm/typescript/decode.ts``.

    Python's scalar decoders use f64; ``_decode_bounded_scalar`` also uses a
    different operand order. Keep this transcription aligned with the
    TypeScript and Rust kernels.
    """
    enc = array.attrs.get("encoding", None)
    if not isinstance(enc, dict):
        return None
    name = enc.get("name")
    if not isinstance(name, str):
        return None

    if name == "array_ref":
        target = enc.get("target")
        if not isinstance(target, str):
            return None
        target_array = root[target]
        if not isinstance(target_array, zarr.Array):
            return None
        return _viewer_kernel_decode(target_array, root)

    if name in _LINEAR_KERNEL_ENCODINGS:
        data = np.asarray(array[:])
        top = int(np.iinfo(data.dtype).max)
        if "bounds" in enc:
            min_val, max_val = enc["bounds"]
        elif "min" in enc and "max" in enc:
            min_val, max_val = enc["min"], enc["max"]
        elif name in {"rgb_uint8", "rgb_uint16"}:
            min_val, max_val = 0.0, 1.0
        else:
            raise ValueError(f"Missing bounds for viewer kernel expectation: {name}")

        lo = float(np.float32(min_val))
        hi = float(np.float32(max_val))
        value_range = float(np.float32(hi - lo))
        scale = float(np.float32(value_range / top))
        scaled = np.asarray(data.astype(np.float64) * scale, dtype="<f4")
        return np.asarray(lo + scaled.astype(np.float64), dtype="<f4")

    if name in _GEOLOG_KERNEL_ENCODINGS:
        data = np.asarray(array[:])
        top = int(np.iinfo(data.dtype).max)
        lo = float(np.float32(enc["min_log"]))
        hi = float(np.float32(enc["max_log"]))
        inv = max(hi - lo, 0.0) / max(top - 1, 1)
        result = np.zeros(data.shape, dtype="<f4")
        nonzero = data > 0
        if nonzero.any():
            exponent = lo + (data[nonzero].astype(np.float64) - 1.0) * inv
            result[nonzero] = np.asarray(np.exp(exponent), dtype="<f4")
        return result

    return None


def _shape_class(decoded_shape: list[int]) -> str:
    """Classify decoded shapes for coverage-manifest assertions."""
    if not decoded_shape:
        return "scalar"
    if 0 in decoded_shape:
        return "empty"
    if decoded_shape[0] == 1:
        return "singleton"
    if len(decoded_shape) == 1:
        return "vector_1d"
    if len(decoded_shape) == 2:
        return "matrix_2d"
    return "tensor_nd"


def _operation_expectation(
    kind: str,
    array: np.ndarray,
    *,
    viewer_array: np.ndarray | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    flat = _float32_view(array)
    item: dict[str, Any] = {
        "kind": kind,
        "decoded_shape": [int(v) for v in array.shape],
        "flat_length": int(flat.size),
        "float32_sha256": _sha256_float32(array),
        "samples": _samples(array),
        "stats": _stats(array),
    }
    if viewer_array is not None:
        item["viewer_float32_sha256"] = _sha256_float32(viewer_array)
        item["viewer_samples"] = _samples(viewer_array)
    if start is not None or end is not None:
        item["start"] = int(start or 0)
        item["end"] = int(end or 0)
    return item


def _range_operations(
    decoded: np.ndarray, viewer_decoded: np.ndarray | None
) -> list[dict[str, Any]]:
    """Representative first-axis range expectations for TS range decoding."""
    operations = [_operation_expectation("full", decoded, viewer_array=viewer_decoded)]
    if decoded.ndim == 0:
        return operations

    n_items = int(decoded.shape[0])
    if n_items == 0:
        return operations

    candidates = [
        (0, min(1, n_items)),
        (0, min(3, n_items)),
        (max(0, n_items // 2 - 1), min(n_items, n_items // 2 + 2)),
        (max(0, n_items - 3), n_items),
        (0, n_items),
    ]
    if n_items >= 9:
        candidates.append((2, 9))
    if n_items >= 257:
        candidates.append((250, min(n_items, 260)))

    seen: set[tuple[int, int]] = set()
    for start, end in candidates:
        if start >= end or (start, end) in seen:
            continue
        seen.add((start, end))
        operations.append(
            _operation_expectation(
                "range",
                decoded[start:end],
                viewer_array=(
                    viewer_decoded[start:end] if viewer_decoded is not None else None
                ),
                start=start,
                end=end,
            )
        )
    return operations


def _contract_case_metadata(array: zarr.Array) -> dict[str, Any] | None:
    raw = array.attrs.get("contract_case", None)
    if isinstance(raw, dict):
        return dict(raw)
    return None


def _array_expectation(
    path: str, array: zarr.Array, root: zarr.Group
) -> dict[str, Any]:
    decoder = ArrayDecoder()
    decoded = decoder.decode(array, root)
    decoded = np.asarray(decoded)
    viewer_decoded = _viewer_kernel_decode(array, root)
    if viewer_decoded is not None and decoded.size:
        max_abs = np.float32(np.abs(decoded).max())
        tolerance = 4 * np.spacing(max_abs)
        max_error = np.abs(decoded.astype(np.float64) - viewer_decoded).max()
        assert max_error <= tolerance, (
            f"Viewer kernel decode diverged from Python for {path}: "
            f"max error {max_error} exceeds {tolerance}"
        )
    flat = _float32_view(decoded)
    decoded_shape = [int(v) for v in decoded.shape]

    expected_elements = int(decoded.shape[0]) if decoded.ndim > 0 else int(decoded.size)
    item = {
        "path": path,
        "encoding": _encoding_name(array),
        "storage_shape": [int(v) for v in array.shape],
        "storage_dtype": str(array.dtype),
        "decoded_shape": decoded_shape,
        "decoded_dtype": str(decoded.dtype),
        "expected_elements": expected_elements,
        "shape_class": _shape_class(decoded_shape),
        "flat_length": int(flat.size),
        "float32_sha256": _sha256_float32(decoded),
        "samples": _samples(decoded),
        "stats": _stats(decoded),
        "operations": _range_operations(decoded, viewer_decoded),
    }
    if viewer_decoded is not None:
        item["viewer_float32_sha256"] = _sha256_float32(viewer_decoded)
        item["viewer_samples"] = _samples(viewer_decoded)
    contract_case = _contract_case_metadata(array)
    if contract_case is not None:
        item["contract_case"] = contract_case
    return item


def _manifest(fixtures: dict[str, Any]) -> dict[str, Any]:
    """Summarize observed contract coverage and required coverage targets."""
    encodings: Counter[str] = Counter()
    storage_dtypes: Counter[str] = Counter()
    decoded_dtypes: Counter[str] = Counter()
    shape_classes: Counter[str] = Counter()
    semantic_types: Counter[str] = Counter()

    for fixture in fixtures.values():
        for item in fixture["arrays"].values():
            encodings[item["encoding"]] += 1
            storage_dtypes[item["storage_dtype"]] += 1
            decoded_dtypes[item["decoded_dtype"]] += 1
            shape_classes[item["shape_class"]] += 1
            contract_case = item.get("contract_case")
            if isinstance(contract_case, dict):
                semantic = contract_case.get("semantic_type")
                if isinstance(semantic, str):
                    semantic_types[semantic] += 1

    required = {
        "encodings": [
            "none",
            "float32",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "rgb_uint8",
            "rgb_uint16",
            "broadcasted",
            "array_ref",
            "lut_uint8",
            "lut_uint16",
            "bounded_scalar_uint8",
            "bounded_scalar_uint16",
            "log_scalar_uint8",
            "log_scalar_uint16",
            "geolog_scalar_uint8",
            "geolog_scalar_uint16",
            "geolog_perchannel_u8",
            "geolog_perchannel_u16",
        ],
        "storage_dtypes": ["float32", "uint8", "uint16", "uint32", "uint64"],
        "semantic_types": [
            "coordinate",
            "color",
            "bounded_scalar",
            "positive_scalar",
            "cholesky",
            "index",
        ],
        "shape_classes": ["empty", "singleton", "vector_1d", "matrix_2d"],
    }

    return {
        "fixture_count": len(fixtures),
        "array_count": sum(len(item["arrays"]) for item in fixtures.values()),
        "observed": {
            "encodings": dict(sorted(encodings.items())),
            "storage_dtypes": dict(sorted(storage_dtypes.items())),
            "decoded_dtypes": dict(sorted(decoded_dtypes.items())),
            "shape_classes": dict(sorted(shape_classes.items())),
            "semantic_types": dict(sorted(semantic_types.items())),
        },
        "required": required,
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
        "version": 3,
        "description": "Python ArrayDecoder expectations for TypeScript round-trip tests.",
        "manifest": _manifest(fixtures),
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
        aprint(
            f"  Encodings: {sorted(expectations['manifest']['observed']['encodings'])}"
        )


if __name__ == "__main__":
    main()
