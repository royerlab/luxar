"""Deep alignment checks for current record positional-pair generations.

Run this slow test from the staging checkout after editing the record pins to
the new digests, with the new bytes under ``delme/``, before committing or
uploading the re-pin.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from luxar.demos import demo_gsplats_3d_ct_totalsegmentator as ct_demo
from luxar.demos import demo_gsplats_3d_visible_human_head as vh_demo
from luxar.demos import voxel_sampled_payload_agreement
from luxar.gsplats.gsplat_data import GSplatData

_REPO = Path(__file__).resolve().parents[6]
_MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest.json"
_CACHE = Path.home() / ".cache" / "luxar"
_STAGING_ROOT = _REPO / "delme"

_PairCheck = tuple[
    str, str, Callable[[Path], np.ndarray], float, Callable[[GSplatData], bool]
]
_PAIR_CHECKS: dict[str, _PairCheck] = {
    ct_demo.DEMO_NAME: (
        ct_demo.FIT_FILE,
        ct_demo.LABELS_FILE,
        ct_demo._load_labels,
        ct_demo.MIN_LABEL_AGREEMENT,
        lambda fit: ct_demo._native_labels(fit) is not None,
    ),
    vh_demo.DEMO_NAME: (
        vh_demo.FIT_FILE,
        vh_demo.COLORS_FILE,
        vh_demo._load_colors_f32,
        vh_demo.MIN_COLOR_AGREEMENT,
        lambda fit: vh_demo._native_colors(fit) is not None,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidates(dataset: str, filename: str) -> Iterator[Path]:
    yield _CACHE / dataset / filename
    if _STAGING_ROOT.is_dir():
        yield from _STAGING_ROOT.rglob(filename)


def _locate(dataset: str, entry: dict[str, Any]) -> tuple[Path | None, list[str]]:
    digest = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    rejections: list[str] = []
    if not digest:
        return None, rejections
    for candidate in _candidates(dataset, entry["name"]):
        try:
            if not candidate.is_file():
                continue
            actual_bytes = candidate.stat().st_size
            if expected_bytes is not None and actual_bytes != expected_bytes:
                rejections.append(
                    f"{candidate}: size {actual_bytes} != bytes {expected_bytes}"
                )
                continue
            actual_digest = _sha256(candidate)
            if actual_digest == digest:
                return candidate, rejections
            rejections.append(
                f"{candidate}: sha256 {actual_digest} != manifest sha256 {digest}"
            )
        except OSError as exc:
            rejections.append(f"{candidate}: {exc}")
    return None, rejections


def _pair_cases() -> list[Any]:
    manifest = json.loads(_MANIFEST.read_text())
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset, spec in manifest["datasets"].items():
        for entry in spec.get("files", ()):
            pair = entry.get("positional_pair")
            if pair:
                groups.setdefault((dataset, pair), []).append(entry)

    assert {dataset for dataset, _pair in groups} == set(_PAIR_CHECKS), (
        "every declared positional-pair dataset needs a deep alignment check"
    )
    cases: list[Any] = []
    for (dataset, pair), entries in sorted(groups.items()):
        fit_name, sidecar_name, payload_loader, threshold, native_check = _PAIR_CHECKS[
            dataset
        ]
        by_name = {entry["name"]: entry for entry in entries}
        assert set(by_name) == {fit_name, sidecar_name}, (
            f"{dataset}/{pair} no longer matches its declared demo payload files"
        )
        cases.append(
            pytest.param(
                dataset,
                by_name[fit_name],
                by_name[sidecar_name],
                payload_loader,
                threshold,
                native_check,
                id=f"{dataset}/{pair}",
            )
        )
    return cases


def test_cache_hit_does_not_walk_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = "toy"
    filename = "fit.zip"
    payload = b"current record bytes"
    cache = tmp_path / "cache"
    staging = tmp_path / "staging"
    cache_file = cache / dataset / filename
    cache_file.parent.mkdir(parents=True)
    staging.mkdir()
    cache_file.write_bytes(payload)
    entry = {
        "name": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }

    def fail_rglob(_path: Path, _pattern: str) -> Iterator[Path]:
        raise AssertionError("staging was walked despite a valid cache hit")

    monkeypatch.setattr(Path, "rglob", fail_rglob)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_CACHE", cache)
    monkeypatch.setattr(module, "_STAGING_ROOT", staging)

    assert _locate(dataset, entry) == (cache_file, [])


def test_size_mismatch_is_not_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = "toy"
    filename = "fit.zip"
    payload = b"current record bytes"
    cache = tmp_path / "cache"
    staging = tmp_path / "staging"
    cache_file = cache / dataset / filename
    staged_file = staging / filename
    cache_file.parent.mkdir(parents=True)
    staging.mkdir()
    cache_file.write_bytes(b"wrong size")
    staged_file.write_bytes(payload)
    entry = {
        "name": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    hashed: list[Path] = []

    def recording_sha256(path: Path) -> str:
        hashed.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_CACHE", cache)
    monkeypatch.setattr(module, "_STAGING_ROOT", staging)
    monkeypatch.setattr(module, "_sha256", recording_sha256)

    located, rejections = _locate(dataset, entry)

    assert located == staged_file
    assert rejections == [
        f"{cache_file}: size {len(b'wrong size')} != bytes {len(payload)}"
    ]
    assert hashed == [staged_file]


def test_lookup_reports_a_stale_record_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = "toy"
    filename = "fit.zip"
    payload = b"new record bytes"
    staging = tmp_path / "staging"
    staged_file = staging / filename
    staging.mkdir()
    staged_file.write_bytes(payload)
    entry = {
        "name": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload) - 1,
    }

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_CACHE", tmp_path / "cache")
    monkeypatch.setattr(module, "_STAGING_ROOT", staging)

    located, rejections = _locate(dataset, entry)

    assert located is None
    assert rejections == [
        f"{staged_file}: size {len(payload)} != bytes {len(payload) - 1}"
    ]

    sidecar_entry = {
        "name": "payload.npz",
        "sha256": "0" * 64,
        "bytes": 1,
    }
    with pytest.raises(pytest.skip.Exception, match="bytes"):
        test_locatable_record_positional_pair_is_aligned(
            dataset,
            entry,
            sidecar_entry,
            lambda _path: np.empty(0),
            1.0,
            lambda _fit: True,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "dataset",
        "fit_entry",
        "sidecar_entry",
        "payload_loader",
        "threshold",
        "native_check",
    ),
    _pair_cases(),
)
def test_locatable_record_positional_pair_is_aligned(
    dataset: str,
    fit_entry: dict[str, Any],
    sidecar_entry: dict[str, Any],
    payload_loader: Callable[[Path], np.ndarray],
    threshold: float,
    native_check: Callable[[GSplatData], bool],
) -> None:
    if not fit_entry.get("sha256") or not sidecar_entry.get("sha256"):
        pytest.skip("positional pair has no current record pin")
    fit_path, fit_rejections = _locate(dataset, fit_entry)
    sidecar_path, sidecar_rejections = _locate(dataset, sidecar_entry)
    if fit_path is None or sidecar_path is None:
        details: list[str] = []
        for entry, path, rejections in (
            (fit_entry, fit_path, fit_rejections),
            (sidecar_entry, sidecar_path, sidecar_rejections),
        ):
            if path is None:
                details.extend(
                    rejections or [f"{entry['name']}: no matching filename found"]
                )
        pytest.skip(
            "current record pair is not present in cache or staging: "
            + "; ".join(details)
        )

    fit = GSplatData.load(fit_path, include_stats=False)
    payload = payload_loader(sidecar_path)

    assert native_check(fit), f"{dataset} record fit lacks its native payload channel"
    assert len(payload) == len(fit.centers), (
        f"{dataset} record pair has {len(payload):,} payload rows for "
        f"{len(fit.centers):,} splats"
    )
    agreement = voxel_sampled_payload_agreement(fit.centers, payload)
    assert agreement is not None, f"{dataset} has too few same-voxel splats to verify"
    assert agreement >= threshold, (
        f"{dataset} record pair agrees at {agreement:.5f} < {threshold}; "
        "the fit and sidecar are not from the same aligned generation"
    )
