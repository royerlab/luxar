"""Deep alignment checks for the current hosted positional-pair generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from luxar.demos import demo_gsplats_3d_ct_totalsegmentator as ct_demo
from luxar.demos import demo_gsplats_3d_visible_human_head as vh_demo
from luxar.demos import voxel_sampled_payload_agreement
from luxar.gsplats.gsplat_data import GSplatData

_REPO = Path(__file__).resolve().parents[6]
_MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest.json"
_CACHE = Path.home() / ".cache" / "luxar"
_STAGING_ROOT = _REPO / "delme"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate(dataset: str, filename: str, digest: str) -> Path | None:
    candidates = [_CACHE / dataset / filename]
    if _STAGING_ROOT.is_dir():
        candidates.extend(_STAGING_ROOT.rglob(filename))
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == digest:
            return candidate
    return None


@pytest.mark.slow
@pytest.mark.parametrize(
    ("dataset", "fit_name", "sidecar_name", "payload_loader", "threshold"),
    [
        (
            "gsplats_ct_totalsegmentator",
            "ct_atlas.gsplats.zarr.zip",
            "ct_atlas_labels.npz",
            ct_demo._load_labels,
            ct_demo.MIN_LABEL_AGREEMENT,
        ),
        (
            "gsplats_visible_human_head",
            "vh_head.gsplats.zarr.zip",
            "vh_head_colors.npz",
            vh_demo._load_colors_f32,
            vh_demo.MIN_COLOR_AGREEMENT,
        ),
    ],
)
def test_locatable_hosted_positional_pair_is_aligned(
    dataset: str,
    fit_name: str,
    sidecar_name: str,
    payload_loader,
    threshold: float,
) -> None:
    manifest = json.loads(_MANIFEST.read_text())
    entries = {entry["name"]: entry for entry in manifest["datasets"][dataset]["files"]}
    fit_path = _locate(dataset, fit_name, entries[fit_name]["hosted_sha256"])
    sidecar_path = _locate(
        dataset, sidecar_name, entries[sidecar_name]["hosted_sha256"]
    )
    if fit_path is None or sidecar_path is None:
        pytest.skip("current hosted pair is not present in cache or staging")

    fit = GSplatData.load(fit_path, include_stats=False)
    payload = payload_loader(sidecar_path)

    assert len(payload) == len(fit.centers), (
        f"{dataset} hosted pair has {len(payload):,} payload rows for "
        f"{len(fit.centers):,} splats"
    )
    agreement = voxel_sampled_payload_agreement(fit.centers, payload)
    assert agreement is not None, f"{dataset} has too few same-voxel splats to verify"
    assert agreement >= threshold, (
        f"{dataset} hosted pair agrees at {agreement:.5f} < {threshold}; "
        "the fit and sidecar are not from the same aligned generation"
    )
