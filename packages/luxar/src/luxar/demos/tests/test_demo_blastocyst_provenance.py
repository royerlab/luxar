"""The two invariants the blastocyst demos exist to guarantee.

Both were introduced as fixes and neither had a test, which is how the second
one regressed into the first one's mirror image during review:

1. A failed acquisition RAISES. It used to catch every exception and return
   procedurally generated blobs, which the scene then published under Blin et
   al.'s DOI and CC BY 4.0 with a description of a Leica SP8 acquisition.

2. ``--synthetic`` never touches real data. The first attempt at the opt-in
   still called ``resolve_gsplats()`` before anything read the flag, so on any
   machine that could reach the pinned record the REAL microscopy came back and
   was published under the synthetic identity — citation stripped, titled "not
   microscopy", with the real artifact's digest stamped in as an input digest.
   Same defect, opposite direction.

The demos are loaded by path, once per flag state, because ``SYNTHETIC`` and
the cache names are module-level constants resolved from ``sys.argv`` at import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import zarr

from luxar.demos import DatasetUnavailable
from luxar.gsplats.gsplat_data import GSplatData

DEMOS = Path(__file__).resolve().parent.parent
DAPI = "demo_gsplats_3d_blastocyst_dapi_nuclei"
MULTI = "demo_gsplats_3d_blastocyst_multichannel"


def _load(stem: str, argv: list[str]) -> ModuleType:
    """Import a demo module under a given argv, isolated from other loads."""
    saved = sys.argv
    sys.argv = ["demo", *argv]
    try:
        spec = importlib.util.spec_from_file_location(
            f"{stem}__{'syn' if '--synthetic' in argv else 'real'}",
            DEMOS / f"{stem}.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved


def _tiny_gsplat_data(seed: int = 0) -> GSplatData:
    """A handful of valid splats, sufficient to compile either scene offline."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (8, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, 8).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (8, 1)).astype(np.float32),
    )


def _overlay_text(root: zarr.Group) -> str:
    """Return every authored text overlay as one string."""
    parts: list[str] = []
    overlays = root.get("overlays")
    if overlays is not None:
        parts.extend(
            str(dict(overlays[name].attrs).get("text", ""))
            for name in overlays.group_keys()
        )
    return "\n".join(parts)


@pytest.mark.parametrize(
    ("stem", "loader"),
    [(DAPI, "load_dapi_data"), (MULTI, "load_multichannel_data")],
)
def test_an_unreachable_source_raises_instead_of_fabricating(stem, loader) -> None:
    module = _load(stem, [])
    module.ZARR_URL = "https://127.0.0.1:1/unreachable.zarr"

    with pytest.raises(DatasetUnavailable, match="idr0062"):
        getattr(module, loader)()


@pytest.mark.parametrize("stem", [DAPI, MULTI])
def test_synthetic_never_consults_the_pinned_manifest(stem) -> None:
    """The critical one: the manifest resolves the REAL artifact.

    Reaching it under --synthetic republished real CC BY 4.0 microscopy with
    its attribution stripped, and meant the flag did not work offline at all —
    which is the only reason it exists.
    """
    module = _load(stem, ["--synthetic"])
    assert module.SYNTHETIC is True

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("resolve_gsplats() reached the manifest under --synthetic")

    module.load_dataset_gsplats = fail_if_called
    for door in ("load_local_fit_gsplats_at", "load_local_fit_gsplats"):
        if hasattr(module, door):
            setattr(module, door, lambda *a, **k: None)

    assert module.resolve_gsplats() is None  # nothing cached -> caller synthesizes


@pytest.mark.parametrize("stem", [DAPI, MULTI])
def test_synthetic_uses_its_own_scene_name(stem) -> None:
    """A synthetic run must not overwrite, or be mistaken for, the real scene."""
    real, synthetic = _load(stem, []), _load(stem, ["--synthetic"])

    assert not real.SCENE_STEM.endswith("_SYNTHETIC")
    assert synthetic.SCENE_STEM.endswith("_SYNTHETIC")
    assert real.SCENE_STEM != synthetic.SCENE_STEM


def test_multichannel_reads_back_the_cache_it_writes() -> None:
    """Writer and reader derived the synthetic names separately.

    The writer stored ``synthetic_*`` and the reader asked for the real names,
    so the synthetic cache was write-only and every run refit from scratch.
    """
    module = _load(MULTI, ["--synthetic"])
    written: list[Path] = []
    read: list[str] = []

    def fake_fit_channel(_volume, _channel_name, cache_file, source_dtype=None):
        written.append(cache_file)
        return object()

    def fake_load_local_fit(_demo_name, names):
        read.extend(names)
        return None

    module.fit_channel = fake_fit_channel
    module.load_local_fit_gsplats = fake_load_local_fit

    module.fit_all_channels([object(), object()])
    assert module.resolve_gsplats() is None

    assert [path.name for path in written] == read


@pytest.mark.parametrize("stem", [DAPI, MULTI])
def test_synthetic_scene_publishes_no_real_provenance(stem, tmp_path) -> None:
    module = _load(stem, ["--synthetic"])
    output = tmp_path / f"{stem}.luxar.zarr"
    splats = _tiny_gsplat_data()
    scene_input = [splats, _tiny_gsplat_data(seed=1)] if stem == MULTI else splats

    scene_path = module.create_luxar_scene(scene_input, output)
    root = zarr.open_group(str(scene_path), mode="r")
    attrs = dict(root.attrs)
    overlays = _overlay_text(root)

    assert attrs.get("citation") is None
    assert "SYNTHETIC" in attrs["title"]
    if stem == DAPI:
        assert "deliberately NOT attached" in attrs["description"]
        published = overlays
    else:
        published = f"{attrs['description']}\n{overlays}"
    for borrowed_claim in (
        "Leica SP8",
        "idr0062",
        "10.1371/journal.pbio.3000388",
        "CC BY",
        "ab16048",
        "immunostain",
    ):
        assert borrowed_claim not in published


def test_synthetic_volumes_are_reproducible_and_declare_no_acquisition() -> None:
    """Seeded, and ``acquisition is None`` — synthesized data IS its own source."""
    module = _load(DAPI, ["--synthetic"])

    first, acquisition = module.synthesize_nuclei_volume(seed=7)
    second, _ = module.synthesize_nuclei_volume(seed=7)

    assert acquisition is None
    assert first.shape == second.shape
    assert (first == second).all()
