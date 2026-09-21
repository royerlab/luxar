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

import pytest

from luxar.demos import DatasetUnavailable

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

    assert module.SYNTHETIC_GSPLATS_FILES == [
        f"synthetic_{name}" for name in module.GSPLATS_FILES
    ]
    assert all(
        name not in module.GSPLATS_FILES for name in module.SYNTHETIC_GSPLATS_FILES
    )


def test_synthetic_volumes_are_reproducible_and_declare_no_acquisition() -> None:
    """Seeded, and ``acquisition is None`` — synthesized data IS its own source."""
    module = _load(DAPI, ["--synthetic"])

    first, acquisition = module.synthesize_nuclei_volume(seed=7)
    second, _ = module.synthesize_nuclei_volume(seed=7)

    assert acquisition is None
    assert first.shape == second.shape
    assert (first == second).all()
