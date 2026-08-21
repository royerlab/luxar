"""The CMU-1 channel order is load-bearing, so the guard that checks it is gated.

``create_luxar_scene`` assigns the red/green/blue colormaps by POSITION in the
resolved path list, so a channel arriving out of order paints hematoxylin red —
a wrong picture that renders perfectly happily. ``resolve_data`` refuses that
case, but its only live trigger is a real run against the hosted 114 MB
artifacts, and demos are omitted from coverage: the refusal had no test at all.

The refusal cases cover both shapes the guard exists for, not just the tidy one:

* a **permutation** of the three channel paths — the mis-colouring the guard is
  named for, though the generator's ``sorted(...)`` glob means no committed
  manifest can actually produce it;
* an **extra sidecar** entry, which is the REACHABLE trigger: two sibling
  datasets already ship exactly such a file next to their fit
  (``gsplats_ct_totalsegmentator`` → ``ct_atlas_labels.npz``,
  ``gsplats_visible_human_head`` → ``vh_head_colors.npz``). A sidecar that sorts
  *before* ``cmu1_ch0`` is the dangerous one — it shifts every channel one
  position along the positional colormap list — so it is exercised separately
  from one appended at the end.

The last test is the valuable one. It lifts the check off the runtime path
entirely by asserting the packaged ``data_manifest.json`` still lists exactly the
three channel files the demo expects, in order — so a manifest edit that would
make the demo die (or, worse, mis-colour) is caught in CI instead of on someone's
machine after a download.
"""

from __future__ import annotations

import importlib
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from luxar.demos import registry

_MANIFEST = registry._DEMOS_DIR / "data_manifest.json"

demo = importlib.import_module("luxar.demos.demo_gsplats_2d_cmu1_pathology")

# A sidecar named like this sorts BEFORE ``cmu1_ch0.gsplats.zarr.zip`` under the
# generator's ``sorted(...)`` glob, so it lands at index 0 and shifts every
# channel one position along ``CHANNEL_COLORMAPS``.
_SIDECAR_FIRST = Path("/nowhere/cmu1_atlas_labels.npz")
_SIDECAR_LAST = Path("/nowhere/cmu1_thumbnail.npz")


def _channel_paths() -> list[Path]:
    """The names ``ensure_dataset`` is contracted to hand back, in order."""
    return [
        Path(f"/nowhere/cmu1_ch{i}.gsplats.zarr.zip") for i in range(demo.N_CHANNELS)
    ]


@pytest.mark.parametrize(
    "corrupt",
    [
        # Rotate rather than index: a hardcoded paths[2] would raise IndexError
        # instead of failing readably if N_CHANNELS ever dropped to 2.
        pytest.param(lambda paths: paths[-1:] + paths[:-1], id="rotated_channels"),
        pytest.param(lambda paths: [*paths, _SIDECAR_LAST], id="sidecar_appended"),
        pytest.param(lambda paths: [_SIDECAR_FIRST, *paths], id="sidecar_sorts_first"),
    ],
)
def test_resolve_data_refuses_anything_but_the_exact_channel_list(
    monkeypatch: pytest.MonkeyPatch,
    corrupt: Callable[[list[Path]], list[Path]],
) -> None:
    bad = corrupt(_channel_paths())
    monkeypatch.setattr(demo, "ensure_dataset", lambda _name: bad)

    with pytest.raises(RuntimeError, match="does not match the expected"):
        demo.resolve_data()


def test_resolve_data_accepts_the_manifest_order_unchanged(monkeypatch) -> None:
    """Positive control: without it, the refusal above could pass for any reason."""
    paths = _channel_paths()
    monkeypatch.setattr(demo, "ensure_dataset", lambda _name: paths)

    assert demo.resolve_data() == paths


def test_the_packaged_manifest_lists_the_channels_the_demo_expects() -> None:
    datasets = json.loads(_MANIFEST.read_text())["datasets"]
    files = [f["name"] for f in datasets[demo.DATASET]["files"]]

    assert files == [f"cmu1_ch{i}.gsplats.zarr.zip" for i in range(demo.N_CHANNELS)], (
        f"{demo.DATASET} no longer lists exactly the three channel fits in "
        f"channel order, so the demo's resolve_data() would refuse it: {files}"
    )


def test_local_fit_paths_rejects_a_truncated_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo, "local_fit_path", lambda _dataset, name: tmp_path / name)
    for name in demo.GSPLATS_FILES:
        with zipfile.ZipFile(tmp_path / name, "w"):
            pass
    (tmp_path / demo.GSPLATS_FILES[1]).write_bytes(b"a Ctrl-C mid-save, not a zip")

    assert demo.local_fit_paths() is None


def test_local_fit_paths_accepts_a_complete_zip_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo, "local_fit_path", lambda _dataset, name: tmp_path / name)
    expected = [tmp_path / name for name in demo.GSPLATS_FILES]
    for path in expected:
        with zipfile.ZipFile(path, "w"):
            pass

    assert demo.local_fit_paths() == expected
