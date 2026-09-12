"""Tests for the neuromast demo's recompute and scene-compilation paths.

Six things here are worth pinning, each with a quiet failure mode:

1. **The source contract.** The fit reads the RAW assembled array (the camera
   pedestal is kept, see the demo docstring's step 2), so the only thing between
   the source and the GPU is the shape check; a wrong file must fail there.
2. **Two sources, not one.** The channels were acquired and assembled
   separately, so the recompute needs a path per channel and the two recorded
   camera pedestals differ. A single ``--source`` would fit one channel twice.
3. **The recipe constants.** ``--jobs-per-gpu`` must not be ``auto``: on the
   acquisition box ``auto`` sized 100 concurrent workers for 100 tasks and every
   one was OOM-killed before a tile landed.
4. **The upstream provenance.** The durable HPC directories, numeric TIFF order,
   axis convention and camera-pedestal measurement are the recipe for rebuilding
   the two assembled arrays if the current copies disappear.
5. **The download estimate.** It must stay derived from both manifest archives,
   or metadata can silently drift when either channel is re-uploaded.
6. **The shipped appearance.** The compiled scene must preserve the tuned
   additive blending, windows, gamma, opacity, order, and exposure.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import open_group
from luxar.demos import demo_gsplats_4d_neuromast_2ch as demo
from luxar.gsplats.calibration.noise_floor import estimate_floor
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import save_gsplats
from luxar.gsplats.tree import GSplatLeaf

SMALL = (4, 3, 5, 6)
_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data_manifest.json"

#: Captured at IMPORT, before the autouse fixture below shrinks the module
#: constant. The constants tests must compare against the real recorded extent;
#: reading ``demo.SOURCE_SHAPE`` inside a test would read the 360-voxel stand-in
#: and assert nothing.
REAL_SOURCE_SHAPE = demo.SOURCE_SHAPE


def test_resolve_channel_paths_prefers_the_record_over_a_local_copy(
    tmp_path, monkeypatch
) -> None:
    """A hand-placed local pair must not shadow the published record.

    The local store is the fallback, so precedence is only observable with BOTH
    sources present: a complete unzipped pair on disk AND a fetch that works.
    The sibling test below has an empty ``DATA_DIR``, which proves the fetch
    path but says nothing about order — and getting the order backwards would
    quietly serve a stale hand-placed copy for ever, with the checksum gate
    never consulted.
    """
    local_dir = tmp_path / "local_store"
    local_dir.mkdir()
    monkeypatch.setattr(demo, "DATA_DIR", local_dir)
    for channel in demo.CHANNELS:
        (local_dir / channel["file"]).touch()

    fetched = [
        tmp_path / "cache" / f"{channel['file']}.zip" for channel in demo.CHANNELS
    ]
    calls = []

    def fetch(name: str):
        calls.append(name)
        return list(fetched)

    monkeypatch.setattr(demo, "ensure_dataset", fetch)

    assert demo.resolve_channel_paths() == fetched
    assert calls == ["gsplats_4d_neuromast_2ch"]


def test_resolve_channel_paths_fetches_the_published_pair(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(demo, "DATA_DIR", tmp_path)
    expected = [tmp_path / f"{channel['file']}.zip" for channel in demo.CHANNELS]
    calls = []

    def fetch(name: str):
        calls.append(name)
        return list(reversed(expected))

    monkeypatch.setattr(demo, "ensure_dataset", fetch)

    assert demo.resolve_channel_paths() == expected
    assert calls == ["gsplats_4d_neuromast_2ch"]


def test_manifest_and_download_size_pin_the_channel_pair() -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text())
    files = manifest["datasets"]["gsplats_4d_neuromast_2ch"]["files"]

    assert demo.DEMO_META["requirements"]["download_mb"] == round(
        sum(file["bytes"] for file in files) / 1024**2
    )


@pytest.fixture(autouse=True)
def _small_shape(monkeypatch):
    """Run source-shape probes against 360 voxels, not an 11 GB array."""
    monkeypatch.setattr(demo, "SOURCE_SHAPE", SMALL)


def _source(tmp_path, name="src", *, values=None, shape=SMALL):
    """A plain zarr array store whose ROOT is the array, as the sources are."""
    from luxar._zarr_compat import create_array

    group = open_group(tmp_path / f"{name}_store", mode="w")
    array = create_array(group, "root", shape=shape, dtype="float32", compressor=None)
    array[:] = (
        values
        if values is not None
        else np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    )
    # Re-open as an array-rooted store by pointing at the array itself, which is
    # what `zarr.open` on the real sources returns.
    return tmp_path / f"{name}_store" / "root"


def _leaf_with_rungs(rungs, *, splats_per_rung=1):
    """A tiny real leaf whose splat and rung counts are independently visible."""
    sublod = AdditiveSubLOD(
        centers=np.zeros((splats_per_rung, 4), dtype=np.float32),
        amplitudes=np.ones(splats_per_rung, dtype=np.float32),
        cholesky_factors=np.ones((splats_per_rung, 10), dtype=np.float32),
    )
    return GSplatLeaf(additive_sublods=[sublod] * rungs, meta={})


def _tiny_gsplat_store(path: Path) -> Path:
    save_gsplats(
        path,
        centers=np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32),
        amplitudes=np.ones(2, dtype=np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1, 0, 0, 0, 1], (2, 1)).astype(
            np.float32
        ),
        ordering="none",
    )
    return path


class TestTheSourceIsValidatedBeforeTheGpu:
    def test_the_full_recording_is_accepted(self, tmp_path):
        src = _source(tmp_path)
        assert demo._validate_source(src) == SMALL

    def test_a_mismatched_extent_is_refused_before_the_gpu(self, tmp_path):
        src = _source(tmp_path, shape=(SMALL[0], SMALL[1] + 1) + SMALL[2:])
        with pytest.raises(SystemExit, match="has shape"):
            demo._validate_source(src)

    def test_a_store_with_no_array_at_its_root_is_refused(self, tmp_path):
        open_group(tmp_path / "empty.zarr", mode="w")
        with pytest.raises(SystemExit, match="no array at its root"):
            demo._validate_source(tmp_path / "empty.zarr")

    def test_a_zip_store_is_closed_after_the_shape_probe(self, tmp_path):
        import zarr
        from zarr.storage import ZipStore

        src = tmp_path / "source.zip"
        store = ZipStore(str(src), mode="w")
        array = zarr.create_array(store=store, shape=SMALL, dtype="float32")
        array[:] = 20.0
        store.close()

        with demo._open_source(src) as opened:
            opened_store = opened.store
            assert opened_store._is_open
        assert not opened_store._is_open


class TestBothChannelsNeedTheirOwnSource:
    def test_a_missing_source_flag_names_every_channel_it_needs(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setitem(demo.SOURCE_ARGS, "membranes", None)
        monkeypatch.setitem(demo.SOURCE_ARGS, "nuclei", None)
        with pytest.raises(SystemExit) as exc:
            demo.recompute_channel_paths(tmp_path / "work")
        message = str(exc.value)
        assert "--source-membranes" in message
        assert "--source-nuclei" in message

    def test_one_source_alone_is_not_enough(self, monkeypatch, tmp_path):
        monkeypatch.setitem(demo.SOURCE_ARGS, "membranes", tmp_path / "m.zarr")
        monkeypatch.setitem(demo.SOURCE_ARGS, "nuclei", None)
        with pytest.raises(SystemExit) as exc:
            demo.recompute_channel_paths(tmp_path / "work")
        assert "--source-nuclei" in str(exc.value)
        assert "--source-membranes" not in str(exc.value)

    def test_the_two_channels_carry_different_pedestals(self):
        """They were assembled separately from different acquisitions, so a
        shared pedestal value would be wrong for at least one of them."""
        pedestals = {ch["name"]: ch["camera_pedestal"] for ch in demo.CHANNELS}
        assert len(set(pedestals.values())) == 2

    def test_the_expected_counts_are_the_unculled_seed_budget(self):
        """With no cull anywhere (fit-time retention 0, no post-fit step) every
        frame keeps exactly its seeds, so both channels record SEEDS x frames.
        The 2026-08 culled build had 5,864,440 / 5,530,300 and the two differed."""
        for ch in demo.CHANNELS:
            assert ch["expected_splats"] == demo.SEEDS * REAL_SOURCE_SHAPE[0]

    def test_every_channel_carries_a_layer_order(self):
        """``create_luxar_scene`` reads ``ch["layer_order"]`` per channel.

        Nothing else in this suite builds the scene — these tests exercise the
        recompute recipe — so that subscript is executed by no test and a
        misspelled or missing key would be a latent ``KeyError``, surfacing only
        when someone runs the demo against its multi-gigabyte sources. Checking
        the config directly costs nothing and closes that gap.
        """
        orders = {ch["name"]: ch["layer_order"] for ch in demo.CHANNELS}
        assert len(orders) == 2
        for name, order in orders.items():
            assert isinstance(order, int) and not isinstance(order, bool), name

    def test_the_channels_do_not_share_a_layer_order(self):
        """Distinct orders, or the two layers land in one band and the viewer
        goes back to inferring their order from bounding-sphere radii — which
        for this fit is backwards (the nuclei sphere is marginally the larger,
        so containment draws nuclei first even though the membrane shell
        encloses them). Sharing a band would silently reinstate that."""
        orders = [ch["layer_order"] for ch in demo.CHANNELS]
        assert len(set(orders)) == len(orders)

        # The enclosing structure must composite FIRST (lower order).
        by_name = {ch["name"]: ch["layer_order"] for ch in demo.CHANNELS}
        assert by_name["membranes"] < by_name["nuclei"]

    def test_a_nonexistent_source_path_is_reported_with_its_flag(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setitem(demo.SOURCE_ARGS, "membranes", tmp_path / "nope.zarr")
        monkeypatch.setitem(demo.SOURCE_ARGS, "nuclei", tmp_path / "also-nope.zarr")
        with pytest.raises(FileNotFoundError, match="source-membranes"):
            demo.recompute_channel_paths(tmp_path / "work")


def test_compiled_scene_preserves_the_tuned_appearance(tmp_path) -> None:
    channel_paths = [
        _tiny_gsplat_store(tmp_path / f"{channel['name']}.gsplats.zarr")
        for channel in demo.CHANNELS
    ]

    output = demo.create_luxar_scene(channel_paths, tmp_path / "scene.luxar.zarr")
    root = open_group(output, mode="r")

    # Retuned on 2026-09-10 against the archives currently pinned in the manifest.
    expected = {
        "membranes": {
            "window": (0.0, 1.719),
            "gamma": 0.71,
            "opacity": 0.5,
            "layer_order": 10,
        },
        "nuclei": {
            "window": (0.0, 0.459),
            "gamma": 0.69,
            "opacity": 0.47,
            "layer_order": 20,
        },
    }
    for name, appearance in expected.items():
        attrs = dict(root[name].attrs)
        lo, hi = appearance["window"]
        assert attrs["blending_mode"] == "additive"
        assert attrs.get("absorption", 1.0) == pytest.approx(1.0)
        assert attrs["gamma"] == pytest.approx(appearance["gamma"])
        assert attrs["opacity"] == pytest.approx(appearance["opacity"])
        assert attrs["layer_order"] == appearance["layer_order"]
        assert attrs["intensity"] == pytest.approx(1.0 / (hi - lo))
        assert attrs["offset"] == pytest.approx(-lo / (hi - lo))

    assert dict(root.attrs)["viewer_config"]["exposure"] == pytest.approx(-3.4)


class TestTheRecipeConstantsMatchTheRecordedRun:
    def test_the_per_timepoint_sources_are_recorded_outside_scratch(self):
        source_dirs = {ch["name"]: ch["hpc_source_dir"] for ch in demo.CHANNELS}
        root = (
            "/hpc/projects/jacobo_group/Adrian/RU_Processed_Data/"
            "No_Ablations_Aligned/"
            "04192022_she_gfp_cldn_mscarlet_Timelapse3_3dpf/S1"
        )
        assert source_dirs == {
            "membranes": f"{root}/Membranes/Deconvolved",
            "nuclei": f"{root}/Nuclei/Deconvolved",
        }

    def test_the_source_files_are_stacked_in_numeric_timepoint_order(self):
        assert demo.SOURCE_TIMEPOINT_LABELS == tuple(range(1, REAL_SOURCE_SHAPE[0] + 1))
        assert demo.SOURCE_FILE_PATTERN.format(timepoint=1).endswith("_t1.tiff")
        assert demo.SOURCE_FILE_PATTERN.format(timepoint=100).endswith("_t100.tiff")

    def test_the_assembly_axis_contract_is_recorded(self):
        assert demo.SOURCE_FRAME_AXES == "z,y,x"
        assert demo.SOURCE_AXES == "time,z,y,x"
        assert demo.SOURCE_AXES == "time," + demo.SOURCE_FRAME_AXES

    def test_the_camera_pedestal_recipe_is_recorded(self):
        assert demo.CAMERA_PEDESTAL_SAMPLE_INDICES == (
            0,
            11,
            22,
            33,
            44,
            55,
            66,
            77,
            88,
            99,
        )
        assert demo.CAMERA_PEDESTAL_SAMPLE_INDICES == tuple(
            int(round(value)) for value in np.linspace(0, REAL_SOURCE_SHAPE[0] - 1, 10)
        )
        assert demo.CAMERA_PEDESTAL_HISTOGRAM_PERCENTILE == 95.0
        assert demo.CAMERA_PEDESTAL_HISTOGRAM_BINS == 512
        assert demo.CAMERA_PEDESTAL_REDUCTION == "median"
        floors = {ch["name"]: ch["camera_pedestal"] for ch in demo.CHANNELS}
        assert floors == {
            "membranes": 105.9911880493164,
            "nuclei": 103.88801574707031,
        }

    def test_the_recorded_per_frame_pedestal_recipe_matches_estimate_floor(self):
        rng = np.random.default_rng(0)
        frame = rng.normal(104.0, 2.0, (32, 32)).astype(np.float32)
        frame[12:16, 12:16] += 400.0
        assert np.all(frame > 0.0)
        upper = np.percentile(frame, demo.CAMERA_PEDESTAL_HISTOGRAM_PERCENTILE)
        background = frame[frame <= upper]
        histogram, edges = np.histogram(
            background.astype(np.float64),
            bins=demo.CAMERA_PEDESTAL_HISTOGRAM_BINS,
        )
        peak = int(np.argmax(histogram))
        recorded_floor = float(0.5 * (edges[peak] + edges[peak + 1]))

        assert recorded_floor < float(np.median(frame))
        assert estimate_floor(frame, method="mode") == recorded_floor

    def test_jobs_per_gpu_is_pinned_not_auto(self):
        """`auto` OOM-killed all 100 workers on the acquisition box."""
        assert demo.JOBS_PER_GPU == 12
        assert not isinstance(demo.JOBS_PER_GPU, str)

    def test_the_tile_size_keeps_the_real_volume_whole(self):
        """One tile per timepoint means no Hann halo and no seams to apodize.

        Written against the literal recorded extent, not ``demo.SOURCE_SHAPE``:
        this module's autouse fixture shrinks that constant, so reading it here
        would compare 640 against 6 and pass no matter what.
        """
        assert demo.TILE_SIZE >= max((84, 580, 576))
        assert REAL_SOURCE_SHAPE == (100, 84, 580, 576)

    def test_the_recorded_source_shape_is_the_full_recording(self):
        """The fixture must be shrinking the real constant, not a stale one."""
        assert REAL_SOURCE_SHAPE[0] == 100, "100 timepoints per channel"

    def test_the_stacked_axis_is_last_in_the_fitted_output(self):
        """The source is time-FIRST; the fit emits spatial-first, stacked-last.
        The Z scale therefore applies to index 0, not index 1.

        Placement only — the factor's VALUE is pinned by the sibling test below.
        """
        assert demo.SOURCE_AXES.startswith("time")
        assert demo.VOXEL_SCALE[0] != 1.0, "index 0 is the scaled (Z) axis"
        assert demo.VOXEL_SCALE[1:] == (1.0, 1.0, 1.0)

    def test_the_z_scale_is_the_measured_voxel_anisotropy(self):
        """The MetaMorph headers record a 0.25 um z-step over a 0.1083 um lateral
        pitch, so the factor is 2.3084x — not the historical 2.5, which stretched
        Z by 8.3 % in the published scene.

        Pinned against those two measured numbers rather than against the
        expression that defines the constant: comparing it to its own source
        passes for whatever instrument someone typed in, so it would not catch
        the defect coming back with a different pitch.
        """
        z_step_um = 0.25
        lateral_pitch_um = 0.1083
        assert demo.VOXEL_SCALE[0] == pytest.approx(z_step_um / lateral_pitch_um)
        assert demo.VOXEL_SCALE[0] == pytest.approx(2.3084, abs=1e-4)

    def test_the_recipe_has_no_cull_at_all(self, monkeypatch, tmp_path):
        """The 2026-08 build redundancy-culled every tile at 0.20 and lost 17-26 dB
        of foreground; the recipe now keeps every splat the fit produces. Pin both
        halves: no `gsplat cull` invocation anywhere, and the batch fit itself
        told to keep everything (`--cull-retention 0.0`) while building the
        streaming ladder in its own merge."""
        calls: list[tuple[str, ...]] = []

        def fake_cli(*args):
            calls.append(tuple(args))
            if args[:3] == ("gsplat", "batch-fit", "run"):
                (Path(args[4]) / "merged" / "final.gsplats.zarr").mkdir(parents=True)

        monkeypatch.setattr(demo, "run_luxar_cli", fake_cli)
        monkeypatch.setattr(demo, "_validate_source", lambda src: None)
        monkeypatch.setattr(demo, "load_gsplat_node", lambda *a, **k: (None, None))
        monkeypatch.setattr(demo, "iter_leaves", lambda node: [])
        monkeypatch.setattr(
            demo, "_validate_rebuilt_channel", lambda ch, leaves: ch["expected_splats"]
        )

        demo.recompute_channel(demo.CHANNELS[0], tmp_path / "src.zarr", tmp_path)

        assert [c[:2] for c in calls] == [
            ("gsplat", "batch-fit"),
            ("gsplat", "transform"),
        ], "a cull step crept back into the recipe"
        fit = calls[0]
        assert fit[fit.index("--cull-retention") + 1] == "0.0"
        assert fit[fit.index("--merge-recipe") + 1] == "stream"
        assert fit[fit.index("--merge-n-lods") + 1] == str(demo.EXPECTED_RUNGS)
        # The raw frames have minimum 0, so the legacy hard-min floor is a no-op.
        assert fit[fit.index("--floor") + 1] == "none"
        # The raw root-level array is the fit input; nothing is pre-subtracted.
        assert fit[3] == str(tmp_path / "src.zarr")
        assert "--array-key" not in fit
        assert not hasattr(demo, "REDUNDANCY_THRESHOLD")

        # The transform is where the anisotropy correction reaches the archive.
        # Pinned as the EMITTED string: reversing the tuple or dropping --scale
        # entirely leaves every other assertion in this suite green. Same for
        # --normalize-intensity: change its value or drop the pair and the
        # rebuilt amplitudes no longer match the display windows that
        # ``test_compiled_scene_preserves_the_tuned_appearance`` pins, with
        # nothing else in this suite going red. Membership is asserted first so
        # a dropped flag names itself instead of raising a bare ValueError.
        transform = calls[1]
        assert "--scale" in transform
        assert (
            transform[transform.index("--scale") + 1] == "2.308402585410896,1.0,1.0,1.0"
        )
        assert "--normalize-intensity" in transform
        assert transform[transform.index("--normalize-intensity") + 1] == "1.0"

    def test_the_seed_budget_is_the_calibrated_k_star(self):
        assert demo.SEEDS == 64_000
        assert demo.PRESET == "n2s"

    def test_the_merged_leaf_has_the_recorded_progressive_rungs(self):
        channel = {"name": "membranes"}
        splats_per_rung = 2
        got = demo._validate_rebuilt_channel(
            channel,
            [_leaf_with_rungs(demo.EXPECTED_RUNGS, splats_per_rung=splats_per_rung)],
        )
        assert got == demo.EXPECTED_RUNGS * splats_per_rung

    def test_a_changed_progressive_recipe_is_rejected(self):
        channel = {"name": "membranes"}
        with pytest.raises(RuntimeError, match="progressive rungs"):
            demo._validate_rebuilt_channel(
                channel, [_leaf_with_rungs(demo.EXPECTED_RUNGS - 1)]
            )
