# Handoff: first-class background/floor suppression for `gsplat fit` (+ `cal`, `batch-fit`)

**Status:** proposed, not started. Self-contained — you should be able to implement from this doc alone.
**Owner hand-off date:** 2026-07-07.
**Priority (per user):** high — "DC is difficult to encode with splats… should be on by default."

---

## 1. Problem & motivation

A Gaussian splat represents a *localized* blob. A constant **background pedestal / DC offset** (common in real microscopy: camera offset, autofluorescence, scattered light) is the worst case for a splat basis — you cannot represent a flat, volume-filling constant efficiently with localized Gaussians, so the optimizer either:

- wastes splat capacity tiling the background with low-amplitude blobs (payload bloat + a visible background "haze"/labyrinth texture), and/or
- under-fits the real signal because the loss is dominated by the pedestal.

### Evidence gathered 2026-07-07 (neuromast iSIM dataset)

Dataset: `she_gfp_cldnb_mscarlet_3dpf` (obsidian `/mnt/raid0/Downloads/...`), timepoint 0, shape `(Z=84, Y=580, X=576)`.

- Raw intensities sit on a **~110-count pedestal** (min 100.5, mode ≈ 110, median 115, max 953; only ~1% of voxels > 333).
- A naïve fit (probe, no background handling) rendered washed-out at **PSNR 18.5 dB**.
- Manually subtracting the histogram-mode floor (~110, clip at 0) before calibrating jumped held-out PSNR to **~43 dB** (region cal) — a ~25 dB swing attributable almost entirely to removing the pedestal.
- Uniform tiling of the pedestal produced **visible tile seams** (each background tile independently fit ~31K noise splats; texture mismatch at boundaries).
- A whole-volume flat fit removed the seams but left a **continuous background "fingerprint" haze** — the residual near-floor noise still gets modeled.

Conclusion: subtracting the floor before fitting is not a niche tweak; it is the single highest-leverage preprocessing step for real microscopy. It should be **on by default** with an `auto` estimator, and overridable.

---

## 2. Current behavior (what happens today)

The volume is normalized to `[0,1]` in exactly one place:

**`packages/luxar/src/luxar/gsplats/fitting/preprocessing.py::_normalize_data` (lines 862–893).**

```python
if norm_percentile == 0.0:                 # default
    image_min = float(np.min(V))           # <-- hard minimum
    image_max = float(np.max(V))
else:
    image_min = float(np.percentile(V, norm_percentile))
    image_max = float(np.percentile(V, 100.0 - norm_percentile))
intensity_range = image_max - image_min
V = np.clip((V - image_min) / intensity_range, 0.0, 1.0)   # line 891
```

Two important facts:

1. **`image_min` today is the hard `min()`** (or a low percentile if `norm_percentile>0`). For data with a pedestal *above* the darkest voxel (i.e. essentially always), the background bulk survives normalization as a small-but-nonzero floor across most of the volume — which is exactly what splats then waste capacity on.
2. **The subtracted `image_min` is never added back to the output.** In `fitting/results.py::finalize_results` (lines 225–230) amplitudes are rescaled by `intensity_range` only:
   ```python
   amps_np = amps_np * preprocessed_data.intensity_range   # line 226; NO "+ image_min"
   ```
   So a floor subtraction is *already implicit and non-configurable*. A `--floor` feature simply makes the subtracted baseline **explicit, larger, and controllable**, and (critically) does not require touching the amplitude reconstruction — output amplitudes are already "baseline-relative."

`norm_percentile` itself is **not currently exposed as a CLI flag** — it only reaches the fitter through `get_fit_defaults()` reading the `fit_gaussian_splats` signature. `--floor` would be the first normalization-related CLI flag.

---

## 3. Goal / desired behavior

Add a `--floor` option to `gsplat fit` (and `cal`, `batch-fit`), **on by default**:

| Value | Meaning |
|---|---|
| `auto` (**default**) | Estimate the background floor and subtract it (clip at 0) before normalization. |
| `<float>` | Subtract this fixed intensity value. |
| `pN` (e.g. `p10`) | Subtract the Nth intensity percentile. |
| `none` / `0` | Disable — reproduce today's behavior (hard-min normalization). |

Semantics: floor subtraction raises the effective `image_min` used in `_normalize_data`. Because normalization already does `clip((V - image_min)/range, 0, 1)`, setting `image_min = floor` **is** "subtract floor, clip sub-floor to 0, normalize" — no separate subtract/clip pass needed. Output amplitudes come out background-relative automatically (see §2 fact 2), which is what the viewer wants (background → 0, no blow-out).

---

## 4. Recommended design (minimal, low-risk)

**Core idea: `--floor` overrides how `image_min` is chosen in `_normalize_data`. Everything downstream is unchanged.**

1. Thread a `floor` parameter (type `str | float | None`, default `"auto"`) through the same chain `norm_percentile` uses (see §5 checklist).
2. In `_normalize_data`, resolve `floor` to a concrete `image_min`:
   - `auto` → `image_min = estimate_floor(V)` (see §6).
   - `pN` → `image_min = np.percentile(V, N)`.
   - `<float>` → `image_min = float(value)`.
   - `none`/`0` → keep current logic (`np.min(V)` or `norm_percentile`).
   - Always clamp `image_min` to `[V.min(), <below image_max>]` and keep the near-uniform guard (lines 885–889).
3. `image_max` stays `V.max()` (or the high percentile). `intensity_range = image_max - image_min`. The existing `np.clip(..., 0, 1)` at line 891 does the rest.
4. **Do not** change `results.py:226` — amplitudes stay `* intensity_range`, which is already baseline-relative. (Add a decision note if you conclude reconstruction fidelity needs the floor recorded; see §7.)
5. **Record the resolved floor** in fit stats/metadata (so it is inspectable via `gsplat info` and reproducible). Suggest adding `floor` alongside the existing normalization metadata in `PreprocessedData` (config.py:250–252) and surfacing it in the saved stats.

Why this is safe: on already-clean data (synthetic, pre-normalized, no pedestal) the `auto` estimate lands at ≈ `min(V)`, so behavior is unchanged. On real data it removes the pedestal. It reuses the one existing clip; no new normalization path.

---

## 5. Exact change points (file:line)

Thread `floor` end-to-end. All line numbers are as of 2026-07-07 — re-verify.

1. **`packages/luxar/src/luxar/gsplats/fit_gsplats.py`**
   - `GaussianSplatFitter.fit`: declare `floor` (~line 101, by `norm_percentile`); pass to `prepare_fit_config` (~line 176).
   - `fit_gaussian_splats` (module-level API): declare `floor` (~line 251); forward into the fitter (~line 560).
   - Adding it to this signature auto-populates `get_fit_defaults()` (see item 6) — no separate default wiring needed there.
2. **`packages/luxar/src/luxar/gsplats/fitting/validation.py`**
   - `prepare_fit_config`: add `floor` to signature (~line 25); validate it (next to the `norm_percentile` check at 177–179 — accept `auto`/`none`/`pN`/float ≥ 0); set it on `FitConfig` (~line 276).
3. **`packages/luxar/src/luxar/gsplats/fitting/config.py`**
   - Add `floor` field to `FitConfig` (near `norm_percentile`, line 120). Consider adding a resolved-`floor` field to `PreprocessedData` (near 250–252) for metadata.
4. **`packages/luxar/src/luxar/gsplats/fitting/preprocessing.py`** — the real work.
   - `_normalize_data` (862–893): accept resolved floor, choose `image_min` accordingly (§4).
   - Call site `preprocess_data` (243–246): pass `config.floor` through. Pre-init amplitude rescale at 252–255 uses the same `image_min`/`intensity_range` — it will inherit the new floor correctly; double-check.
5. **`packages/luxar/src/luxar/gsplats/fitting/results.py`**
   - Lines 225–230: **no change expected** (amplitudes already `* intensity_range`, baseline-relative). Add a code comment documenting that `image_min` is intentionally not restored. Only revisit if §7 decision says otherwise.
6. **CLI `fit` — `packages/luxar/src/luxar/cli/gsplat_ops/fitting.py`**
   - `fit_volume` typer options (319–701): add `--floor` (default `"auto"`) near `lr`/`loss` (~383–387).
   - `cli_overrides` dict (989–998): add `"floor": floor`.
   - Fit invocation at 1321–1327 (`**fit_config`) and progressive path 1309–1317 pick it up automatically.
7. **CLI `cal` — same file `fitting.py`**
   - `calibrate_command` (from 1486): add `--floor` (default `"auto"`); add to the `cli_overrides` in the `load_fit_config(...)` call at 1728–1734 so every K-sweep fit subtracts the floor. **Important:** calibration should default-on too, so K* is measured on floor-suppressed data (matches how you will fit).
8. **Config — `packages/luxar/src/luxar/cli/gsplat_config.py`**
   - `get_fit_defaults()` (96–107) reads the `fit_gaussian_splats` signature via `inspect` → `floor` appears automatically once item 1 is done.
   - `load_fit_config` (110–158) does not whitelist keys → passes through freely.
   - `dump_default_config()` (182–283): add a commented `floor:` line near `norm_percentile` (line 229) for `--dump-config` discoverability.
   - Optional: per-preset `floor` in `PRESETS` (54–89) if any preset should differ (probably not; the signature default suffices).
9. **`batch-fit` — `packages/luxar/src/luxar/cli/gsplat_ops/batch_planning.py`**
   - Add a `floor` field to the `FitConfig` dataclass (35–47).
   - `_assemble_fit_args` (313–358): add `fit_args["floor"] = <value>` (key must equal the CLI flag name).
10. **`batch-fit` — `packages/luxar/src/luxar/cli/gsplat_ops/batch.py`**
    - Add `--floor` typer option to both subcommands (near 142–165).
    - Pass `floor` into the `FitConfig(...)` constructions at 551–561 (run/local) and ~1254+ (submit/Slurm).
    - `packages/luxar/src/luxar/gsplats/batch/fit_command.py` (`iter_fit_arg_flags`/`fit_args_to_tokens`, 26–43): generic → **no change**; the `floor` key becomes `--floor <v>` automatically. Workers literally re-invoke `luxar gsplat fit … --floor <v>`, so item 6 is the load-bearing change.

---

## 6. The `auto` floor estimator

Reuse the calibration background helpers so estimation is consistent across `fit` and `cal`.

**`packages/luxar/src/luxar/gsplats/calibration.py`** already has:
- `_background_mad(V, percentile=10.0)` (630–643): threshold = `np.percentile(V, 10.0)` (line 637).
- `foreground_mask_otsu(V)` (260–263) + `_otsu_threshold` (~247).
- `estimate_noise_floor` (646–682), `NoiseFloor.sigma_background` (575).

**Recommended estimator** (matches what worked by hand on the neuromast set):

```python
def estimate_floor(V, method="mode"):
    if method == "mode":
        # histogram mode of the low-intensity bulk = the pedestal peak
        hi = np.percentile(V, 95)
        hist, edges = np.histogram(V[V <= hi], bins=512)
        i = int(hist.argmax())
        return 0.5 * (edges[i] + edges[i + 1])
    elif method == "percentile":
        return float(np.percentile(V, 10.0))   # cheaper, reuses _background_mad threshold
```

Put this in a shared location (e.g. `calibration.py` next to `_background_mad`, or a small `preprocessing` helper) and call it from both `_normalize_data` (fit) and the calibration path.

**Guard rails (decide & test):**
- The mode is robust for pedestal data. For an image that is *mostly* signal (rare here), the mode is still the darkest mode — verify it does not eat real signal. Consider capping: `floor = min(mode, percentile(V, 50))` or similar.
- On clean/synthetic data with no pedestal, `mode ≈ min(V)` → effectively a no-op → backward-compatible.
- Consider excluding exact-zero voxels (masked/out-of-FOV) before estimating, so padding doesn't dominate the histogram.

---

## 7. Decisions to make (flag these in the PR)

1. **Default value.** User wants `auto` ON by default for `fit` **and** `cal`. This changes default output for every fit. Repo philosophy is "no backwards-compat burden," so acceptable — but call it out, and make sure the manuscript `n2s`/blind-spot protocol is still reproducible (either `--floor none` reproduces the old numbers, or re-baseline the manuscript figures deliberately).
2. **Estimator: mode vs percentile-10.** Mode matched the manual result; percentile-10 is simpler and already in the codebase. Pick one default, expose the other.
3. **Record floor for reconstruction?** Today `image_min` is dropped from output amplitudes (§2). If exact intensity reconstruction ever matters (e.g. `gsplat compare` against the *raw* volume, or round-tripping), you may want to store the floor and optionally add it back. For *display* (the driving use case) you explicitly do **not** want it back. Recommend: store `floor` in metadata, do not add back, document clearly.
4. **Interaction with `norm_percentile`.** If both `--floor` and a nonzero `norm_percentile` are given, define precedence (suggest: `floor` sets `image_min`, `norm_percentile` still governs `image_max` clipping of bright outliers). Keep them orthogonal.
5. **Auto tiling.** Related but separate: the default `--tiling auto` splits any volume with a dim > 256 (see `fitting.py::_resolve_tiling`), which needlessly tiled this small stack and caused the seams. Not in scope here, but worth a companion issue — floor suppression + not-tiling-small-volumes together fix the whole neuromast experience.

---

## 8. Testing plan

Unit:
- `gsplats/fitting/tests/test_fitting_preprocessing.py` — `test_normalization_percentile` (line 119) exercises `_normalize_data`; add floor cases: pedestal volume → assert background maps to 0, peak preserved, `auto` ≈ manual mode subtraction; `none` reproduces old `image_min == min(V)`.
- `gsplats/fitting/tests/test_fitting_validation.py` — validation of `floor` values (`auto`/`none`/`pN`/float/negative rejected); mirror `norm_percentile` tests.
- `gsplats/fitting/tests/test_fitting_config.py` — `FitConfig.floor` field/default.
- `gsplats/fitting/tests/test_results.py` — confirm amplitude rescale unchanged (Area 5); output_space cases 1013–1103.
- `gsplats/tests/test_calibration.py` — `estimate_floor` (+ existing `_background_mad`, `foreground_mask_otsu`).
- `gsplats/tests/test_fit_command.py` — assert `fit_args` with `floor` expands to `--floor <v>`.

CLI:
- `cli/tests/test_gsplat_cli_extended.py` — fit + cal accept `--floor`; default is `auto`.
- `cli/tests/test_batch_run.py` — batch `FitConfig.floor` → worker `--floor` token.

Integration / real-data validation (the proof):
- Reproduce the neuromast result. Volume + artifacts live on obsidian `/mnt/raid0/_sandbox/`:
  - `neuromast_t0_bgsub.npy` — the manually floor-subtracted tp0 (target behavior).
  - `neuromast_cal.json` / `neuromast_cal_wholevol.json` — region vs whole-vol calibrations.
- Acceptance: `luxar gsplat fit <raw tp0> out.zarr --floor auto` should land within ~0.5 dB of fitting the pre-subtracted `neuromast_t0_bgsub.npy`, and clearly beat `--floor none` (expect a large gap, ~20 dB on this data). Compare held-out PSNR and eyeball the viewer (no background haze/seams).

---

## 9. Reference material

- Session that motivated this (2026-07-07): calibrated + fit the neuromast iSIM tp0; pedestal drove PSNR 18.5 → ~43 dB; tiling seams and flat-fit background haze both trace to un-suppressed DC.
- Memory notes: `project_gsplat_floor_suppression.md` (this feature) and the amplitude-scale/`output_space` display gotcha recorded alongside it.
- Raw data: obsidian `/mnt/raid0/Downloads/she_gfp_cldnb_mscarlet_3dpf/she_gfp_cldnb_mscarlet_3dpf.zarr.zip` (100 tp, TZYX float32).
- Related code the estimator can borrow: `calibration.py` `_background_mad` (637), `estimate_noise_floor` (646), `foreground_mask_otsu` (260).
