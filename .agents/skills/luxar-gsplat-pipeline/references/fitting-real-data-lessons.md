# Fitting someone else's data: checks that changed the result

Loaded from the `luxar-gsplat-pipeline` skill ("Symptom → knob"). Each section is
a check that moved a real fit, with the measurement behind it. The worked
examples are two bundled demos: `demo_gsplats_4d_organoid_budding.py` (a
deconvolved two-channel light-sheet time-lapse exported as 8-bit RGB) and
`demo_gsplats_3d_easifish_hypothalamus.py` (four-channel expanded-tissue FISH,
DAPI plus mRNA puncta).

## Check whether an export mixes channels, and how

An author's RGB export is a display product, and its channels may be blended.
Find out whether they are, and HOW (max, mean, sum), from the data and from the
authors' code before choosing an unmixing; the right inverse depends on the
blend. The LSTree organoid export is a MAX composite, `(max(m, n), n, n)`:
green and blue are the nuclei `n`, red is the brighter of membranes `m` and
nuclei. Fitted as membranes, red draws every nucleus a second time.

The evidence, which together rules out both "independent" and "additive":

- red is never below green (100% of voxels on T0301 and T0601), so red carries
  the nuclei;
- where red exceeds a non-zero green, red stays flat as green rises while
  red − green falls with a slope of about −1: a max, not a sum, and saturation
  does not explain it (red = 255 in at most 1.1% of voxels below green 192);
- the authors' code agrees: LSTree's `[VolumeGridTask]` sets `blending_mode=max`
  and `mesh_tasks.py` calls `make_composite(..., mode='max')`.

(Red equals green in 99% / 94% of nuclear voxels, but under a max that only says
the nuclei outshine the membranes there; it does not distinguish the blends.)

The inverse of a max is `np.where(R > G, R, 0)`: exact wherever the membrane is
brighter, 0 where a nucleus hides it (unrecoverable). `clip(R − G, 0)` is the
inverse of an additive blend; on this export it under-reads the membrane by the
nuclear value at every mixed voxel, keeping 0.93 / 0.92 / 0.68 of the membrane
mass on T0301 / T0320 / T0601. The exact inverse leaves hard holes in the
target, which changes how its fits must be scored (see "A target with hard
edges" below). Appearance settings cannot repair a mixed channel after the fit:
the duplicate is in the splats.

The same `mesh_tasks.py` also rescales each frame and channel to its own
0.2 / 99.99th percentiles before export, so the 8-bit values are not comparable
between frames. Read an export's provenance for both: how channels were blended,
and how intensity was scaled.

## Budget by occupancy, and cap each channel on its largest frame

On deconvolved data there is no noise peak for `cal` to find, so the organoid
budgets seeds per occupied voxel (8-bit value > 16), capped per channel, and
checks foreground PSNR against the frame (1,000 iterations, RTX 3070). The cap
bounds the REQUESTED seeds; the stored count is lower after the prune and cull.
The membrane rows were scored against the earlier `clip(R − G, 0)` target:

| channel, frame | budget | foreground PSNR |
|---|---|---|
| nuclei, 332 | 4k / 8k / 16k seeds | 39.8 / 44.0 / 47.0 dB |
| membranes, 320 | 1 seed per 24 / 12 / 6 voxels | 38.4 / 40.5 / 42.9 dB |
| membranes, 600 | 60k / 82k / 120k / 165k seeds | 38.6 / 40.4 / 44.3 / 44.7 dB |

One seed per 12 voxels held both channels at about 40-46 dB. The cap is what
differs: the membrane sheets occupy far more voxels than the nuclei on the late
frames, so a 60k cap set for the nuclei cost those frames about 6 dB. Raising it
to 120k bought +5.7 dB for 1.9x the bytes on those frames (0.57 → 1.06 MB);
beyond 120k, +0.4 dB. Size each channel's cap on its largest frame, not on the
channel next to it.

Under the max-composite target 6-24% more voxels count as occupied, the most on
the late frames: T0601 requests 203,858 seeds (164,639 before), both clipped to
120,000, with at most 93,949 stored. Raising the membrane cap to about 200k buys
about +1 dB headline (+3 dB away from the occlusion holes, below) on late frames
for about 1.6x the splats; the demo keeps 120k.

Sheets are not filaments here: on these membranes more splats kept paying, while
on 1-voxel filaments more seeds made beading worse
([thin-structure-beading.md](thin-structure-beading.md)).

## A target with hard edges caps the headline PSNR

The max-composite membrane target is zero wherever a nucleus hides the membrane,
so it steps from the membrane value straight to 0 at those holes, where the old
`clip(R − G, 0)` target faded. No sum of Gaussians reproduces a step: on the frames with nucleus occlusion
(4 of the 5 measured), 57-94% of the new fits' foreground error sits within 2 voxels of the hole edges, and the
median membrane foreground PSNR is 35.9 dB against this target, where the old
fits scored 42.6 dB against theirs (robust peak). Those are different targets, not a quality
loss. Scored like-for-like on five frames, outside a 2-voxel xy band around the
holes the new fits match their target within about ±3 dB of the old fits'
scores against the old target (−2.7 to +2.9 dB, about equal on average). More
seeds do not lift the headline: at 1 seed per 12 / 6 / 3 voxels, T0321 scored
35.7 / 36.1 / 35.2 dB overall but 42.7 / 45.2 / 46.2 dB outside the band; T0601
uncapped at 1 per 12 scored 35.5 / 44.2 dB. Judge a fit to a stepped target
(a mask, a hard threshold, an occlusion) away from the steps, and never set its
headline against a smooth target's.

## Small frames need a seed floor

A per-voxel rule gives the early, few-cell frames a few hundred seeds. Organoid
frame 24's membranes scored 21-22 dB at 500 and 1,000 seeds and 48 dB at 2,000
(old target), so the demo floors every frame at 2,000. The default `--min-extent 0.6` prune
still removes up to two thirds of those seeds on the smallest frames, leaving the
early membrane frames uneven (25.4-48.1 dB foreground against the hole target
above; frame 64 keeps 631 splats). The prune and cull are cheap on large fits (0.0-0.1 dB once seeds start
alive), so look at the stored count on the smallest frames rather than on the
median one.

## Per-frame time-lapse fits from Python

`batch-fit` resolves one floor and one intensity range for the whole time-lapse
([background-floor.md](background-floor.md)). A Python loop over frames has to
do the same by hand, or brightness flickers frame to frame:

- **One fixed intensity scale, no second normalization.** Pass a fixed
  `norm_range` so the fitter does not rescale each frame on its own. On raw
  data that keeps brightness comparable across frames. The organoid uses
  `(0, 255)` on an export the authors had ALREADY rescaled per frame (see
  above): there the fixed range fits each frame as exported rather than
  normalizing it a second time, and brightness trends were lost before the fit.
- **`floor="none"` on deconvolved 8-bit exports** that are already zero outside
  the specimen: a per-frame `auto` floor is another per-frame scale.
- **A discrete time axis with an exactly representable step.** Use the round
  interval (10 min), not a measured `0.1667 h`; otherwise `max = (n−1)·step`
  can land past the last slider stop and the final frame becomes unreachable
  (see "Dimensions" in the `luxar-visualization` skill's
  `references/scene-api.md`).
- **A stand-in for an empty source frame.** The LSTree export's T0245 is a
  2x15x15 crop of zeros; drawing it blinks the scene to black. The demo draws
  T0244 in that slot at T0245's nominal time and says so in the description.
- **Cache names that encode the recipe.** The organoid's per-frame cache names
  carry the unmixing, the voxels-per-seed, the floor and cap, the iteration count
  and the frame actually fitted, so changing any of them misses the cache
  instead of silently reusing an old fit.

## Sparse puncta (FISH spots)

- **K\* under-seeds sparse spots.** Held-out PSNR is flat on sparse puncta, so
  `cal` reports a small K; its spot check flags it (see "Sparse puncta" in
  [calibration-sweep.md](calibration-sweep.md)).
- **`--min-extent 0` for spot-level work.** On EASI-FISH Gpr83 crops at 30,000
  seeds, the default prune kept spot recall at 0.983 against 0.992 unpruned at
  full resolution (both scored against the same 1,496 detected spots), and cost
  0.1-1.0 dB of foreground PSNR. The default stays; the cli README's
  `--min-extent` paragraph has the full measurement.
- **Score recall against one reference.** Spots detected on downsampled data are
  a different set: crops of the same footprint gave 1,496 spots at full
  resolution and 398 at xy/2, plausibly because dense or dim puncta merge or
  fade at xy/2 (not checked like-for-like). A recall of 1.00 at xy/2 and 0.99 at
  full resolution are therefore not comparable; score both fits against the
  full-resolution spots.

## Anisotropic voxels: downsample the fine axes toward isotropy

The EASI-FISH CZI samples 0.42 µm in z and 0.23 µm in xy (expanded units).
Halving xy gives 0.42 x 0.46 x 0.46 µm, nearly isotropic, at a quarter of the
voxels (1.03 G against 4.1 G per channel) and so about a quarter of the fit time
and memory. The cost is lateral detail below the coarser step: puncta closer than
about 0.5 µm can merge. Decide by what the scene is for; spot counting wants full
resolution (and the like-for-like recall check above).

## Expanded tissue: the file's voxel size is in expanded units

Expansion microscopy images a swollen hydrogel, so the voxel size a CZI or
OME-Zarr records describes the expanded sample. Divide by the linear expansion
factor for biological micrometres: the EASI-FISH authors' own code uses `ex=2`
(`easi_fish/roi_prop.py`). Fit on the expanded grid (the quality figures are
there) and apply the 1/ex scale last, e.g. with `luxar gsplat transform
--scale`.

## Background pedestals and refits

Seeds used to start dead on a camera pedestal; that is fixed, and refits of
pedestal data store more splats than before. See "Seeds start above the
pedestal" in [background-floor.md](background-floor.md).

## Reproducibility across machines

Seed order is platform-independent: every sort that ranks seeds or peaks is
stable, ties broken by index (`seeds/README.md`). The same CPU fit starts from
the same seeds on Linux x86-64 and macOS arm64, and agrees up to floating-point
rounding, which a short fit can amplify to a few tenths of a dB. Refits made
before that change can differ by up to about 1 dB on a short fit.

## Stating a compression ratio

Compare on-disk bytes with on-disk bytes: the scene as served (sum of its file
sizes, after `luxar optimize --pack` if you pack) against the source as stored.
A ratio against computed voxel bytes (shape x dtype) rather than the stored file
flatters the fit, and so does a baseline the demo never had (the organoid's ~700
GB acquisition); name the baseline next to the number, and say so when the fit
used a downsampled level. EASI-FISH quotes 111 MB against the 34.9 GB raw CZI it
was built from (≈315x) and lists the block-mean downsampling among its changes;
the organoid quotes 89.1 MB against the 462 MB of extracted VTI frames and
lineage it read (≈5x), because those frames are already deconvolved, cropped,
8-bit and zlib-compressed. The demos' wording lives in `luxar/demos/_caption.py`
(`compression_caption`, `compression_note`).
