# Gallery harness

Manifest-driven tooling that captures a **still PNG + seamlessly-looping orbit
video (WebP + WebM)** for every demo, so we can generate the full set and then curate
the best for the README gallery (TODO **R19**).

## Files

| File | Role |
|------|------|
| `manifest.json` | **Single source of truth** — the demo list + per-demo capture hints. Consumed by both the dataset generator and the capture spec. Add a demo here and nothing else needs editing. |
| `generate_gallery_datasets.py` | Generates each demo's `.luxar.zarr` under `datasets/demos/` (idempotent; skips ones already present; best-effort). |
| `../../packages/luxar-viewer/src/tests/screenshots/generate-gallery.spec.ts` | Playwright capture: auto-center + fill-to-frame, auto-exposure, orbit, still + video. |
| `../../packages/luxar-viewer/src/tests/screenshots/exposure-policy.ts` | The auto-exposure **decision** + its tuning constants, split out of the spec so it is unit-testable without a browser (`src/tests/unit/gallery-exposure-policy.test.ts`). |
| `../../packages/luxar-viewer/src/tests/screenshots/crop-policy.ts` | The border-lit (**cropped subject**) verdict + its warning floor, split out of the spec so it is unit-testable without a browser (`src/tests/unit/gallery-crop-policy.test.ts`). |
| `../../packages/luxar-viewer/playwright.gallery.config.ts` | Playwright config (GPU flags, viewer + data servers, video recording). |
| `score_exposure.py` | Offline scorer for the captured stills: flags `OVER` (blown highlights) and `FLAT` (narrow, uniformly over-exposed). Hand-synced with `exposure-policy.ts`. |
| `tests/` | Unit tests for `score_exposure.py`, incl. a parity test that pins its mirrored thresholds to the ones in `exposure-policy.ts`. On the default Python suite. |

## Usage

```bash
# 1. Generate the datasets the manifest needs (idempotent, resumable)
make generate-gallery-datasets              # all
make generate-gallery-datasets ONLY=lorenz,desi_galaxies

# 2. Capture stills + orbit videos → docs/images/gallery/ (staging, gitignored)
make generate-gallery                       # datasets (if missing) + capture
make generate-gallery ONLY=desi_galaxies    # a subset

# Under the hood (from packages/luxar-viewer/):
GALLERY_ONLY=lorenz pnpm gallery
```

Output lands in `docs/images/gallery/<id>.{png,webp,webm}`. That directory is
**gitignored** — it's a review staging area. Once you pick the winners, copy
them into `docs/images/readme/` (which **is** committed) and wire them into the
README gallery table.

## How capture works

- **Center + fill:** presses `F` (the viewer's FOV-aware fit) then closed-loop
  dollies until the subject's measured screen coverage reaches `FILL_TARGET`
  (0.95 of the frame's smaller dimension) — the built-in fit leaves a margin.
  Coverage is measured from a screenshot using a 3rd–97th-percentile bounding
  box of the lit pixels, so a few stray outliers can't keep the scene tiny.
  Override per demo with `"fillTarget"`, or nudge afterwards with `"zoom"`.
- **Auto-exposure:** measures the composited frame (a Playwright screenshot,
  decoded back inside the page — the renderer runs with
  `preserveDrawingBuffer: false`, so an in-page `gl.readPixels` reads an empty
  buffer) and picks an exposure in three phases: drive the lit foreground's p99
  just below clipping; then, if the lit histogram turns out to be *narrow*
  (p99 − p10 < `NARROW_SPREAD_MAX` — a headlit shaded mesh, where p99 says
  nothing about where the subject sits), re-target the lit *median* to
  `TARGET_MID` so the surface keeps its colour instead of washing out in the
  ACES shoulder; finally step down while the subject is blown to white or the
  background is lifted to grey. The decision logic lives in
  `../../packages/luxar-viewer/src/tests/screenshots/exposure-policy.ts` (pure
  and unit-tested). The narrow-spread gate is **empirical** — when it fires, the
  capture log says `(auto, flat subject)`, so a manifest-wide sweep can spot a
  false positive. `score_exposure.py` re-derives the same metrics offline from
  hand-synced copies of the thresholds (pinned to `exposure-policy.ts` by
  `tests/test_score_exposure.py`; the scorer's `FLAT` rule adds a p50 margin the
  harness gate deliberately does not have).
  Override per demo with `"exposure": <log2 stops>` in the manifest when auto
  misses.
- **Crop check:** counts the **lit pixels on the frame's outermost row/column**
  (the still plus the two rock extremes — plus the last frame of a `timelapse`,
  where a developing subject is largest — measured off the lossless PNGs the
  harness already captures, since JPEG ringing next to a bright edge would fake a
  crop). Lit content on the edge means the subject runs off frame, and the fill
  loop is blind to it *by construction*: its 3rd–97th-percentile bbox discards
  exactly the outliers that touch the edge, so a tile can report "82% — a fit"
  while a nucleus leaves the frame in a third of the orbit frames (measured on
  `mesh_isosurface_cells3d`: **summed over the 120 orbit frames**, 3389
  border-lit pixels at `fillTarget` 0.84 — worst single frame 218 — then 1256 at
  0.75 and 0 at 0.69). It **warns and never fails**, naming the pose and
  suggesting the knob that actually governs that demo's framing (`fillTarget`,
  `zoom`, `distance`, or a skipped fill), because cropping is a judgement call at
  the margin. Honest caveat: a non-zero count looks **common** — 25 of the 28
  committed README tiles are non-zero at the 0.04 lit cutoff, and luma alone does
  not separate a real crop from a faint background wash or an unhinted full-bleed
  composition. So the harness logs the **number** on every demo as a per-tile
  regression signal, and the number matters more than the boolean; the audit
  behind that (and `BORDER_LIT_MAX`, the one floor to raise if the warning gets
  noisy) is documented in `crop-policy.ts`.
- **Seamless orbit:** a small-angle **sinusoidal rock** of ±20° about the
  subject's up axis (`"orbitUp"`, default world-Y), captured as 120 explicit
  per-angle screenshots — not Playwright's passive video, which does not
  reliably record the viewer's rAF repaints in headless. `sin` returns to its
  start, so the loop is seamless with no pre-roll to trim. ffmpeg assembles the
  frames 1:1 (no motion interpolation, which warps fine structure) into a VP9
  WebM master and a smaller animated WebP for the README.

## Manifest fields

Required: `id` (media + dataset stem), `title` (caption), `geometry`,
`category`, `script` (`demo_*.py`, or `null` for a feature-branch demo whose
dataset must be pre-generated), `dataset` (path under `datasets/demos/`).

Optional capture hints (see the `DemoEntry` interface in the capture spec for
the authoritative list and defaults): `exposure` (log2 stops, overrides
auto-exposure), `autoExpose` / `autoFrame` (set `false` to use the scene's baked
`viewer_config` instead), `fillTarget`, `zoom`, `distance`, `viewAngle`
(`{azimuth, elevation}`), `orbitUp` (`'x' | 'y' | 'z'`), `dimensionNav`
(`{key, steps}` for nD), `timelapse` (`{framePoint}` for 4D series), `lodFinest`,
`readme` (a current top README pick), `note` (free-text human annotation; the
capture code never reads it).

## Requirements

- Datasets: run the generator first (heavy demos download data / fit splats).
- `ffmpeg` on `PATH` for the WebP/WebM conversion.
