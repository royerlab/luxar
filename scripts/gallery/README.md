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
| `../../packages/luxar-viewer/playwright.gallery.config.ts` | Playwright config (GPU flags, viewer + data servers, video recording). |
| `score_exposure.py` | Offline scorer for the captured stills: flags `OVER` (blown highlights) and `FLAT` (narrow, uniformly over-exposed). Hand-synced with `exposure-policy.ts`. |

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
  hand-synced copies of the thresholds (no test asserts the two agree; the
  scorer's `FLAT` rule also adds a p50 margin the harness gate does not have).
  Override per demo with `"exposure": <log2 stops>` in the manifest when auto
  misses.
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
