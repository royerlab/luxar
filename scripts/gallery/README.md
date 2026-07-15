# Gallery harness

Manifest-driven tooling that captures a **still PNG + seamless 360° orbit video
(WebP + GIF)** for every demo, so we can generate the full set and then curate
the best for the README gallery (TODO **R19**).

## Files

| File | Role |
|------|------|
| `manifest.json` | **Single source of truth** — the demo list + per-demo capture hints. Consumed by both the dataset generator and the capture spec. Add a demo here and nothing else needs editing. |
| `generate_gallery_datasets.py` | Generates each demo's `.luxar.zarr` under `datasets/demos/` (idempotent; skips ones already present; best-effort). |
| `../../packages/luxar-viewer/src/tests/screenshots/generate-gallery.spec.ts` | Playwright capture: auto-center + fill-to-frame, auto-exposure, orbit, still + video. |
| `../../packages/luxar-viewer/playwright.gallery.config.ts` | Playwright config (GPU flags, viewer + data servers, video recording). |

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

Output lands in `docs/images/gallery/<id>.{png,webp,gif}`. That directory is
**gitignored** — it's a review staging area. Once you pick the winners, copy
them into `docs/images/readme/` (which **is** committed) and wire them into the
README gallery table.

## How capture works

- **Center + fill:** presses `F` (the viewer's FOV-aware fit) then dollies in to
  a `fill` factor (default 0.7) so the subject fills the frame — the built-in
  fit leaves ~25% margin.
- **Auto-exposure:** reads back the tone-mapped canvas via `gl.readPixels` and
  picks an exposure so the lit foreground sits near a target luminance. Override
  per demo with `"exposure": <log2 stops>` in the manifest when auto misses.
- **Seamless orbit:** a horizontal azimuth sweep of exactly 2π (continuous
  loop); the load/center/expose pre-roll is trimmed from the recorded video.

## Manifest fields

`id` (media + dataset stem), `title` (caption), `geometry`, `category`,
`script` (`demo_*.py`, or `null` for a feature-branch demo whose dataset must be
pre-generated), `dataset` (path under `datasets/demos/`), and the optional
capture hints `exposure`, `fill`, `dimensionNav` (`{key, steps}` for nD),
`rotationAxis`, `readme` (a current top README pick).

## Requirements

- Datasets: run the generator first (heavy demos download data / fit splats).
- `ffmpeg` on `PATH` for the WebP/GIF conversion.
