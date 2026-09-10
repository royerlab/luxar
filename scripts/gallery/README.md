# Gallery harness

Manifest-driven tooling that captures a **still PNG + seamlessly-looping orbit
video (WebP + WebM)** for every demo, so we can generate the full set and then curate
the best for the README gallery (TODO **R19**).

## Files

| File | Role |
|------|------|
| `manifest.json` | **Single source of truth** — the demo list + per-demo capture hints. Consumed by both the dataset generator and the capture spec. Add a demo here and nothing else needs editing. |
| `media-manifest.json` | Published root-README media: demo id → content-addressed WebP/WebM keys, with byte counts, digests, and content types. The README and capture selection both consume it. |
| `generate_gallery_datasets.py` | Generates each demo's `.luxar.zarr` under `datasets/demos/` (idempotent; skips ones already present; best-effort), then runs the LOD-ladder and scene-credit auditors with `--require-scenes` against exactly the stores produced by that invocation. Scene-credit failures gate the build; ladder findings are report-only until the corpus is laddered. A second report-only pass covers the complete local inventory, so already-present neighbours stay visible without gating a fresh store. A failed generated-store gate is not re-gated once that store is already present: fix it and regenerate with `--force` (or delete it), and require the direct pre-upload full-inventory credit audit to pass. `SCENE_AUDITOR_NAMES` is the single registration seam for future built-scene auditors and records whether each one gates generated output. A demo whose `DEMO_META` declares machine-local `local_data` (`manual-file` / `kaggle-auth` / `git-lfs`) is still run, but a **positive non-zero exit status** is reported in the soft `manual-data` bucket instead of failing the build — for `git-lfs`, only while one of its manifest-declared checkout payloads is missing or still an unpulled pointer. Record-backed downloads declare no local-data requirement and fail hard. A `timeout`, a `no-output` or a death by signal (negative return code) stays hard. One case remains hard on a cold checkout: `arxiv_papers_kaggle` can exhaust the per-demo timeout during its ~30 GB download. A demo listed in `UNBUILDABLE_IDS` is the third disposition: it is **never spawned at all** and lands in the soft `unbuildable` bucket, for a demo whose shipped input is known-broken and whose fallback would blow the timeout. The list is empty today (`gsplats_3d_visible_human_head` was removed once its colors sidecar was regenerated, #1670) and is meant to stay that way — delete an entry as soon as its input is fixed. |
| `check_tile_staleness.py` | Report-only check for published README still/video pairs older than the gallery dataset/capture policies, their manifest-listed demo generator and directly imported private helpers, applicable `luxar.shading` production code, or their own manifest entry. Also reports manifest-recorded media sizes. Uses entry-specific blame in both manifests so editing one demo does not mark every tile stale. |
| `verify_media.py` | Checks manifest/README consistency offline, then optionally fetches every hosted object and verifies its content type, byte count, and SHA-256 digest. |
| `../../packages/luxar-viewer/src/tests/screenshots/generate-gallery.spec.ts` | Playwright capture: auto-center + fill-to-frame, auto-exposure, orbit, still + video. |
| `../../packages/luxar-viewer/src/tests/screenshots/exposure-policy.ts` | The auto-exposure **decision** + its tuning constants, split out of the spec so it is unit-testable without a browser (`src/tests/unit/gallery-exposure-policy.test.ts`). |
| `../../packages/luxar-viewer/src/tests/screenshots/crop-policy.ts` | The under-fill and border-lit (**cropped subject**) verdicts + warning floors, split out of the spec so they are unit-testable without a browser (`src/tests/unit/gallery-{underfill,crop}-policy.test.ts`). |
| `../../packages/luxar-viewer/src/tests/screenshots/frame-similarity.ts` | The still-vs-orbit-frame **correlation** (`COMPARE_SIZE`, `DISAGREEMENT_THRESHOLD`), split out of the spec so it is unit-testable without a browser (`src/tests/unit/gallery-frame-similarity.test.ts`). |
| `../../packages/luxar-viewer/src/tests/screenshots/orbit-axis.ts` | Which **signed world axis** the rock revolves about, derived from the camera's own up-vector, split out of the spec so it is unit-testable without a browser (`src/tests/unit/gallery-orbit-axis.test.ts`). |
| `../../packages/luxar-viewer/playwright.gallery.config.ts` | Playwright config (GPU flags, viewer + data servers, video recording). |
| `gen_redirects.py` | Generates the demo site's stable `/d/<demo-key>` routes as a Cloudflare Pages `_redirects`. **Load-bearing**: the root README links 29 tile titles at these routes, and `--check-contract` fails the build naming any linked key that lost one. Takes the store from each entry's `dataset`, never its `id` — 8 of 87 differ, and a mechanical mapping emits routes to stores that do not exist (which the origin answers `200 text/html`, i.e. a blank viewer, not a 404). Regenerate on **every** deploy so the dated prefix inside stays current. |
| `audit_readme_demo_count.py` | Report-only: does the README's stated live-demo count match the gallery being deployed? Runs at deploy time, because that is when the fact changes. Never blocks — a stale README must not stop correct data reaching the site. Nothing else watches this number (`sync_demo_counts.py` owns the *bundled* count) and it has drifted twice. |
| `score_exposure.py` | Offline scorer for the captured stills: flags `OVER` (blown highlights) and `FLAT` (narrow, uniformly over-exposed). Hand-synced with `exposure-policy.ts`. |
| `tests/` | Unit tests for the Python gallery tools, including real temporary Git histories for tile-staleness comparisons and a `score_exposure.py` parity test that pins its mirrored thresholds to `exposure-policy.ts`. On the default Python suite. |

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
GALLERY_ONLY=readme pnpm gallery             # recapture the demos selected by the README

# Report published README tile staleness and media size margins
make check-gallery-staleness

# Verify every hosted README media object (opt-in network audit)
make check-gallery-media
```

Output lands in `docs/images/gallery/<id>.{png,webp,webm}`, except that a
`noOrbitVideo` demo has no `.webm`. That directory is **gitignored** — it's a
review staging area. To publish a reviewed refresh:

1. Compute each selected WebP/WebM file's SHA-256 and use its first 16 hex
   characters plus the extension as the object key.
2. Upload the files to `data.luxarviewer.dev/media/` with the recorded content
   types; content-addressed keys make existing URLs immutable.
3. Replace the affected demo entries in `media-manifest.json` with the key,
   full digest, byte count, and content type.
4. Replace the corresponding root-README image and link URLs with the new keys,
   then run `hatch run pytest scripts/gallery/tests/test_verify_media.py` and
   `make check-gallery-media` before merging.

`make check-gallery-staleness` is intentionally non-gating: it prints `STALE`
rows and still exits zero, because refreshing media is a reviewed batch action.
For each demo it takes the newest blamed line in that demo's
`media-manifest.json` entry as the publish timestamp, then compares it with the
gallery dataset generator; the capture spec together with its media-reporting,
timelapse-settle, and orbit-axis helpers and Playwright gallery config; the
exposure and crop policies; the manifest-listed demo generator and its directly
imported private `luxar.demos` helpers; and the lines of that demo's own manifest
object. Production `luxar.shading` history applies only when the committed demo
module imports it; shading tests/docs remain excluded.
Global inputs are printed once above the rows, while per-demo failures print
`UNKNOWN` and do not hide the rest of the report. All reads use committed
`HEAD`, so an in-progress manifest edit cannot create a fake commit timestamp.
Each row also reports the manifest-recorded README WebP/WebM sizes. The 20 MiB warning
and 25 MiB limit shelves are borrowed from the gallery capture size guard
(#2263) as a sanity check; the demo site's Cloudflare Pages `/media/*` deploy
corpus is a separate set measured during capture. The footer prints the corpus
total and five largest files. The total is informational because this check does
not enforce a corpus-size cap, and all size findings remain report-only.
The check intentionally does not inspect external dataset pins or
machine-local payload contents. It refuses shallow clones rather than silently
producing incomplete history; run it from a full checkout.

## How capture works

- **Center + fill:** presses `F` (the viewer's FOV-aware fit) then closed-loop
  dollies until the subject's measured screen coverage reaches `FILL_TARGET`
  (0.95 of the frame's smaller dimension) — the built-in fit leaves a margin.
  Coverage is measured from a screenshot using a 3rd–97th-percentile bounding
  box of the lit pixels, so a few stray outliers can't keep the scene tiny.
  Override per demo with `"fillTarget"`, or nudge afterwards with `"zoom"`.
  Every capture logs `final coverage=NN.N% final lit=NN.N%` from the settled
  still it ships, regardless of framing path; record those values in the
  manifest `"note"`.
  The fill loop is a pure dolly and cannot re-target an off-centre or
  translucency-biased subject; use an absolute `"distance"` to bypass it in
  those cases, and record the tested sweep in the manifest `"note"`.
- **Auto-exposure:** measures the composited frame (a Playwright screenshot,
  decoded back inside the page — the renderer runs with
  `preserveDrawingBuffer: false`, so an in-page `gl.readPixels` reads an empty
  buffer) and picks an exposure in three phases: drive the lit foreground's p99
  just below clipping; then, if the lit histogram turns out to be *narrow*
  (p99 − p10 < `NARROW_SPREAD_MAX`, where p99 says nothing about where the
  subject sits), re-target the lit *median* to
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
- **Under-fill check:** measures the final still from every framing path after
  `zoom`, exposure and LOD settle, then always logs both the 3rd–97th-percentile
  bbox **span** and the **lit screen area**. Span alone misses a long, thin
  subject, so the harness warns when either signal falls below its measured
  floor: 50% span or 10% lit area. The snapshot basis is frame 0 of the 29
  committed 340 px animated WebP tiles, a mixed-vintage set rendered from
  2026-07-15 through 2026-08-26, whose minima were 51.0% and 12.7%. The runtime
  check instead reads the settled PNG. WebP and PNG agree closely for the same
  current render, but an older committed tile can drift enough to warn; that is
  the staleness signal working, not a reason to hide it. Re-derive the floors
  from a full run's `final coverage=` / `final lit=` lines when the tile set or
  renderer changes (including #2176 and #2160); do not use WebP frame 0 for a
  `timelapse`, whose settled still may select a different timepoint. The warning
  never fails a capture, and a failed in-page decode is reported as
  `final coverage=unmeasured final lit=unmeasured` rather than losing the media.
- **Crop check:** counts the **lit pixels on the frame's outermost row/column**
  (the still plus orbit poses on a ~5° grid across the whole rock — plus the last
  frame of a `timelapse`, where a developing subject is largest — measured off
  the lossless PNGs the harness already captures, since JPEG ringing next to a
  bright edge would fake a crop). Sampling only the two rock extremes is not
  enough: a subject whose projected extent peaks at an intermediate angle comes
  back inside the frame by the endpoint, so an endpoints-only check reads clean
  on exactly the tiles it is meant to catch.
  Lit content on the edge means the subject runs off frame, and the fill
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
  regression signal, and the number matters more than the boolean; it also logs
  `poses=<measured>/<attempted>`, since a measurement that fails is skipped rather
  than fatal and the worst-of-the-rest would otherwise read as complete; the audit
  behind that (and `BORDER_LIT_MAX`, the one floor to raise if the warning gets
  noisy) is documented in `crop-policy.ts`.
- **Seamless orbit:** a small-angle **sinusoidal rock** of ±20° about the
  subject's up axis (`"orbitUp"`, defaulting to whichever signed world axis the
  **camera's own up-vector** most nearly points along — so a scene that bakes
  `up=(0,0,1)` in its `viewer_config` rocks about +Z without being told to),
  captured as 120 explicit per-angle screenshots — not Playwright's passive
  video, which does not reliably record the viewer's rAF repaints in headless.
  `sin` returns to its start, so the loop is seamless with no pre-roll to trim.
  ffmpeg assembles the frames 1:1 (no motion interpolation, which warps fine
  structure) into a VP9 WebM master and a smaller animated WebP for the README.

  The orbit **hard-sets `cam.up` every frame**, so that axis has to be right. It
  used to default to world-Y unconditionally, which silently discarded any baked
  non-Y up: the still keeps the baked pose (`F` restores it) but the animation
  did not, so a demo baking a non-Y up renders an animation rolled ~90° away from
  its own poster, turntable degenerating into tumble — `mesh_isosurface_cells3d`
  swung 130% in subject aspect over one rock before its `orbitUp: "x"` was set
  (#1377). Five manifest demos bake a non-Y up and only two of them declare
  `orbitUp`, so three were left on the wrong axis; two of those three were
  measurably shipping the roll (the asteroids/cosmicflows correlations pinned in
  `gallery-frame-similarity.test.ts`), and the third is a 4D timelapse whose
  camera was barely orbiting in the first place (#1383).
  Deriving the default from the camera makes `orbitUp` a true **override**,
  needed only to rock about something other than the scene's own up. The derived
  axis keeps its **sign**, so a baked `up=(0,0,-1)` rocks about −Z rather than
  being flipped 180° from the still. Setting `orbitUp` explicitly also runs
  `positionForOrbitUp`, which re-parks the camera on an axis and discards the
  baked framing — so an explicit `orbitUp` usually wants a `viewAngle` beside it
  whenever that parked pose lands edge-on (as this mesh slab's does, hence its
  `viewAngle` 45/40, whereas `gsplats_3d_ct_totalsegmentator`'s `orbitUp: "z"`
  parks coronal and needs none), while the derived default leaves framing alone.

  **Timelapse demos need the rAF loop re-frozen every frame.** The orbit freezes
  it once at setup, but a timelapse slice load restarts it, and a live loop runs
  `controls.update()`, which re-derives the camera from the controls' stored
  spherical state and snaps it back to the pre-orbit pose. Before this was
  handled, a 4D tile's camera sat at the baked pose in 6 of 8 frames — it barely
  orbited at all while its time dimension advanced (#1383). The reset lands
  *after* the per-frame `page.evaluate` returns, so re-applying the pose inside
  that evaluate does not help; the loop has to be stopped. The freeze is gated on
  the demo being a timelapse, so captures that were never broken are untouched.

  After each capture the harness compares the still against orbit frame 0 — the
  same nominal pose — and **warns** if they correlate below 0.85. That is the
  check whose absence let #1377 ship. Timelapse demos are exempt: their still is
  framed at `timelapse.framePoint` while frame 0 sits at the clip start, so the
  two legitimately differ.

  **Judge an animated tile on its orbit frames, not on the still.** The README
  embeds the animated WebP; the PNG is a byproduct that ships nowhere. A framing
  tuned on the still can crop at rock extremes the still never visits. For a
  `noOrbitVideo` tile, judge the curated still instead: it is the source of the
  published static WebP, and the rock extremes do not ship.

## Manifest fields

Required: `id` (gallery media filename stem; must equal the demo key), `title`
(caption), `geometry`, `category`, `script` (`demo_*.py`, or `null` for a
feature-branch demo whose dataset must be pre-generated; currently unused), `dataset`
(authoritative store path under `datasets/demos/`).

Optional capture hints (see the `DemoEntry` interface in the capture spec for
the authoritative list and defaults): `exposure` (log2 stops, overrides
auto-exposure), `autoExpose` / `autoFrame` (set `false` to use the scene's baked
`viewer_config` instead), `fillTarget`, `zoom`, `distance`, `viewAngle`
(`{azimuth, elevation}`), `orbitUp` (`'x' | 'y' | 'z'`, the world axis the rock
revolves about; default = the camera's own signed up axis, so set it only to
override — and note it re-parks the camera and so usually wants a `viewAngle`
beside it), `dimensionNav`
(`{key, steps}` for nD), `timelapse` (`{framePoint}` for 4D series), `lodFinest`,
`noOrbitVideo` (see below), `note` (free-text human annotation; the capture code
never reads it).

`noOrbitVideo` (optional): capture a **static single-frame** tile and skip the
orbit video entirely. For a subject whose apparent extent changes sharply with
view angle — a row of objects foreshortening, a flat wall going edge-on — where
the rock reads as flashing rather than motion. `gsplats_lod_embryo_line` swings
13x in mean luminance twice per loop at the default ±20°, and still 8.2x at ±6°,
so no amplitude fixes it.

Note the `.webp` **is** the animated loop (`build_gallery_data.py` uses it as the
tile's `still`, and `has_media` is `video or still`), so dropping only the
`.webm` would leave the pulsing in place. Set on `gsplats_lod_embryo_line` and
`gsplats_2d_codex_pancreas`. The capture removes a stale staged `.webm` whenever
the flag is set, so the summary and the publish sweep cannot mistake it for a
fresh encode.

Root-README tiles currently require both WebP and WebM entries in
`media-manifest.json` and `verify_media.py`. Before promoting a `noOrbitVideo`
demo there, extend that publishing contract to support a WebP-only tile.

`citation` (optional, not a capture hint): the dataset credit, copied verbatim
from the demo's `DEMO_META["citation"]["short"]`. It is here so a tile's credit
is reviewable in the repo instead of only on the rendered page;
`test_demo_meta.py` pins it equal to the demo's own value. Only demos that
declare a real credit carry it — an absent key means "unknown or not yet
recorded", which is deliberately *not* the same claim as "nothing to credit", so
a procedurally generated demo has no `citation` here either.

## Requirements

- Datasets: run the generator first (heavy demos download data / fit splats).
- `ffmpeg` on `PATH` for the WebP/WebM conversion.

## Troubleshooting

Both `webServer` entries keep their **stdout** quiet by default (a sweep is long
and vite logs every request), so a server that starts but never becomes reachable
at the port Playwright is watching shows up as nothing more than
`Timed out waiting 90000ms from config.webServer`. Re-run with
`GALLERY_DEBUG=1 pnpm gallery` (or `GALLERY_DEBUG=1 make generate-gallery`) to
forward vite's and `luxar serve`'s stdout to the reporter. (The data server
always runs with `PYTHONUNBUFFERED=1`; without it a piped, block-buffered stdout
loses its last lines when Playwright kills the process on timeout.) **stderr is
never suppressed** — a server that dies outright still reports why without the
flag.

The usual cause is the data server binding a *different* port than the one
Playwright waits on: `luxar serve` shifts to the next free port when its probe
says the requested one is busy, and prints
`⚠️  Data port 9899 busy, using 9900 instead` — which `GALLERY_DEBUG=1` makes
visible. If something really is on the port, free it or point the run elsewhere
with `GALLERY_DATA_PORT`.
