# Viewer Developer Tools

Standalone TypeScript drivers that automate a real Chromium against the
running viewer — for AI-assisted debugging and for capturing publication-
grade screenshots without a physical monitor. Sibling to `scripts/` (which
holds build / quality / perf-diff utilities); these are interactive
browser-driving tools, not part of any build pipeline.

The two browser drivers share the same recipe: launch headless Chromium
via `@playwright/test` with GPU acceleration flags, navigate to a Luxar
URL with `?debug` enabled, wait for the scene to initialize, poke at
`window.__luxarDebug`, and write a PNG to disk. The E2E server-identity
helper is test-harness infrastructure rather than a browser driver.

## Contents

```
tools/
├── agent-driver.ts          # Browser debugging driver
├── capture-hires.ts         # High-resolution figure capture for papers
└── e2e-server-identity.ts   # Checkout identity + Playwright preflight helpers
```

## `e2e-server-identity.ts`

Prevents local Playwright runs from silently reusing Vite or dataset
servers rooted in another clone/worktree. It derives a deterministic
identity from the checkout's canonical repository path, writes a small
gitignored marker under `.luxar-e2e-identities/`, exposes the marker
through Vite middleware, and validates the exact response during global
setup. The repository-root `python3 -m http.server` serves the same
marker directly.

The standard, performance, screenshot, and video Playwright configs use
the checkout-specific Vite marker as their `webServer.url` readiness
probe. A sibling checkout therefore returns 404, while a same-checkout
Vite server remains reusable. The standard and performance configs also
probe the repository-root data marker; screenshot/video generation uses
`luxar serve`, which has no identity endpoint, so those configs disable
data-server reuse instead. The helper also performs the HTTP availability
checks for required E2E datasets so filesystem presence cannot mask a
mis-rooted server.

## `agent-driver.ts`

Lets Claude Code (or any AI agent) "see" the viewer without a monitor.
Mirrors all browser console output to the terminal — including
`pageerror`, `requestfailed`, and any HTTP response ≥400 — then dumps a
JSON snapshot of `__luxarDebug.scene` / `.camera` / `.renderer` / the
`getState()` performance block, and screenshots the current view.

Wired into the viewer package via two pnpm scripts (see
`package.json`):

```bash
pnpm agent:debug              # headless
pnpm agent:debug:visible      # visible browser (--headless=false)
```

Direct invocation accepts `--url=`, `--headless=true|false`, and
`--wait=<ms>` flags (the wait controls how long to give Three.js to
finish first-frame initialization). The default URL is
`http://localhost:5173/?debug` (or `APP_URL` from the environment); the
driver appends `?debug` if you forget to. Output paths:

- `test-results/debug/debug-view.png` — success screenshot.
- `test-results/debug/error-state.png` — written on any thrown error
  before the process exits 1.

The driver requires the viewer to be running locally and the scene to
have been opened with `?debug` so `window.__luxarDebug` is populated —
otherwise the state dump returns `{ error: 'Debug mode not enabled' }`.

See the parent README's "AI-Assisted Debugging" section for the full
debug-loop workflow.

## `capture-hires.ts`

High-resolution figure capture for paper and slide figures. Same
launch pattern as `agent-driver.ts`, but driven entirely by environment
variables so it slots cleanly into figure-generation scripts:

| Env var            | Purpose                                                | Default                        |
| ------------------ | ------------------------------------------------------ | ------------------------------ |
| `APP_URL`          | Viewer URL (required; must include `?debug`)           | —                              |
| `OUT`              | Output PNG path                                        | `test-results/debug/hires.png` |
| `WAIT`             | Total ms budget for scene init + post-camera streaming | `15000`                        |
| `WIDTH` / `HEIGHT` | Viewport in CSS pixels                                 | `2400` / `1600`                |
| `DSF`              | `deviceScaleFactor` (for retina-quality output)        | `1`                            |
| `CAMERA_ZOOM`      | Scale distance from orbit target                       | `1`                            |
| `CAMERA_POS`       | Absolute camera position `x,y,z` (world units)         | —                              |
| `CAMERA_TARGET`    | Orbit target `x,y,z`                                   | —                              |
| `CAMERA_FOV`       | Vertical FOV override (degrees)                        | —                              |
| `SLICE`            | nD slice override, e.g. `"time:240,channel:0"`         | —                              |

Camera and slice overrides are applied after the first 15s of the
`WAIT` budget is consumed (so auto-fit and initial chunks land first);
the remainder of the budget then gives the new frustum a chance to
stream in fresh chunks before the screenshot fires.

```bash
APP_URL="http://localhost:5173/?src=/data/demo.zarr&debug" \
OUT=figure-1.png WAIT=20000 WIDTH=3200 HEIGHT=2400 DSF=2 \
  pnpm exec tsx tools/capture-hires.ts

# nD navigation + camera framing for a multi-channel time series
APP_URL="http://localhost:5173/?src=/data/4d.zarr&debug" \
SLICE="time:240,channel:0" CAMERA_ZOOM=1.5 \
  pnpm exec tsx tools/capture-hires.ts
```

The script logs the resolved `getState()` snapshot before screenshotting
(`totalPoints`, `totalGSplats`, `totalElements`, point-cloud and gsplat
counts) and warns if `totalElements === 0` — that almost always means
the slice position is wrong or the dataset URL is stale.

## Requirements

- The viewer dev server must be running (`pnpm dev`) — both tools talk
  to a live browser instance.
- `@playwright/test` Chromium must be installed (`pnpm exec playwright
install chromium` if first-time setup).
- The target URL must include `?debug` so `window.__luxarDebug` is
  exposed; `agent-driver.ts` auto-appends it, `capture-hires.ts` does
  not.

## See Also

- `../scripts/README.md` — build / lint / perf-diff scripts.
- Parent `../README.md` — "AI-Assisted Debugging" section.
- `../src/core/app/debug/debug-interface.ts` — assembles the
  `window.__luxarDebug` object (`scene`, `camera`, `renderer`,
  `controls`, `sceneDimsManager`, `getState()`, `renderOnce()`).
- `../src/core/app/debug/debug-state.ts` — the pure scene-walking
  computer behind `__luxarDebug.getState()` (point / gsplat / line
  counts + camera + dim reporting).
