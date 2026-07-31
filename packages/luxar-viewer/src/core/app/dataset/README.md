# Dataset routing

Helpers extracted from `LuxarApp` that govern how a `src` URL becomes a
loaded scene. The orchestrator (`core/app.ts`) decides _whether_ to load
a dataset directly or open the `DatasetBrowser` modal first; this folder
owns the _how_ of each branch — URL classification, the async zarr
probe, the dataset browser lifecycle, and the scene-dependent UI
initialization sequence that follows a successful load.

Every file here is consumed by `LuxarApp` exclusively (via thin
delegate methods on the class). Nothing in this folder is re-exported
from the package barrel. Each helper takes a `*Ports` interface so the
orchestrator wires its long-lived components (SceneManager,
InputHandler, RenderingControls, …) and overlay-init callbacks in
without the helper needing to know how they're implemented.

## Files

```
dataset/
├── browser-decision.ts       # Pure URL classifier (sync; no fetch)
├── should-show-browser.ts    # Sync classify + async zarr-marker HEAD probe
├── show-browser.ts           # Open the DatasetBrowser modal + wire callbacks
├── browser-shortcut.ts       # `open-dataset-browser` window-event listener
└── load-dataset.ts           # Scene load + scene-dependent UI init sequence
```

| File                     | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `browser-decision.ts`    | Pure synchronous URL classifier. Exports the `BrowserDecision` union (`'must-browse' \| 'maybe-zarr'`) and `classifyBrowserUrl(src)`. Empty / whitespace-only input and any URL ending with `/` are `'must-browse'`; everything else is `'maybe-zarr'` and warrants a zarr-metadata probe. Extracted so the sync decision can be unit-tested without `fetch`.                                                                                                                                                                                                                                                                                                                         |
| `should-show-browser.ts` | Async wrapper used by `LuxarApp.init`. Runs `classifyBrowserUrl` first; on `'maybe-zarr'`, fires three parallel `HEAD` requests for v2 (`.zgroup`, `.zattrs`) and v3 (`zarr.json`) markers with a 5 s `AbortController` timeout. Returns `false` (load directly) as soon as any probe responds 2xx, otherwise `true` (show the browser).                                                                                                                                                                                                                                                                                                                                              |
| `show-browser.ts`        | Constructs a `DatasetBrowser` modal and routes its callbacks through a `ShowDatasetBrowserPorts` interface. Clears the error overlay on open, normalizes the chosen URL (strips trailing slashes), optionally calls `replaceBrowserDataSourceUrl` when `updateBrowserUrl` is opted in, propagates the new src back to the orchestrator via `onSrcChange`, then dispatches through the orchestrator-supplied guarded dataset-switch callback. Both side effects are skipped when `isSwitchInFlight()` reports an active switch — the dispatch is going to reject, and the host URL / src snapshot must not end up pointing at a dataset that never loaded. Hands a close handle to `InputHandler` so the Escape key routes through the browser's own `close()` rather than yanking the DOM node. Caller must guard against double-open and an active switch.                                        |
| `browser-shortcut.ts`    | Installs a window-level `open-dataset-browser` custom-event listener via the supplied `EventGroup` (so it tears down on dispose). Calls `showBrowser()` only when `hasOpenBrowser()` is false; the orchestrator additionally suppresses `showBrowser()` while a dataset switch is active.                                                                                                                                                                                                                                                                                                                                                                                             |
| `load-dataset.ts`        | The post-load UI init pipeline — the longest helper in the folder. Runs twelve steps in a strict order observable by users and tests: clear dimension UI → dispose previous overlays → `setSceneId` (must precede `loadSceneData` so persisted settings reach materials at construction time) → `loadSceneData` → push `viewer_config` into `renderingControls` (apply zarr defaults only when no localStorage entry exists) → update fly-speed range from scene scale → init dimension sliders + scale bar → init layers panel + colormap legend → init overlays + GPU picking → apply `viewer_config` UI state → optionally open the cache-stats monitor → kick the animation loop. |

## Lifecycle

```
LuxarApp.init(src)
  │
  ├─ shouldShowBrowser(src) ───┐  classifyBrowserUrl()  (sync)
  │                            └─ HEAD probe + Promise.any  (async, ≤5s)
  │
  ├─ true  → showDatasetBrowser({ ports })
  │            │
  │            └─ onDatasetSelect(url) → switchDataset(url) → loadDataset(url, ports)
  │
  └─ false → loadDataset(src, ports)

installBrowserShortcut({ events, hasOpenBrowser, showBrowser })
  └─ window 'open-dataset-browser'  → showDatasetBrowser (re-entry-safe)

LuxarApp.dispose()
  └─ EventGroup tears down the browser-shortcut listener
```

Browser selection and the public embedder API both enter through
`LuxarApp.switchDataset`, whose shared in-flight guard serializes every
post-init teardown+reload. Initial loading during `init()` calls
`loadDataset` directly because the app is not initialized yet. Every path
still funnels through the same load helper, so the ordering invariants
apply uniformly.

## Invariants

- **Sync decision before any network call.** `shouldShowBrowser` calls
  `classifyBrowserUrl` first and only probes when the URL might
  plausibly be a zarr root. Trailing slashes never trigger a fetch.
- **`setSceneId` precedes `loadSceneData`.** `load-dataset.ts` orders
  these explicitly so `RenderingControls` can apply persisted
  per-scene settings (HDR intensity, blending mode, …) before
  materials are constructed inside the loader.
- **Overlays dispose in lockstep with scene clear.** `disposeOverlays`
  runs before `loadSceneData`; if the new scene fails to load, the
  user sees a blank canvas, not stale overlay DOM on top of nothing.
- **Browser callbacks never mutate orchestrator state directly.** All
  side effects (src update, browser handle reset, host-URL
  replacement) flow through `ShowDatasetBrowserPorts`, keeping the
  helper testable without a full `LuxarApp`.
- **Opt-in URL mutation.** `replaceBrowserDataSourceUrl` is only
  called when `updateBrowserUrl === true`. Embedded callers default
  to no host-page URL changes; the standalone bootstrap opts in.
- **Re-entry-safe shortcut and switching.** `installBrowserShortcut`
  short-circuits when a browser is already open, and the orchestrator refuses
  to reopen it while `switchInFlight` is set. Browser selections also route
  through that same guard, so stale callbacks cannot start an overlapping
  teardown+reload.

## See also

- `../../app.ts` — `LuxarApp` orchestrator that owns the long-lived
  components passed in via `*Ports` and the delegate methods that
  call each helper here.
- `../../bootstrap.ts` — populates `LuxarAppOptions.updateBrowserUrl`
  for the standalone entry point.
- `../../../ui/dataset-browser` — `DatasetBrowser` modal class
  instantiated by `show-browser.ts`.
- `../../../config/url-params.ts` — `replaceBrowserDataSourceUrl`
  used by `show-browser.ts`.
- `../../../utils/cross-layer/event-group.ts` — `EventGroup` used by
  `browser-shortcut.ts` for auto-disposing listeners.
