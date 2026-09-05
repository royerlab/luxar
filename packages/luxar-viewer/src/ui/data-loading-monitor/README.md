# Data Loading Monitor internals

Private helpers behind the `M`-key data-loading monitor. The public
facade lives at `../data-loading-monitor.ts` and owns the panel
lifecycle, event subscription, and three-state UI (hidden → mini →
expanded). This folder holds the focused helpers it coordinates each
tick: the stateful provider registry, HTML templates, the loading
advisor, the event queue, the polling loop, and the hierarchical
timing panel.

## Files

| File                   | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `templates/`           | HTML-string templates split by concern: shared primitives and formatting, Overview, Cache, Memory, Insights, and scene-graph rendering.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `advisor.ts`           | `LoadingAdvisor` — consumes `MonitorEvent`s and rolled-up `LoaderMetrics` / `MemoryMetrics` and emits `Recommendation`s (slow query, slow load, high query time, low query efficiency, high error rate, low GPU reuse rate, excessive accumulator growth, …).                                                                                                                                                                                                                                                                                           |
| `cache-actions.ts`     | Cache-clear workflows for L0, SliceCache, L1, L2, and all tiers, including destructive confirmation, completion toasts, and the orchestrator refresh callback.                                                                                                                                                                                                                                                                                                                                                                                          |
| `event-queue.ts`       | Generic `EventQueue<T>` — bounded ring buffer with non-blocking `push` and atomic `drain()` used to decouple loader event producers from the polling consumer.                                                                                                                                                                                                                                                                                                                                                                                          |
| `polling-loop.ts`      | `PollingLoop` — restartable interval timer. Errors thrown from `onTick` are logged via `utils/log` and never stop the loop.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `providers.ts`         | `MonitorProviderRegistry` — stateful owner of the scene-scoped provider slots and live LOD/draw-order snapshots, plus the app-scoped density-guard snapshot behind the tree's `1/K` density chip (wired once by the init pipeline; survives `resetSceneProviders`). It remains private to the parent monitor and marks the orchestrator's structure dirty when provider changes require a repaint.                                                                                                                                                      |
| `headline-counts.ts`   | The shared per-geometry headline table (label, DOM field id, unit noun, presence rule) read by the Overview hero cards, their incremental patcher, and the collapsed compact badge — one source, so the three cannot disagree about which types exist or what they are called.                                                                                                                                                                                                                                                                          |
| `scene-graph-model.ts` | `SceneGraphModel` — owns scene-graph statistics, expansion state, per-path visible counts, and the memoized path index while preserving the monitor's public facade.                                                                                                                                                                                                                                                                                                                                                                                    |
| `timing-panel.ts`      | Renderer + in-place updater for the collapsible per-frame timing tree, fed by `profiling/update-profiler`. Module-level `expandedState` map persists collapse state across rerenders. The footer also carries the depth-sort verdict: passed `depthSortUnavailable` (from `rendering/depth-sort-coordinator::isDepthSortAvailable`) it prints `depth sort UNAVAILABLE` in place of the sort count and suppresses the "No timing data yet" empty state, so a session drawing order-dependent geometry in storage order cannot be buried by zero timings. |

`README.md` for this folder; per-subpackage READMEs live under
`templates/`, `metrics/`, and `tabs/`.

## How the pieces fit together

```
LoaderMonitor events ──► EventQueue ──► PollingLoop.onTick ─┐
                                                            │
                ┌──────────────────────────────────────────┘
                ▼
     metrics/*   (roll up rates, cache, global, and memory state)
                │
                ▼
   templates/* (full repaint)   ◄──┐
                │                  │ structure missing
                ▼                  │
   tabs/* (incremental)  ──────────┘ patch-by-`data-field`
                │
                ▼
        advisor.ts (emits Recommendations into the Insights tab)
                │
                ▼
   timing-panel.ts (hierarchical timing tree, separate render path)
```

The orchestrator at `../data-loading-monitor.ts` owns:

- the `EventQueue<MonitorEvent>` instance and a `PollingLoop` that
  drains it on each tick,
- a `LoadingAdvisor` instance that sees every drained event plus
  rolled-up metrics,
- a `MonitorProviderRegistry` that owns the provider slots and live
  snapshots while reporting structural changes back to the orchestrator,
- the painted-once-then-patched tab DOM (templates for structure;
  `tabs/` for per-tick value patches),
- the timing panel's container and its expand/collapse state.

## Status badges and color classes

`templates/cache.ts` is the single source of truth for the cache-status
pills surfaced in the Cache tab. `CACHE_BADGE_COLOR` maps each
`CacheStatusBadge` (`cache-enabled`, `no-cache`, `disabled-config`,
`opfs-unavailable`, `quota-constrained`, `cache-errors-detected`,
`unvalidated-external-dataset`, `provider-missing`) to a
`luxar-color--*` modifier class, and `CACHE_BADGE_TOOLTIP` gives each
badge a didactic hover explanation (what it means, why it appears,
what to do about it). `renderCacheStatusBadges` emits one
`<span class="luxar-badge …" data-badge="…" title="…">` per badge; the
row wrapper carries `data-field="cache-status-row"` and a `join('|')`
signature so `tabs/cache.ts` can skip `innerHTML` replacement when the
badge set hasn't changed across ticks.

All other dynamic colors flow through `getColorClass(SemanticColor)`
which returns `luxar-color--{success|warning|error|info|muted|dimmed|primary}`.

## Tooltip policy

Every label, value, badge, and table header the monitor paints carries
a `title` tooltip, and the tooltips are deliberately **didactic**: they
explain what the metric is, why it matters, and what a good/bad value
looks like (e.g. hit-rate tooltips note that a low rate right after
load is normal). Two tooltips are _state-dependent_ and therefore
re-patched by `tabs/cache.ts` on every tick alongside their values:
`validationModeTooltip(mode)` and `lastValidatedTooltip(mode)`. Under
the source-validation modes the freshness timestamp updates on each
successful online check (a real confirmation), but under
`validationMode: 'ttl'`/`'none'` it is a fixed known-good **baseline** —
when the cache was established — that does NOT advance on repeat offline
checks and does not confirm the cached data still matches the server; the
tooltip must say so for whichever mode is currently displayed. For the
same reason the row label itself is mode-aware
(`lastValidatedLabel(mode)`): "Last Validated" under the
content-hash/.zattrs-hash/archive-etag modes, "Cached Since" for ttl/none.
The timing panel's per-operation explanations live in the `TOOLTIPS` map
in `timing-panel.ts`.

## Contracts and invariants

- **Templates produce structure, updaters patch values.**
  The concern modules under `templates/` paint the full HTML on a tab
  switch or a structural change; the per-tick updaters under `tabs/`
  only rewrite values via `data-field` selectors. If a selector misses,
  the updater returns `false` and the orchestrator rebuilds via the
  matching template module.
- **`EventQueue.drain()` is atomic** — the internal array is
  reassigned in one step so producers pushing concurrently never
  observe a half-drained queue.
- **`PollingLoop` swallows `onTick` errors** (logged via
  `utils/log`) so a single bad tick can't stop monitoring.
- **`LoadingAdvisor` is idempotent per recommendation id** — every
  `addX…Recommendation` writes a fixed `id` (e.g. `slow-query`,
  `high-avg-query`, `low-gpu-reuse-points`) into a `Map`, so the latest
  values win and stale entries don't accumulate across ticks. The
  history buffer is capped at
  `config.dataLoading.monitor.limits.maxAdvisorHistory`.
- **Thresholds come from `config.dataLoading.monitor.thresholds`** —
  `advisor.ts` never hard-codes magic numbers; tuning happens in the
  unified config.
- **LOD-progress is polled, not pushed.** The orchestrator refreshes a
  `path → LODProgressState` snapshot from the injected
  `LODProgressProvider` each tick (additive `loaded/total/refining`,
  substitutive `activeLevel/levelCount`), then patches the tree's LOD
  chips (`data-lod-path`) and header summary in place — no structural
  rebuild. Dataset totals for substitutive `kind=lod` groups count the
  finest level only (alternatives, not cumulative); the K level loaders
  collapse to one logical layer in `getGlobalStats`.
- **Additive-chip glyphs**: `LOD x/N` = detail levels loaded of total;
  `●` = last refinement pass fully cache-resident, `◌` = still streaming
  from the network; `⏳` = refinement in progress; `LOD –/N` = the node
  has N additive levels but no live streaming loader (typically an
  inactive substitutive level). Every glyph is spelled out in the chip's
  `title` tooltip.
- **Active substitutive level rows** are highlighted (and inactive levels
  dimmed) via `data-level-of`/`data-level-index` attributes that the
  per-tick patcher re-marks from the group's `activeLevel` — the tree
  shows _what renders_, not just what the file contains.
- **Per-node visible counts** arrive via
  `updateVisibleCountsByPath` (pushed by the SceneLoader's
  visible-counts walk, keyed by mesh `name` = scene-graph path) and are
  merged into the tree nodes so badge tooltips read
  "N elements (M visible after slicing)" — symmetric across all four
  geometry types. For mesh the suffix means "how much of this surface the
  current nD slab indexes" rather than a streaming residency, since a mesh
  is resident in full either way. The walk prunes non-visible subtrees, so nodes whose
  path is absent from the latest map (hidden layer, switched-away
  substitutive level) have their count cleared back to unknown — the
  suffix disappears rather than showing a stale number. The same walk pushes
  the visible-scene dropped-element aggregate to the overview card via
  `updateDroppedElementCount`.
- **Timing panel expand/collapse state is module-level** on purpose so
  it survives full DOM repaints triggered by tab switches.
- **All user-supplied strings flow through `utils/escape-html`** before
  being interpolated into template literals (loader paths, entry
  names, recommendation messages).

## Four-type parity

Every surface in the panel reports all four geometry types — points, lines,
gsplats and mesh — and each uses that type's own DRAWN-PRIMITIVE noun (points /
segments / splats / triangles), the convention the whole monitor follows.

Mesh is the type most recently brought up to parity, and where it differs it is
because a whole-node loader genuinely differs, not because a number is missing:

- **Hero cards / compact badge** — one card per type present, from
  `headline-counts.ts`. A mesh-only scene shows `VISIBLE TRIANGLES`, not the
  "LOADING …" placeholder.
- **Scene-graph tree** — icons, `faceCount` badge, LOD chips, and the
  `(M visible after slicing)` suffix.
- **Loader list / loader totals** — mesh loaders implement the `LoaderMonitor`
  surface (`mesh-whole-node`), so their bytes, loads and resident memory join
  the panel's totals. Whole-node latency does not trigger the advisor's
  chunk-size slow-load recommendation. They report NO
  `queries` / `avgQueryTime` / `spatialIndex`: there is no spatial index and a
  view change re-serves the resident mesh, so a query sample would be a ~0 ms
  entry for work that never touched the store. For the same reason mesh is not
  counted in `activeSpatialLoaders`.
- **Performance tab** — mesh sessions aggregate into one `Mesh` row like the
  other three (`NODE_TYPE_COUNTERS` in `timing-panel.ts`) and carry a typed
  `triangles` count that SUMS across a multi-layer row.
- **Memory tab** — mesh is deliberately ABSENT from the GPU-pool and
  accumulator tables. Those are keyed by `POOLED_GEOMETRY_TYPES` (the
  instanced-quad element-texture path); a mesh uploads its own
  `BufferGeometry` and keeps no per-slice working set, so it has no row to
  show there. Its resident bytes appear in the loader totals instead.

## Failed-load recovery surface

The Overview tab shows a warning banner while the SceneLoader has recorded
load failures (`renderFailedLoadsBanner` in `templates/overview.ts`), fed by the
`FailedLoadsProviderPort` injected through `setFailedLoadsProvider` (wired in
`data/scene-loader/monitor/monitor-wiring.ts`). Its Retry button
(`data-action="retryFailedLoads"`) runs `SceneLoader.retryAllFailedLoaders`
(serialized against the update lock by the loader) with an in-flight guard and
toasts the outcome. Failed loads are also retried automatically when the
window comes back online (`core/app/lifecycle/online-retry.ts`).

## Why this layout

The monitor's public file is already large because it orchestrates an
event loop, multi-tab DOM, scene-graph viewer, and timing tree. Each
helper here was extracted to keep that orchestrator focused on
coordination rather than HTML strings, rate math, or advisory logic.
Everything in this folder is reachable only via the parent monitor;
no external module imports it directly, including the provider registry.

## Subpackages

- [`templates/`](./templates/README.md) — HTML-string renderers split
  by concern, their dependency direction, and the Cache tab's stable
  incremental-update markup contract.
- [`metrics/`](./metrics/README.md) — Pure roll-up helpers for cache,
  global, and memory metrics plus per-second event rates.
- [`tabs/`](./tabs/README.md) — Per-tick tab updaters that patch the
  static structure painted by `templates/` (`updateOverviewTab`,
  `updateCacheTab`, `updateMemoryTab`, badge patching, and DOM helpers).

## See Also

- [`../data-loading-monitor.ts`](../data-loading-monitor.ts) —
  Orchestrator that wires the queue, polling loop, advisor, templates,
  and tab updaters together.
- [`../README.md`](../README.md) — UI package overview; the Data
  Loading Monitor section describes the user-facing behaviour
  (keyboard shortcut `M`, three-state UI, event types).
- `../../types/data-monitor-types.ts` — Shared types
  (`MonitorEvent`, `LoaderMetrics`, `CacheMetrics`,
  `CacheStatusBadge`, `Recommendation`, `SceneGraphState`, …).
- `../../profiling/update-profiler.ts` — Source of the `TimingEntry`
  tree rendered by `timing-panel.ts`.
