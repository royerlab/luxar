# Data Loading Monitor internals

Private helpers behind the `M`-key data-loading monitor. The public
facade lives at `../data-loading-monitor.ts` and owns the panel
lifecycle, event subscription, and three-state UI (hidden → mini →
expanded). This folder holds the pure-ish helpers it pulls in each
tick: HTML templates, the loading advisor, the event queue, the
polling loop, and the hierarchical timing panel.

## Files

| File              | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `templates.ts`    | HTML-string template functions for every cell, card, progress bar, status badge, tab content, scene-graph tree, and memory section the monitor paints. Also re-exports `MemoryMetrics` and friends. The scene-graph tree renders LOD/partition kind badges (`K LODs` / `N parts`) and a live LOD chip (`L{i}/{n}` active level, or `LOD {loaded}/{total}` + refining/residency) via `renderKindBadge` / `lodChipContent`; `summariseLodStates` builds the header summary. |
| `advisor.ts`      | `LoadingAdvisor` — consumes `MonitorEvent`s and rolled-up `LoaderMetrics` / `MemoryMetrics` and emits `Recommendation`s (slow query, high memory, low GPU reuse rate, frequent evictions, …).                                                                                                                                                                                                                                                                             |
| `event-queue.ts`  | Generic `EventQueue<T>` — bounded ring buffer with non-blocking `push` and atomic `drain()` used to decouple loader event producers from the polling consumer.                                                                                                                                                                                                                                                                                                            |
| `polling-loop.ts` | `PollingLoop` — restartable interval timer with tick stats. Errors thrown from `onTick` are logged via `utils/log` and never stop the loop.                                                                                                                                                                                                                                                                                                                               |
| `timing-panel.ts` | Renderer + in-place updater for the collapsible per-frame timing tree, fed by `profiling/update-profiler`. Module-level `expandedState` map persists collapse state across rerenders.                                                                                                                                                                                                                                                                                     |

`README.md` for this folder; per-subpackage READMEs live under
`metrics/` and `tabs/`.

## How the pieces fit together

```
LoaderMonitor events ──► EventQueue ──► PollingLoop.onTick ─┐
                                                            │
                ┌──────────────────────────────────────────┘
                ▼
     metrics/rates.ts + metrics/cache.ts   (roll up rates & cache state)
                │
                ▼
   templates.ts (full repaint)  ◄──┐
                │                  │ structure missing
                ▼                  │
   tabs/cache.ts (incremental)  ───┘ patch-by-`data-field`
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
- the painted-once-then-patched tab DOM (templates for structure;
  `tabs/` for per-tick value patches),
- the timing panel's container and its expand/collapse state.

## Status badges and color classes

`templates.ts` is the single source of truth for the cache-status
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
load is normal). Two tooltips are *state-dependent* and therefore
re-patched by `tabs/cache.ts` on every tick alongside their values:
`validationModeTooltip(mode)` and `lastValidatedTooltip(mode)` — under
`validationMode: 'none'` the freshness timestamp records only a check
*attempt* (nothing to compare against), not a confirmation, and the
tooltip must say so for whichever mode is currently displayed. For the
same reason the row label itself is mode-aware
(`lastValidatedLabel(mode)`): "Last Validated" only under
content-hash, "Last Checked" otherwise. The timing panel's per-operation explanations
live in the `TOOLTIPS` map in `timing-panel.ts`.

## Contracts and invariants

- **Templates produce structure, updaters patch values.**
  `templates.ts` paints the full HTML on a tab switch or a structural
  change; `tabs/cache.ts` (and the per-tick updaters in the
  orchestrator) only rewrite values via `data-field` selectors. If a
  selector misses, the updater returns `false` and the orchestrator
  rebuilds via `templates.ts`.
- **`EventQueue.drain()` is atomic** — the internal array is
  reassigned in one step so producers pushing concurrently never
  observe a half-drained queue.
- **`PollingLoop` swallows `onTick` errors** (logged via
  `utils/log`) so a single bad tick can't stop monitoring.
- **`LoadingAdvisor` is idempotent per recommendation id** — every
  `addX…Recommendation` writes a fixed `id` (e.g. `slow-query`,
  `high-memory`, `low-gpu-reuse-points`) into a `Map`, so the latest
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
  "N elements (M visible after slicing)" — symmetric across points /
  lines / gsplats. The walk prunes non-visible subtrees, so nodes whose
  path is absent from the latest map (hidden layer, switched-away
  substitutive level) have their count cleared back to unknown — the
  suffix disappears rather than showing a stale number.
- **Timing panel expand/collapse state is module-level** on purpose so
  it survives full DOM repaints triggered by tab switches.
- **All user-supplied strings flow through `utils/escape-html`** before
  being interpolated into template literals (loader paths, entry
  names, recommendation messages).

## Failed-load recovery surface

The Overview tab shows a warning banner while the SceneLoader has recorded
load failures (`renderFailedLoadsBanner` in `templates.ts`), fed by the
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
no external module imports it directly.

## Subpackages

- [`metrics/`](./metrics/README.md) — Pure roll-up helpers for cache
  metrics and per-second event rates (`aggregateCacheMetrics`,
  `calculateRates`).
- [`tabs/`](./tabs/README.md) — Per-tick tab updaters that patch the
  static structure painted by `templates.ts` (currently
  `updateCacheTab` plus `dom-helpers`).

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
