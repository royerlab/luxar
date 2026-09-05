# Data-monitor tab renderers

Per-tick updater modules for the expanded Data Loading Monitor's
tabs. Each module patches an already-rendered tab's value cells in
place rather than rebuilding the DOM every tick.

The static structure for every tab is painted once by its module under
`../templates/`; the orchestrator (`../../data-loading-monitor.ts`)
then calls into these updaters each polling tick.

## Files

| File                    | Role                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cache.ts`              | `updateCacheTab(container, cacheMetrics)` — patches L0/L1/L2 stats, status pills, totals, error row                                                                |
| `memory.ts`             | `updateMemoryTab(container, metrics)` — patches GPU-pool, accumulator, and total-memory rows                                                                       |
| `overview.ts`           | `updateOverviewTab(container, stats, cacheMetrics, badges)` — patches headline metrics then scene-graph badges                                                     |
| `dom-helpers.ts`        | `patchField` + `updateColorClass` — shared selector-based patch primitives                                                                                         |
| `scene-graph-badges.ts` | Patches scene-graph counts, LOD/draw-order/density chips (glyph + text via `chipInnerHtml`), active-level rows, and the header summary from live monitor snapshots |

## The `data-field` selector pattern

Each value cell in a tab's rendered HTML carries a stable
`data-field="<key>"` attribute. The painted-once renderer in
`../templates/cache.ts` produces the structure; the per-tab updater here
patches values by selector:

```ts
patchField(container, 'l0-hitrate', `${hitRate.toFixed(1)}%`);
updateColorClass(el, getColorClass('success'));
```

`patchField` returns `false` when its selector misses, and
`updateCacheTab` returns `false` when the tab structure isn't yet
present in the container — the caller uses that signal to fall back
to a full rebuild via `../templates/cache.ts`.

## Invariants

- Updaters are **idempotent within a tick**: each call rewrites every
  field it owns, so partial state never leaks across ticks.
- Updaters **never construct structure** — only `../templates/`
  paints HTML. If a `data-field` selector is missing the updater
  bails out and lets the orchestrator request a rebuild.
- Color classes are managed exclusively via `updateColorClass`, which
  strips any prior `luxar-color--*` class before adding the new one
  so the cell never accumulates stale color modifiers.
- The status-pill row uses a cheap `join('|')` signature on
  `cacheMetrics.status` to skip `innerHTML` replacement when the
  badge set is unchanged across ticks.

## Related

- `../templates/cache.ts` — paints the static Cache-tab structure (the renderer
  paired with these updaters).
- `../metrics/cache.ts` — aggregates raw events into the
  `CacheMetrics` payload consumed by `updateCacheTab`.
- `../scene-graph-model.ts` — owns the tree state and path index consumed by
  `updateSceneGraphBadges`.
- `../../data-loading-monitor.ts` — orchestrator that wires both
  halves together each polling tick.
