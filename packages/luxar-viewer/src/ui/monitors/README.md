# Monitors

Runtime monitoring UI for cache state, data-loading activity, and
performance. The monitors are read-only views over state owned
elsewhere — they never mutate loader/cache state directly.

## Files

| File | Role |
|------|------|
| `data-monitor-manager.ts` | Singleton manager owning the monitor lifecycle, factory hooks, and shared event bus |
| `data-loading-monitor.ts` | The "Data" tab — per-loader progress, recent events, throughput |
| `cache-metrics-aggregator.ts` | Pure aggregator: rolls L0/L1/L2 stats + loader-side counters into a single `CacheMetrics` snapshot |
| `data-monitor-templates.ts` | DOM templates and small helpers (`renderCacheStatusBadges`, `formatValidationMode`, `l2ErrorTotal`) |
| `performance-monitor.ts` | FPS / draw-call / VRAM panel |
| `rate-calculator.ts` | Sliding-window per-second rate counters (hits/sec, misses/sec, bandwidth) |

## Public surface

- `DataMonitorManager.getInstance().createMonitor(id, container)` —
  factory used by `core/app.ts` to wire a monitor into the scene
  loader through the `SceneLoaderMonitorPort`.
- `DataMonitorManager.disposeInstance()` — explicit teardown called
  from `LuxarApp.dispose()`.
- `PerformanceMonitor.getInstance()` — global FPS panel.

## Invariants

- Aggregators are **pure**: `aggregateCacheMetrics(params)` reads
  `params.l0Provider`, `params.cacheStatsProvider`, `params.loaders`,
  and mutates only the provided `metricsCache` map. No singleton
  state.
- Status badges and validation-mode formatting (`CACHE_BADGE_COLOR`,
  `formatValidationMode`, `formatLastValidated`, `l2ErrorTotal`) are
  pure helpers tested directly in
  `tests/unit/ui/monitors/data-monitor-templates-cache-helpers.test.ts`.
- The monitor never reaches into the data layer to read loader
  internals; it pulls everything through the `LoaderMonitor`
  structural interface.
