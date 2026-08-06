# Data-Monitor Metrics Extractors

Pure helpers that the data-loading monitor's main file calls each
poll tick to turn raw event streams and provider snapshots into the
`CacheMetrics` / `RatesSnapshot` shapes its tabs render. Keeping the
math here makes the monitor's main file a thin dispatcher and lets
these helpers be unit-tested without DOM, timers, or tab state.

## Files

| File       | Role                                                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| `cache.ts` | Aggregates L0 / L1 / L2 / network stats from optional provider ports and refreshes the per-loader metrics snapshot map. |
| `rates.ts` | Walks the monitor's event ring buffer once per call to compute per-second rolling rates (queries, loads) and bandwidth. |

## Public surface

```typescript
// rates.ts
export interface RatesSnapshot {
  queriesPerSec: number;
  loadsPerSec: number;
  bandwidth: number;
  lastCalculated: number;
}

export function calculateRates(params: CalculateRatesParams): void;
// Mutates `params.rates` in place. No-ops when the cache is still
// fresh (`now - rates.lastCalculated < cacheTimeoutMs`).
```

```typescript
// cache.ts
export function aggregateCacheMetrics(params: AggregateCacheMetricsParams): CacheMetrics;
// Returns a fresh CacheMetrics object. Mutates `params.metricsCache`
// (refreshes the monitor's per-loader snapshots) and reads `params.rates`
// for the rolling per-second fields.
```

`cache.ts` also exports three small input-shape interfaces —
`CacheRatesSnapshot` (the subset of the rolling rates the aggregator
reads: `queriesPerSec` / `loadsPerSec` / `bandwidth`), `L0Provider`
(`{ getStats(), clear?() }`, the port for the in-memory
decompressed-chunk cache), and `SliceProvider` (the same-shaped port
for the SliceCache / "S-cache") — and re-exports
`CacheTelemetryState` from `types/data-monitor-types` so existing
consumers can import it from the same module that owns the aggregation.

## Contracts and invariants

- **`calculateRates` is read-only on `events`** and mutates only
  `params.rates`. It iterates newest-first so it can early-exit once
  the rate-window cutoff is crossed.
- **`bandwidth` is bytes/sec**, normalized by `bandwidthWindowMs` so
  the value is comparable across configurations. `cache.ts` passes
  it through to `CacheMetrics.bandwidth` without recomputation.
- **`aggregateCacheMetrics` mutates `metricsCache` in place** — for
  every loader in `params.loaders` it writes the current
  `getMetrics()` snapshot under the loader's path. Callers depend on
  this side effect to keep their "last seen metrics" map fresh.
- **`telemetryState` precedence**: caller-supplied wins; otherwise
  inferred from `cacheStatsProvider` presence + `isEnabled()`. With
  no provider the default is `not-wired` (not `enabled`) so
  `?no-cache` runs do not surface as enabled.
- **Cross-tier demand hit rate**: consumers read
  `effectiveDemandHitRate`, which combines L0 hits with the
  L1 / L2 / network demand counters when available.
- **Cache-status badges**: emitted into `CacheMetrics.status` and
  driven by telemetry state plus L2 health counters
  (`quotaWriteSkipped`, `writeFailures`, `corruptedEntries`,
  `metadataParseFailures`) and `health.unvalidatedExternalDataset` /
  `health.opfsAvailable`. `opfs-unavailable` is emitted only when the
  provider explicitly reports `opfsAvailable === false`; older
  providers that omit the field are treated as available.

## Why pure

The monitor itself is event-driven and stateful (ring buffer,
timing panel, tab DOM patching). Pushing the metric roll-ups into
pure functions here keeps the monitor's main file focused on event
dispatch and DOM patching, and means these helpers can be exercised
by unit tests that fabricate a `MonitorEvent[]` or a provider stub
without booting the monitor.

## Dependencies

- Internal: `types/data-monitor-types` (`MonitorEvent`,
  `LoaderMonitor`, `LoaderMetrics`, `CacheMetrics`,
  `CacheStatsProvider`, `CacheStatusBadge`, `CacheTelemetryState`).
- External: none.

## See Also

- [`../README.md`](../README.md) — data-loading monitor internals
  overview (orchestrator wiring, event flow, status-badge source).
- [`../../README.md`](../../README.md) — UI package overview;
  the Data Loading Monitor section describes the surfaced metrics
  in user-facing terms.
- [`../tabs/cache.ts`](../tabs/cache.ts) — primary consumer of
  `aggregateCacheMetrics`'s output.
