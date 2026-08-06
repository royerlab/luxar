# monitor

Data-loading monitor slice. Owns the timings, alert thresholds, and ring-buffer limits used by the runtime data-loading monitor that tracks events, cache rates, query timings, and memory pressure.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md), minus a validator: `data.ts` exports the literal and `types.ts` defines the interface. Unlike its sibling slices, this section has no `validate.ts` — its values are passive instrumentation knobs, not safety-critical inputs, and are not wired into the central dispatcher. Composed into `dataLoading.monitor` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingMonitorConfig: DataLoadingMonitorConfig`. Three nested groups:
  - `timings` — `eventCleanupInterval` (30 s), `maxEventAge` (5 min), `ratesCacheTimeout` (1 s), `defaultUpdateInterval` (100 ms), `queryCleanupCheckInterval` (10), `maxQueryAge` (60 s).
  - `thresholds` — `highQueryTime` (100 ms), `highLoadTime` (500 ms), `highErrorRate` (0.05), `lowQueryEfficiency` (0.5). Used to flag degraded monitor states.
  - `limits` — `maxEvents` (1000), `maxAdvisorHistory` (100), `rateCalculationWindow` (5 s), `bandwidthCalculationWindow` (1 s). Ring-buffer caps and sliding-window sizes for rate/bandwidth computation.
- `types.ts` — `DataLoadingMonitorConfig` and its three sub-interfaces `MonitorTimings`, `MonitorThresholds`, `MonitorLimits`.

## Public API

- `dataLoadingMonitorConfig` — composed into `dataLoadingConfig.monitor` by `../data.ts`.
- `DataLoadingMonitorConfig`, `MonitorTimings`, `MonitorThresholds`, `MonitorLimits` — re-exported through `../types.ts` (the data-loading types barrel) and then `../../../types.ts`.
