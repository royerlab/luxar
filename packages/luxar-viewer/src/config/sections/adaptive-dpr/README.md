# adaptive-dpr

Adaptive device-pixel-ratio configuration slice. Owns the construction-time
defaults for the FPS-driven DPR control loop: refresh-rate-relative scaling
thresholds, multiplicative scaling factors, the static DPR floor, the U-shape
probe knobs, the learned floor/ceiling TTLs with exponential backoff, and the
session-hygiene timings (gap reset, content-change recheck).

The MAXIMUM DPR is not here either: the adaptive loop scales below a ceiling
that `rendering/pixel-ratio-cap.ts` owns, which is the display's own DPR only
when the `allowHighDPR` setting is on. It is off by default, so on a HiDPI
display the ceiling this loop works under is 1.0.

Conforms to the section-trio pattern documented in [../README.md](../README.md):
`data.ts` exports the literal, `types.ts` defines the interface, and
`validate.ts` checks the cross-field invariants (threshold ordering, factor
ranges, TTL consistency). The runtime on/off toggle lives in
`renderingControls.defaults.adaptiveDPREnabled`, not here.

## Contents

- `data.ts` — `adaptiveDPRConfig: AdaptiveDPRConfig`. Key defaults:
  `enabled: true`, `minDPR: 0.5`, `scaleDownFactor: 0.9`,
  `scaleUpFactor: 1.05` (asymmetric scaling — slower up, faster down —
  prevents quality flicker), `hysteresisSeconds: 3`,
  `evaluationIntervalMs: 500`; refresh-relative ratios
  `scaleDownFpsRatio: 0.75` / `scaleUpFpsRatio: 0.90` over a
  `refreshRateFallback: 60` warmup cap (and under a `refreshRateCeiling`,
  `0` = none here; the mobile runtime constructs the manager with 60); probe knobs
  `probeWindowMs: 1500` / `probeImprovement: 1.05` / `probeMinSamples: 8`;
  floor/backoff `floorTtlMs: 30_000` × `backoffMultiplier: 2` capped at
  `backoffMaxTtlMs: 300_000`; ceiling demotion
  (`ceilingTtlMs`, `punishedAscentWindowMs`, `punishedAscentThreshold`);
  hygiene (`gapResetMs: 350`, `contentChangeRecheckMs: 5000`,
  `midbandGraceSamples: 1`).

  The `AdaptiveDPRManager` constructor merges this literal directly
  underneath `config.adaptiveDPR`, so partial overrides (including the unit
  tests' fixed-shape config mock) can never leave a knob `undefined`.

- `types.ts` — `AdaptiveDPRConfig` interface. `enabled` is the
  construction-time default; runtime toggling is delegated to
  `renderingControls.defaults.adaptiveDPREnabled`.
- `validate.ts` — `validateAdaptiveDPR`: errors on broken invariants
  (inverted FPS ratios, non-contracting factors, non-positive timings,
  `floorTtlMs <= probeWindowMs`, `backoffMaxTtlMs < floorTtlMs`); warns on
  legal-but-self-defeating values (probe window inside the 1s FPS sample
  window, a gap threshold too large to ever fire).

## Public API

- `adaptiveDPRConfig` — re-exported through `../../index.ts` into `AppConfig.adaptiveDPR`.
- `AdaptiveDPRConfig` — re-exported through `../../types.ts`.
- `validateAdaptiveDPR` — invoked by the central dispatcher in `../../validation.ts`.
