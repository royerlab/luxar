# adaptive-dpr

Submodules of the `AdaptiveDPRManager` (`../adaptive-dpr-manager.ts` — the
facade and sole public export). Each module is pure, timestamp-driven, and
dependency-injected: no `window`, no config imports, no clocks of its own, so
each is unit-testable without mocks. The manager orchestrates them and owns
everything environmental (live `devicePixelRatio`, the renderer, config merge,
callbacks).

| Module | Owns |
|---|---|
| `fps-tracker.ts` | Sliding-window FPS estimation: O(1) trim/compaction, interval-normalized FPS, sample count / span for probe quality gates |
| `refresh-rate-estimator.ts` | The display's achievable rAF cap (high-water mark, fallback lower bound, sustained-uniform-low throttle downshift) that the relative FPS thresholds derive from |
| `hysteresis-tracker.ts` | The sustained-high scale-up streak, with a mid-band grace so isolated dropped-frame samples don't restart the wait |
| `probe-controller.ts` | U-shape probe lifecycle: arm on scale-down, settle after the probe window on a CLEAN sample (accept / reject vs. the pre-change baseline / inconclusive-void), void on pause or display change |
| `bounds-ledger.ts` | Learned operating bounds: the rejected-probe floor with exponential rejection backoff, the punished-ascent ceiling (native → 1.0 demotion), TTL decay, and content-change softening |

State classification (the manager's contract with its hooks):

- **Session state** (FPS window, hysteresis streak, pending probe) is cheap,
  display-and-moment specific, and cleared aggressively — on pause, on a
  native-DPR change, on disable.
- **Learned state** (the bounds ledger) is expensive evidence and survives
  pauses and idle restores; only a display change or disable wipes it, and
  TTLs decay it.

Tests: `src/tests/unit/rendering/adaptive-dpr/` (one file per module) plus the
manager-level integration suite in `src/tests/unit/rendering/adaptive-dpr-manager.test.ts`.
