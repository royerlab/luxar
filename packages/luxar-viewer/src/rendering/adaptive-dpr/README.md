# adaptive-dpr

Submodules of the `AdaptiveDPRManager` (`../adaptive-dpr-manager.ts` — the
facade and sole public export). Each module is pure, timestamp-driven, and
dependency-injected: no `window`, no config imports, no clocks of its own, so
each is unit-testable without mocks. The manager orchestrates them and owns
everything environmental (live `devicePixelRatio`, the renderer, config merge,
callbacks).

| Module                      | Owns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fps-tracker.ts`            | Sliding-window FPS estimation: O(1) trim/compaction, interval-normalized FPS, sample count / span for probe quality gates. The window is a maximum AGE, not a sample budget — trimming stops at `minRetainedSamples` (default 2) so the estimate stays defined below one frame per window instead of collapsing to "not enough data" at the frame rates that need adapting most                                                                                                                                                                                                                                                      |
| `stall-detector.ts`         | Telling DEAD TIME (GC pause, synchronous decode, idle resume) apart from a genuinely slow FRAME RATE: an interval counts as dead time only when it clears the absolute `gapResetMs` floor AND is strictly more than 4× the median of the 4 PRECEDING inter-frame intervals — never of a memory including itself, which with a short post-boundary memory made a 5 s startup stall read as "the frame rate". Its cadence memory deliberately survives the gap reset it drives — that is what lets a session that genuinely slows down converge after one or two misread intervals — and is cleared only where the frame STREAM breaks |
| `refresh-rate-estimator.ts` | The display's achievable rAF cap (high-water mark, fallback lower bound, sustained-uniform-low throttle downshift) that the relative FPS thresholds derive from — plus the one-shot sub-throttle DISTRESS verdict (a sustained plateau below ~22fps can't be a real display mode — 23.976Hz film/TV modes are the slowest genuine regime, so instead of latching the cap onto it, which would invert the thresholds and park the DPR at native, the manager answers with a ceiling demotion to 1.0)                                                                                                                                  |
| `hysteresis-tracker.ts`     | The sustained-high scale-up streak, with a mid-band grace so isolated dropped-frame samples don't restart the wait                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `probe-controller.ts`       | U-shape probe lifecycle: arm on scale-down, settle after the probe window on a CLEAN sample (accept / reject vs. the pre-change baseline / inconclusive-void), void on pause or display change, and carry the content-confounded flag a kept-across-a-content-change probe is marked with                                                                                                                                                                                                                                                                                                                                            |
| `bounds-ledger.ts`          | Learned operating bounds: the rejected-probe floor with exponential rejection backoff, the punished-ascent ceiling (ceiling → 1.0 demotion; inert by construction when the allow-high-DPR cap already pins 1.0, since an ascent above 1.0 is what arms it), TTL decay, and content-change softening                                                                                                                                                                                                                                                                                                                                  |

State classification (the manager's contract with its hooks):

- **Session state** (FPS window, the stall detector's cadence memory, hysteresis
  streak, pending probe, the estimator's sample-stream transients — recent
  window / plateau clock / unconsumed distress latch) is cheap,
  display-and-moment specific, and cleared aggressively — on pause, on a
  native-DPR change, on disable, on a content change. Three deliberate
  exceptions. The first two are about the CADENCE memory, which is cleared only
  where the frame stream itself breaks (pause, display change, disable,
  dispose):
  - it survives the frame-gap reset it drives — that reset is not a session
    boundary, and forgetting the cadence there is what made the old rule unable
    to converge;
  - it survives a content change too, which is not a stream boundary either:
    frames keep arriving at the same rate while new content uploads. A cold
    memory falls back to the absolute floor alone, so wiping it would misread
    the very next ordinary interval of any loop slower than
    `1000/gapResetMs` fps as dead time — clearing the freshly-restarted FPS
    window a second time and voiding the in-flight probe.

  The third is the PENDING PROBE on a content change: it is voided only when the
  loop can actually run the replacement experiment (at least two frames per
  `probeWindowMs`). Below that every armed probe was voided before it could ever
  be judged, so a slow loop keeps its probe and settles it as confounded — see
  below.

- **Cadence trust.** Once the median absorbs a run of dead intervals as "the
  frame rate" (by the third in a row, half the memory is dead time), the window
  it lands in can hold nothing but dead time, so the manager treats the next
  couple of intervals as unrepresentative — exactly like data-loading jank:
  scale-downs still apply, but no probe is armed or settled and the estimator is
  not fed. The guarantee is BOUNDED and counted in INTERVALS: a burst of up to
  four teaches nothing, and past that the "burst" is a slow regime the manager
  must be free to learn from. Intervals is the only form the guarantee takes:
  what it buys in wall clock depends on how long they are and on how much of its
  own window a probe can still gather, so it is measured per cadence rather than
  stated as a number (measured with the production 0.9 step, for consecutive
  400 ms hitches inside a 60 fps session: eight, 3.2 s, leave the floor
  untouched; nine, 3.6 s, pin one).
- **Confounded probes.** A probe whose measurement window a content change ran
  through compared two different scenes, so its verdict may not be acted on in
  either direction and teaches nothing at all: the reduction is kept, the DPR is
  never reverted upward, no floor is pinned, and the backoff streak is untouched.
  The walk is NOT held — under sustained churn no clean experiment exists, so the
  ratio keeps descending unratified (bounded by the floor, lifted again by the
  normal scale-up hysteresis once content settles); what discarding the verdict
  buys is the absence of up/down thrash and of 30 s floors pinned on meaningless
  comparisons.
- **Learned state** (the bounds ledger) is expensive evidence and survives
  pauses and idle restores; only a display change or disable wipes it, and
  TTLs decay it.

Tests: `src/tests/unit/rendering/adaptive-dpr/` (one file per module) plus the
manager-level integration suite in `src/tests/unit/rendering/adaptive-dpr-manager.test.ts`.
