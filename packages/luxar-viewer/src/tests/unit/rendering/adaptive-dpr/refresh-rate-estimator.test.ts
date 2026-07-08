import { describe, it, expect } from 'vitest';
import { RefreshRateEstimator } from '../../../../rendering/adaptive-dpr/refresh-rate-estimator';

/** Feed `count` samples of `fps` at 500ms cadence starting at `start`. */
function feed(
  est: RefreshRateEstimator,
  fps: number,
  count: number,
  start: number,
  jitter = 0
): number {
  let t = start;
  for (let i = 0; i < count; i++) {
    t = start + i * 500;
    est.addSample(fps + (i % 2 === 0 ? jitter : -jitter), t);
  }
  return t;
}

describe('RefreshRateEstimator', () => {
  it('returns the fallback before any samples', () => {
    const est = new RefreshRateEstimator(60);
    expect(est.getCap()).toBe(60);
  });

  it('learns a higher cap immediately from a single fast sample (120Hz display)', () => {
    const est = new RefreshRateEstimator(60);
    est.addSample(119.5, 0);
    expect(est.getCap()).toBeCloseTo(119.5, 5);
  });

  it('never lets a heavy scene drag the cap below the fallback', () => {
    const est = new RefreshRateEstimator(60);
    // A session that STARTS heavy: minutes of stable 20fps. The cap must
    // hold at the fallback so scale-down stays armed.
    feed(est, 20, 8, 0, 3); // 17..23 — spread 26% > 15%, NOT uniform
    expect(est.getCap()).toBe(60);
  });

  it('a steady heavy scene is NEVER misclassified as a throttled display (proven-rate guard)', () => {
    const est = new RefreshRateEstimator(60);
    // GPU-bound scene parked at the DPR floor: perfectly uniform 20fps
    // for far longer than the 10s downshift delay. The display never
    // proved it can beat 80% of the fallback, so the throttle detector
    // must stay cold — otherwise the cap collapses to 20, the relative
    // thresholds read 20fps as healthy (down 15 / up 18), scale-down is
    // disarmed and scale-up fires on the scene that most needs help.
    feed(est, 20, 60, 0); // 30s of uniform 20fps
    expect(est.getCap()).toBe(60);
  });

  it('keeps the high-water mark under subsequent load (60fps session, then heavy)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 61, 4, 0);
    feed(est, 20, 8, 2000, 3); // heavy but jittery
    expect(est.getCap()).toBeCloseTo(61, 5);
  });

  it('downshifts after a sustained uniform-low signature (120Hz → 30Hz throttle)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    expect(est.getCap()).toBe(120);

    // rAF throttled to 30: perfectly uniform 30fps samples. Uniform-low
    // must persist 10s beyond the first detection before the downshift.
    const t = feed(est, 30, 25, 2000); // 12s of uniform 30
    void t;
    expect(est.getCap()).toBe(30);
  });

  it('recovers instantly when the throttle lifts', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    feed(est, 30, 25, 2000); // downshifted to 30
    expect(est.getCap()).toBe(30);

    est.addSample(118, 20_000); // throttle lifted
    expect(est.getCap()).toBe(118);
  });

  it('a session throttled from the very first frame keeps the fallback cap (never proved a rate)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 30, 25, 0); // 30Hz-throttled from the start — mark never beat 48
    // Without a proven higher rate the uniform-low signature is
    // indistinguishable from a heavy scene, so no downshift: the cap
    // holds at the fallback and the (futile) scale-down probes are
    // contained by the rejection-backoff machinery instead.
    expect(est.getCap()).toBe(60);
  });

  it('a downshifted sub-fallback cap survives slow samples until a fast one appears', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0); // prove the display can do 120
    feed(est, 30, 25, 2000); // genuine 120→30 throttle → downshift
    expect(est.getCap()).toBe(30);

    // Still throttled: a slightly higher-but-slow sample doesn't restore
    // the fallback bound (48 = 0.8 × 60 is the un-throttle line).
    est.addSample(35, 20_000);
    expect(est.getCap()).toBeLessThan(60);
  });

  it('clear() forgets the mark and the throttle state', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    est.clear();
    expect(est.getCap()).toBe(60);
  });

  it('light-then-heavy across a content change is NOT misclassified as a throttle', () => {
    const est = new RefreshRateEstimator(60);
    // Scene loads light: the display proves ~60 (mark pinned >= 48)...
    feed(est, 61, 4, 0);
    expect(est.getCap()).toBeCloseTo(61, 5);

    // ...then the user navigates into dense data (content change) and
    // the render parks at a steady uniform 20fps. The proof was earned
    // on the OLD content — it must not license a throttle downshift on
    // the new one, or the cap collapses to 20 and the thresholds invert
    // (scale-up armed on a GPU-bound scene).
    est.noteContentChanged();
    feed(est, 20, 60, 10_000); // 30s of uniform 20fps
    expect(est.getCap()).toBeCloseTo(61, 5); // mark kept; NO downshift
  });

  it('re-proving the rate after a content change re-arms the throttle downshift', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    est.noteContentChanged();
    feed(est, 120, 4, 5000); // new content also renders light — re-proven
    feed(est, 30, 25, 10_000); // genuine throttle signature
    expect(est.getCap()).toBe(30); // downshift correctly allowed again
  });

  it('a content change resets an already-latched throttle verdict (not just the proof)', () => {
    const est = new RefreshRateEstimator(60);
    // Genuine throttle detected on the OLD content: 120 proven, then
    // uniform 30 → latched, cap collapsed to the plateau.
    feed(est, 120, 4, 0);
    feed(est, 30, 25, 2000);
    expect(est.getCap()).toBe(30);

    // Content change: the verdict was earned against the old content's
    // frame stream. Carrying it forward would keep the cap collapsed
    // for the NEW content — a heavy scene at 35fps would then read as
    // healthy (upThr 31.5) and get scaled UP, with no reachable exit
    // sample (35 < 48). The latch must reset with the proof.
    est.noteContentChanged();
    expect(est.getCap()).toBe(60);

    // And a heavy new scene keeps the sane fallback-floored cap.
    feed(est, 35, 30, 20_000);
    expect(est.getCap()).toBe(60);
  });

  it('recovers from a 30Hz-class plateau at 25% above it — no dead zone up to 48fps', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    feed(est, 30, 25, 2000); // latched at plateau 30 (exit line 37.5)
    expect(est.getCap()).toBe(30);

    // 38fps is unreachable under a genuine tight 30Hz throttle but well
    // below 80% of the fallback (48) — the old mark/0.55 exit line was
    // inert here. The plateau-anchored line must fire.
    est.addSample(38, 60_000);
    expect(est.getCap()).toBe(60);
  });

  it('a mis-capped scene recovers as soon as FPS clearly exceeds the throttle plateau', () => {
    const est = new RefreshRateEstimator(60);
    // Proven at 60, then (no content signal) a uniform regime in the
    // 24-48fps ambiguity band earns a downshift — the residual
    // rotation-into-heavy ambiguity.
    feed(est, 61, 4, 0);
    feed(est, 30, 60, 2000);
    expect(est.getCap()).toBe(30);

    // The scene lightens a little: 45fps is impossible under a genuine
    // 30Hz throttle (> plateau × 1.25 = 37.5), so the estimator must
    // un-throttle immediately — not wait for 48 (80% of fallback),
    // which a still-heavy scene may never reach.
    est.addSample(45, 60_000);
    expect(est.getCap()).toBe(60);
  });

  // ── Sub-throttle distress (plateau < MIN_THROTTLE_PLATEAU = 22) ──

  it('a sub-throttle plateau NEVER collapses the cap, even with a proven rate — it raises distress', () => {
    const est = new RefreshRateEstimator(60);
    // The pre-fix catastrophic latch: 120 proven, then a GPU-bound
    // scene parks at uniform 10fps. The old detector latched the cap
    // onto 10, the relative thresholds then read 10fps as healthy, and
    // the manager scaled UP to native and parked there forever.
    feed(est, 120, 4, 0);
    feed(est, 10, 30, 2000); // 15s of uniform 10fps
    expect(est.getCap()).toBe(120); // cap intact — scale-down stays armed
    expect(est.consumeDistress()).toBe(true); // distress raised instead
    expect(est.consumeDistress()).toBe(false); // one-shot until re-earned
  });

  it('distress fires without a proven rate (heavy from the very first frame)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 10, 30, 0); // 15s of 10fps, nothing ever proven
    expect(est.getCap()).toBe(60);
    expect(est.consumeDistress()).toBe(true);
  });

  it('distress tolerates jitter — no uniformity requirement below the plateau line', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 12, 30, 0, 4); // 8..16fps — spread ~50%, all < 22
    expect(est.consumeDistress()).toBe(true);
  });

  it('spiky FPS straddling the plateau line does NOT fire distress (max governs, not min)', () => {
    // Alternating 20/40: dips below 22 but the scene demonstrably still
    // reaches 40fps — not sustained distress. The normal relative
    // scale-down machinery (cap stays 60) handles this regime instead.
    const spikyOnly = new RefreshRateEstimator(60);
    feed(spikyOnly, 30, 30, 0, 10); // 20..40 for ~15s
    expect(spikyOnly.consumeDistress()).toBe(false);
    expect(spikyOnly.getCap()).toBe(60);

    // A spiky phase must not BANK latch credit either: if the scene
    // then turns genuinely low, the verdict still requires the full
    // sustained period from that point. No consume in between — a
    // min-governs mutant would have latched during the straddle and
    // the banked verdict would pass re-validation on the fresh low
    // window without ever meeting the sustained requirement.
    const est = new RefreshRateEstimator(60);
    const t = feed(est, 30, 30, 0, 10); // spiky straddle, unconsumed
    feed(est, 20, 4, t + 500); // 4 low windows ≈ 2s — far under 10s
    expect(est.consumeDistress()).toBe(false);
  });

  it('distress needs the sustained period — a brief dip does not fire', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 10, 12, 0); // only ~5.5s below the line
    expect(est.consumeDistress()).toBe(false);
  });

  it('distress re-arms and re-fires while the scene stays distressed', () => {
    const est = new RefreshRateEstimator(60);
    const t = feed(est, 10, 30, 0);
    expect(est.consumeDistress()).toBe(true);
    feed(est, 10, 30, t + 500);
    expect(est.consumeDistress()).toBe(true);
  });

  it('a content change clears an unconsumed distress verdict', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 10, 30, 0);
    est.noteContentChanged();
    expect(est.consumeDistress()).toBe(false);
  });

  it('a low-band grind (21fps) IS distress — the lower edge of the line is pinned', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 21, 30, 0); // 15s of uniform 21fps — just under the 22 line
    expect(est.consumeDistress()).toBe(true);
  });

  it('uniform ~23fps heavy content NEVER latches the throttle (no real display runs that slow)', () => {
    const est = new RefreshRateEstimator(60);
    // The regression the MIN_REAL_DISPLAY_RATE floor closes: a proven
    // -fast display grinding at uniform 23fps sits between the distress
    // line (22) and the slowest real display mode (23.976). Without the
    // floor, the throttle signature latches (23 < 0.55×120, uniform),
    // collapsing the cap onto 23 and re-opening the parked-at-native
    // inversion one band up from the one the distress verdict closed.
    feed(est, 120, 4, 0);
    feed(est, 23, 60, 2000); // 30s of uniform 23fps
    expect(est.getCap()).toBe(120); // cap intact — no latch
    expect(est.consumeDistress()).toBe(false); // and not distress either
  });

  it('a genuine 120Hz → 23.976Hz display-mode throttle still latches', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    feed(est, 23.9, 60, 2000); // dragged onto a 24Hz TV: uniform ~23.976
    expect(est.getCap()).toBeCloseTo(23.9, 5); // real display rate — latched
  });

  it('a healthy 24Hz film/TV display mode (23.976fps) is NOT distress', () => {
    const est = new RefreshRateEstimator(60);
    // Macs driving 4K TVs over HDMI 1.4 run the whole desktop at
    // ~23.976Hz — a vsync-bound session there is healthy, not heavy.
    feed(est, 23.9, 40, 0); // 20s of uniform ~24fps
    expect(est.consumeDistress()).toBe(false);
    expect(est.getCap()).toBe(60); // and no throttle latch (unproven)
  });

  it('a stale latched verdict is dropped at consumption if the window has recovered', () => {
    const est = new RefreshRateEstimator(60);
    // Verdict latches during a grind but is not consumed (probe in
    // flight / load suppression on the manager side)...
    const t = feed(est, 10, 30, 0);
    // ...then the workload lightens BEFORE the next clean tick.
    feed(est, 120, 4, t + 500);
    // Consumption re-validates against the live window: no demotion
    // of a session that has since recovered.
    expect(est.consumeDistress()).toBe(false);
    // And the latch is spent — it does not linger for a later tick.
    expect(est.consumeDistress()).toBe(false);
  });

  it('noteSessionInterrupted() stops pause dead-time from counting as "sustained"', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 10, 12, 0); // ~5.5s below the line — under the 10s bar
    est.noteSessionInterrupted(); // idle pause / gap reset
    // Two minutes later a single janky post-resume window arrives.
    // Without the reset, the stale plateau clock (wall-clock 120s) and
    // stale recent[] samples would fire distress immediately.
    est.addSample(15, 120_000);
    expect(est.consumeDistress()).toBe(false);
  });

  it('noteSessionInterrupted() clears an unconsumed distress latch but keeps learned state', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0); // prove the display (learned mark)
    feed(est, 10, 30, 2000); // latch distress
    est.noteSessionInterrupted();
    expect(est.consumeDistress()).toBe(false);
    expect(est.getCap()).toBe(120); // high-water mark survives
  });

  it('noteSessionInterrupted() keeps a latched THROTTLE verdict (learned, not session, state)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    feed(est, 30, 25, 2000); // genuine 120→30 throttle → cap collapsed
    expect(est.getCap()).toBe(30);
    est.noteSessionInterrupted(); // idle pause
    // The throttle verdict describes the display, not the interrupted
    // sample stream — the collapsed cap must survive the pause.
    expect(est.getCap()).toBe(30);
  });

  it('a 30Hz-class plateau is still a throttle, not distress (above the plateau line)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    feed(est, 30, 25, 2000);
    expect(est.getCap()).toBe(30); // genuine throttle downshift kept
    expect(est.consumeDistress()).toBe(false);
  });
});
