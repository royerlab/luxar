/**
 * `LodSettleDrain` directly, without a capture session.
 *
 * The behaviours below were all already covered end-to-end through
 * `OfflineCaptureStrategy.run` (offline-capture-strategy.test.ts), and those
 * tests still pass unchanged — this file is not re-proving them. What it adds
 * is reach: the state machine's awkward corners (an abort landing mid-poll, the
 * latch re-arming, the exact-vs-vague report branch) are one object away here
 * instead of behind a driver, an animation controller and a DOM overlay, so
 * they can be provoked directly rather than staged.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  LOD_SETTLE_MAX_FRAMES,
  LodSettleDrain,
  MAX_CONSECUTIVE_LOD_TIMEOUTS,
} from '../../../../ui/recording-panel/offline-lod-settle';

vi.mock('../../../../utils/log', () => ({
  Modules: { RECORDING: 'RECORDING' },
  log: { warning: vi.fn(), error: vi.fn(), info: vi.fn() },
}));

/** A drain whose clock, predicate and liveness are all under the test's thumb. */
function makeDrain(
  answers: (boolean | null | 'throw')[] | (() => boolean | null),
  overrides: { isRecording?: () => boolean; isAborted?: () => boolean } = {}
) {
  let frames = 0;
  let call = 0;
  const isLODSettled = Array.isArray(answers)
    ? (): boolean | null => {
        const a = answers[Math.min(call++, answers.length - 1)];
        if (a === 'throw') throw new Error('predicate exploded');
        return a;
      }
    : answers;
  const drain = new LodSettleDrain({
    isLODSettled,
    isRecording: overrides.isRecording ?? ((): boolean => true),
    isAborted: overrides.isAborted ?? ((): boolean => false),
    nextFrame: async (): Promise<void> => {
      frames++;
    },
  });
  return { drain, frames: () => frames, calls: () => call };
}

describe('LodSettleDrain', () => {
  beforeEach(() => vi.clearAllMocks());

  it('spends no frame at all when the scene has no LOD groups to wait for', async () => {
    const { drain, frames } = makeDrain([null, null, null]);
    await drain.waitForFrame();
    // Not even the mandatory selector-catch-up tick: a plain points scene must
    // cost exactly what it did before the drain existed.
    expect(frames()).toBe(0);
    expect(drain.settleTimeouts).toBe(0);
  });

  it('spends exactly the one mandatory tick when the scene is already settled', async () => {
    const { drain, frames } = makeDrain([true, true]);
    await drain.waitForFrame();
    expect(frames()).toBe(1);
    expect(drain.settleTimeouts).toBe(0);
  });

  it('keeps ticking until the predicate flips to settled', async () => {
    const { drain, frames } = makeDrain([false, false, false, true]);
    await drain.waitForFrame();
    // 1 mandatory + polls until the `true`.
    expect(frames()).toBe(3);
    expect(drain.settleTimeouts).toBe(0);
  });

  it('gives up at the frame cap and counts one timeout', async () => {
    const { drain, frames } = makeDrain(() => false);
    await drain.waitForFrame();
    expect(frames()).toBe(LOD_SETTLE_MAX_FRAMES);
    expect(drain.settleTimeouts).toBe(1);
  });

  it('treats a throwing predicate as "do not wait" rather than failing the run', async () => {
    const { drain, frames } = makeDrain(['throw']);
    await expect(drain.waitForFrame()).resolves.toBeUndefined();
    expect(frames()).toBe(0);
    // A throw is not a timeout — it must not appear in the report.
    expect(drain.settleTimeouts).toBe(0);
  });

  it('does not count a timeout when a Stop breaks it out of the wait', async () => {
    let recording = true;
    const { drain, frames } = makeDrain(() => false, { isRecording: () => recording });
    // Stop lands during the mandatory tick, i.e. after the drain has committed
    // to waiting but before it has polled even once.
    const pending = drain.waitForFrame();
    recording = false;
    await pending;
    expect(frames()).toBe(1);
    // Giving up because the user stopped is not the scene failing to settle,
    // so it must not reach the end-of-run report.
    expect(drain.settleTimeouts).toBe(0);
    expect(drain.report(3, 3)).toBeNull();
  });

  it('latches the wait off after a streak, then re-arms when the scene settles', async () => {
    // Always-false: every frame times out.
    let answer: boolean | null = false;
    const { drain, frames } = makeDrain(() => answer);

    for (let i = 0; i < MAX_CONSECUTIVE_LOD_TIMEOUTS; i++) await drain.waitForFrame();
    expect(drain.settleTimeouts).toBe(MAX_CONSECUTIVE_LOD_TIMEOUTS);

    // Latched: the next frame costs a free probe and NO rAF at all.
    const before = frames();
    await drain.waitForFrame();
    expect(frames()).toBe(before);
    expect(drain.settleTimeouts).toBe(MAX_CONSECUTIVE_LOD_TIMEOUTS);

    // The free probe now reports settled -> the latch re-arms, and the frame
    // after it drains normally again.
    answer = true;
    await drain.waitForFrame();
    expect(frames()).toBe(before);
    await drain.waitForFrame();
    expect(frames()).toBe(before + 1);
  });

  it('reports nothing at all when no frame ever timed out', () => {
    const { drain } = makeDrain([true]);
    expect(drain.report(10, 10)).toBeNull();
  });

  it('reports an EXACT count while the wait stayed on for the whole run', async () => {
    const { drain } = makeDrain(() => false);
    await drain.waitForFrame();
    expect(drain.report(5, 5)).toBe('1 frame(s) captured before LOD settled');
  });

  it('refuses to claim a number once the wait was ever paused', async () => {
    const { drain } = makeDrain(() => false);
    for (let i = 0; i < MAX_CONSECUTIVE_LOD_TIMEOUTS; i++) await drain.waitForFrame();
    // The count is now meaningless — frames captured while the wait was off
    // were never counted at all — so the report must not quote one.
    const toast = drain.report(600, 600);
    expect(toast).toBe('Paused waiting for LOD — some frames may not show the settled level');
    expect(toast).not.toMatch(/\d/);
  });
});
