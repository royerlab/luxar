/**
 * Test helpers for the two globals the DPR machinery reads: the
 * display's `window.devicePixelRatio` and the viewer's pixel-ratio cap
 * (`rendering/pixel-ratio-cap`).
 *
 * Both were previously stubbed by four near-identical local copies
 * across the DPR test files. They are shared here because the cap made
 * the pairing load-bearing: nearly every DPR assertion now depends on
 * BOTH values, and a test that sets one without the other is asserting
 * against a state the viewer never reaches.
 *
 * `window.devicePixelRatio` is not writable in jsdom, hence
 * `Object.defineProperty` rather than assignment.
 */

import { getMaxPixelRatioCap, setMaxPixelRatioCap } from '../../rendering/pixel-ratio-cap';

/**
 * Set `window.devicePixelRatio`, returning a restore closure.
 *
 * @param value - the DPR the display should report
 * @returns a function that puts the original value back
 */
export function setNativeDPR(value: number): () => void {
  const original = window.devicePixelRatio;
  Object.defineProperty(window, 'devicePixelRatio', {
    configurable: true,
    value,
    writable: true,
  });
  return () => {
    Object.defineProperty(window, 'devicePixelRatio', {
      configurable: true,
      value: original,
      writable: true,
    });
  };
}

/** Run `fn` with `window.devicePixelRatio` set to `dpr`, then restore. */
export function withNativeDPR<T>(dpr: number, fn: () => T): T {
  const restore = setNativeDPR(dpr);
  try {
    return fn();
  } finally {
    restore();
  }
}

/**
 * Lift the pixel-ratio cap so the viewer may render at the display's
 * full DPR, returning a restore closure.
 *
 * Most pre-existing DPR tests were written against the uncapped
 * behaviour and assert against native values throughout; they call this
 * in `beforeEach` so their coverage of the adaptive loop, the U-shape
 * probe and the learned ceiling demotion (a 2.0 -> 1.0 choreography that
 * is unreachable, and so silently vacuous, when the cap already pins
 * 1.0) survives the change of default.
 */
export function allowHighDPR(): () => void {
  const original = getMaxPixelRatioCap();
  setMaxPixelRatioCap(Infinity);
  return () => setMaxPixelRatioCap(original);
}

/** Run `fn` with high DPR allowed, then restore the previous cap. */
export function withHighDPRAllowed<T>(fn: () => T): T {
  const restore = allowHighDPR();
  try {
    return fn();
  } finally {
    restore();
  }
}
