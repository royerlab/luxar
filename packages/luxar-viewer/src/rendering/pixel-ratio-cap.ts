/**
 * The pixel-ratio cap — the maximum device pixel ratio the viewer will
 * ever render at.
 *
 * HiDPI rendering is expensive out of all proportion to what it buys on
 * soft-edged emissive geometry: a 2x display is 4x the fragment work for
 * a scene made of points and gsplats. So the viewer renders at
 * CSS resolution (DPR 1.0) unless high DPR is explicitly allowed —
 * `renderingControls.defaults.allowHighDPR`, authorable per scene as
 * `viewer_config.allow_high_dpr`.
 *
 * # What the cap costs, measured
 *
 * On a Retina panel against DPR 2: mean luminance, lit coverage and p99
 * luminance all hold to within 2.5% on every geometry type — the
 * minimum-drawn-size widening and the shader's energy compensation very
 * nearly cancel. The entire visible effect is a 15-35% loss of
 * high-frequency detail.
 *
 * That is mild on points and gsplats, whose sprites are soft to begin
 * with. It is NOT mild on dense thin lines: a river network or a
 * tractogram stops being separable, which loses information rather than
 * polish. Those scenes are also the least fill-bound, so they gain least
 * from the cap in the first place (measured 1.06-1.17x, against 2.6-2.7x
 * on a point cloud) — which makes them both the place the cap hurts most
 * and the cheapest place to switch it off. Line-dominant demos author
 * `allow_high_dpr=True` for exactly that reason.
 *
 * # Why this is its own module, and why it lives in `rendering/`
 *
 * The cap is read by modules in two different layers —
 * `AdaptiveDPRManager` here, and `scene/scene-manager/viewport/dpr-policy`
 * — and dependency-cruiser's layer rules forbid `rendering/` from
 * importing `scene/`. A leaf module in the lower of the two layers is the
 * only place both can reach. It imports only `utils/input-capabilities`,
 * which is itself import-free, so it cannot take part in a cycle.
 *
 * `dpr-policy.getActivePixelRatio` is what actually applies the cap, and
 * that is the ONE function the renderer boundary calls (see
 * resize-orchestrator). Enforcing it there makes it structurally
 * impossible for a DPR above the cap to reach the renderer: the opening
 * frame, every resize and every monitor drag are covered, without
 * threading a flag through ResizeCtx.
 *
 * # The three writers
 *
 * The cap is stored as a NUMBER rather than the boolean it usually comes
 * from, because two callers besides the setting legitimately need to
 * raise it. **All three, listed here so they stay reviewable — if you
 * add a fourth, document it here:**
 *
 * 1. The `allowHighDPR` setting — `Infinity` when allowed, else 1.0.
 *    The normal path (`RenderingControls`, on load and on toggle). On a
 *    MOBILE device class "allowed" resolves to {@link MOBILE_MAX_PIXEL_RATIO}
 *    instead of `Infinity`: a scene authored with `allow_high_dpr` is
 *    tuned for a desktop Retina panel, and on a DPR-3 phone the same flag
 *    means 9× the fragment work of CSS resolution from the very first
 *    frame — before the adaptive loop's evaluation interval can react.
 *    Measured under iPhone emulation on the protein-stories scene: DPR
 *    3.00 into a 2340×3984 target, then a slow walk down at 0.8–3.9 FPS.
 * 2. `AdaptiveDPRManager.pinManualDPR` (the `?dpr=` URL param) raises the
 *    cap to the pinned value for the session, so `?dpr=2` renders at 2
 *    even with the setting off. An explicit request wins, and
 *    deterministic E2E / repro pins keep working.
 * 3. `RecordingSession.saveRecordingState` / `restoreRecordingState` lift
 *    the cap to the recording panel's `captureDPR` for the duration of a
 *    capture and put it back afterwards. Without a hatch at this level
 *    the seam would clamp an explicitly requested high-DPR export back
 *    down to whatever is on screen.
 */

import { getInputProfile } from '../utils/input-capabilities';

/**
 * The cap applied when high DPR is not allowed: exactly CSS resolution.
 *
 * 1.0 is a Schelling point, not a tuned estimate — it is what every
 * non-HiDPI display renders, and it is the same value
 * `BoundsLedger.dprCeiling` demotes to when the adaptive loop earns that
 * conclusion the hard way from FPS evidence. This setting is that
 * demotion made the default.
 */
export const DEFAULT_MAX_PIXEL_RATIO = 1.0;

/**
 * The cap `allowHighDPR` resolves to on a MOBILE device class (phones and
 * tablets, iPhone and iPad included). 2 keeps the high-frequency detail the
 * setting exists for on the line-dominant scenes that author it, at 4× the
 * fragment work of CSS resolution rather than the 9× a DPR-3 panel would
 * otherwise take. Explicit requests (`?dpr=`, the recording panel's
 * `captureDPR`) are not clamped by this — see the writer list above.
 */
export const MOBILE_MAX_PIXEL_RATIO = 2;

/**
 * The cap, as an absolute DPR. `Infinity` means "no cap — use whatever
 * the display offers".
 *
 * Defaults to `DEFAULT_MAX_PIXEL_RATIO` so a viewer that never touches
 * the setting (an embedder, a unit test constructing a SceneManager
 * directly) still gets the cheap default rather than silently
 * supersampling.
 */
let maxPixelRatioCap: number = DEFAULT_MAX_PIXEL_RATIO;

/**
 * Uncapped `window.devicePixelRatio`, with the 0/undefined guard.
 *
 * Also guards `window` itself, so a node test or the published library bundle
 * under SSR gets a missing display rather than a ReferenceError from a bare
 * `window.devicePixelRatio`. 1 is the honest answer when there is no display
 * at all: CSS resolution, which is what the cap defaults to anyway.
 */
export function getNativePixelRatio(): number {
  return (typeof window === 'undefined' ? 1 : window.devicePixelRatio) || 1;
}

/**
 * Set the cap. See the three legitimate writers in the module doc.
 *
 * A non-positive or NaN value falls back to the default cap rather than
 * disabling the ceiling by accident.
 */
export function setMaxPixelRatioCap(cap: number): void {
  maxPixelRatioCap = cap > 0 ? cap : DEFAULT_MAX_PIXEL_RATIO;
}

/** The stored cap, before clamping to what the display actually offers. */
export function getMaxPixelRatioCap(): number {
  return maxPixelRatioCap;
}

/**
 * The cap "high DPR allowed" resolves to on this device: no cap on a
 * laptop/desktop, {@link MOBILE_MAX_PIXEL_RATIO} on a phone or tablet.
 */
export function deviceHighDprCeiling(): number {
  return getInputProfile().deviceClass === 'mobile' ? MOBILE_MAX_PIXEL_RATIO : Infinity;
}

/**
 * Convenience for the common case: allow high DPR (up to the device's
 * ceiling — see {@link deviceHighDprCeiling}) or not (CSS resolution).
 */
export function setHighDPRAllowed(allowed: boolean): void {
  setMaxPixelRatioCap(allowed ? deviceHighDprCeiling() : DEFAULT_MAX_PIXEL_RATIO);
}

/**
 * True when the cap is not restricting the display's own DPR.
 *
 * Derived from the cap rather than stored separately, so the two can
 * never disagree — but note a session that raised the cap via a `?dpr=2`
 * pin also reads as "allowed" here. That is deliberate: the question this
 * answers is "may the viewer render above CSS resolution", and after an
 * explicit pin the answer genuinely is yes.
 */
export function isHighDPRAllowed(): boolean {
  return maxPixelRatioCap > DEFAULT_MAX_PIXEL_RATIO;
}

/**
 * The effective ceiling: the display's own DPR or the cap, whichever is
 * lower. A cap never RAISES the pixel ratio above what the display has.
 */
export function getMaxPixelRatio(): number {
  return Math.min(getNativePixelRatio(), maxPixelRatioCap);
}
