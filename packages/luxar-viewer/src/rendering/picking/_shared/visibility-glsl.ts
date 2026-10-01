/**
 * GLSL pick visibility weight: the per-element factors the VISUAL pass scales
 * an element's contribution by, which therefore decide whether it can be seen
 * at all — and so whether the pick pass may report it.
 *
 * A pick shader re-derives its element's falloff from the geometry, but the
 * draw also multiplies it by
 *
 *   - the per-element alpha — a linear contribution scale in every mode, and
 *     mapped into optical depth `w(a) = −ln(1 − a)` under `volumetric` (gated
 *     on `uHasElementAlpha`, so RGB data's identity 1.0 stays 1.0);
 *   - the node opacity `uOpacity`;
 *   - the gain: the visual discards are gain-aware through `max(uIntensity, 1)`
 *     (a dim element a high gain lifts above the floor stays visible; a gain
 *     below 1 never moves the threshold).
 *
 * Without them the pick pass treated a fully transparent floater as fully
 * visible (it won tooltips) and a dim element under a high gain as invisible
 * (it could not be picked). The four uniforms are synced from the visual
 * material per pick render (`picking-system/visibility-sync.ts`).
 *
 * Declare the snippet in the ONE stage that evaluates the weight (gsplat: the
 * vertex stage, which also sizes its quad from it; points/lines: the fragment
 * stage, which already holds the per-element alpha the draw uses).
 *
 * TSL twin: `./visibility-tsl.ts`.
 *
 * @module rendering/picking/_shared/visibility-glsl
 */
import { ALPHA_CLAMP } from '../../materials/_shared/volumetric';

export const GLSL_PICK_VISIBILITY = /* glsl */ `
// Pick visibility inputs, synced from the visual material per pick render.
uniform mediump float uIntensity;   // node gain
uniform mediump float uOpacity;     // node opacity
uniform lowp float uHasElementAlpha; // 1 when colors carry a real alpha column
uniform int uVolumetric;            // 1 when the visual node blends 'volumetric'

// The factor the visual pass scales an element by for its per-element alpha.
float luxarPickAlphaFactor(float alpha) {
  return (uVolumetric == 1)
    ? mix(1.0, -log(1.0 - min(alpha, ${ALPHA_CLAMP})), uHasElementAlpha)
    : alpha;
}

// alpha factor x node opacity x the visual discards' max(gain, 1).
float luxarPickWeight(float alpha) {
  return luxarPickAlphaFactor(alpha) * uOpacity * max(uIntensity, 1.0);
}
`;
