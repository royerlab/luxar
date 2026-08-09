/**
 * Shared-math family for the TSL ↔ GLSL parity harness: the #1352
 * volumetric-line ray integral (`materials/line/ray-integral.ts`).
 * 1 registry entry.
 *
 * The GLSL side injects `GLSL_ERF_FUNCTIONS` + the module's own
 * `GLSL_LINE_RAY_INTEGRAL_FUNCTIONS` verbatim; the TSL side builds the
 * same three functions through `lineRayIntegralTSL`,
 * `raySegmentDistanceTSL` and `lineCapsuleProfileTSL`. This entry exists
 * because sharing CONSTANTS between the two backends does not constrain
 * a hand-written expression TREE: without a rendered comparison the TSL
 * builder can carry an inverted lane select, a reversed `step` or a
 * dropped 1/sqrt2 and every CPU-side test still passes (the unit suite
 * pins the GLSL string against the TS mirror, and neither touches the
 * node graph). The float32 finiteness guards are NOT visible here —
 * they only ever alter the mix arm that is multiplied by zero, so the
 * codegen snapshot pins their presence in the graph and the unit
 * suite's `Math.fround` harness pins their behaviour.
 *
 * The sweep is deliberately two-dimensional and covers BOTH numerical
 * lanes, since the lane select is one of the things that can drift:
 *
 *   - `x` drives the erf-window gap `Δ ∈ [0, 3]`, crossing the
 *     `LINE_WINDOW_GAP_THRESHOLD = 0.5` lane boundary at x = -2/3 — on a
 *     [-1, 1] quad that is the leftmost sixth, measured as 11 of the 64
 *     columns (704 of 4096 pixels) in the derivative lane and the rest
 *     in the closed form, so both lanes are amply covered;
 *   - `y` drives the window midpoint `M ∈ [-3, 3]`, i.e. the ray's axial
 *     closest approach sweeping from well before the segment start to
 *     well past its end — which also drives the capsule profile's
 *     `clamp(s*, 0, L)` through both of its clamped arms;
 *   - the radial distance is a nonzero constant so the `exp(-(D·k)²)`
 *     factor is live rather than an identity.
 *
 * Red channel = the sum-mode ray integral, green = the peak-mode capsule
 * profile fed by `raySegmentDistance`, so a single frame exercises all
 * three TSL builders and a divergence in any one of them separates the
 * two backends' pixels.
 *
 * @module tests/e2e/harnesses/tsl-harness/line-ray-integral
 */

import { NodeMaterial } from 'three/webgpu';
import { vec4, float, positionGeometry } from 'three/tsl';
import { GLSL_ERF_FUNCTIONS } from '../../../../rendering/materials/_shared/erf';
import {
  GLSL_LINE_RAY_INTEGRAL_FUNCTIONS,
  LINE_INV_SQRT2,
  lineCapsuleProfileTSL,
  lineRayIntegralTSL,
  raySegmentDistanceTSL,
} from '../../../../rendering/materials/line/ray-integral';
import type { ShaderSource } from '../../../../rendering/materials/_shared/shader-source';
import type { RegistryEntry } from './types';

/**
 * Harness scaffolding shared by both backends. `sigma = 1`, so the
 * module's internal `k = LINE_INV_SQRT2 / sigma` is exactly
 * {@link LINE_INV_SQRT2} and the (Δ, M) targets below invert cleanly to
 * the `(segLength, sStar)` the entry points actually take.
 */
const SIGMA = 1;
/** Radial closest approach — nonzero so the `exp(-(D·k)²)` factor is live. */
const DISTANCE_TO_AXIS = 0.6;
/** Δ sweep half-range: x ∈ [-1, 1] maps to Δ ∈ [0, 3]. */
const GAP_SCALE = 1.5;
/** M sweep half-range: y ∈ [-1, 1] maps to M ∈ [-3, 3]. */
const MIDPOINT_SCALE = 3.0;

const glslNum = (v: number) => v.toFixed(9);

const LINE_RAY_INTEGRAL_SOURCE: ShaderSource = {
  name: 'line-ray-integral',
  webgl: {
    vertex: /* glsl */ `
      out vec2 vSweep;
      void main() {
        vSweep = position.xy;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragment: /* glsl */ `
      precision highp float;
      in vec2 vSweep;
      out vec4 fragColor;
      ${GLSL_ERF_FUNCTIONS}
      ${GLSL_LINE_RAY_INTEGRAL_FUNCTIONS}
      void main() {
        float k = ${glslNum(LINE_INV_SQRT2)};
        float gapTarget = (vSweep.x + 1.0) * ${glslNum(GAP_SCALE)};
        float midTarget = vSweep.y * ${glslNum(MIDPOINT_SCALE)};
        float segLength = gapTarget / k;
        float sStar = segLength * 0.5 + midTarget / k;
        float d = ${glslNum(DISTANCE_TO_AXIS)};
        float sum = luxarLineRayIntegral(${glslNum(SIGMA)}, segLength, d, sStar, 1.0);
        float dist = luxarRaySegmentDistance(segLength, d, sStar, 1.0);
        float peak = luxarLineCapsuleProfile(${glslNum(SIGMA)}, dist);
        fragColor = vec4(sum, peak, 0.0, 1.0);
      }
    `,
  },
  webgpu: () => {
    const material = new NodeMaterial();
    const k = float(LINE_INV_SQRT2);
    const gapTarget = positionGeometry.x.add(1.0).mul(GAP_SCALE);
    const midTarget = positionGeometry.y.mul(MIDPOINT_SCALE);
    const segLength = gapTarget.div(k);
    const sStar = segLength.mul(0.5).add(midTarget.div(k));
    const distanceToAxis = float(DISTANCE_TO_AXIS);
    const absU = float(1.0);
    const sum = lineRayIntegralTSL({
      sigma: float(SIGMA),
      length: segLength,
      distanceToAxis,
      sStar,
      absU,
    });
    const dist = raySegmentDistanceTSL(segLength, distanceToAxis, sStar, absU);
    const peak = lineCapsuleProfileTSL(float(SIGMA), dist);
    material.fragmentNode = vec4(sum, peak, float(0.0), float(1.0));
    material.toneMapped = false;
    material.depthTest = false;
    material.depthWrite = false;
    material.transparent = false;
    return material;
  },
};

/**
 * Registry of volumetric-line-math parity entries, merged into
 * `SHADER_REGISTRY` by `index.ts`.
 */
export const LINE_RAY_INTEGRAL_SHADERS: Record<string, RegistryEntry> = {
  'line-ray-integral': {
    source: LINE_RAY_INTEGRAL_SOURCE,
    buildUniforms: () => ({}),
  },
};
