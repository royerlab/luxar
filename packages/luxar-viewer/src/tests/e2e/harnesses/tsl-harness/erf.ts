/**
 * Shared-math family for the TSL ↔ GLSL parity harness: the erf
 * polynomial (`materials/_shared/erf.ts`). 1 registry entry.
 *
 * The GLSL side injects `GLSL_ERF_FUNCTIONS` verbatim; the TSL side
 * builds the same polynomial through `erfPolyTSL`. Both encode
 * `(erf(x) + 1) / 2` for `x` swept across [-4, 4] (covering both clamp
 * regions and the full transition), so ANY structural divergence
 * between the two builders — Horner order, sign handling, clamp radius,
 * a drifted coefficient — shows up as a pixel difference. This is the
 * value-level backend-parity guarantee the module header promises
 * (literal formatting is the code generator's; see `_shared/erf.ts`).
 *
 * @module tests/e2e/harnesses/tsl-harness/erf
 */

import { NodeMaterial } from 'three/webgpu';
import { vec4, float, positionGeometry } from 'three/tsl';
import { GLSL_ERF_FUNCTIONS, erfPolyTSL } from '../../../../rendering/materials/_shared/erf';
import type { ShaderSource } from '../../../../rendering/materials/_shared/shader-source';
import type { RegistryEntry } from './types';

const ERF_SOURCE: ShaderSource = {
  name: 'erf',
  webgl: {
    vertex: /* glsl */ `
      out float vX;
      void main() {
        vX = position.x * 4.0;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragment: /* glsl */ `
      precision highp float;
      in float vX;
      out vec4 fragColor;
      ${GLSL_ERF_FUNCTIONS}
      void main() {
        float v01 = (luxarErf(vX) + 1.0) * 0.5;
        fragColor = vec4(v01, v01, v01, 1.0);
      }
    `,
  },
  webgpu: () => {
    const m = new NodeMaterial();
    const x = positionGeometry.x.mul(4.0);
    const v01 = erfPolyTSL(x).add(1.0).mul(0.5);
    m.fragmentNode = vec4(v01, v01, v01, float(1.0));
    m.toneMapped = false;
    m.depthTest = false;
    m.depthWrite = false;
    m.transparent = false;
    return m;
  },
};

/**
 * Registry of shared-math parity entries, merged into `SHADER_REGISTRY`
 * by `index.ts`.
 */
export const ERF_SHADERS: Record<string, RegistryEntry> = {
  erf: {
    source: ERF_SOURCE,
    buildUniforms: () => ({}),
  },
};
