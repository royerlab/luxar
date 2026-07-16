/**
 * Regression: gamma + display range must operate on the scalar VALUE
 * (pre-LUT), not on the output color, when a colormap (LUT) is active.
 *
 * Symptom this guards against: changing gamma on a colormapped layer
 * warped the mapped colors (pow(LUT(t), invGamma)) instead of warping
 * which value mapped to which color (LUT(pow(t, invGamma))). The fix
 * applies gamma to the normalized scalar in the VERTEX stage before the
 * LUT lookup and skips the per-fragment color gamma pow() in colormap
 * mode. Intensity (gain) + offset are NOT part of that bypass: they
 * apply POST-LUT to the mapped color in ALL modes (matching the gsplat
 * shader) so the layer intensity/offset controls work on colormapped
 * nodes too — see the companion describe block below.
 *
 * Direct-color mode is unchanged: full GOG operates on the color.
 *
 * These assertions inspect the GLSL3 source strings (the canonical
 * reference shaders) for the structural invariant. The TSL counterparts
 * mirror them and are checked for parity by `tsl-shader-parity.spec.ts`.
 */
import { describe, it, expect } from 'vitest';
import {
  POINT_VERTEX_SHADER,
  POINT_FRAGMENT_SHADER,
} from '../../../rendering/materials/point/shader-glsl';
import {
  LINE_VERTEX_SHADER,
  LINE_FRAGMENT_SHADER,
} from '../../../rendering/materials/line/shader-glsl';
import {
  GSPLAT_VERTEX_SHADER,
  GSPLAT_FRAGMENT_SHADER,
} from '../../../rendering/materials/gsplat/shader-glsl';

/** Strip GLSL line/block comments so assertions match code, not prose. */
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/** Index of the colormap LUT lookup in a vertex shader. */
function lutLookupIndex(src: string): number {
  const i = src.indexOf('texture(uColormapTex');
  expect(i, 'expected a colormap LUT lookup in the vertex shader').toBeGreaterThan(-1);
  return i;
}

interface ShaderPair {
  name: string;
  vertex: string;
  fragment: string;
  /** Vertex uniform name for the inverse gamma. */
  invGamma: string;
}

const SHADERS: ShaderPair[] = [
  {
    name: 'points',
    vertex: POINT_VERTEX_SHADER,
    fragment: POINT_FRAGMENT_SHADER,
    invGamma: 'invGamma',
  },
  {
    name: 'lines',
    vertex: LINE_VERTEX_SHADER,
    fragment: LINE_FRAGMENT_SHADER,
    invGamma: 'uInvGamma',
  },
  {
    name: 'gsplats',
    vertex: GSPLAT_VERTEX_SHADER,
    fragment: GSPLAT_FRAGMENT_SHADER,
    invGamma: 'uInvGamma',
  },
];

describe('colormap gamma operates on the value, pre-LUT', () => {
  for (const s of SHADERS) {
    describe(s.name, () => {
      const vtx = stripComments(s.vertex);
      const frag = stripComments(s.fragment);

      it('applies gamma to the scalar BEFORE the LUT lookup (vertex stage)', () => {
        const lut = lutLookupIndex(vtx);
        const before = vtx.slice(0, lut);
        // A `pow(..., <invGamma>)` warps the normalized scalar prior to
        // the texture sample.
        const gammaPow = new RegExp(`pow\\([^;]*${s.invGamma}\\s*\\)`);
        expect(
          gammaPow.test(before),
          `${s.name}: expected gamma pow(..., ${s.invGamma}) on the value before the LUT lookup`
        ).toBe(true);
      });

      it('declares the inverse-gamma uniform in the vertex stage (colormap block)', () => {
        expect(vtx).toMatch(new RegExp(`uniform[^;]*\\b${s.invGamma}\\b`));
      });

      it('does NOT apply the color GOG pow() inside the USE_COLORMAP fragment branch', () => {
        // The fragment still has a direct-color gamma pow() — but it must
        // be gated so it does NOT run when USE_COLORMAP is defined.
        expect(frag).toMatch(/pow\(adjusted/);
        // The colormap branch must pass `adjusted` straight to the final
        // color (no pow). Verify a guarded passthrough exists.
        const passthrough = /USE_COLORMAP[\s\S]*?(finalColor|gammaColor)\s*=\s*adjusted\s*;/;
        expect(
          passthrough.test(frag),
          `${s.name}: expected a USE_COLORMAP branch that passes adjusted through without gamma pow()`
        ).toBe(true);
      });
    });
  }
});

describe('intensity/offset apply POST-LUT to the mapped color (all modes)', () => {
  // The gamma bypass above must not extend to gain/offset: the layer
  // Intensity/Offset sliders drive uIntensity/uOffset, and colormapped
  // Points/Lines must honor them exactly like GSplats do. The old
  // shaders short-circuited the whole GOG chain under USE_COLORMAP
  // (`adjusted = max(vColor, vec3(0.0))`), which made the sliders
  // no-ops on colormapped nodes.
  for (const s of SHADERS) {
    describe(s.name, () => {
      const frag = stripComments(s.fragment);

      it('applies vColor * uIntensity + uOffset unconditionally', () => {
        expect(
          /vColor\s*\*\s*uIntensity\s*\+\s*uOffset/.test(frag),
          `${s.name}: expected the GOG gain/offset chain on vColor in the fragment shader`
        ).toBe(true);
        // The gain/offset chain must not be gated on colormap mode. The
        // only permitted compile-out is the line shader's LUXAR_NO_GOG
        // identity fast path (stamped solely from intensity==1 &&
        // offset==0, never from colormap state).
        const gogIndex = frag.search(/vColor\s*\*\s*uIntensity\s*\+\s*uOffset/);
        const directivesBefore = frag.slice(0, gogIndex).match(/#(?:el)?if[^\n]*/g) ?? [];
        const lastGate = directivesBefore[directivesBefore.length - 1] ?? '';
        expect(
          lastGate.includes('USE_COLORMAP'),
          `${s.name}: the GOG chain must not sit behind a USE_COLORMAP gate (found "${lastGate}")`
        ).toBe(false);
      });

      it('has no colormap-only GOG bypass (max(vColor, vec3(0.0)))', () => {
        expect(
          /max\(\s*vColor\s*,\s*vec3\(0\.0\)\)/.test(frag),
          `${s.name}: found the legacy USE_COLORMAP GOG bypass — intensity/offset would be no-ops on colormapped nodes`
        ).toBe(false);
      });
    });
  }
});
