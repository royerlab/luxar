/**
 * Mesh appearance table + the mode → emission-shape mapping.
 *
 * Small surface, but it is the one both backends and the node factory read, so a
 * disagreement here is a disagreement between the two shaders.
 */

import { describe, it, expect } from 'vitest';
import {
  MESH_DEFAULTS,
  MESH_SUPPORTED_BLENDING_MODES,
  resolveMeshBlendingMode,
  resolveMeshOutput,
} from '../../../../../rendering/materials/mesh/appearance';
import { MESH_NORMAL_EPS_SQ } from '../../../../../rendering/materials/mesh/shader-glsl';
import { MESH_NORMAL_EPS_SQ_VALUE } from '../../../../../rendering/materials/mesh/shader-tsl';
import { BLENDING_MODES } from '../../../../../types/blending';

describe('resolveMeshBlendingMode', () => {
  it("maps only 'volumetric' — and maps it to 'opaque'", () => {
    // Enumerated over the WHOLE mode union rather than a hand-written list, so a new
    // blending mode has to state its mesh behavior here instead of silently passing
    // through into a shader branch that does not exist.
    for (const mode of BLENDING_MODES) {
      const resolved = resolveMeshBlendingMode(mode);
      if (mode === 'volumetric') {
        expect(resolved).toBe('opaque');
      } else {
        expect(resolved, `${mode} must pass through unchanged`).toBe(mode);
      }
    }
  });

  it('is idempotent, so a material may re-resolve an already-resolved mode', () => {
    // `applyBlendingMode` resolves, then stamps `userData.blendingMode`, which
    // `clone()` feeds back through the constructor — a second pass must be a no-op.
    for (const mode of BLENDING_MODES) {
      expect(resolveMeshBlendingMode(resolveMeshBlendingMode(mode))).toBe(
        resolveMeshBlendingMode(mode)
      );
    }
  });
});

describe('MESH_SUPPORTED_BLENDING_MODES', () => {
  it('is every mode except volumetric', () => {
    expect([...MESH_SUPPORTED_BLENDING_MODES].sort()).toEqual(
      BLENDING_MODES.filter((m) => m !== 'volumetric')
        .slice()
        .sort()
    );
  });
});

describe('resolveMeshOutput', () => {
  it.each([
    ['opaque', 'opaque'],
    ['volumetric', 'opaque'],
    ['max', 'rgb-contribution'],
    ['additive', 'alpha-weighted'],
    ['luminous', 'alpha-weighted'],
    ['normal', 'alpha-weighted'],
  ] as const)('%s → %s emission', (mode, expected) => {
    expect(resolveMeshOutput(mode)).toBe(expected);
  });

  it('never returns premultiplied-alpha, the one shape mesh has no branch for', () => {
    // `getCompleteBlendingState` can produce it (gsplat `normal` / `volumetric` do),
    // and the mesh fragment has exactly three emissions. If a future mode routed a
    // mesh into that state the shader would silently emit the wrong one.
    for (const mode of BLENDING_MODES) {
      expect(resolveMeshOutput(mode)).not.toBe('premultiplied-alpha');
    }
  });
});

describe('MESH_DEFAULTS', () => {
  it('shades: an ambient floor below 1 and a positive exponent', () => {
    // `ambient === 1` would collapse the shade term to the emissive look of the
    // other three types, which is the documented escape hatch — not the default.
    expect(MESH_DEFAULTS.ambient).toBeGreaterThan(0);
    expect(MESH_DEFAULTS.ambient).toBeLessThan(1);
    expect(MESH_DEFAULTS.shadeExponent).toBeGreaterThan(0);
  });

  it('cuts out at the alpha-test midpoint', () => {
    expect(MESH_DEFAULTS.alphaCutoff).toBe(0.5);
  });
});

describe('the normal-validity epsilon', () => {
  it('is the SAME value in both backends', () => {
    // The GLSL side needs a source string and the TSL side a number, so the constant
    // exists twice. If they drift, the two backends switch to the derivative
    // fallback on different fragments — a divergence that renders as a subtle
    // shading difference nobody would trace back to a literal.
    expect(Number(MESH_NORMAL_EPS_SQ)).toBe(MESH_NORMAL_EPS_SQ_VALUE);
  });

  it('is a squared length, i.e. small enough not to reject a unit normal', () => {
    // Compared against dot(N, N). A value anywhere near 1 would reject every
    // legitimate normal and shade the whole scene from derivatives.
    expect(MESH_NORMAL_EPS_SQ_VALUE).toBeLessThan(1e-6);
    expect(MESH_NORMAL_EPS_SQ_VALUE).toBeGreaterThan(0);
  });
});
