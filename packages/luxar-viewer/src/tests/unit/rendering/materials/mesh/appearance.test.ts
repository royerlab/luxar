/**
 * Mesh appearance table + the mode → emission-shape mapping.
 *
 * Small surface, but it is the one both backends and the node factory read, so a
 * disagreement here is a disagreement between the two shaders.
 */

import { describe, it, expect } from 'vitest';
import {
  MESH_DEFAULTS,
  MESH_NORMAL_EPS_SQ,
  clampShadeExponent,
  syncMeshEmissionDefines,
  MESH_SUPPORTED_BLENDING_MODES,
  resolveMeshBlendingMode,
  resolveMeshOutput,
} from '../../../../../rendering/materials/mesh/appearance';
import {
  MESH_FRAGMENT_SHADER,
  MESH_VERTEX_SHADER,
} from '../../../../../rendering/materials/mesh/shader-glsl';
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
  it('reaches the emitted GLSL from the shared constant, not a hardcoded literal', () => {
    // One constant now serves both backends (the TSL factory takes the number
    // directly), so "do the two agree" is unrepresentable rather than merely tested.
    // What CAN still go wrong is the GLSL side: it needs the value as source text, so
    // a literal typed into the template string would compile fine and drift silently.
    // Pin the interpolation instead — the guard must compare against exactly this
    // value, and it must appear in the shader as a valid float literal.
    const literal = String(MESH_NORMAL_EPS_SQ);
    expect(literal).toMatch(/^\d(\.\d+)?e-\d+$/); // valid GLSL ES 3.0 exponent form
    expect(MESH_FRAGMENT_SHADER).toContain(`nn >= ${literal}`);
    expect(MESH_FRAGMENT_SHADER).toContain(`max(nn, ${literal})`);
  });

  it('is a squared length, i.e. small enough not to reject a unit normal', () => {
    // Compared against dot(N, N). A value anywhere near 1 would reject every
    // legitimate normal and shade the whole scene from derivatives.
    expect(MESH_NORMAL_EPS_SQ).toBeLessThan(1e-6);
    expect(MESH_NORMAL_EPS_SQ).toBeGreaterThan(0);
  });

  it('is read by the FRAGMENT stage only — the vertex stage has no normal guard', () => {
    // Placement check: the guard reads an INTERPOLATED normal, so it can only live
    // downstream of the rasterizer. A vertex-stage copy would guard the wrong value
    // (the un-interpolated per-vertex one) and silently let a cancelled-to-zero
    // interior normal through.
    expect(MESH_VERTEX_SHADER).not.toContain(String(MESH_NORMAL_EPS_SQ));
  });
});

describe('clampShadeExponent — pow(0, y) is undefined for y <= 0', () => {
  it('floors a zero or negative exponent, which the shade term would otherwise hit at the silhouette', () => {
    // `wrap = saturate(N·V · 0.5 + 0.5)` is EXACTLY 0 for a fragment facing directly
    // away, so `pow(wrap, 0)` is undefined GLSL — driver-dependent 1, 0 or NaN. The
    // clamp keeps that fragment defined (pow(0, 0.001) == 0 → shades at `ambient`).
    expect(clampShadeExponent(0)).toBeGreaterThan(0);
    expect(clampShadeExponent(-3)).toBeGreaterThan(0);
    expect(clampShadeExponent(0)).toBe(0.001);
  });

  it('passes sane values through and defaults when undefined', () => {
    expect(clampShadeExponent(1.5)).toBe(1.5);
    expect(clampShadeExponent(undefined)).toBe(MESH_DEFAULTS.shadeExponent);
  });
});

describe('syncMeshEmissionDefines — at most one emission flag', () => {
  it.each([
    ['opaque', ['LUXAR_MESH_ALPHA_CUTOUT']],
    ['rgb-contribution', ['LUXAR_MAX_RGB_CONTRIBUTION']],
    ['alpha-weighted', []],
    ['premultiplied-alpha', []],
  ] as const)('%s → %j', (output, expected) => {
    const defines: Record<string, unknown> = {};
    syncMeshEmissionDefines(defines, output);
    expect(Object.keys(defines).sort()).toEqual([...expected].sort());
  });

  it('clears a stale flag when the shape changes, and reports the change', () => {
    const defines: Record<string, unknown> = {};
    expect(syncMeshEmissionDefines(defines, 'opaque')).toBe(true);
    expect(syncMeshEmissionDefines(defines, 'opaque')).toBe(false); // idempotent
    expect(syncMeshEmissionDefines(defines, 'rgb-contribution')).toBe(true);
    expect(Object.keys(defines)).toEqual(['LUXAR_MAX_RGB_CONTRIBUTION']);
    expect(syncMeshEmissionDefines(defines, 'alpha-weighted')).toBe(true);
    expect(Object.keys(defines)).toEqual([]);
  });

  it('leaves unrelated defines alone', () => {
    const defines: Record<string, unknown> = { USE_COLORMAP: '', LUXAR_MESH_FLAT_NORMAL: '' };
    syncMeshEmissionDefines(defines, 'opaque');
    expect(defines.USE_COLORMAP).toBe('');
    expect(defines.LUXAR_MESH_FLAT_NORMAL).toBe('');
  });
});

describe('the normal guard is TWO-SIDED in both shader sources', () => {
  it('rejects an infinite dot(N, N), not just a vanishing one', () => {
    // `inf >= eps` is TRUE, and `inf * inversesqrt(inf)` is `inf * 0` = NaN — the
    // guard's own failure mode arriving from the upper end. The writer rejects
    // non-finite normals, so this is the hand-crafted-store case every sibling
    // shader sanitizes for. Asserted on the SOURCE because no unit test can run GLSL.
    expect(MESH_FRAGMENT_SHADER).toMatch(/nn < 1e30/);
  });
});
