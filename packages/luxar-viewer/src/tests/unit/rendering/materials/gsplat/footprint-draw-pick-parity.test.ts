/**
 * Draw and pick size a gsplat's quad to the SAME reach radius (#2944 B10).
 *
 * Both vertex stages shrink a dim splat's quad to the ellipse its fragments
 * can reach. If the pick stage's radius ever drifted below the draw stage's,
 * picking would quietly miss the visible edge of a dim splat — and no
 * pixel-identity gate sees that, because the drawn image is unchanged. So the
 * rule is ONE function in each language (`gsplatFootprintPeakScale` +
 * `gsplatVisibleMahalSq` in GLSL, `gsplatQuadFootprintTSL` in TSL), and the
 * two stages differ ONLY in the alpha factor they hand it: the pick pass
 * weighs by the same per-splat alpha factor and gain as the draw, and
 * additionally by the node opacity (a fully transparent node is invisible
 * however bright its splats, so it must not be pickable either).
 *
 * Pinned three ways:
 *   1. GLSL sources: both vertex shaders call the shared helpers with the
 *      same amplitude, uniforms and gain; pick's alpha factor is
 *      `pickAlphaFactor` (the draw's alpha factor times uOpacity).
 *   2. Numerics, through the shared CPU mirror: over a grid of amplitudes and
 *      truncation radii, pick's radius and extents equal draw's EXACTLY at a
 *      neutral appearance (opaque, gain <= 1).
 *   3. TSL codegen snapshots (`src/tests/__codegen__`, themselves pinned to the
 *      live TSL output by the tsl-codegen-snapshot e2e spec): every draw
 *      variant and the pick shader emit the same reach-radius block, up to
 *      variable names, and the same extent + cull use of it.
 */
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  GLSL_GSPLAT_VISIBLE_FOOTPRINT,
  gsplatFootprintExtent,
  gsplatFootprintPeakScale,
  gsplatVisibleMahalSq,
} from '../../../../../rendering/materials/gsplat/math';
import { GSPLAT_VERTEX_SHADER } from '../../../../../rendering/materials/gsplat/shader-glsl';
import { GSPLAT_PICK_VERTEX_SHADER } from '../../../../../rendering/picking/gsplat/shaders';

const CODEGEN_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../../__codegen__'
);

function count(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

/** The peak-scale arguments of the one `visibleMahalSq` assignment in `vs`. */
function glslPeakScaleArgs(vs: string): string[] {
  const calls = [
    ...vs.matchAll(
      /float visibleMahalSq = gsplatVisibleMahalSq\(\s*gsplatFootprintPeakScale\(([^)]*)\),\s*uShiftC,\s*uTruncateSq\);/g
    ),
  ];
  expect(calls).toHaveLength(1);
  return calls[0][1].split(',').map((arg) => arg.trim());
}

describe('GLSL: draw and pick vertex shaders share one reach-radius rule', () => {
  const shaders = { draw: GSPLAT_VERTEX_SHADER, pick: GSPLAT_PICK_VERTEX_SHADER };

  it('both embed the shared helper snippet once, and define no helper of their own', () => {
    for (const vs of Object.values(shaders)) {
      expect(count(vs, GLSL_GSPLAT_VISIBLE_FOOTPRINT)).toBe(1);
      expect(count(vs, 'float gsplatFootprintPeakScale(')).toBe(1);
      expect(count(vs, 'float gsplatVisibleMahalSq(')).toBe(1);
      expect(count(vs, 'float gsplatFootprintExtent(')).toBe(1);
    }
  });

  it('the GLSL peak scale is the CPU mirror, term for term', () => {
    expect(GLSL_GSPLAT_VISIBLE_FOOTPRINT).toContain(
      'return amplitude2D * invOneMinusC * alphaFactor * max(gain, 1.0);'
    );
  });

  it('same amplitude, uniforms and gain; pick folds the node opacity into the alpha factor', () => {
    const draw = glslPeakScaleArgs(shaders.draw);
    const pick = glslPeakScaleArgs(shaders.pick);
    expect(draw.slice(0, 2)).toEqual(['vAmplitude2D', 'uInvOneMinusC']);
    expect(pick).toEqual(['vAmplitude2D', 'uInvOneMinusC', 'pickAlphaFactor', 'uIntensity']);
    expect(draw.slice(2)).toEqual(['footprintAlpha', 'uIntensity']);
  });

  it('both cull on the same sign test and tighten both extents the same way', () => {
    for (const vs of Object.values(shaders)) {
      expect(count(vs, 'if (visibleMahalSq < 0.0) {')).toBe(1);
      expect(count(vs, 'extent1 = gsplatFootprintExtent(extent1, lambda1, visibleMahalSq);')).toBe(
        1
      );
      expect(count(vs, 'extent2 = gsplatFootprintExtent(extent2, lambda2, visibleMahalSq);')).toBe(
        1
      );
    }
  });
});

describe('numerics: pick reaches exactly the draw radius at a neutral appearance', () => {
  const AMPLITUDES = Array.from({ length: 81 }, (_, i) => 10 ** (-6 + i * 0.1)); // 1e-6 .. 1e2
  const TRUNCATIONS = [1, 1.5, 2, 2.5, 3, 4, 6, 40]; // 40: C underflows to 0
  const NEUTRAL_GAINS = [0, 0.25, 0.5, 1];
  const LAMBDAS = [0.01, 1, 37.5, 4096];
  const LEGACY = 200;

  it('same squared radius, same extents, same cull decision', () => {
    for (const t of TRUNCATIONS) {
      const truncateSq = t * t;
      const shiftC = Math.exp(-0.5 * truncateSq);
      const invOneMinusC = 1 / (1 - shiftC);
      for (const amplitude of AMPLITUDES) {
        const pick = gsplatVisibleMahalSq(
          gsplatFootprintPeakScale(amplitude, invOneMinusC, 1, 1),
          shiftC,
          truncateSq
        );
        for (const gain of NEUTRAL_GAINS) {
          const draw = gsplatVisibleMahalSq(
            gsplatFootprintPeakScale(amplitude, invOneMinusC, 1, gain),
            shiftC,
            truncateSq
          );
          expect(draw).toBe(pick);
          if (pick < 0) continue;
          for (const lambda of LAMBDAS) {
            expect(gsplatFootprintExtent(LEGACY, lambda, draw)).toBe(
              gsplatFootprintExtent(LEGACY, lambda, pick)
            );
          }
        }
      }
    }
  });

  it('the draw-only factors are the only difference: a brighter gain only grows the draw radius', () => {
    const truncateSq = 9;
    const shiftC = Math.exp(-4.5);
    const invOneMinusC = 1 / (1 - shiftC);
    for (const amplitude of AMPLITUDES) {
      const pick = gsplatVisibleMahalSq(
        gsplatFootprintPeakScale(amplitude, invOneMinusC),
        shiftC,
        truncateSq
      );
      const draw = gsplatVisibleMahalSq(
        gsplatFootprintPeakScale(amplitude, invOneMinusC, 1, 4),
        shiftC,
        truncateSq
      );
      expect(draw).toBeGreaterThanOrEqual(pick);
    }
  });
});

describe('TSL codegen: every draw variant and the pick shader emit one reach block', () => {
  interface ReachBlock {
    /** The peak-scale expression, before the 5% margin. */
    peak: string;
    /** The block from the margin to the radius `.toVar()`, identifiers renamed. */
    normalized: string;
    /** The radius variable the extents and the cull read. */
    radiusVar: string;
    source: string;
  }

  /** Rename nodeVar/nodeUniform identifiers in order of first appearance. */
  function normalize(text: string): string {
    const names = new Map<string, string>();
    return text.replace(/\bnode(?:Var|Uniform)\d+\b/g, (id) => {
      if (!names.has(id)) names.set(id, `id${names.size}`);
      return names.get(id)!;
    });
  }

  function reachBlock(file: string): ReachBlock {
    const source = readFileSync(path.join(CODEGEN_DIR, file), 'utf8');
    const lines = source.split('\n');
    const start = lines.findIndex((l) => /^\tnodeVar\d+ = \(.* \* 1\.05 \);$/.test(l));
    expect(start, `${file}: no reach-radius block`).toBeGreaterThanOrEqual(0);
    const peak = /^\tnodeVar\d+ = \( (.*) \* 1\.05 \);$/.exec(lines[start])![1];
    const end = lines.findIndex((l, i) => i > start && /^\tnodeVar\d+ = nodeVar\d+;$/.test(l));
    const block = lines.slice(start, end + 1);
    const radiusVar = /^\t(nodeVar\d+) = /.exec(lines[end])![1];
    const body = ['\tPEAK_MARGIN', ...block.slice(1)].join('\n');
    return { peak, normalized: normalize(body), radiusVar, source };
  }

  const files = readdirSync(CODEGEN_DIR).filter((f) => /^gsplat.*\.vertex\.glsl\.txt$/.test(f));
  const pickFile = 'gsplat-pick.vertex.glsl.txt';
  const drawFiles = files.filter((f) => f !== pickFile);

  it('covers the pick shader and every draw variant', () => {
    expect(files).toContain(pickFile);
    expect(drawFiles.length).toBeGreaterThanOrEqual(6);
  });

  it('the radius computation is identical up to variable names', () => {
    const pick = reachBlock(pickFile);
    for (const file of drawFiles) {
      expect(reachBlock(file).normalized, file).toBe(pick.normalized);
    }
  });

  it('draw peak = pick peak times the draw-only alpha factor and max(gain, 1)', () => {
    const pickPeak = reachBlock(pickFile).peak;
    expect(pickPeak).toMatch(/^\( nodeVar\d+ \* nodeUniform\d+ \)$/);
    for (const file of drawFiles) {
      const m =
        /^\( \( (\( nodeVar\d+ \* nodeUniform\d+ \)) \* .+ \) \* max\( nodeUniform\d+, 1\.0 \) \)$/.exec(
          reachBlock(file).peak
        );
      expect(m, file).not.toBeNull();
      expect(normalize(m![1])).toBe(normalize(pickPeak));
    }
  });

  it('both extents read the radius through the same tightening, and the cull tests its sign', () => {
    for (const file of files) {
      const { radiusVar, source } = reachBlock(file);
      const tightening = new RegExp(
        `min\\( \\( nodeVar\\d+ \\* nodeVar\\d+ \\), \\( sqrt\\( \\( max\\( ${radiusVar}, 0\\.0 \\) \\* nodeVar\\d+ \\) \\) \\+ 1\\.0 \\) \\)`,
        'g'
      );
      expect(source.match(tightening), file).toHaveLength(2);
      expect(count(source, `( ${radiusVar} < 0.0 )`), file).toBe(1);
    }
  });
});
