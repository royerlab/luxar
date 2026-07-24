/**
 * TSL opacity-tail tripwire (all three geometry types).
 *
 * THREE's NodeMaterial appends `DiffuseColor.w *= material.opacity` to
 * every generated fragment (visible in the codegen snapshots as
 * `DiffuseColor.w = DiffuseColor.w * nodeUniformN`). Under the
 * volumetric blending mode the output alpha is the PHYSICAL absorption
 * `1 − e^(−τ)` — Luxar's opacity is already folded into τ through the
 * `uOpacity`/`opacity` custom uniform — so any code that ever writes
 * the THREE `.opacity` property on a TSL material would double-apply
 * opacity to the destination-screening term (a GLSL/TSL divergence:
 * the GLSL twins have no such tail).
 *
 * Today the tail is inert because nothing writes `.opacity` (it stays
 * at the THREE default 1.0) and `updateOpacity` deliberately targets
 * the custom uniform only. These pins turn that implicit invariant into
 * a loud failure the day either half changes.
 */
import { describe, it, expect } from 'vitest';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';

describe('TSL opacity tail stays inert (volumetric double-check tripwire)', () => {
  const make = () =>
    [
      ['PointTSLMaterial', new PointTSLMaterial({ blendingMode: 'volumetric' })],
      ['LineTSLMaterial', new LineTSLMaterial({ blendingMode: 'volumetric' })],
      ['GSplatTSLMaterial', new GSplatTSLMaterial({ blendingMode: 'volumetric' })],
    ] as const;

  it("THREE .opacity is 1.0 on construction (the NodeMaterial tail's multiplier is identity)", () => {
    for (const [name, mat] of make()) {
      expect(mat.opacity, `${name}: THREE .opacity must stay at the 1.0 default`).toBe(1.0);
    }
  });

  it('updateOpacity writes the CUSTOM uniform, never THREE .opacity', () => {
    for (const [name, mat] of make()) {
      (mat as { updateOpacity(v: number): void }).updateOpacity(0.4);
      expect(mat.opacity, `${name}: updateOpacity leaked into THREE .opacity`).toBe(1.0);
      const uniforms = (mat as { uniforms: Record<string, { value: unknown }> }).uniforms;
      const custom = uniforms.uOpacity ?? uniforms.opacity;
      expect(custom.value, `${name}: custom opacity uniform not updated`).toBe(0.4);
    }
  });
});
