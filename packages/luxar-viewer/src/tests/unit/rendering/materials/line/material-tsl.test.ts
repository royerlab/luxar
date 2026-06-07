/**
 * Unit tests for LineTSLMaterial clone semantics.
 *
 * Pins the clone contract for the `isOrtho` graph-specialized config.
 * The constructor builds the TSL graph using its default value
 * (perspective); the node-factory / camera updates flip `uIsOrtho`
 * post-construction. A naïve clone copies uniforms but drops the
 * rebuild, so the clone would silently render with the wrong
 * projection branch. These tests pin that the clone re-applies it.
 */

import { describe, expect, it } from 'vitest';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';

describe('LineTSLMaterial clone', () => {
  it('preserves the orthographic uIsOrtho uniform value', () => {
    const original = new LineTSLMaterial();
    original.uniforms.uIsOrtho.value = 1;

    const cloned = original.clone();

    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
  });

  it('does not introduce a LUXAR_SHARPNESS_TWO define (fast path removed)', () => {
    const original = new LineTSLMaterial();
    const cloned = original.clone();
    expect('LUXAR_SHARPNESS_TWO' in (cloned.defines ?? {})).toBe(false);
  });
});
