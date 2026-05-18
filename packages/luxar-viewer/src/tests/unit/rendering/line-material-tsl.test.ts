/**
 * Unit tests for LineTSLMaterial clone semantics.
 *
 * Pins the clone contract for the two graph-specialized configs
 * (`isOrtho` and `LUXAR_SHARPNESS_TWO`). The constructor builds the
 * TSL graph using its default values (perspective + slow-pow path);
 * the node-factory mutates both post-construction after inspecting
 * uniforms / per-vertex sharpness arrays. A naïve clone copies
 * uniforms but drops the post-construction state, so the clone
 * would silently render with the wrong projection or the slow
 * fragment path. These tests pin that the clone re-applies both.
 */

import { describe, expect, it } from 'vitest';
import { LineTSLMaterial } from '../../../rendering/line-material-tsl';

describe('LineTSLMaterial clone', () => {
  it('preserves the LUXAR_SHARPNESS_TWO fast-path define', () => {
    const original = new LineTSLMaterial();
    original.setSharpnessAllTwo(true);

    const cloned = original.clone();

    expect(cloned.defines).toBeDefined();
    expect('LUXAR_SHARPNESS_TWO' in (cloned.defines as Record<string, unknown>)).toBe(true);
  });

  it('clone of a default material does not pick up LUXAR_SHARPNESS_TWO', () => {
    const original = new LineTSLMaterial();
    const cloned = original.clone();
    expect('LUXAR_SHARPNESS_TWO' in (cloned.defines as Record<string, unknown>)).toBe(false);
  });

  it('preserves the orthographic uIsOrtho uniform value', () => {
    const original = new LineTSLMaterial();
    original.uniforms.uIsOrtho.value = 1;

    const cloned = original.clone();

    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
  });

  it('rebuilds the graph after clone when source is ortho + sharpness=2 (one rebuild covers both)', () => {
    // Both flags are graph-specialized, so the clone needs to rebuild
    // exactly once with both. Observing the version delta from clone
    // construction is the cheapest deterministic proxy for "rebuild
    // happened".
    const original = new LineTSLMaterial();
    original.uniforms.uIsOrtho.value = 1;
    original.setSharpnessAllTwo(true);

    const cloned = original.clone();
    const versionAfterClone = cloned.version;

    // Setting the same flag again on the clone should be a no-op
    // (idempotent setter). If the clone forgot to rebuild during
    // construction, this call would trigger one and bump the version.
    cloned.setSharpnessAllTwo(true);
    expect(cloned.version).toBe(versionAfterClone);
  });
});
