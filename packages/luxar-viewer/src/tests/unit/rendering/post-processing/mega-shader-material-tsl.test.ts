/**
 * Unit tests for the TSL `MegaShaderTSLMaterial` capture-mode toggles.
 *
 * Before the dual-stack r184 fix, `toggleRawHdrCapture` /
 * `toggleLinearLdrCapture` were warning-only no-ops on the TSL path,
 * so EXR exports under WebGPU silently fell through to the full
 * normal-pipeline output. These tests pin the contract:
 *
 *   1. Toggling capture-on flips internal state and forces a graph
 *      rebuild (observable via `material.version` bumping — THREE
 *      exposes `needsUpdate` only as a setter that bumps the version).
 *   2. Toggling to the same value is idempotent (version stays).
 *   3. Toggle-off rebuilds back to the baseline graph.
 *
 * We avoid spinning up a renderer — the material constructor + toggle
 * methods exercise the JS-side flag plumbing without ever dispatching
 * GPU work.
 */

import { describe, expect, it } from 'vitest';
import { MegaShaderTSLMaterial } from '../../../../rendering/post-processing/mega-shader-material-tsl';

describe('MegaShaderTSLMaterial capture-mode toggles', () => {
  it('toggleRawHdrCapture(true) rebuilds the TSL graph (version bumps)', () => {
    const material = new MegaShaderTSLMaterial();
    const v0 = material.version;
    material.toggleRawHdrCapture(true);
    expect(material.version).toBeGreaterThan(v0);

    material.dispose();
  });

  it('toggleRawHdrCapture(true) twice is idempotent (no second rebuild)', () => {
    const material = new MegaShaderTSLMaterial();
    material.toggleRawHdrCapture(true);
    const v0 = material.version;

    material.toggleRawHdrCapture(true);
    expect(material.version).toBe(v0);

    material.dispose();
  });

  it('toggleRawHdrCapture(false) after on returns to baseline (rebuilds)', () => {
    const material = new MegaShaderTSLMaterial();
    material.toggleRawHdrCapture(true);
    const v0 = material.version;

    material.toggleRawHdrCapture(false);
    expect(material.version).toBeGreaterThan(v0);

    material.dispose();
  });

  it('toggleLinearLdrCapture(true) rebuilds independently of raw-HDR', () => {
    const material = new MegaShaderTSLMaterial();
    material.toggleRawHdrCapture(true);
    const v0 = material.version;

    material.toggleLinearLdrCapture(true);
    expect(material.version).toBeGreaterThan(v0);

    material.dispose();
  });

  it('toggleLinearLdrCapture(false) when already off is idempotent', () => {
    const material = new MegaShaderTSLMaterial();
    const v0 = material.version;

    material.toggleLinearLdrCapture(false);
    expect(material.version).toBe(v0);

    material.dispose();
  });
});
