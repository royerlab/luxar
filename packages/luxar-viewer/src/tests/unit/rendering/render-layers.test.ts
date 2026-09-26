/**
 * The viewer's layer bits stay clear of three's WebXR eye layers.
 *
 * three's WebXRManager enables layer 1 on the left-eye camera and layer 2 on the
 * right-eye camera, so a mesh on either during an XR frame draws in one eye only.
 */

import { describe, it, expect } from 'vitest';
import {
  RENDER_LAYER_DEFAULT,
  RENDER_LAYER_REFRACTING_GLASS,
  RENDER_LAYER_UNPARTITIONED,
} from '../../../rendering/render-layers';

const XR_EYE_LAYERS = [1, 2];
const TRANSIENT = [RENDER_LAYER_REFRACTING_GLASS, RENDER_LAYER_UNPARTITIONED];

describe('render layers', () => {
  it('draws everything persistent on three default layer 0', () => {
    expect(RENDER_LAYER_DEFAULT).toBe(0);
  });

  it('keeps the transient refraction-split layers off the XR eye layers', () => {
    for (const layer of TRANSIENT) expect(XR_EYE_LAYERS).not.toContain(layer);
  });

  it('uses distinct, valid layer bits', () => {
    const all = [RENDER_LAYER_DEFAULT, ...TRANSIENT];
    expect(new Set(all).size).toBe(all.length);
    for (const layer of all) {
      expect(Number.isInteger(layer)).toBe(true);
      expect(layer).toBeGreaterThanOrEqual(0);
      expect(layer).toBeLessThan(32);
    }
  });
});
