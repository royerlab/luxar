/** Mesh TSL materials of one configuration share one node graph (see graph-sharing-cases.ts). */
import * as THREE from 'three';
import { expect, it } from 'vitest';
import { MeshTSLMaterial } from '../../../../../rendering/materials/mesh/material-tsl';
import { colormapTexture, describeGraphSharing } from '../graph-sharing-cases';

describeGraphSharing('MeshTSLMaterial', {
  make: (config = {}) => new MeshTSLMaterial(config),
  valueConfigs: [{ opacity: 0.3 }, { opacity: 0.7, ambient: 0.4, specular: 0.2, shininess: 12 }],
  variants: [
    ['normal', { blendingMode: 'normal' }],
    ['additive', { blendingMode: 'additive' }],
    ['max', { blendingMode: 'max' }],
    ['gamma', { gamma: 2 }],
    ['gain', { intensity: 2 }],
    ['flat shading', { shading: 'flat' }],
    ['no shading', { shading: 'none' }],
    ['colormap', { colormapTexture: colormapTexture() }],
  ],
  bindColormap: (m, tex) => m.updateColormapTexture(tex),
  runtimeFlag: {
    config: { shading: 'flat' },
    set: (m) => m.updateShading('flat'),
    unset: (m) => m.updateShading('smooth'),
  },
});

it('separates nearest-filtered mesh graphs with different texture wrapping', () => {
  const makeTexture = (wrapS: THREE.Wrapping, wrapT: THREE.Wrapping): THREE.DataTexture => {
    const texture = new THREE.DataTexture(new Uint8Array(16), 2, 2, THREE.RGBAFormat);
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.wrapS = wrapS;
    texture.wrapT = wrapT;
    return texture;
  };
  const clamp = new MeshTSLMaterial({});
  const repeatU = new MeshTSLMaterial({});
  clamp.updateBaseColorTexture(makeTexture(THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping));
  repeatU.updateBaseColorTexture(makeTexture(THREE.RepeatWrapping, THREE.ClampToEdgeWrapping));
  expect(repeatU.colorNode).not.toBe(clamp.colorNode);

  const repeatV = new MeshTSLMaterial({});
  repeatV.updateBaseColorTexture(makeTexture(THREE.RepeatWrapping, THREE.RepeatWrapping));
  expect(repeatV.colorNode).not.toBe(repeatU.colorNode);
});
