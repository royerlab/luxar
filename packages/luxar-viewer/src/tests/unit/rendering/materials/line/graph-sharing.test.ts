/** Line TSL materials of one configuration share one node graph (see graph-sharing-cases.ts). */
import * as THREE from 'three';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';
import { colormapTexture, describeGraphSharing } from '../graph-sharing-cases';

describeGraphSharing('LineTSLMaterial', {
  make: (config = {}) => new LineTSLMaterial(config),
  valueConfigs: [{ opacity: 0.3 }, { opacity: 0.7, absorption: 2 }],
  variants: [
    ['normal', { blendingMode: 'normal' }],
    ['volumetric', { blendingMode: 'volumetric' }],
    ['max', { blendingMode: 'max' }],
    ['gamma', { gamma: 2 }],
    ['gain', { intensity: 2 }],
    ['colormap', { colormapTexture: colormapTexture() }],
    ['screen-space quad', { primitive: 'screen-space' }],
    ['no join', { join: 'none' }],
  ],
  bindElementTexture: (m, tex) => m.updateLineTexture(tex),
  bindColormap: (m, tex) => m.updateColormapTexture(tex),
  runtimeFlag: {
    config: { gamma: 2 },
    set: (m) => m.updateGamma(2),
    unset: (m) => m.updateGamma(1),
  },
});

describeGraphSharing('LineTSLMaterial (ortho camera)', {
  make: (config = {}) => {
    const m = new LineTSLMaterial(config);
    m.updateCameraParams(new THREE.Vector2(800, 600), true);
    return m;
  },
  valueConfigs: [{ opacity: 0.3 }, { opacity: 0.7 }],
  variants: [['screen-space quad', { primitive: 'screen-space' }]],
});
