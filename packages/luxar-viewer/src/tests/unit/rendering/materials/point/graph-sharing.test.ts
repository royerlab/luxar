/** Point TSL materials of one configuration share one node graph (see graph-sharing-cases.ts). */
import { PointTSLMaterial } from '../../../../../rendering/materials/point/material-tsl';
import { colormapTexture, describeGraphSharing } from '../graph-sharing-cases';

describeGraphSharing('PointTSLMaterial', {
  make: (config = {}) => new PointTSLMaterial(config),
  valueConfigs: [{ opacity: 0.3 }, { opacity: 0.7, absorption: 2, radiusScale: 3 }],
  variants: [
    ['normal', { blendingMode: 'normal' }],
    ['volumetric', { blendingMode: 'volumetric' }],
    ['max', { blendingMode: 'max' }],
    ['gamma', { gamma: 2 }],
    ['gain', { intensity: 2 }],
    ['colormap', { colormapTexture: colormapTexture() }],
  ],
  bindElementTexture: (m, tex) => m.updatePointTexture(tex),
  bindColormap: (m, tex) => m.updateColormapTexture(tex),
  runtimeFlag: {
    config: { gamma: 2 },
    set: (m) => m.updateGamma(2),
    unset: (m) => m.updateGamma(1),
  },
});
