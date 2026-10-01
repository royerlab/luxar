/** Mesh TSL materials of one configuration share one node graph (see graph-sharing-cases.ts). */
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
