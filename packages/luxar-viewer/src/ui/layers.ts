/**
 * Layers — public entrypoint
 *
 * Re-exports the layers panel, state manager, and shared widget pieces.
 * Internal helpers live under ui/layers/.
 */

export { LayersPanel } from './layers/layers-panel';
export { LayerStateManager, computeUniforms, computeDisplayRange } from './layers/layer-state';
export type { LayerInfo, DisplayUniforms, SelectionMode } from './layers/layer-state';
export { RangeSlider } from './layers/range-slider';
