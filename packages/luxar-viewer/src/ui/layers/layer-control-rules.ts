/**
 * LAYER_CONTROL_RULES — the one table of WHEN each Layers-panel control is shown.
 *
 * A control the panel shows must change something on screen (or in the audio
 * graph); one that would reach no uniform, material state or engine call is
 * hidden instead. That rule used to live in scattered visibility syncs, and it
 * drifted: a sound row showed sliders that reached no material, a
 * `material="physical"` mesh showed the house lighting sliders, Gamma and Blend.
 * Every gate now reads this table — the appearance section
 * (`LayerControls`), the row's own controls and the row menu's submenus
 * (`LayersPanel`) — and the control-effect test
 * (`src/tests/unit/ui/layers/control-effect.test.ts`) drives every VISIBLE
 * control on a fixture of each layer kind and asserts it has an observable
 * effect, and that Reset restores it. A new control therefore needs a row here,
 * a `data-control` id on its DOM, and an effect probe in that test.
 *
 * `visibleFor` answers for the control's EFFECTIVE visibility: an appearance
 * control on a `sound` layer is hidden because the whole section steps aside,
 * and its rule says so rather than relying on the ancestor.
 */

import { canTakeColormap, resolveLayerBlendingMode, type LayerInfo } from './layer-state';

/** Where a control lives: the appearance section below the list, or the layer's row. */
export type LayerControlPlace = 'section' | 'row';

/** When one control is shown. */
export interface LayerControlRule {
  readonly place: LayerControlPlace;
  readonly visibleFor: (layer: LayerInfo) => boolean;
}

/** Every control id, in panel order (also the `data-control` attribute values). */
export const LAYER_CONTROL_IDS = [
  'displayRange',
  'gamma',
  'opacity',
  'absorption',
  'ambient',
  'shadeExponent',
  'specular',
  'shininess',
  'alphaCutoff',
  'physical',
  'blend',
  'layerOrder',
  'colormap',
  'customColormap',
  'labels',
  'activeLevel',
  'gain',
] as const;

export type LayerControlId = (typeof LAYER_CONTROL_IDS)[number];

/** A sound row has no material: its one control (gain) lives in the row. */
export function layerHasAppearance(layer: LayerInfo): boolean {
  return layer.type !== 'sound';
}

/** three's PBR material: none of the house shader's uniforms, modes or gamma term. */
function isPhysicalMesh(layer: LayerInfo): boolean {
  return layer.type === 'mesh' && layer.material === 'physical';
}

function isHouseMesh(layer: LayerInfo): boolean {
  return layer.type === 'mesh' && !isPhysicalMesh(layer);
}

/** The house mesh lighting uniforms exist unless shading resolves to `none`. */
function isLitHouseMesh(layer: LayerInfo): boolean {
  return isHouseMesh(layer) && layer.shading !== 'none';
}

/** The mode that renders (a mesh maps `volumetric` to `opaque`; see `resolveLayerBlendingMode`). */
function renderedMode(layer: LayerInfo): string {
  return resolveLayerBlendingMode(layer.type, layer.blendingMode);
}

/**
 * Whether a blending-mode control reaches this layer's material. A `sound` row has
 * no material, and a `material="physical"` mesh runs three's PBR material, which
 * implements none of the house modes (`applyBlendingMode` is a no-op on it).
 */
function layerHasBlending(layer: LayerInfo): boolean {
  return layerHasAppearance(layer) && !isPhysicalMesh(layer);
}

/**
 * True when `layer` is a kind=partition layer wrapping one or more nested
 * lod_groups, so the "Active level" dropdown broadcasts to them.
 */
export function isBroadcastPartition(layer: LayerInfo): boolean {
  return (
    layer.kind === 'partition' &&
    layer.nestedLodGroupPaths != null &&
    layer.nestedLodGroupPaths.length > 0 &&
    (layer.nestedLodMaxChildCount ?? 0) > 0
  );
}

function hasActiveLevel(layer: LayerInfo): boolean {
  if (layer.kind === 'lod') return (layer.lodGroupChildCount ?? 0) > 0;
  return isBroadcastPartition(layer);
}

const section = (visibleFor: (layer: LayerInfo) => boolean): LayerControlRule => ({
  place: 'section',
  visibleFor: (layer) => layerHasAppearance(layer) && visibleFor(layer),
});

/** The table. Keyed by {@link LayerControlId}, so a missing row is a compile error. */
export const LAYER_CONTROL_RULES: Readonly<Record<LayerControlId, LayerControlRule>> = {
  displayRange: section(() => true),
  // The physical family has no gamma term.
  gamma: section((l) => !isPhysicalMesh(l)),
  opacity: section(() => true),
  // κ is read only by the volumetric branch; a mesh never renders volumetric.
  absorption: section((l) => renderedMode(l) === 'volumetric'),
  // The four house lighting uniforms, compiled out under `shading="none"`. A GROUP
  // over meshes does not get them: they do not compose along the ancestry.
  ambient: section(isLitHouseMesh),
  shadeExponent: section(isLitHouseMesh),
  specular: section(isLitHouseMesh),
  shininess: section(isLitHouseMesh),
  // The cutout exists only in the opaque branch of the house mesh shader.
  alphaCutoff: section((l) => isHouseMesh(l) && renderedMode(l) === 'opaque'),
  physical: section(isPhysicalMesh),
  blend: section(layerHasBlending),
  layerOrder: section(() => true),
  colormap: section((l) => l.supportsColormap),
  // The authored non-builtin LUT, offered only to a layer that owns one.
  customColormap: section((l) => l.supportsColormap && canTakeColormap(l, 'custom')),
  labels: section((l) => (l.labelVocabulary?.length ?? 0) > 0),
  activeLevel: section(hasActiveLevel),
  gain: { place: 'row', visibleFor: (l) => l.sound !== undefined },
};

/** Whether control `id` is shown for `layer`. */
export function isControlVisible(id: LayerControlId, layer: LayerInfo): boolean {
  return LAYER_CONTROL_RULES[id].visibleFor(layer);
}
