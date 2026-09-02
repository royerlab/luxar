/**
 * Attribute composition across the scene graph hierarchy.
 *
 * Rendering attributes compose from root to leaf rather than override:
 *
 *   effective_opacity    = clamp(∏ opacity_i,  0, 1)
 *   effective_absorption = max(0, ∏ absorption_i)   (volumetric κ; identity 1)
 *   effective_gamma      = clamp(∏ gamma_i,    0.1, 10)
 *   effective_intensity  = max(0, ∏ intensity_i)
 *   effective_offset     = Σ offset_i
 *   effective_blending   = nearest ancestor (root-to-leaf) that sets blending_mode,
 *                          else undefined (each consumer applies its own per-type
 *                          default: `additive` for points/lines/gsplats, `opaque`
 *                          for mesh — docs/specs/MESH_NODE_SPEC.md §6.3)
 *   effective_join       = nearest ancestor (root-to-leaf) that sets join,
 *                          else undefined (lines apply DEFAULT_LINE_JOIN)
 *   effective_layer_order = nearest ancestor (root-to-leaf) that sets layer_order,
 *                          else undefined (the renderer treats unset as band 0
 *                          and keeps its inferred containment ordering — see
 *                          docs/guides/specs/LAYER_ORDER_SPEC.md)
 *   effective_colormap   = nearest ancestor (root-to-leaf) that sets colormap,
 *                          else undefined (the leaf renders direct colours).
 *                          `customLutBytes` travels WITH it, from the same node
 *                          — a name and a LUT from different nodes would paint
 *                          the wrong palette.
 *
 * Note: the offset composition is additive per the spec. This is mathematically
 * different from chaining the shader's `color * I + O` model through successive
 * nodes (which would yield `O_parent * I_child + O_child`). The additive form
 * is simpler to reason about at the authoring layer and is what the Python
 * API documents; viewer and spec match intentionally.
 */

import type { SceneNode } from './data-loader-types';
import { clamp } from '../utils/clamp';
import { normalizeBlendingMode } from '../rendering/blending-state';
import type { BlendingMode } from '../types/blending';
import { isGeometryType } from '../types/geometry-capabilities';

export interface ComposableAttrs {
  opacity?: number;
  /** Absorption coefficient κ (volumetric mode); multiplicative, identity 1. */
  absorption?: number;
  gamma?: number;
  intensity?: number;
  offset?: number;
  /** Raw (unvalidated) mode string as authored; normalized at compose time. */
  blending_mode?: string;
  /**
   * Raw (unvalidated) line join style as authored. Unlike `blending_mode` it
   * is NOT normalized here — `createLinesNode` runs it through
   * `parseLineJoinStyle` and warns once on an unknown value, and validating
   * in both places would warn twice.
   */
  join?: string;
  /**
   * Palette name as authored (a builtin name, or the `'custom'` sentinel that
   * points at the node's sibling `colormap_lut` array). Nearest-setter-wins,
   * like `blending_mode`: the Python writer no longer manufactures a per-leaf
   * `'gray'` when an ancestor authored one (#1600), so an ancestor's palette
   * is what a bare leaf should render through.
   */
  colormap?: string;
  /**
   * The `colormap === 'custom'` LUT bytes the scene loader stashed on the node
   * that declared it. Carried as part of the SAME record as `colormap` so the
   * two can never be composed from different nodes.
   */
  customLutBytes?: Uint8Array;
  /**
   * Authored cross-layer draw order (`LAYER_ORDER_SPEC.md`). Higher =
   * nearer the camera = drawn later. Nearest-setter-wins like `blending_mode`,
   * and deliberately kept `number | undefined` rather than defaulted here:
   * "nobody authored an order" must stay distinguishable from "someone authored
   * 0", because an EXPLICIT order suppresses the containment rule while an
   * unset one must not (spec D2/D3). The renderer applies `unset ⇒ band 0`.
   */
  layer_order?: number;
}

export interface EffectiveAttrs {
  opacity: number;
  /** Composed absorption coefficient κ (≥ 0; only read in volumetric mode). */
  absorption: number;
  gamma: number;
  intensity: number;
  offset: number;
  /**
   * `undefined` when no level of the chain set a mode (each consumer applies
   * its own per-type default); otherwise canonical — a set value (even a
   * malformed one) is run through `normalizeBlendingMode`.
   */
  blending_mode: BlendingMode | undefined;
  /**
   * Nearest ancestor (root-to-leaf) that sets `join`, `undefined` when no
   * level of the chain does — which is what lets `createLinesNode` apply
   * `DEFAULT_LINE_JOIN`. Left raw on purpose: the consumer validates.
   */
  join: string | undefined;
  /**
   * Nearest ancestor (root-to-leaf) that sets `colormap`, `undefined` when no
   * level of the chain does (the leaf renders its direct colours). Left raw:
   * `getColormapTexture` owns the unknown-name fallback.
   *
   * Whether an inherited palette actually applies is the consumer's call and
   * differs by geometry: points / lines / mesh gate on `has_scalars` (no
   * scalar channel, no colormap), while a gsplats leaf is always
   * colormap-capable — its amplitude IS the scalar — so an ancestor-authored
   * palette overrides even per-splat colours there. That matches the Layers
   * panel's imperative fan-out (`layer-apply.ts::applyColormap`), so the
   * authored and interactive routes agree.
   */
  colormap: string | undefined;
  /**
   * The custom LUT bytes belonging to whichever node supplied `colormap`;
   * `undefined` when that node declared no `'custom'` palette.
   */
  customLutBytes: Uint8Array | undefined;
  /**
   * Nearest ancestor (root-to-leaf) that sets `layer_order`; `undefined` when
   * no level of the chain does. The `undefined` is load-bearing — see the
   * `ComposableAttrs` field.
   */
  layer_order: number | undefined;
}

/**
 * Compose a root-to-leaf chain of attribute records.
 *
 * The chain is ordered from the outermost ancestor (root) to the leaf node
 * whose effective attributes we want. Unset values are treated as identity:
 * opacity/gamma/intensity = 1, offset = 0. A set `blending_mode` at any
 * level overrides the cumulative choice; a set winning string is validated
 * through `normalizeBlendingMode` (unknown string → 'normal' + one-time
 * warning), so every consumer sees a canonical mode. An unset chain yields
 * `undefined` — each consumer then applies its own per-type default (spec
 * §6.3: `additive` for points/lines/gsplats, `opaque` for mesh).
 */
export function composeAttrs(chainRootToLeaf: readonly ComposableAttrs[]): EffectiveAttrs {
  let opacity = 1.0;
  let absorption = 1.0;
  let gamma = 1.0;
  let intensity = 1.0;
  let offset = 0.0;
  let blending_mode: string | undefined;
  let join: string | undefined;
  let colormap: string | undefined;
  let customLutBytes: Uint8Array | undefined;
  let layer_order: number | undefined;

  for (const a of chainRootToLeaf) {
    if (a.opacity !== undefined) opacity *= a.opacity;
    if (a.absorption !== undefined) absorption *= a.absorption;
    if (a.gamma !== undefined) gamma *= a.gamma;
    if (a.intensity !== undefined) intensity *= a.intensity;
    if (a.offset !== undefined) offset += a.offset;
    if (a.blending_mode !== undefined) blending_mode = a.blending_mode;
    if (a.join !== undefined) join = a.join;
    // Name and LUT move together, always from the SAME record: a node that
    // sets a builtin palette must also clear an ancestor's `'custom'` bytes,
    // or the pair would describe two different nodes' intent.
    if (a.colormap !== undefined) {
      colormap = a.colormap;
      customLutBytes = a.customLutBytes;
    }
    if (a.layer_order !== undefined) layer_order = a.layer_order;
  }

  // Clamp per spec
  opacity = clamp(opacity, 0, 1);
  absorption = Math.max(0, absorption); // κ is unbounded above
  gamma = clamp(gamma, 0.1, 10);
  intensity = Math.max(0, intensity);

  return {
    opacity,
    absorption,
    gamma,
    intensity,
    offset,
    // Preserve the unset state: only a truly-unset chain yields `undefined`
    // (consumers apply their per-type default). A set-but-malformed value
    // (e.g. '') is NOT undefined, so it still normalizes → 'normal'.
    blending_mode: blending_mode === undefined ? undefined : normalizeBlendingMode(blending_mode),
    // Passed through as authored — `createLinesNode` is the one place that
    // validates a join style, so it also owns the unknown-value warning.
    join,
    // Passed through as authored, for the same reason: `getColormapTexture`
    // owns the unknown-name → viridis fallback and its warning.
    colormap,
    customLutBytes,
    // Never defaulted to 0 here: the renderer needs to tell an authored level
    // from an absent one, because only the former suppresses containment.
    layer_order,
  };
}

/**
 * Walk from `root` toward the node whose path equals `targetPath`,
 * returning the chain of SceneNodes in root-to-leaf order. The scene-root
 * node is included iff its own `type` is not `'scene'` — scene roots are
 * carriers-only and should contribute no rendering attrs.
 *
 * This implementation is path-format-agnostic: it matches children by
 * direct prefix comparison on the already-stored `path` field, so it
 * works for both the leading-slash format produced by the real scene
 * loader (`'/'`, `'/group/pts'`) and the plain format used in unit tests
 * (`''`, `'group/pts'`).
 */
export function collectAncestorNodes(root: SceneNode, targetPath: string): SceneNode[] {
  const chain: SceneNode[] = [];

  // Empty target means "nothing requested" — return nothing even if the
  // root itself is a data node (callers that want root should pass its
  // path explicitly).
  if (!targetPath) return chain;

  // Target equals root: include root iff it carries attrs (non-scene).
  if (targetPath === root.path) {
    if (root.type !== 'scene') chain.push(root);
    return chain;
  }

  if (root.type !== 'scene') chain.push(root);

  let cursor: SceneNode = root;
  while (cursor.path !== targetPath) {
    const next: SceneNode | undefined = cursor.children?.find(
      (c) => c.path === targetPath || targetPath.startsWith(c.path + '/')
    );
    if (!next) break;
    chain.push(next);
    cursor = next;
  }

  return chain;
}

/**
 * Walk the scene graph from `root` toward `targetPath`, collecting the
 * composable attrs of every node on the path.
 */
export function collectAncestorAttrs(root: SceneNode, targetPath: string): ComposableAttrs[] {
  return collectAncestorNodes(root, targetPath).map((n) => toComposable(n.attrs));
}

/**
 * Convenience: compose effective attrs for a target path in the scene graph.
 */
export function getEffectiveAttrs(root: SceneNode, targetPath: string): EffectiveAttrs {
  return composeAttrs(collectAncestorAttrs(root, targetPath));
}

/**
 * Collect every data-leaf descended from `start`.
 * Used by group-layer controls that need to fan out to actual materials.
 *
 * "Data-leaf" is the whole geometry vocabulary, not a fixed list: a group-level
 * opacity / gamma / colormap control must reach every leaf underneath it, so a
 * geometry type omitted here would silently ignore its ancestors' attributes.
 */
export function collectDataDescendants(start: SceneNode): SceneNode[] {
  const result: SceneNode[] = [];
  const visit = (n: SceneNode): void => {
    if (isGeometryType(n.type)) {
      result.push(n);
    }
    if (n.children) for (const c of n.children) visit(c);
  };
  visit(start);
  return result;
}

function toComposable(attrs: SceneNode['attrs']): ComposableAttrs {
  return {
    opacity: attrs.opacity as number | undefined,
    absorption: attrs.absorption as number | undefined,
    gamma: attrs.gamma as number | undefined,
    intensity: attrs.intensity as number | undefined,
    offset: attrs.offset as number | undefined,
    blending_mode: attrs.blending_mode as string | undefined,
    join: attrs.join as string | undefined,
    colormap: attrs.colormap as string | undefined,
    customLutBytes: attrs.customLutBytes as Uint8Array | undefined,
    layer_order: sanitizeLayerOrder(attrs.layer_order),
  };
}

/**
 * Read an authored `layer_order` off a raw zarr attrs record.
 *
 * TOLERANT read against a STRICT write: the Python writer refuses anything but
 * an `int`, but a hand-edited or third-party store can carry anything, and the
 * viewer must still render. A non-number, `NaN` or `±Infinity` would poison the
 * band comparator, so it is treated as *absent* — the node falls back to today's
 * inferred ordering rather than to an arbitrary band.
 *
 * A finite non-integer is accepted as authored rather than rounded: the value's
 * only meaning is its order, so `1.5` bands perfectly well, and rounding would
 * silently merge two bands the author separated.
 */
function sanitizeLayerOrder(raw: unknown): number | undefined {
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}
