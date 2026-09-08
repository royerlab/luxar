/**
 * Layer state management for the Layers panel.
 *
 * Pure data model (no DOM) that tracks per-layer state and provides
 * the min/max-to-intensity/offset mapping for the shader uniforms.
 */

import type { SceneNode } from '../../data/data-loader-types';
import { collectDataDescendants, getEffectiveAttrs } from '../../data/attrs-composer';
import type { BlendingMode } from '../../types/blending';
import type { GeometryTypeName, NodeKind } from '../../types/format-contract';
import { defaultBlendingMode, isGeometryType } from '../../types/geometry-capabilities';
import { log, Modules } from '../../utils/log';
// Pure display-window ↔ shader-uniform math now lives in the rendering layer
// (`rendering/display-range`) so `rendering/` modules can import it without
// violating the module layer order. Imported for this module's own internal
// use and re-exported below so every existing `./layer-state` consumer keeps
// working unchanged.
import { computeUniforms, computeDisplayRange } from '../../rendering/display-range';
import {
  MESH_DEFAULTS,
  resolveMeshBlendingMode,
  type MeshShadingMode,
} from '../../rendering/materials/mesh/appearance';
import { resolveMeshShading } from '../../rendering/node-factory/create-mesh-node';
import type { MeshMaterialKind, MeshMetadata } from '../../types/mesh';
import {
  PHYSICAL_MESH_KNOB_KEYS,
  PHYSICAL_MESH_KNOBS,
  type PhysicalMeshKnobKey,
} from '../../rendering/materials/mesh-physical/config';

/**
 * Geometry type of a layer.
 *
 * - ``group`` — composite container; controls fan out to descendants.
 * - every {@link GeometryTypeName} — leaf data layers. Deliberately the whole
 *   geometry vocabulary: any leaf type the format can carry is a layer the panel
 *   must be able to list.
 *
 * Specialized groups (``kind === 'lod'`` / ``'partition'``) appear in the layers
 * panel as their underlying ``display_type`` — never as ``'group'``. That
 * ``display_type`` is drawn from the LOD/partition-capable *subset* of the
 * vocabulary, not from `LayerType` itself (see `types/geometry-capabilities`).
 * The specialized-group nature is surfaced via the ``kind`` field on
 * ``LayerInfo``, which drives the per-layer badge / LOD dropdown.
 */
export type LayerType = GeometryTypeName | 'group' | 'sound';

/**
 * The sound-specific half of a `sound` layer (`SOUND_SPEC.md` §4.2): the live
 * per-node gain the row's slider edits, and the provenance the tooltip shows.
 * A sound layer is HEARD, not drawn, so every appearance field on its
 * {@link LayerInfo} is a neutral placeholder the controls section hides.
 */
export interface SoundLayerInfo {
  /** Live linear gain (authored `gain` at load). */
  gain: number;
  bus: string;
  trigger: string;
  license?: string;
  attribution?: string;
  sourceUrl?: string;
}

/**
 * Coerce a node's raw `layer` attr into "exposed in the Layers panel".
 *
 * The Python writer (`validate_layer`) always normalises to a JSON boolean,
 * but hand-edited or third-party zarr may carry a number (`1`/`0`). Accept
 * strict `true` and truthy finite numbers so such values don't silently drop
 * the node from the panel; everything else (incl. `undefined`, strings) is
 * not-a-layer. Pure — callers log a warning for malformed (non-boolean) values.
 */
export function isLayerEnabled(value: unknown): boolean {
  if (value === true) return true;
  if (typeof value === 'number') return Number.isFinite(value) && value !== 0;
  return false;
}

/**
 * Derive a SCALAR display data-range for a composite group (kind=lod /
 * kind=partition) that carries no range of its own, by walking to the FINEST
 * descendant leaf (largest `n_splats`) that declares one. The finest level is
 * the full-detail data, so its range is the representative signal window; using
 * it avoids the near-black render a [0, 1] fallback produces for a colormapped
 * gsplat layer. Returns undefined when no descendant declares a range.
 *
 * Deliberately does NOT consider `color_data_range`: that attr describes the
 * spread of authored RGB, not a scalar signal, and windowing on it
 * contrast-stretches the author's colours (see `initialDisplayRange`).
 */
function deriveScalarRangeFromDescendants(node: SceneNode): [number, number] | undefined {
  let best: [number, number] | undefined;
  let bestCount = -1;
  const visit = (n: SceneNode): void => {
    const r = (n.attrs.scalar_data_range || n.attrs.amplitude_data_range) as
      [number, number] | undefined;
    const count = (n.attrs.n_splats as number | undefined) ?? 0;
    if (r && count > bestCount) {
      best = r;
      bestCount = count;
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return best;
}

type LabelVocabulary = Array<{ id: string; name: string }>;

/**
 * Read an authored `layer_order` off a raw attrs record, or `undefined`.
 *
 * The SAME predicate the renderer uses (`render-order.ts::authoredLayerOrder`)
 * and the composer uses (`attrs-composer.ts::sanitizeLayerOrder`): a
 * non-number, NaN or ±Infinity is ABSENT, not a band. Kept identical on purpose
 * — if the panel and the renderer disagreed about what counts as authored, the
 * field would show a value the render is not using.
 */
function sanitizedLayerOrder(raw: unknown): number | undefined {
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

function labelVocabularyFromAttrs(node: SceneNode): LabelVocabulary | undefined {
  const vocabulary = node.attrs.label_vocabulary as Record<string, string> | undefined;
  return vocabulary ? Object.entries(vocabulary).map(([id, name]) => ({ id, name })) : undefined;
}

function deriveLabelVocabularyFromDescendants(node: SceneNode): LabelVocabulary | undefined {
  const gsplatDescendants = collectDataDescendants(node).filter(
    (descendant) => descendant.type === 'gsplats'
  );
  if (gsplatDescendants.length === 0) return undefined;

  const first = labelVocabularyFromAttrs(gsplatDescendants[0]);
  if (!first) return undefined;
  for (const descendant of gsplatDescendants.slice(1)) {
    const candidate = labelVocabularyFromAttrs(descendant);
    if (
      !candidate ||
      candidate.length !== first.length ||
      candidate.some((entry) => first.find((item) => item.id === entry.id)?.name !== entry.name)
    ) {
      return undefined;
    }
  }
  return first;
}

/** Direct element-colour or texture range stamped on this node. */
function directColorRange(node: SceneNode): [number, number] | undefined {
  return (node.attrs.color_data_range || node.attrs.texture_data_range) as
    [number, number] | undefined;
}

/**
 * Direct colour range, else the UNION of every descendant's.
 * A partition's parts each declare their own spread, and the slider bounds
 * must cover the whole layer — taking the first part's range alone would
 * leave a later HDR part's colours unreachable.
 */
function deriveColorRangeFromDescendants(node: SceneNode): [number, number] | undefined {
  const own = directColorRange(node);
  if (own) return own;
  let min = Infinity;
  let max = -Infinity;
  const visit = (n: SceneNode): void => {
    const r = directColorRange(n);
    if (r) {
      min = Math.min(min, r[0]);
      max = Math.max(max, r[1]);
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return min <= max ? [min, max] : undefined;
}

/**
 * True when `node` — or a descendant it OWNS — renders through a colormap LUT.
 *
 * The walk stops at a nested `layer=true` descendant: that node is its own row
 * in the panel with its own colormap control, so its palette can change at any
 * time. Deriving THIS layer's mode from it would freeze a snapshot that goes
 * stale the moment the inner control is used, leaving the outer layer claiming
 * a scalar window over a leaf that has since gone back to direct colour (its
 * range control then routes to the identity and does nothing). Every writer
 * routes `layer` through `COMPOSITING_ATTRS` onto the wrapper only, so a
 * kind=partition / kind=lod layer never has layer descendants — the boundary
 * only bites on explicitly nested authored layers.
 */
function usesColormap(node: SceneNode): boolean {
  if (node.attrs.colormap) return true;
  return (node.children ?? []).some((c) => !isLayerEnabled(c.attrs.layer) && usesColormap(c));
}

/**
 * The palette a wrapper layer EFFECTIVELY renders through when it carries no
 * `colormap` attr of its own: the single distinct palette authored on its
 * descendants, or undefined when there is none or they disagree (a mixed
 * subtree has no single palette the dropdown could show). `colormap` is
 * deliberately NOT a compositing attr on the Python side — the writer copies
 * it onto every leaf — so a kind=partition / kind=lod layer from
 * `add_gsplats_from_file(..., colormap=...)` always has this shape. Surfacing
 * the descendant palette keeps the dropdown and the legend truthful and makes
 * "(direct colors)" an actual off-switch there (a select only emits `change`
 * when its value moves).
 *
 * Stops at nested `layer=true` descendants for the same reason
 * {@link usesColormap} does — they own their palette.
 */
function deriveColormapFromDescendants(node: SceneNode): string | undefined {
  let found: string | undefined;
  let mixed = false;
  const visit = (n: SceneNode): void => {
    if (mixed || isLayerEnabled(n.attrs.layer)) return;
    const cm = n.attrs.colormap as string | undefined;
    if (cm) {
      if (found === undefined) found = cm;
      else if (found !== cm) mixed = true;
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return mixed ? undefined : found;
}

/**
 * A numeric mesh appearance attr (`ambient` / `shade_exponent` / `alpha_cutoff` /
 * `specular` / `shininess`) read from `node`, else from the first descendant that
 * carries one.
 *
 * Same shape and same reason as {@link deriveColormapFromDescendants}: these are
 * NOT compositing attrs on the Python side, so `add_mesh(partition=…, ambient=…)`
 * stamps them on every `part_<i>` leaf and leaves the kind=partition wrapper bare.
 * The panel presents that wrapper as the mesh layer, so without this the sliders
 * would open at the material defaults while the surface renders the authored
 * values — and the first drag would jump.
 *
 * Stops at nested `layer=true` descendants, which own their own appearance row.
 */
function deriveMeshAttrFromDescendants(node: SceneNode, attr: string): number | undefined {
  const own = node.attrs[attr];
  if (typeof own === 'number') return own;
  let found: number | undefined;
  const visit = (n: SceneNode): void => {
    if (found !== undefined || isLayerEnabled(n.attrs.layer)) return;
    const value = n.attrs[attr];
    if (typeof value === 'number') {
      found = value;
      return;
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return found;
}

/**
 * Resolve the effective mesh shading for a layer row.
 *
 * A mesh leaf resolves directly. Specialized mesh groups carry the appearance attrs
 * on their leaves, so walk owned descendants while stopping at nested layer rows. A
 * malformed mixed group remains controllable when any descendant is lit; only an
 * all-unlit layer hides the four lighting sliders. The resolver intentionally receives
 * raw `has_normals`: only its `none` arm is authoritative here, while the material's
 * smooth/flat choice remains view-dependent on `normal_dims` matching `displayDims`.
 */
function deriveMeshShadingFromDescendants(node: SceneNode): MeshShadingMode {
  const resolve = (candidate: SceneNode): MeshShadingMode =>
    resolveMeshShading(
      candidate.attrs as unknown as MeshMetadata,
      candidate.attrs.has_normals === true
    );

  if (node.type === 'mesh') return resolve(node);

  let found: MeshShadingMode | undefined;
  const visit = (candidate: SceneNode): void => {
    if (found !== undefined && found !== 'none') return;
    if (isLayerEnabled(candidate.attrs.layer)) return;
    if (candidate.type === 'mesh') {
      found = resolve(candidate);
      return;
    }
    candidate.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return found ?? 'flat';
}

/**
 * The AUTHORED physical knobs of a mesh layer, as the panel lists them read-only.
 *
 * Only knobs actually present are returned — an absent knob renders as three's
 * default, and the panel says so rather than printing a number the author never
 * wrote. Read off the node, else off its first mesh descendant, for the same
 * kind=partition reason as `deriveMeshAttrFromDescendants`.
 */
export type PhysicalKnobValues = Record<PhysicalMeshKnobKey, number> & {
  sheen_color?: string;
  attenuation_color?: string;
  alpha_cutoff?: number;
  /**
   * The LIVE `refract_data` switch (spec §3.4 Phase 3): the glass draws after, and
   * refracts, the emissive data. Always present on a physical layer (absent authored
   * attr = false) so the toggle always has a state to show.
   */
  refract_data?: boolean;
};

/**
 * Which material family a mesh layer renders with (spec
 * `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.1). A partition wrapper reads it off its
 * parts; an absent attr is the house shader.
 */
function deriveMeshMaterialFromDescendants(node: SceneNode): MeshMaterialKind {
  const read = (candidate: SceneNode): MeshMaterialKind | undefined =>
    candidate.attrs.material === 'physical' ? 'physical' : undefined;
  if (node.type === 'mesh') return read(node) ?? 'luxar';
  let found: MeshMaterialKind | undefined;
  const visit = (candidate: SceneNode): void => {
    if (found !== undefined || isLayerEnabled(candidate.attrs.layer)) return;
    if (candidate.type === 'mesh') {
      found = read(candidate);
      return;
    }
    candidate.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return found ?? 'luxar';
}

/**
 * The DENSE physical knob record a layer's sliders start from: the authored value
 * where one exists, else the knob's default — in the MATERIAL domain. Keeping the
 * authored value here matters for unbounded knobs: their finite slider tracks are a
 * presentation detail and cannot round-trip a thickness or attenuation distance above
 * the track maximum.
 */
function derivePhysicalKnobsFromDescendants(node: SceneNode): PhysicalKnobValues {
  const knobs = {} as PhysicalKnobValues;
  for (const key of PHYSICAL_MESH_KNOB_KEYS) {
    const authored = deriveMeshAttrFromDescendants(node, key);
    knobs[key] = authored ?? PHYSICAL_MESH_KNOBS[key].default;
  }
  const cutoff = deriveMeshAttrFromDescendants(node, 'alpha_cutoff');
  if (cutoff !== undefined) knobs.alpha_cutoff = cutoff;
  const sheen = deriveMeshStringAttrFromDescendants(node, 'sheen_color');
  if (sheen !== undefined) knobs.sheen_color = sheen;
  const attenuation = deriveMeshStringAttrFromDescendants(node, 'attenuation_color');
  if (attenuation !== undefined) knobs.attenuation_color = attenuation;
  knobs.refract_data = deriveMeshBoolAttrFromDescendants(node, 'refract_data') ?? false;
  return knobs;
}

/** The boolean twin of `deriveMeshStringAttrFromDescendants` (the `refract_data` flag). */
function deriveMeshBoolAttrFromDescendants(node: SceneNode, attr: string): boolean | undefined {
  const read = (candidate: SceneNode): boolean | undefined =>
    typeof candidate.attrs[attr] === 'boolean' ? (candidate.attrs[attr] as boolean) : undefined;
  let value = read(node);
  const visit = (candidate: SceneNode): void => {
    if (value !== undefined || isLayerEnabled(candidate.attrs.layer)) return;
    value = read(candidate);
    if (value === undefined) candidate.children?.forEach(visit);
  };
  if (value === undefined) node.children?.forEach(visit);
  return value;
}

/** The string-valued twin of `deriveMeshAttrFromDescendants` (the two colour knobs). */
function deriveMeshStringAttrFromDescendants(node: SceneNode, attr: string): string | undefined {
  const read = (candidate: SceneNode): string | undefined =>
    typeof candidate.attrs[attr] === 'string' ? (candidate.attrs[attr] as string) : undefined;
  let value = read(node);
  const visit = (candidate: SceneNode): void => {
    if (value !== undefined || isLayerEnabled(candidate.attrs.layer)) return;
    value = read(candidate);
    if (value === undefined) candidate.children?.forEach(visit);
  };
  if (value === undefined) node.children?.forEach(visit);
  return value;
}

/**
 * The display window a layer starts at — i.e. what `[displayMin, displayMax]`
 * the panel pushes into the material before the user touches anything.
 *
 * The window maps the rendered VALUE to [0, 1], so which range is right depends
 * on what that value is:
 *
 * * **Colormapped layers** window a scalar (gsplat amplitude / points-lines
 *   scalar), whose useful range is the data range — a linear [0, 1] window on a
 *   heavily right-skewed amplitude renders near-black (#522).
 * * **Direct-colour layers** window authored RGB, whose range IS [0, 1]. Any
 *   other window is a contrast stretch of colours the author already chose:
 *   a uniform grey `(0.72, 0.74, 0.78)` has `color_data_range` [0.72, 0.78],
 *   which windows to gain 16.7 / offset −12 and renders as saturated BLUE.
 *   So direct colour starts at the identity window.
 */
function initialDisplayRange(node: SceneNode, scalarWindow: boolean): [number, number] {
  if (!scalarWindow) return [0, 1];
  const scalarRange = (node.attrs.scalar_data_range || node.attrs.amplitude_data_range) as
    [number, number] | undefined;
  return scalarRange ?? deriveScalarRangeFromDescendants(node) ?? [0, 1];
}

/**
 * Specialized-group discriminant on a layer. Single-sourced from the
 * cross-language format contract (format-contract/contract.yaml).
 */
export type LayerKind = NodeKind;

/** Information about a single layer in the Layers panel */
export interface LayerInfo {
  /** Zarr path (e.g. "group/channel_gfp") — used as unique key */
  path: string;
  /** Display name (last segment of path) */
  name: string;
  /**
   * Geometry type the user sees this layer as. For specialized groups
   * (kind=lod / kind=partition), this is the resolved ``display_type`` attr
   * from disk — not ``'group'``.
   */
  type: LayerType;
  /**
   * Specialized-group kind, if the underlying scene-graph node is a
   * kind=lod or kind=partition ``Group``. Drives the layer-header badge and
   * the inline LOD-level dropdown.
   */
  kind?: LayerKind;
  /** Whether the layer is visible in the scene */
  visible: boolean;
  /** Opacity (0–1) */
  opacity: number;
  /** Absorption coefficient κ (≥ 0; only meaningful in volumetric mode) */
  absorption: number;
  /**
   * Mesh shade floor (0–1) — the §6.2 wrapped-diffuse `ambient`. Only meaningful on a
   * mesh layer, where it is what keeps a silhouette readable rather than black; `1.0`
   * removes the diffuse gradient; specular remains independently controlled.
   */
  ambient: number;
  /**
   * Mesh wrapped-diffuse falloff exponent (> 0) — the §6.2 `shade_exponent`. `1.0` is the
   * plain linear wrap. Mesh-only, like the three around it.
   */
  shadeExponent: number;
  /** Mesh additive specular strength (0–1). */
  specular: number;
  /** Mesh specular highlight exponent (> 0). */
  shininess: number;
  /**
   * Mesh shading capability for the layer controls. Only `none` is authoritative;
   * smooth versus flat ignores the material's view-dependent normal-frame check.
   */
  shading: MeshShadingMode;
  /**
   * Which material family a mesh layer renders with. `'luxar'` (the house shader,
   * and what every non-mesh layer carries) shows the shading sliders;
   * `'physical'` hides them and shows the {@link physicalKnobs} sliders instead.
   */
  material?: MeshMaterialKind;
  /**
   * The LIVE physical knobs in material space, seeded from the authored attrs and
   * knob defaults. Slider mapping is presentation-only, so values beyond a finite
   * track remain intact. Absent for every non-physical layer.
   */
  physicalKnobs?: PhysicalKnobValues;
  /**
   * Mesh `opaque`-mode cutout threshold (0–1) — the §6.2 `alpha_cutoff`.
   *
   * Only meaningful in `opaque`, which is a NARROWER condition than the other four
   * (they apply in every blending mode when shading is lit), so the panel gates its
   * slider on the mode as well as the type — the same shape as absorption's
   * volumetric gate.
   */
  alphaCutoff: number;
  /** Current display-range minimum (maps to intensity+offset in shader) */
  displayMin: number;
  /** Current display-range maximum */
  displayMax: number;
  /**
   * Slider lower bound. Wide enough to reach every window the layer can
   * usefully take: the mode's default (scalar range / identity), the authored
   * `intensity`/`offset` window, and — for a direct-colour layer — its
   * `color_data_range`, so "stretch these colours" stays one drag away.
   */
  dataMin: number;
  /** Slider upper bound (see {@link LayerInfo.dataMin}). */
  dataMax: number;
  /** Gamma correction value */
  gamma: number;
  /** Blending mode */
  blendingMode: BlendingMode;
  /**
   * Whether this layer OWNS a blend mode — authored on the node ITSELF on disk,
   * or chosen by the user via the panel. When false, `blendingMode` is merely the
   * inherited/defaulted value shown in the dropdown, and the layer must not act
   * as a `blending_mode` SETTER: `liveLayerAttrs` emits the attr only when this
   * is true (a plain group layer would otherwise push its `additive` placeholder
   * onto a mesh leaf and bury the mesh's own `opaque` default — #1272), and
   * `composeEffective`'s subtree-drop fires only when this is true (a wrapper
   * owning no mode has no control value to impose, so it must not suppress a
   * descendant's authored mode — #1275).
   *
   * Deliberately the node's OWN attr, not the composed ancestry: a layer that
   * merely INHERITS an ancestor's mode must not re-emit it as its own setter —
   * the panel snapshot would go stale the moment the ancestor layer's live pick
   * diverges from disk, and the re-emitted copy (being nearer the leaf) would
   * shadow the ancestor's newer choice.
   */
  blendingModeExplicit: boolean;
  /**
   * Effective cross-layer draw order (`LAYER_ORDER_SPEC.md`), including an
   * inherited value, or `undefined` when the ancestry states none. Higher =
   * nearer the camera = drawn later.
   */
  layerOrder?: number;
  /** Effective order inherited from ancestors when this layer owns none. */
  inheritedLayerOrder?: number;
  /**
   * Whether the level is EXPLICIT — the node's OWN attr, or a user pick in the
   * panel — as opposed to inherited or absent. Same reasoning as
   * `blendingModeExplicit`: re-emitting an inherited level as this layer's own
   * would change which node owns the setter and make clearing the ancestor no
   * longer restore `auto` here.
   */
  layerOrderExplicit: boolean;
  /** Whether this layer is selected in the list */
  selected: boolean;
  /** Present on `type === 'sound'` layers only. */
  sound?: SoundLayerInfo;
  /** Active colormap name (undefined = direct RGB colors) */
  colormap?: string;
  /** Whether this node supports colormap (has scalars or amplitudes) */
  supportsColormap: boolean;
  /** Exact categorical vocabulary exposed by gsplat label_ids. */
  labelVocabulary?: Array<{ id: string; name: string }>;
  /** Render deterministic categorical colours instead of authored RGB. */
  colorByLabel: boolean;
  /** Exact class id selection; undefined shows all classes. */
  labelFilterId?: string;
  /** Scalar data range for colormap normalization */
  scalarDataRange?: [number, number];
  /**
   * Authored RGB spread (`color_data_range`). Never the starting window — see
   * {@link LayerInfo.dataMin} — but kept so a colormap toggle can restore the
   * direct-colour slider bounds, symmetric with {@link LayerInfo.scalarDataRange}.
   */
  colorDataRange?: [number, number];
  /**
   * Whether the current display window is a SCALAR window (a colormap is in
   * play for the layer — on the node or a descendant) rather than an
   * authored-RGB one. `LayerApplyEngine.applyComposed` routes on this per
   * leaf: a leaf the C1 guard keeps on direct colour must NOT receive the
   * scalar window as a colour gain (a mixed group layer contains both kinds).
   * Follows the colormap toggle via {@link LayerStateManager.setColormapWindow}.
   */
  scalarWindow: boolean;
  /**
   * For ``kind === 'lod'`` layers: number of child levels. Drives the
   * "Active level" dropdown's option count and the "N LODs" badge.
   * Absent for non-LOD layers.
   */
  lodGroupChildCount?: number;
  /**
   * For ``kind === 'partition'`` layers: number of BSP child parts. Drives
   * the "N parts" badge on the layer header. Absent for non-Partition
   * layers.
   */
  partCount?: number;
  /**
   * For ``kind === 'partition'`` layers that wrap kind=lod descendants:
   * absolute paths of every nested lod_group in the subtree. The
   * "Active level" dropdown on the wrapper broadcasts to every path
   * (clamped per-group by ``LODGroupRegistry.setSelectorMode`` on
   * ragged ladders). Absent for non-Partition layers and for Partition layers
   * with no nested LODs.
   *
   * Also used to compute the ``[N parts × M LODs]`` combined badge:
   * when this list is non-empty, M is the maximum child count across
   * nested groups (matches the dropdown's "highest level offered"
   * affordance).
   */
  nestedLodGroupPaths?: string[];
  /**
   * Maximum LOD child count across the layer's nested lod_groups.
   * Used by the dropdown to size its option list and by the combined
   * badge. Absent unless ``nestedLodGroupPaths`` is non-empty.
   */
  nestedLodMaxChildCount?: number;
}

// Re-export the relocated display-range helpers (imported at the top of this
// file) so every existing `./layer-state` consumer keeps working unchanged.
export { computeUniforms, computeDisplayRange };
export type { DisplayUniforms } from '../../rendering/display-range';

/** Selection mode for layer clicks */
export type SelectionMode = 'single' | 'add' | 'range';

/** Listener callback type */
export type LayerChangeListener = () => void;

/**
 * Manages the state of all layers in the Layers panel.
 *
 * Walks the SceneNode tree to collect nodes with `layer: true`,
 * tracks selection, and provides mutators that notify listeners.
 */
/**
 * The blending mode a layer of `type` will ACTUALLY render with.
 *
 * For mesh this is not the composed value: `volumetric` has no meaning for a
 * zero-thickness surface, so the mesh material maps it to `opaque` and stamps the
 * RESOLVED mode into `userData.blendingMode` (spec §6.3). Storing the unresolved value
 * in `LayerInfo` made the panel disagree with the render in two visible ways at once —
 * it showed the Absorption slider (which no mesh shader reads) and HID the Alpha-cutoff
 * slider precisely when the cutout was active. The pick pass reads the material's
 * resolved mode, so it was correct and the UI was not.
 *
 * Resolved at the point of STORAGE rather than at each display gate, so every consumer
 * of `LayerInfo.blendingMode` — the Blend dropdown's own displayed value included — sees
 * the mode that renders. A user who explicitly picks `volumetric` on a mesh sees the
 * dropdown snap back to `opaque`, which is honest: it is what the surface is doing, and
 * it matches the one-time warning the loader already emits.
 */
export function resolveLayerBlendingMode(
  type: LayerType | undefined,
  mode: BlendingMode
): BlendingMode {
  return type === 'mesh' ? resolveMeshBlendingMode(mode) : mode;
}

export class LayerStateManager {
  /** Ordered list of layer paths (insertion order from scene graph walk) */
  private layerOrder: string[] = [];
  /** Map from path to LayerInfo */
  private layers = new Map<string, LayerInfo>();
  /** Path of the last clicked layer (anchor for Shift+click range) */
  private lastClickedPath: string | null = null;
  /** Change listeners */
  private listeners = new Set<LayerChangeListener>();

  /**
   * Initialize layers from the scene graph.
   * Walks children recursively and collects nodes with `layer: true`.
   */
  initFromSceneGraph(root: SceneNode): void {
    this.soloState = null;
    this.layers.clear();
    this.layerOrder = [];
    this.lastClickedPath = null;

    this.walkSceneGraph(root, root, undefined);
  }

  private walkSceneGraph(
    node: SceneNode,
    root: SceneNode,
    inheritedLayerOrder: number | undefined
  ): void {
    const ownLayerOrder =
      node.type === 'scene' ? undefined : sanitizedLayerOrder(node.attrs.layer_order);
    const effectiveLayerOrder = ownLayerOrder ?? inheritedLayerOrder;
    // Skip the root scene node; collect anything else with layer=true.
    // Groups exposed as layers act as composites — their controls fan out
    // to every data descendant when applied in the scene.
    const layerAttr = node.attrs.layer;
    if (layerAttr !== undefined && typeof layerAttr !== 'boolean') {
      // Producers should write a JSON boolean; coerce truthy values but warn
      // so a malformed (e.g. string) attr doesn't silently drop the node.
      log.warning(
        Modules.UI,
        `Node "${node.path}" has a non-boolean 'layer' attr (${typeof layerAttr}); coercing.`
      );
    }
    if (node.type !== 'scene' && isLayerEnabled(layerAttr)) {
      // Vocabulary question, not a capability one: any geometry leaf the
      // format can carry must be listable in the panel (see `LayerType`).
      const isLayerType = isGeometryType(node.type) || node.type === 'group';
      if (node.type === 'sound') {
        this.pushSoundLayer(node, effectiveLayerOrder);
      } else if (isLayerType) {
        const name = node.path.split('/').pop() || node.path;

        const ampRange = node.attrs.amplitude_data_range as [number, number] | undefined;
        const scalarRange = node.attrs.scalar_data_range as [number, number] | undefined;

        // Colormap support — gsplats inherit palettes directly; points / lines /
        // mesh inherit one only when they have scalars. A group inherits only
        // when at least one descendant can consume the palette.
        // Gsplats only support a colormap when they actually have scalar
        // data (`has_scalars`) or an authored `colormap`; a bare gsplats
        // node with no scalars must NOT advertise colormap support, or
        // the UI offers a no-op colormap dropdown.
        const groupCanUseInheritedColormap =
          node.type === 'group' &&
          collectDataDescendants(node).some(
            (descendant) => descendant.type === 'gsplats' || !!descendant.attrs.has_scalars
          );
        const canUseInheritedColormap =
          groupCanUseInheritedColormap || node.type === 'gsplats' || !!node.attrs.has_scalars;
        const inheritedColormap = canUseInheritedColormap
          ? getEffectiveAttrs(root, node.path).colormap
          : undefined;
        const colormap =
          (node.attrs.colormap as string | undefined) ||
          deriveColormapFromDescendants(node) ||
          inheritedColormap;
        const scalarWindow = !!colormap || usesColormap(node);
        const supportsColormap =
          groupCanUseInheritedColormap || !!node.attrs.has_scalars || !!colormap;
        const colormapScalarRange =
          scalarRange || ampRange || deriveScalarRangeFromDescendants(node);
        const labelVocabulary =
          labelVocabularyFromAttrs(node) || deriveLabelVocabularyFromDescendants(node);

        // The window the layer starts at. Colormapped layers window a scalar
        // (from this node or, for a composite kind=lod / kind=partition group,
        // its finest descendant leaf); direct-colour layers window authored RGB
        // and so start at the identity [0, 1] — see `initialDisplayRange`.
        const dataRange = initialDisplayRange(node, scalarWindow);

        // Initialize display range from existing intensity/offset if present,
        // otherwise default to full data range
        const intensity = (node.attrs.intensity as number) ?? 1.0;
        const offset = (node.attrs.offset as number) ?? 0.0;
        let displayMin: number;
        let displayMax: number;

        if (intensity === 1.0 && offset === 0.0) {
          displayMin = dataRange[0];
          displayMax = dataRange[1];
        } else {
          const recovered = computeDisplayRange(intensity, offset);
          displayMin = recovered.min;
          displayMax = recovered.max;
        }

        // Slider bounds must encompass the current display range. The recovered
        // [displayMin, displayMax] can extend beyond color_data_range when the
        // node has authored intensity/offset (e.g., intensity=0.2 implies a
        // display range of [0, 5] regardless of where the actual color values
        // lie). If the slider's <input type="range"> min/max stayed at the
        // narrower data range, the browser would silently clamp the thumb
        // values on first render, and the first slider interaction would write
        // the clamped values back — snapping intensity from the authored value
        // to the slider-implied one and causing a sudden brightness jump.
        // A direct-colour layer no longer STARTS at its colour range, but the
        // slider must still reach it so "stretch these colours" stays a
        // one-drag operation. Kept on the layer either way (even when a
        // colormap currently wins) so `setColormapWindow` can restore these
        // bounds when the colormap is switched off.
        const colorDataRange = deriveColorRangeFromDescendants(node);
        const colorRange = scalarWindow ? undefined : colorDataRange;
        const dataMin = Math.min(dataRange[0], displayMin, colorRange?.[0] ?? Infinity);
        const dataMax = Math.max(dataRange[1], displayMax, colorRange?.[1] ?? -Infinity);

        // Honor the authoring-time `visible` attr (default true). Allows
        // Python authors to start a layer hidden via add_points(..., visible=False).
        const initialVisible = node.attrs.visible !== false;

        // Specialized-group discriminant: read ``kind`` from the on-disk
        // attrs of a ``type === 'group'`` node. ``undefined`` for plain
        // groups and leaf nodes.
        const rawKind = node.attrs.kind as LayerKind | undefined;
        const kind: LayerKind | undefined =
          node.type === 'group' && (rawKind === 'lod' || rawKind === 'partition')
            ? rawKind
            : undefined;

        // The layer's user-facing geometry type. For a specialized
        // group, use its ``display_type`` (resolved at write time on the
        // Python side) so the layer reads as e.g. "gsplats" not "group".
        // Fall back to the node's raw type for plain groups / leaves.
        const displayType = node.attrs.display_type as LayerType | undefined;
        const layerType: LayerType =
          kind !== undefined && displayType ? displayType : (node.type as LayerType);

        // Kind-specific badge counts. lod → "N LODs" + dropdown; partition →
        // "N parts" status chip. Other layers leave both undefined.
        const lodGroupChildCount = kind === 'lod' ? (node.children?.length ?? 0) : undefined;
        const partCount = kind === 'partition' ? (node.children?.length ?? 0) : undefined;

        // Partition-of-LOD discovery. A kind=partition layer that wraps
        // kind=lod descendants gets a broadcast dropdown over every
        // nested lod_group. The walk stops at the first lod_group it
        // hits per branch — an lod_group's own children are leaves of
        // the LOD ladder, not further LOD wrappers.
        let nestedLodGroupPaths: string[] | undefined;
        let nestedLodMaxChildCount: number | undefined;
        if (kind === 'partition') {
          const collected: string[] = [];
          let maxCount = 0;
          const visit = (n: SceneNode): void => {
            const childKind = n.attrs.kind as LayerKind | undefined;
            if (
              n.type === 'group' &&
              childKind === 'lod' &&
              n !== node // skip self
            ) {
              collected.push(n.path);
              maxCount = Math.max(maxCount, n.children?.length ?? 0);
              return; // do not descend into this lod_group's children
            }
            for (const c of n.children ?? []) visit(c);
          };
          for (const c of node.children ?? []) visit(c);
          if (collected.length > 0) {
            nestedLodGroupPaths = collected;
            nestedLodMaxChildCount = maxCount;
          }
        }

        // Composed blend mode: `undefined` when NO ancestry level set one, else
        // the canonical inherited/authored mode (#1272). The `undefined` case is
        // exactly what tells us to fall back to the per-type default AND to leave
        // the layer non-explicit so it does not impose that default on descendants.
        const effectiveAttrs = getEffectiveAttrs(root, node.path);
        const composedBlendingMode = effectiveAttrs.blending_mode;
        const material = layerType === 'mesh' ? deriveMeshMaterialFromDescendants(node) : 'luxar';

        this.layerOrder.push(node.path);
        this.layers.set(node.path, {
          path: node.path,
          name,
          type: layerType,
          kind,
          visible: initialVisible,
          opacity: (node.attrs.opacity as number) ?? 1.0,
          // RAW like opacity, NOT composed: composeEffective substitutes
          // each layer's live values per ancestry node, so a composed
          // init would multiply ancestor κ in twice.
          absorption: (node.attrs.absorption as number) ?? 1.0,
          // RAW, and defaulted from the material's own constants rather than
          // re-spelled here: these are the values `MeshMaterial` starts at when the
          // attr is absent, so the slider must open on the same number the surface is
          // already rendering with. They do NOT compose along the ancestry (unlike
          // opacity/gamma) — a shade floor is a per-surface appearance choice, not a
          // multiplicative attr. The writer never stamps them on a PLAIN group, but it
          // does stamp them on every part of a kind=partition mesh (they are not
          // compositing attrs), so a wrapper layer reads them off its parts.
          ambient: deriveMeshAttrFromDescendants(node, 'ambient') ?? MESH_DEFAULTS.ambient,
          shadeExponent:
            deriveMeshAttrFromDescendants(node, 'shade_exponent') ?? MESH_DEFAULTS.shadeExponent,
          specular: deriveMeshAttrFromDescendants(node, 'specular') ?? MESH_DEFAULTS.specular,
          shininess: deriveMeshAttrFromDescendants(node, 'shininess') ?? MESH_DEFAULTS.shininess,
          shading: deriveMeshShadingFromDescendants(node),
          material,
          physicalKnobs:
            layerType === 'mesh' && material === 'physical'
              ? derivePhysicalKnobsFromDescendants(node)
              : undefined,
          alphaCutoff:
            deriveMeshAttrFromDescendants(node, 'alpha_cutoff') ?? MESH_DEFAULTS.alphaCutoff,
          displayMin,
          displayMax,
          dataMin,
          dataMax,
          gamma: (node.attrs.gamma as number) ?? 1.0,
          // Blending mode is COMPOSED along the ancestry (nearest set
          // ancestor wins, normalized by composeAttrs) — a geometry leaf's
          // dropdown must show the mode the material actually renders with,
          // not the node's own (possibly absent / malformed) raw attr. An
          // unset chain composes to `undefined`; show the per-type default
          // (mesh → opaque, emissive → additive) the material would use — but
          // leave it NON-owned so it is not pushed onto descendants (a
          // plain group layer merely displays `additive` as a neutral default;
          // each descendant keeps its own default until the control is used).
          //
          // Then RESOLVED for the layer's type, which is a separate concern from the
          // default and applies to an OWNED mode too: a mesh cannot render
          // `volumetric`, so its material maps that to `opaque` and stamps the resolved
          // value. Without this wrap the panel showed Absorption (which no mesh shader
          // reads) and hid Alpha cutoff exactly when the cutout was active. A no-op for
          // the default path, since `defaultBlendingMode('mesh')` is already `opaque`.
          //
          // The default is keyed on `layerType`, NOT `node.type`: a kind=partition /
          // kind=lod wrapper's raw type is `group` (default `additive`) while the layer —
          // and every leaf it wraps — is its `display_type`. For the emissive types the
          // two agree, but a partitioned MESH renders `opaque`, so keying on the raw type
          // showed "Additive" for an opaque surface and hid the Alpha-cutoff slider
          // (`layer-controls.ts::syncMeshAppearanceVisibility`) exactly when the cutout
          // was active. Plain groups and leaves have `layerType === node.type`.
          blendingMode: resolveLayerBlendingMode(
            layerType,
            composedBlendingMode ?? defaultBlendingMode(layerType)
          ),
          // Ownership reads the node's OWN attr — see the `LayerInfo` doc for why the
          // composed ancestry would be wrong (stale-snapshot shadowing) and why a
          // merely-inherited or defaulted mode must not make this layer a setter.
          // Uses `node.attrs`, so a composite kind=lod/partition wrapper (whose
          // display type is a geometry name but whose node authored no mode) stays
          // non-owning, exactly like a plain group.
          blendingModeExplicit: node.attrs.blending_mode != null,
          // Display the COMPOSED value the renderer uses, while ownership
          // remains the node's OWN sanitized attr. A loose `!= null` on the raw
          // attr (the shape `blendingModeExplicit` above can afford, because a
          // mode has a per-type fallback) would report `explicit: true` for a
          // junk value like `'front'` that the renderer discards.
          layerOrder: effectiveLayerOrder,
          inheritedLayerOrder,
          layerOrderExplicit: ownLayerOrder !== undefined,
          selected: false,
          colormap,
          supportsColormap,
          labelVocabulary,
          colorByLabel: false,
          labelFilterId: undefined,
          scalarDataRange: colormapScalarRange,
          colorDataRange,
          scalarWindow,
          lodGroupChildCount,
          partCount,
          nestedLodGroupPaths,
          nestedLodMaxChildCount,
        });
      }
    }

    // Recurse into children
    if (node.children) {
      for (const child of node.children) {
        this.walkSceneGraph(child, root, effectiveLayerOrder);
      }
    }
  }

  /**
   * A `sound` node exposed as a layer: eye = mute, slider = gain, tooltip =
   * licence. Every appearance field is a neutral placeholder — nothing here
   * reaches a material — so the row can sit in the same list as the geometry.
   */
  private pushSoundLayer(node: SceneNode, effectiveLayerOrder: number | undefined): void {
    const name = node.path.split('/').pop() || node.path;
    const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : undefined);
    const gain = typeof node.attrs.gain === 'number' && node.attrs.gain >= 0 ? node.attrs.gain : 1;
    this.layerOrder.push(node.path);
    this.layers.set(node.path, {
      path: node.path,
      name,
      type: 'sound',
      kind: undefined,
      visible: node.attrs.visible !== false,
      opacity: 1,
      absorption: 1,
      ambient: MESH_DEFAULTS.ambient,
      shadeExponent: MESH_DEFAULTS.shadeExponent,
      specular: MESH_DEFAULTS.specular,
      shininess: MESH_DEFAULTS.shininess,
      shading: 'none',
      alphaCutoff: MESH_DEFAULTS.alphaCutoff,
      displayMin: 0,
      displayMax: 1,
      dataMin: 0,
      dataMax: 1,
      gamma: 1,
      blendingMode: defaultBlendingMode('points'),
      blendingModeExplicit: false,
      layerOrder: effectiveLayerOrder,
      inheritedLayerOrder: effectiveLayerOrder,
      layerOrderExplicit: false,
      selected: false,
      sound: {
        gain,
        bus: str(node.attrs.bus) ?? 'ambient',
        trigger: str(node.attrs.trigger) ?? 'continuous',
        license: str(node.attrs.license),
        attribution: str(node.attrs.attribution),
        sourceUrl: str(node.attrs.source_url),
      },
      supportsColormap: false,
      colorByLabel: false,
      labelFilterId: undefined,
      scalarWindow: false,
    });
  }

  /** Set a sound layer's live gain (clamped to `[0, 2]`); a no-op on other layers. */
  setSoundGain(path: string, gain: number): void {
    const layer = this.layers.get(path);
    if (!layer?.sound || !Number.isFinite(gain)) return;
    layer.sound.gain = Math.min(2, Math.max(0, gain));
    this.notify();
  }

  /** Get all layers in display order */
  getLayers(): LayerInfo[] {
    return this.layerOrder.map((path) => this.layers.get(path)!);
  }

  /** Get a single layer by path */
  getLayer(path: string): LayerInfo | undefined {
    return this.layers.get(path);
  }

  /** Get the number of layers */
  get count(): number {
    return this.layerOrder.length;
  }

  // ─── Selection ───────────────────────────────────────────

  /**
   * Select a layer with napari-style modifiers.
   * - 'single': clear all, select this one
   * - 'add': toggle this one (Ctrl+click)
   * - 'range': select from lastClicked to this (Shift+click)
   */
  select(path: string, mode: SelectionMode): void {
    const layer = this.layers.get(path);
    if (!layer) return;

    switch (mode) {
      case 'single':
        for (const l of this.layers.values()) l.selected = false;
        layer.selected = true;
        this.lastClickedPath = path;
        break;

      case 'add':
        layer.selected = !layer.selected;
        this.lastClickedPath = path;
        break;

      case 'range': {
        const anchor = this.lastClickedPath;
        if (!anchor) {
          // No anchor — behave like single
          for (const l of this.layers.values()) l.selected = false;
          layer.selected = true;
          this.lastClickedPath = path;
          break;
        }

        const anchorIdx = this.layerOrder.indexOf(anchor);
        const targetIdx = this.layerOrder.indexOf(path);
        const lo = Math.min(anchorIdx, targetIdx);
        const hi = Math.max(anchorIdx, targetIdx);

        for (const l of this.layers.values()) l.selected = false;
        for (let i = lo; i <= hi; i++) {
          this.layers.get(this.layerOrder[i])!.selected = true;
        }
        // Don't update lastClickedPath for range — anchor stays
        break;
      }
    }

    this.notify();
  }

  /** Get all currently selected layers */
  getSelected(): LayerInfo[] {
    return this.getLayers().filter((l) => l.selected);
  }

  /** Get the "primary" selected layer (last clicked) for showing in controls */
  getPrimarySelected(): LayerInfo | undefined {
    if (this.lastClickedPath) {
      const layer = this.layers.get(this.lastClickedPath);
      if (layer?.selected) return layer;
    }
    // Fallback: first selected
    return this.getSelected()[0];
  }

  // ─── Mutations ───────────────────────────────────────────

  /** Toggle visibility (independent of selection) */
  /** Toggle-restore solo capture (see {@link solo}); null when not soloed. */
  private soloState: { path: string; previous: Map<string, boolean> } | null = null;

  /** Path of the layer currently soloed via {@link solo}, else null. */
  get soloedPath(): string | null {
    return this.soloState?.path ?? null;
  }

  /**
   * Toggle-restore solo ("hide all others"). The first call captures the
   * current visibility set and shows only `path`; soloing the SAME path
   * again restores the captured set (for layers that still exist); soloing
   * a DIFFERENT path re-targets while keeping the original capture, so the
   * eventual un-solo restores the true pre-solo state. A manual visibility
   * change or a scene reload clears the capture. One notification per
   * transition (never per-layer).
   */
  solo(path: string): void {
    if (!this.layers.has(path)) return;
    if (this.soloState?.path === path) {
      for (const [p, v] of this.soloState.previous) {
        const layer = this.layers.get(p);
        if (layer) layer.visible = v;
      }
      this.soloState = null;
    } else {
      if (!this.soloState) {
        const previous = new Map<string, boolean>();
        for (const layer of this.layers.values()) previous.set(layer.path, layer.visible);
        this.soloState = { path, previous };
      } else {
        this.soloState = { path, previous: this.soloState.previous };
      }
      for (const layer of this.layers.values()) layer.visible = layer.path === path;
    }
    this.notify();
  }

  /**
   * Set visibility for many layers with a SINGLE change notification
   * ({@link setVisible} notifies per call, so an N-layer sweep would
   * re-render the panel N times). Clears any solo capture — a batch
   * visibility change supersedes the remembered pre-solo state.
   */
  setVisibleMany(entries: Array<{ path: string; visible: boolean }>): void {
    this.soloState = null;
    let changed = false;
    for (const { path, visible } of entries) {
      const layer = this.layers.get(path);
      if (layer && layer.visible !== visible) {
        layer.visible = visible;
        changed = true;
      }
    }
    if (changed) this.notify();
  }

  /**
   * Replace a layer's state with a freshly derived {@link LayerInfo}
   * (per-layer reset). Preserves `selected`, and clears any solo capture —
   * the reset rewrites visibility behind the capture's back, so a later
   * un-solo restore would silently overwrite it (and `soloedPath` would lie
   * about which layers are showing in the meantime). Notifies once. The
   * live object is mutated in place, so references held by callers stay
   * valid.
   */
  resetLayerState(path: string, fresh: LayerInfo): void {
    const live = this.layers.get(path);
    if (!live) return;
    this.soloState = null;
    Object.assign(live, fresh, { selected: live.selected });
    this.notify();
  }

  setVisible(path: string, visible: boolean): void {
    // A manual per-layer change invalidates the solo capture (restoring it
    // later would silently overwrite what the user just chose).
    this.soloState = null;
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.visible = visible;
    this.notify();
  }

  /** Set display range for a layer */
  setDisplayRange(path: string, min: number, max: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.displayMin = min;
    layer.displayMax = max;
    this.notify();
  }

  /** Set gamma for a layer */
  setGamma(path: string, gamma: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.gamma = gamma;
    this.notify();
  }

  /** Set opacity for a layer */
  setOpacity(path: string, opacity: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.opacity = opacity;
    this.notify();
  }

  /** Set colormap for a layer */
  setColormap(path: string, colormap: string): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.colormap = colormap;
    this.notify();
  }

  /**
   * Re-default the display window after the layer's colormap was toggled.
   *
   * The window maps the RENDERED VALUE to [0, 1], and toggling a colormap
   * changes what that value is — so the sensible default changes with it:
   *
   * * colormap ON  → the value is the scalar (gsplat amplitude / per-element
   *   scalar), whose useful window is its data range. Keeping the direct-colour
   *   identity here would push `updateScalarRange(0, 1)` onto amplitudes that
   *   live in ~[1e-4, 0.02] and render near-black (the #522 failure).
   * * colormap OFF → the value is authored RGB, whose range IS [0, 1].
   *
   * A window carried over from the other mode is meaningless, so this
   * deliberately overwrites a user-set one.
   *
   * The slider BOUNDS follow the mode too, landing on exactly what a natively
   * authored layer of that mode gets at init. Merely widening them instead
   * would leave the useful window as an unusable sliver of the track — a
   * gsplat amplitude window of [1e-4, 0.02] inside [0, 1] bounds is 2% of it.
   */
  setColormapWindow(path: string, colormapOn: boolean): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.scalarWindow = colormapOn;
    const [min, max] = colormapOn ? (layer.scalarDataRange ?? [0, 1]) : [0, 1];
    layer.displayMin = min;
    layer.displayMax = max;
    // colormap ON  -> the scalar range (what a colormapped layer inits to)
    // colormap OFF -> [0, 1] widened to the authored RGB spread (what a
    //                 direct-colour layer inits to; keeps HDR colours > 1 in reach)
    const [boundMin, boundMax] = colormapOn
      ? [min, max]
      : [Math.min(0, layer.colorDataRange?.[0] ?? 0), Math.max(1, layer.colorDataRange?.[1] ?? 1)];
    layer.dataMin = Math.min(boundMin, min);
    layer.dataMax = Math.max(boundMax, max);
    this.notify();
  }

  /** Set blending mode for a layer. A user pick marks the mode EXPLICIT. */
  setBlendingMode(path: string, mode: BlendingMode): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    // Resolved at the point of STORAGE, same as the panel's dropdown handler:
    // a mesh maps `volumetric` → `opaque`, and the Blend dropdown displays the
    // stored value raw — see `resolveLayerBlendingMode`.
    layer.blendingMode = resolveLayerBlendingMode(layer.type, mode);
    // The user explicitly picked a mode ⇒ this layer now OWNS one, so
    // `liveLayerAttrs` may emit it as a composition setter (even a group).
    layer.blendingModeExplicit = true;
    this.notify();
  }

  /**
   * Set (or clear) a layer's authored cross-layer draw order.
   *
   * `undefined` CLEARS it, which is not the same authored state as setting 0:
   * both resolve to band 0, but only the latter remains a stated value in the
   * panel and authored-band diagnostics. Clearing must therefore also clear
   * `layerOrderExplicit`, or `liveLayerAttrs` would keep emitting the stale
   * value as this layer's own setter.
   */
  setLayerOrder(path: string, level: number | undefined): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    const sanitizedLevel = level !== undefined && Number.isSafeInteger(level) ? level : undefined;
    layer.layerOrder = sanitizedLevel ?? layer.inheritedLayerOrder;
    layer.layerOrderExplicit = sanitizedLevel !== undefined;
    this.notify();
  }

  /** Set absorption (volumetric κ) for a layer */
  setAbsorption(path: string, absorption: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.absorption = absorption;
    this.notify();
  }

  /**
   * Apply a mutation to all selected layers.
   * The mutator receives each selected LayerInfo to modify in place.
   */
  applyToSelected(mutator: (layer: LayerInfo) => void): void {
    for (const layer of this.getSelected()) {
      mutator(layer);
    }
    this.notify();
  }

  // ─── Events ──────────────────────────────────────────────

  /** Subscribe to state changes. Returns an unsubscribe function. */
  onChange(listener: LayerChangeListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }

  /** Clear all state */
  dispose(): void {
    this.layers.clear();
    this.layerOrder = [];
    this.lastClickedPath = null;
    this.listeners.clear();
  }
}
