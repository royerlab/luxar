/**
 * The Layers panel must window each LOD LEVEL on the level's OWN scalar range.
 *
 * A layer composes exactly ONE display window — the panel has one range slider —
 * but `applyComposed` fans that window out over `collectDataDescendants`, so a
 * `kind=lod` layer pushes it into every level. The window is a value→LUT mapping
 * in the SCALAR's own units, and across a LOD ladder those units are not shared:
 * gsplat LOD merging SUMS amplitudes, so a coarsened level carries its own
 * `amplitude_data_range` for the same physical signal, while the layer's
 * reference range is merely whichever descendant
 * `deriveScalarRangeFromDescendants` picked — the one with the largest
 * `n_splats`, or the first one visited when none declares an `n_splats` (points,
 * lines and mesh never do) or several tie.
 *
 * So before #1753 the whole layer rendered on the reference level's window. Three
 * things followed, and this file pins all three:
 *
 * 1. Any per-level window differentiation was discarded. Today's gsplat producer
 *    stamps one independently:
 *    `packages/luxar/src/luxar/io/_compiler/gsplat_assembly.py` computes
 *    `amplitude_data_range = [min, p99.9]` over each splat set's OWN amplitudes,
 *    so a coarsened level's range genuinely differs from the finest's. (#1752,
 *    in flight, would additionally scale each level's window by its
 *    mass-weighted amplitude ratio; none of that pass is in this tree.)
 * 2. It happened ON LOAD, not just on interaction — `layers-panel.ts` runs
 *    `applyDisplayRange(layer)` for every layer during panel construction — so a
 *    level streamed in AFTER the last commit kept its own window until the next
 *    one, and two levels of one object could be windowed differently depending
 *    only on load order.
 * 3. It applied to `scalar_data_range` on points/lines/mesh identically — so a
 *    points `kind=lod` layer is pinned here too.
 *
 * The fix re-expresses the composed window in each level's range, preserving its
 * relative position — "the middle 40% of the layer's signal" stays the middle 40%
 * of each level's signal. Three families of case must NOT be remapped, and all
 * three are guards with teeth rather than defensive noise:
 *
 * * A relation that is not a LOD ladder. A `kind=partition` layer fans out over
 *   disjoint SPATIAL subsets of one field at the same scale, so per-part windows
 *   are auto-contrast and put a colour discontinuity at every seam — the exact
 *   thing `core/group/adders/mesh.py::_shared_scalar_window` exists to prevent.
 * * A degenerate or missing range on either side. A degenerate LEAF range
 *   collapses the window to a point, which `updateScalarRange` hands to
 *   `computeScalarRangeUniforms` — and that answers a sub-eps span with the LUT
 *   MIDPOINT (#631), so every element of the leaf renders as one flat neutral
 *   colour.
 * * A composed window that is not stated in the reference basis at all, because
 *   some OTHER window at or below the edited layer contributed to it. Remapping
 *   one of those corrupts a window that was already correct.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { LayerApplyEngine } from '../../../../ui/layers/layer-apply';
import { LayerStateManager } from '../../../../ui/layers/layer-state';
import { remapWindowToLeafRange, resolveColormapWindow } from '../../../../rendering/display-range';
import type { SceneNode } from '../../../../data/data-loader-types';

/** A vi-free material stub recording exactly the writes this file asserts on. */
interface MaterialStub extends Record<string, unknown> {
  scalarRanges: Array<[number, number]>;
  intensities: number[];
  offsets: number[];
}

/**
 * `isColormapActive` reads the `USE_COLORMAP` shader define, which is what
 * `updateColormapTexture` sets — so a colormapped leaf is modelled by the define
 * being PRESENT, exactly as the real materials carry it.
 */
function materialStub(colormapActive: boolean): MaterialStub {
  const mat: MaterialStub = {
    scalarRanges: [],
    intensities: [],
    offsets: [],
    userData: { blendingMode: 'additive' },
    uniforms: { uOpacity: { value: 1.0 } },
    defines: colormapActive ? { USE_COLORMAP: '' } : {},
    updateGamma: () => {},
    updateOpacity: () => {},
    updateAbsorption: () => {},
    updateIntensity: (v: number) => mat.intensities.push(v),
    updateOffset: (v: number) => mat.offsets.push(v),
    updateScalarRange: (min: number, max: number) => mat.scalarRanges.push([min, max]),
    applyBlendingMode: (mode: string) => {
      (mat.userData as Record<string, unknown>).blendingMode = mode;
    },
  };
  return mat;
}

/**
 * The last scalar window a material was handed, with the sign of zero
 * normalised. `computeDisplayRange` answers a zero-minimum window with `-0`
 * (`-offset` where `offset` is `+0`) — numerically the same window, but not
 * `toEqual`-equal to `0`, and a pre-existing quirk that is not what this file
 * is about.
 */
function windowOf(mat: MaterialStub): [number, number] {
  return norm(mat.scalarRanges.at(-1)!);
}

/** A `[min, max]` pair with the sign of zero normalised — see {@link windowOf}. */
function norm(pair: readonly number[]): [number, number] {
  return [pair[0] + 0, pair[1] + 0];
}

interface LeafSpec {
  /** Last path segment; the full path is `<wrapper>/<name>`. */
  name: string;
  type: 'gsplats' | 'points';
  attrs: Record<string, unknown>;
  /** Whether the leaf's material renders through the LUT. Default: yes. */
  colormapActive?: boolean;
  /**
   * Put this leaf under an intermediate (non-layer) group at
   * `<wrapper>/<under.name>/<name>`. Leaves sharing an `under.name` share the
   * one group node. Used to build the `overview` shape — an lod wrapper whose
   * fine branch is a nested `kind=partition`.
   */
  under?: { name: string; attrs: Record<string, unknown> };
}

/**
 * One `layer=true` wrapper over the given leaves, with a real
 * `LayerStateManager` and a real `LayerApplyEngine` on top — the shape
 * `layers-panel.ts` builds.
 *
 * `ancestorAttrs` inserts a plain (non-layer) group ABOVE the wrapper, at
 * `/anc`, which moves the wrapper to `/anc/obj`. That is how the
 * ancestor-gain control below builds a gain that composes from OUTSIDE the
 * edited layer.
 */
function harness(
  wrapperAttrs: Record<string, unknown>,
  leaves: LeafSpec[],
  ancestorAttrs?: Record<string, unknown>
) {
  const rootGroup = new THREE.Group();
  const mats = new Map<string, MaterialStub>();
  const objPath = ancestorAttrs ? '/anc/obj' : '/obj';

  const children: SceneNode[] = [];
  const nested = new Map<string, SceneNode>();
  for (const spec of leaves) {
    const parentPath = spec.under ? `${objPath}/${spec.under.name}` : objPath;
    const path = `${parentPath}/${spec.name}`;
    const mat = materialStub(spec.colormapActive ?? true);
    mats.set(spec.name, mat);
    const obj = new THREE.Points(new THREE.BufferGeometry(), mat as unknown as THREE.Material);
    obj.name = path;
    // Pre-set the clone-on-first-use flag so `getLeafMaterial` hands back this
    // very stub: the real first-use path calls `mat.clone()` and registers the
    // clone with MaterialManager, which needs the full THREE.Material API this
    // stub deliberately does not implement. Production DOES always clone once
    // (`_layerMaterialCloned` starts unset on every node) — the flag is a test
    // shortcut, and a sound one because the clone is what every later tick
    // writes to, so asserting on the pre-clone object is asserting on the same
    // object the panel keeps using.
    obj.userData._layerMaterialCloned = true;
    rootGroup.add(obj);
    const leafNode = {
      name: spec.name,
      path,
      type: spec.type,
      attrs: spec.attrs,
      children: [],
    } as unknown as SceneNode;
    if (!spec.under) {
      children.push(leafNode);
      continue;
    }
    let group = nested.get(spec.under.name);
    if (!group) {
      group = {
        name: spec.under.name,
        path: parentPath,
        type: 'group',
        attrs: spec.under.attrs,
        children: [],
      } as unknown as SceneNode;
      nested.set(spec.under.name, group);
      children.push(group);
    }
    (group.children as SceneNode[]).push(leafNode);
  }

  const objNode = {
    name: 'obj',
    path: objPath,
    type: 'group',
    attrs: wrapperAttrs,
    children,
  } as unknown as SceneNode;

  const graph: SceneNode = {
    name: 'root',
    path: '',
    type: 'scene',
    attrs: {},
    children: ancestorAttrs
      ? [
          {
            name: 'anc',
            path: '/anc',
            type: 'group',
            attrs: ancestorAttrs,
            children: [objNode],
          } as unknown as SceneNode,
        ]
      : [objNode],
  } as unknown as SceneNode;

  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const engine = new LayerApplyEngine({
    getRootGroup: () => rootGroup,
    getSceneGraph: () => graph,
    state,
    requestRender: () => {},
    requestReprocess: () => {},
  });
  const layer = () => state.getLayer(objPath)!;
  return {
    engine,
    state,
    mats,
    objPath,
    layer,
    apply: () => engine.applyDisplayRange(layer()),
  };
}

/**
 * A kind=lod wrapper carrying NO range and NO colormap of its own.
 *
 * The bare wrapper is what the compiler actually writes: `colormap` is
 * deliberately excluded from `COMPOSITING_ATTRS` on the Python side, so the
 * writer stamps it on every LEAF and leaves the wrapper without one. The panel
 * finds it either way (`usesColormap` / `deriveColormapFromDescendants` both
 * walk descendants), so this is fixture realism rather than behaviour — every
 * derived `LayerInfo` and every window asserted below is identical with the
 * attr on the wrapper instead.
 */
const LOD_WRAPPER = {
  layer: true,
  kind: 'lod',
  display_type: 'gsplats',
};

/**
 * Three LOD levels, coarse → fine, with deliberately DIFFERENT amplitude ranges
 * and increasing `n_splats`. `deriveScalarRangeFromDescendants` keys on
 * `n_splats`, so the layer's reference range is the finest level's `[0, 2]`.
 */
const LOD_LEVELS: LeafSpec[] = [
  {
    name: 'lod_0',
    type: 'gsplats',
    attrs: {
      amplitude_data_range: [0, 8],
      n_splats: 100,
      has_scalars: true,
      colormap: 'viridis',
    },
  },
  {
    name: 'lod_1',
    type: 'gsplats',
    attrs: {
      amplitude_data_range: [0, 4],
      n_splats: 400,
      has_scalars: true,
      colormap: 'viridis',
    },
  },
  {
    name: 'lod_2',
    type: 'gsplats',
    attrs: {
      amplitude_data_range: [0, 2],
      n_splats: 1600,
      has_scalars: true,
      colormap: 'viridis',
    },
  },
];

describe('every LOD level is windowed on its own amplitude range', () => {
  it('at panel-construction time, before any interaction', () => {
    // `layers-panel.ts` calls `applyDisplayRange(layer)` for every layer while
    // building the panel, so this IS the load-time state — not an interaction.
    const h = harness(LOD_WRAPPER, LOD_LEVELS);
    // The reference the panel derived: the finest level's range.
    expect(h.state.getLayer('/obj')!.scalarDataRange).toEqual([0, 2]);
    expect([h.state.getLayer('/obj')!.displayMin, h.state.getLayer('/obj')!.displayMax]).toEqual([
      0, 2,
    ]);

    h.apply();

    // The full-range window maps to each level's OWN full range. Before the fix
    // all three got [0, 2] — the finest level's window — so the coarse levels,
    // whose amplitudes reach 8, rendered clipped at a quarter of their span.
    expect(windowOf(h.mats.get('lod_0')!)).toEqual([0, 8]);
    expect(windowOf(h.mats.get('lod_1')!)).toEqual([0, 4]);
    expect(windowOf(h.mats.get('lod_2')!)).toEqual([0, 2]);
  });

  it('after a slider drag, proportionally, in each level’s own units', () => {
    const h = harness(LOD_WRAPPER, LOD_LEVELS);
    // The middle half of the reference range [0, 2].
    h.state.setDisplayRange('/obj', 0.5, 1.5);
    h.apply();

    // t ∈ [0.25, 0.75] of each level's own span.
    expect(windowOf(h.mats.get('lod_0')!)).toEqual([2, 6]);
    expect(windowOf(h.mats.get('lod_1')!)).toEqual([1, 3]);
    expect(windowOf(h.mats.get('lod_2')!)).toEqual([0.5, 1.5]);
  });

  it('keeps the #936 identity reset on the colour GOG', () => {
    // The scalar window lives in the LUT lookup; the post-LUT colour gain must
    // be identity or the authored gain double-applies.
    const h = harness(LOD_WRAPPER, LOD_LEVELS);
    h.state.setDisplayRange('/obj', 0.5, 1.5);
    h.apply();
    for (const name of ['lod_0', 'lod_1', 'lod_2']) {
      expect(h.mats.get(name)!.intensities.at(-1)).toBe(1);
      expect(h.mats.get(name)!.offsets.at(-1)).toBe(0);
    }
  });
});

describe('leaves the remap must leave alone', () => {
  it('a DEGENERATE level range keeps the composed window', () => {
    // A constant-amplitude level legitimately declares `[x, x]` (every imported
    // classical splat file has `amplitudes = 1`). Mapping onto a zero span would
    // collapse the window to a point, which `updateScalarRange` passes to
    // `computeScalarRangeUniforms` — and that answers a sub-eps span with the LUT
    // MIDPOINT (#631), so every splat of the level would render as one flat
    // neutral colour.
    const h = harness(LOD_WRAPPER, [
      ...LOD_LEVELS,
      {
        name: 'flat',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [3, 3],
          n_splats: 50,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
    ]);
    h.state.setDisplayRange('/obj', 0.5, 1.5);
    h.apply();

    expect(windowOf(h.mats.get('flat')!)).toEqual([0.5, 1.5]);
    // The neighbours still remap, so this is the guard firing and not the whole
    // feature being off.
    expect(windowOf(h.mats.get('lod_0')!)).toEqual([2, 6]);
  });

  it('a level declaring NO range at all keeps the composed window', () => {
    const h = harness(LOD_WRAPPER, [
      ...LOD_LEVELS,
      {
        name: 'bare',
        type: 'gsplats',
        attrs: { n_splats: 50, has_scalars: true, colormap: 'viridis' },
      },
    ]);
    h.state.setDisplayRange('/obj', 0.5, 1.5);
    h.apply();

    expect(windowOf(h.mats.get('bare')!)).toEqual([0.5, 1.5]);
    expect(windowOf(h.mats.get('lod_1')!)).toEqual([1, 3]);
  });

  it('a DIRECT-COLOUR leaf of a mixed layer still takes the identity colour GOG', () => {
    // The `identityLayerWindow` route (a leaf the C1 guard keeps on direct colour
    // while the layer's window is a scalar one). It must not receive the scalar
    // window as a colour gain — and, since it is not colormap-active, it must not
    // receive a scalar range either. It is the one leaf here with no `colormap`
    // attr, which is also what the writer stamps for a direct-colour leaf.
    const h = harness(LOD_WRAPPER, [
      ...LOD_LEVELS,
      {
        name: 'rgb',
        type: 'points',
        attrs: { scalar_data_range: [0, 5] },
        colormapActive: false,
      },
    ]);
    h.state.setDisplayRange('/obj', 0.5, 1.5);
    h.apply();

    const rgb = h.mats.get('rgb')!;
    expect(rgb.scalarRanges).toEqual([]);
    expect(rgb.intensities.at(-1)).toBe(1);
    expect(rgb.offsets.at(-1)).toBe(0);
  });
});

/**
 * The remap re-states the composed window as a POSITION inside the reference
 * range. That premise only holds when the edited layer's own live window is the
 * only gain contributed at or below it — `composeEffective` multiplies over the
 * whole ancestry, substituting live panel state for every `layer=true` node, so
 * several reachable shapes produce a composed window in a different basis
 * entirely. Remapping those does not refine a correct window; it corrupts one.
 */
describe('the remap declines when the composed window is not in the reference basis', () => {
  /**
   * Two nested `layer=true` colormapped children under ONE wrapper, dragged by
   * the wrapper's opacity. The wrapper is `kind=lod` on purpose: that satisfies
   * the LOD-only structural gate, so what these two tests exercise is the
   * nested-layer arm alone and not the structural one (which the
   * `kind=partition` and plain-group tests below cover).
   *
   * `deriveScalarRangeFromDescendants` is the one derivation in `layer-state.ts`
   * that does NOT stop at a nested `layer=true` descendant, so the wrapper's
   * reference is the larger child's range even though each child is its own
   * panel row windowed on its own. And `usesColormap` DOES stop there, so the
   * wrapper itself renders direct colour and contributes the identity window —
   * which is what makes each child's composed window already exactly its own.
   */
  function nestedLayerChildren(ch0Range: [number, number]) {
    return harness({ layer: true, kind: 'lod', display_type: 'gsplats' }, [
      {
        name: 'ch0',
        type: 'gsplats',
        attrs: {
          layer: true,
          colormap: 'viridis',
          amplitude_data_range: ch0Range,
          n_splats: 1000,
          has_scalars: true,
        },
      },
      {
        name: 'ch1',
        type: 'gsplats',
        attrs: {
          layer: true,
          colormap: 'viridis',
          amplitude_data_range: [0, 5],
          n_splats: 10000,
          has_scalars: true,
        },
      },
    ]);
  }

  it('a nested layer=true colormapped child keeps its OWN window', () => {
    const h = nestedLayerChildren([0, 0.02]);
    const grp = h.layer();
    expect(grp.scalarDataRange).toEqual([0, 5]);
    // The wrapper itself is NOT colormapped (`usesColormap` stops at the nested
    // layers), so its own window is the direct-colour identity.
    expect([grp.displayMin, grp.displayMax]).toEqual([0, 1]);

    // An OPACITY drag on the wrapper — every control funnels through
    // `applyComposed`, so this is the same code path a range drag takes.
    h.state.setOpacity(h.objPath, 0.4);
    h.engine.applyOpacity(h.layer());

    // ch0's composed window is `{1, 0} × computeUniforms(0, 0.02)` = its own
    // [0, 0.02], already exactly what its own panel row shows. Remapping
    // [0, 5] → [0, 0.02] would have pushed [0, 8e-5]: 250x too narrow, with
    // nothing re-applying ch0 afterwards.
    expect(windowOf(h.mats.get('ch0')!)).toEqual([0, 0.02]);
    expect(windowOf(h.mats.get('ch1')!)).toEqual([0, 5]);
  });

  it('…including a nested child whose own range is exactly [0, 1]', () => {
    // The reason arm 3 tests "is this node TRACKED AS A LAYER" and not "does it
    // contribute a gain". `computeUniforms(0, 1)` is `{intensity: 1, offset: -0}`
    // — the identity, indistinguishable from a node that authored no window at
    // all — so a gain-only test let this child through and remapped a window
    // that was already right. `[0, 1]` is not an exotic range: normalized
    // scalars, probabilities, masks and fractions all land on it.
    const h = nestedLayerChildren([0, 1]);
    expect(h.layer().scalarDataRange).toEqual([0, 5]);

    h.state.setOpacity(h.objPath, 0.4);
    h.engine.applyOpacity(h.layer());

    // Remapping [0, 5] → [0, 1] would have pushed [0, 0.2]: 5x too narrow.
    expect(windowOf(h.mats.get('ch0')!)).toEqual([0, 1]);
    expect(windowOf(h.mats.get('ch1')!)).toEqual([0, 5]);
  });

  it('a leaf with its OWN authored intensity keeps the composed window', () => {
    // `resolveColormapWindow`'s first branch says the same thing at load time: a
    // non-identity RAW LEAF gain means the composed gain IS the window and the
    // data range is not consulted. The panel must not then re-derive a window
    // from that same data range.
    const h = harness(LOD_WRAPPER, [
      {
        name: 'child_0',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 8],
          intensity: 0.25,
          n_splats: 100,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
      {
        name: 'child_1',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 2],
          n_splats: 1600,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
    ]);
    expect(h.layer().scalarDataRange).toEqual([0, 2]);

    h.apply();

    // Composed = wrapper live {0.5, 0} × leaf authored {0.25, 0} = {0.125, 0}
    // → [0, 8]. Remapping [0, 2] → [0, 8] would have pushed [0, 32], 4x too wide.
    expect(windowOf(h.mats.get('child_0')!)).toEqual([0, 8]);
    // The sibling with no gain of its own still remaps (nothing here is the
    // whole feature switching off).
    expect(windowOf(h.mats.get('child_1')!)).toEqual([0, 2]);
  });

  it('an authored intensity on the LAYER ITSELF declines the remap', () => {
    // `intensity`/`offset` are compositing attrs, so `add_gsplats_from_file(…,
    // layer=True, intensity=0.5)` stamps them on the kind=lod wrapper.
    // `walkSceneGraph` then seeds displayMin/Max from
    // `computeDisplayRange(0.5, 0)` = [0, 2] — a window in the normalized-GAIN
    // basis with no relation to the [0, 0.02] reference span. That window is
    // already wrong (100x too wide); remapping would multiply the error by
    // leafSpan/refSpan on top, so the change must leave it exactly as it was.
    const h = harness({ ...LOD_WRAPPER, intensity: 0.5 }, [
      {
        name: 'coarse',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 0.08],
          n_splats: 100,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
      {
        name: 'fine',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 0.02],
          n_splats: 1600,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
    ]);
    expect(h.layer().scalarDataRange).toEqual([0, 0.02]);
    expect(norm([h.layer().displayMin, h.layer().displayMax])).toEqual([0, 2]);

    h.apply();

    // Both levels get the composed window verbatim — the pre-#1753 behaviour.
    // Remapping would have given `coarse` [0, 8] (2 / 0.02 × 0.08).
    expect(windowOf(h.mats.get('coarse')!)).toEqual([0, 2]);
    expect(windowOf(h.mats.get('fine')!)).toEqual([0, 2]);
  });

  it('a gain on a non-layer ANCESTOR still remaps, and agrees with creation', () => {
    // The positive control for the arms above: an ancestor ABOVE the edited
    // layer is deliberately NOT gated, because the remap re-expresses the
    // LAYER'S OWN window and re-applies the ancestor gain afterwards, landing
    // on exactly the window the node factory computes at creation.
    const h = harness(
      LOD_WRAPPER,
      [
        {
          name: 'coarse',
          type: 'gsplats',
          attrs: {
            amplitude_data_range: [0, 8],
            n_splats: 100,
            has_scalars: true,
            colormap: 'viridis',
          },
        },
        {
          name: 'fine',
          type: 'gsplats',
          attrs: {
            amplitude_data_range: [0, 2],
            n_splats: 1600,
            has_scalars: true,
            colormap: 'viridis',
          },
        },
      ],
      { intensity: 2 }
    );
    expect(h.layer().scalarDataRange).toEqual([0, 2]);

    h.apply();

    // Composed = ancestor {2, 0} × wrapper live computeUniforms(0, 2) = {1, 0}
    // → window [0, 1], i.e. HALF the reference range; remapped onto the coarse
    // level's [0, 8] that is [0, 4].
    expect(windowOf(h.mats.get('coarse')!)).toEqual([0, 4]);
    expect(windowOf(h.mats.get('fine')!)).toEqual([0, 1]);
    // …which is precisely what `resolveColormapWindow` hands the coarse level at
    // creation time (leaf gain identity, ancestor-only composed gain of 2). The
    // panel and the node factory agreeing is the point of not gating here.
    expect(
      norm(resolveColormapWindow([0, 8], { intensity: 1, offset: 0 }, { intensity: 2, offset: 0 }))
    ).toEqual([0, 4]);
  });

  it('…including when the ranges do NOT start at zero', () => {
    // The case above agrees for either implementation, because every range in
    // it starts at 0. Remapping the COMPOSED window (which already carries the
    // ancestor gain) reads it as a position inside the reference range, and
    // that only cancels when `ref₀/refSpan === leaf₀/leafSpan`. Shift the
    // reference off zero and the two part company: the composed window is
    // [0.5, 1.5] and remapping THAT onto [0, 8] gives [-2, 2] — a quarter of
    // the LUT spent below the leaf's own minimum — where creation gives [0, 4].
    const h = harness(
      LOD_WRAPPER,
      [
        {
          name: 'coarse',
          type: 'gsplats',
          attrs: {
            amplitude_data_range: [0, 8],
            n_splats: 100,
            has_scalars: true,
            colormap: 'viridis',
          },
        },
        {
          name: 'fine',
          type: 'gsplats',
          attrs: {
            amplitude_data_range: [1, 3],
            n_splats: 1600,
            has_scalars: true,
            colormap: 'viridis',
          },
        },
      ],
      { intensity: 2 }
    );
    expect(h.layer().scalarDataRange).toEqual([1, 3]);

    h.apply();

    expect(windowOf(h.mats.get('coarse')!)).toEqual([0, 4]);
    expect(
      norm(resolveColormapWindow([0, 8], { intensity: 1, offset: 0 }, { intensity: 2, offset: 0 }))
    ).toEqual([0, 4]);
    // The reference level itself short-circuits, so it keeps the composed
    // window — which is also what creation gives it.
    expect(windowOf(h.mats.get('fine')!)).toEqual([0.5, 1.5]);
    expect(
      norm(resolveColormapWindow([1, 3], { intensity: 1, offset: 0 }, { intensity: 2, offset: 0 }))
    ).toEqual([0.5, 1.5]);
  });
});

describe('points behave identically to gsplats (consequence 3)', () => {
  it('remaps each LOD level onto its own `scalar_data_range`', () => {
    // Same shape as the gsplat ladder, `scalar_data_range` instead of
    // `amplitude_data_range`, and the reference stated on the wrapper — points
    // carry no `n_splats` for `deriveScalarRangeFromDescendants` to rank by, so
    // without a wrapper range it would take whichever level it visited FIRST.
    const h = harness(
      {
        layer: true,
        kind: 'lod',
        display_type: 'points',
        scalar_data_range: [0, 2],
      },
      [
        {
          name: 'lod_0',
          type: 'points',
          attrs: { scalar_data_range: [0, 4], has_scalars: true, colormap: 'viridis' },
        },
        {
          name: 'lod_1',
          type: 'points',
          attrs: { scalar_data_range: [0, 2], has_scalars: true, colormap: 'viridis' },
        },
      ]
    );
    h.state.setDisplayRange('/obj', 0.5, 1.5);
    h.apply();

    expect(windowOf(h.mats.get('lod_0')!)).toEqual([1, 3]);
    expect(windowOf(h.mats.get('lod_1')!)).toEqual([0.5, 1.5]);
  });
});

/**
 * The case this change matters MOST in, and the reason it is not merely a
 * refinement: a `kind=lod` ladder whose levels are not even the same PHYSICAL
 * QUANTITY.
 *
 * `add_points(..., scalars=…, colormap='plasma', layer=True,
 * substitutive_lod=True)` produces a MIXED ladder — the coarse levels are LIFTED
 * to gsplats (direct colour, no `colormap`, `amplitude_data_range` + `n_splats`)
 * and the finest level stays `points` (colormapped, `scalar_data_range`, and NO
 * `n_splats`, because points write `n_points`). The attrs below are read
 * verbatim off such a store.
 *
 * Two facts collide. `deriveScalarRangeFromDescendants` ranks by `n_splats`, and
 * a points leaf never declares one — so the layer's reference is the largest
 * LIFTED GSPLAT level's amplitude range, a different physical quantity from the
 * points scalar it will be pushed into. And the one leaf that is
 * colormap-active is precisely that points leaf. So the composed window landed
 * on the colormapped leaf in units it has nothing to do with: an amplitude
 * window of `[0.037, 2.817]` over a scalar spanning `1.035 … 280.862`, so
 * everything above 2.817 — 99.4% of that span — clamps to the top of the LUT and
 * renders as one flat colour. This is not the ≤13% per-level spread a plain
 * gsplat ladder shows; it is two orders of magnitude.
 */
describe('a MIXED points-lift LOD ladder (the case this matters most in)', () => {
  const MIXED_LIFT_LADDER: LeafSpec[] = [
    // The three lifted coarse levels render DIRECT COLOUR (the lift bakes the
    // palette into per-splat RGB), so none of them is colormap-active.
    {
      name: 'child_0',
      type: 'gsplats',
      colormapActive: false,
      attrs: { amplitude_data_range: [0.0417868047952652, 2.960740089416504], n_splats: 63 },
    },
    {
      name: 'child_1',
      type: 'gsplats',
      colormapActive: false,
      attrs: { amplitude_data_range: [0.06937223672866821, 2.933626413345337], n_splats: 250 },
    },
    {
      name: 'child_2',
      type: 'gsplats',
      colormapActive: false,
      attrs: { amplitude_data_range: [0.03669371083378792, 2.8173763751983643], n_splats: 1000 },
    },
    // The finest level is the original points leaf: colormapped, its own scalar
    // range, and no `n_splats` for the reference derivation to rank it by.
    {
      name: 'child_3',
      type: 'points',
      attrs: {
        colormap: 'plasma',
        scalar_data_range: [1.0351287126541138, 280.862060546875],
        has_scalars: true,
      },
    },
  ];

  it('windows the colormapped points level on its OWN scalar range', () => {
    const h = harness({ layer: true, kind: 'lod', display_type: 'points' }, MIXED_LIFT_LADDER);

    // The reference really is the lifted sibling's AMPLITUDE range, not the
    // points scalar range the only colormapped leaf actually carries.
    expect(h.layer().scalarDataRange).toEqual([0.03669371083378792, 2.8173763751983643]);

    h.apply();

    // Before #1753 this leaf got the composed window verbatim —
    // [0.03669371083378792, 2.8173763751983643] — over scalars reaching 280.86.
    // Now it gets its own range, which is also what the node factory already
    // hands it at creation.
    expect(windowOf(h.mats.get('child_3')!)).toEqual([1.0351287126541138, 280.862060546875]);
    // The lifted levels are direct colour: no scalar window is pushed at all,
    // and their colour GOG stays the identity (the `identityLayerWindow` route,
    // since the layer's window is a SCALAR one).
    for (const name of ['child_0', 'child_1', 'child_2']) {
      expect(h.mats.get(name)!.scalarRanges).toEqual([]);
      expect(h.mats.get(name)!.intensities.at(-1)).toBe(1);
      expect(h.mats.get(name)!.offsets.at(-1)).toBe(0);
    }
  });
});

/**
 * The remap crosses a LOD ladder and NOTHING else.
 *
 * A LOD *level* and a partition *part* are different kinds of sibling. Levels
 * are alternative representations of the WHOLE object and gsplat LOD merging
 * SUMS amplitudes, so a coarse level's amplitude is the same physical signal at
 * a different numeric SCALE — re-expressing the window per level is the
 * correction #1753 asks for. Parts are disjoint SPATIAL subsets of one field at
 * the SAME scale, differing only by CONTENT
 * (`packages/luxar/src/luxar/io/_compiler/gsplat_assembly.py` derives
 * `amplitude_data_range = [min, p99.9]` per splat set), so a per-part window is
 * auto-contrast: one physical value renders as a different colour in different
 * tiles and the colormap goes non-monotone at every BSP seam.
 *
 * That doctrine is the producers', not this file's:
 * `core/group/adders/mesh.py::_shared_scalar_window` stamps ONE window on every
 * child precisely because "a level or a part that stamps its own subset min/max
 * renders the same value as a different colour … which is exactly the
 * discontinuity this helper exists to prevent", and `gsplats/lift.py` normalises
 * beads over the full field so they share the finest node's `scalar_data_range`
 * rather than a per-segment one.
 */
describe('the remap declines across anything that is not a LOD ladder', () => {
  it('a kind=partition layer keeps ONE shared window on every part', () => {
    // A realistic partition: the compiler stamps no range on the wrapper, and
    // the parts' ranges are near-DISJOINT because each covers its own region of
    // a monotone field. Remapping would ramp the LUT black→white across both,
    // so at the seam the field would step 51 (white) → 51.1 (black).
    //
    // The reference is part_0's range: neither part declares `n_splats`, so
    // `deriveScalarRangeFromDescendants` scores both 0 and keeps the first
    // visited.
    const h = harness({ layer: true, kind: 'partition', display_type: 'points' }, [
      {
        name: 'part_0',
        type: 'points',
        attrs: { scalar_data_range: [0.02, 51], has_scalars: true, colormap: 'viridis' },
      },
      {
        name: 'part_1',
        type: 'points',
        attrs: { scalar_data_range: [51.1, 100], has_scalars: true, colormap: 'viridis' },
      },
    ]);
    expect(h.layer().scalarDataRange).toEqual([0.02, 51]);

    h.state.setDisplayRange('/obj', 10, 30);
    h.apply();

    // The user moved ONE slider, so both parts window on the same [10, 30] —
    // saturated where a part has no signal there, but monotone across the seam.
    // Remapping would have given part_1 ≈[60.7, 79.9], its own middle band.
    expect(windowOf(h.mats.get('part_0')!)).toEqual([10, 30]);
    expect(windowOf(h.mats.get('part_1')!)).toEqual([10, 30]);
  });

  it('a PLAIN group layer over two colormapped channels keeps one shared window', () => {
    // Two channels are two different physical fields, not one field at two
    // scales, so there is no scale change for the remap to undo. (Neither child
    // is a layer here — `deriveScalarRangeFromDescendants` ranks them by
    // `n_splats`, so the reference is ch1's [0, 5].)
    const h = harness({ layer: true }, [
      {
        name: 'ch0',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 0.02],
          n_splats: 1000,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
      {
        name: 'ch1',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 5],
          n_splats: 10000,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
    ]);
    expect(h.layer().scalarDataRange).toEqual([0, 5]);

    h.state.setDisplayRange('/obj', 1, 4);
    h.apply();

    expect(windowOf(h.mats.get('ch0')!)).toEqual([1, 4]);
    expect(windowOf(h.mats.get('ch1')!)).toEqual([1, 4]);
  });

  it('an `overview` tree: the coarse cap is eligible, the tiles under the partition are not', () => {
    // `lod --recipe overview`: a kind=lod group whose coarse level is a bare
    // leaf and whose fine level is a nested kind=partition of tiles. The split
    // falls out of the rule rather than being special-cased — the cap's path
    // crosses only the lod wrapper, each tile's path crosses the partition too.
    //
    // The `n_splats` here are chosen to make the cap's ELIGIBILITY observable,
    // and a real `overview` store does not look like this: the recipe gives the
    // cap and every part the same splat count (432 each on a measured store), so
    // `deriveScalarRangeFromDescendants`' strict `count > bestCount` keeps the
    // FIRST visited — the cap — and the cap then remaps onto its own range,
    // which `remapWindowToLeafRange`'s equality short-circuit makes a literal
    // no-op. On a real overview tree this change therefore does nothing in
    // either branch, and the fine parts keep rendering on the cap's window. See
    // the `overview` row of `ui/layers/README.md`.
    const h = harness({ ...LOD_WRAPPER }, [
      {
        name: 'level_0',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 8],
          n_splats: 100,
          has_scalars: true,
          colormap: 'viridis',
        },
      },
      {
        name: 'part_0',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 3],
          n_splats: 800,
          has_scalars: true,
          colormap: 'viridis',
        },
        under: { name: 'level_1', attrs: { kind: 'partition' } },
      },
      {
        name: 'part_1',
        type: 'gsplats',
        attrs: {
          amplitude_data_range: [0, 2],
          n_splats: 1600,
          has_scalars: true,
          colormap: 'viridis',
        },
        under: { name: 'level_1', attrs: { kind: 'partition' } },
      },
    ]);
    // The finest tile is the reference (largest `n_splats`).
    expect(h.layer().scalarDataRange).toEqual([0, 2]);

    h.apply();

    // The coarse cap is a LOD level of the whole object: remapped onto its own
    // range, so its amplitudes (which reach 8) are no longer clipped at 2.
    expect(windowOf(h.mats.get('level_0')!)).toEqual([0, 8]);
    // The tiles share the composed window. Remapping would have given part_0
    // [0, 3] and part_1 [0, 2] — a different LUT per tile on one field.
    expect(windowOf(h.mats.get('part_0')!)).toEqual([0, 2]);
    expect(windowOf(h.mats.get('part_1')!)).toEqual([0, 2]);
  });
});

describe('remapWindowToLeafRange', () => {
  const w = { min: 0.5, max: 1.5 };

  it('preserves the window’s relative position inside the range', () => {
    expect(remapWindowToLeafRange(w, [0, 2], [0, 8])).toEqual({ min: 2, max: 6 });
    // A shifted, not merely scaled, leaf range.
    expect(remapWindowToLeafRange(w, [0, 2], [10, 12])).toEqual({ min: 10.5, max: 11.5 });
    // A window that reaches outside the reference stays outside the leaf range
    // by the same proportion — clamping here would silently narrow the user's
    // choice, and the shader clamps at the LUT edge anyway.
    expect(remapWindowToLeafRange({ min: -1, max: 3 }, [0, 2], [0, 4])).toEqual({
      min: -2,
      max: 6,
    });
  });

  it('returns the window unchanged when either range is missing', () => {
    expect(remapWindowToLeafRange(w, undefined, [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], undefined)).toBe(w);
  });

  it('returns the window unchanged for a degenerate or inverted range', () => {
    // `[x, x]` is legitimate (constant amplitudes), not corrupt.
    expect(remapWindowToLeafRange(w, [2, 2], [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], [3, 3])).toBe(w);
    expect(remapWindowToLeafRange(w, [2, 0], [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], [8, 0])).toBe(w);
  });

  it('returns the window unchanged for a SUB-EPS span', () => {
    // Wider than zero but narrower than `DEGENERATE_SCALAR_RANGE_EPS` (1e-10).
    // `hi > lo` accepted these. On the reference side that is a real bug and not
    // just arithmetic hygiene: `computeUniforms` has ALREADY answered such a span
    // with the identity {1, 0}, so the composed window is [0, 1] and not the
    // reference window at all — remapping would extrapolate by up to 1e10. On the
    // leaf side, `computeScalarRangeUniforms` would answer the collapsed result
    // with the LUT midpoint.
    expect(remapWindowToLeafRange(w, [0, 1e-11], [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], [0, 1e-11])).toBe(w);
  });

  it('returns the window unchanged for a non-finite range', () => {
    // `[0, Infinity]` passes any WIDTH test, so finiteness is its own arm.
    expect(remapWindowToLeafRange(w, [0, Infinity], [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [-Infinity, 2], [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], [0, Infinity])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], [-Infinity, 8])).toBe(w);
    // The worst of them, and the reachable one: a window that starts AT the
    // reference minimum (composed windows routinely do — `[0, x]` on a `[0, y]`
    // reference) gives `t₀ = 0`, and `0 × Infinity` is NaN. Without the guard
    // that NaN reaches `updateScalarRange` and then the shader uniform.
    expect(remapWindowToLeafRange({ min: 0, max: 1 }, [0, 2], [0, Infinity])).toEqual({
      min: 0,
      max: 1,
    });
  });

  it('returns the window unchanged when a range carries NaN', () => {
    // Falls out of the `hi > lo` predicate: every comparison with NaN is false.
    expect(remapWindowToLeafRange(w, [NaN, 2], [0, 8])).toBe(w);
    expect(remapWindowToLeafRange(w, [0, 2], [0, NaN])).toBe(w);
  });

  it('short-circuits an equal range BIT-EXACTLY', () => {
    // The common case: the finest level IS the reference, and every leaf of an
    // ordinary single-range layer. Round-tripping through the two divisions is
    // not lossless — this window on this range comes back as
    // 0.0029999999999999996 — so the equality short-circuit is what keeps an
    // untouched layer's window untouched.
    const range: [number, number] = [1e-4, 0.02];
    const amplitudeWindow = { min: 0.003, max: 0.017 };
    // VALUE equality drives it, not object identity — a leaf and its wrapper
    // hold two separate arrays parsed from two separate zarr attrs.
    const copy: [number, number] = [range[0], range[1]];
    expect(remapWindowToLeafRange(amplitudeWindow, range, copy)).toBe(amplitudeWindow);
    // The premise, asserted rather than assumed: the round trip really is lossy
    // on this range, so the short-circuit does work instead of restating the
    // arithmetic.
    const span = range[1] - range[0];
    expect(range[0] + ((amplitudeWindow.min - range[0]) / span) * span).not.toBe(
      amplitudeWindow.min
    );
  });
});
