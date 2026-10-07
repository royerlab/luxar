// @vitest-environment jsdom
/**
 * Every control the Layers panel SHOWS has an effect, and Reset undoes it.
 *
 * Driven by `LAYER_CONTROL_RULES` (src/ui/layers/layer-control-rules.ts). One
 * fixture per kind of layer — points, lines, gsplats, house mesh, physical mesh,
 * sound, LOD, partition, labelled, custom LUT — is mounted with REAL materials,
 * then:
 *
 * - the controls on screen are exactly the ones the table allows, and exactly
 *   the ones the fixture declares (`shows`), so a control hidden where it works
 *   fails as surely as one shown where it does nothing;
 * - every shown control is driven through its real DOM element to a non-default
 *   value, and the observable state — every leaf's material uniforms, defines and
 *   flags (visual AND pick), each object's render-order slot, the audio graph's
 *   gains and the LOD registry's selector modes — must change;
 * - "Reset this layer" must then put that observable state back exactly;
 * - every `data-control` element the panel renders has a table row, and every
 *   row is exercised by at least one fixture.
 *
 * A new control therefore needs a rule, a `data-control` id and a probe below
 * (`PROBES` is keyed by the id type, so a missing probe does not compile).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { AnimationController } from '../../../../scene/animation/animation-controller';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { PhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/material-glsl';

vi.mock('../../../../ui/toast', () => ({ showToast: vi.fn() }));
vi.mock('../../../../rendering/material-manager', () => ({
  materialManager: { register: vi.fn() },
}));

/** The LOD registry the Active-level control writes; the engine state it observes. */
const selectorModes = new Map<string, unknown>();
vi.mock('../../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: () => ({
      getDefaultLoader: () => ({
        lodGroupRegistry: {
          setSelectorMode: (path: string, mode: unknown) => selectorModes.set(path, mode),
          get: (path: string) => ({
            selectorMode: selectorModes.get(path) ?? 'auto',
            activeChildIndex: 0,
            children: [{}, {}],
          }),
        },
      }),
    }),
  },
}));

import { LayersPanel } from '../../../../ui/layers/layers-panel';
import {
  LAYER_CONTROL_IDS,
  LAYER_CONTROL_RULES,
  type LayerControlId,
} from '../../../../ui/layers/layer-control-rules';
import type { ContextMenuItem } from '../../../../ui/overlay-widgets/context-menu';
import type { LayerInfo } from '../../../../ui/layers/layer-state';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

interface LeafSpec {
  path: string;
  type: 'points' | 'lines' | 'gsplats' | 'mesh';
  attrs?: Record<string, unknown>;
}

interface Fixture {
  name: string;
  /** The layer under test (selected). */
  layer: string;
  graph: SceneNode;
  leaves: LeafSpec[];
  /** The controls this layer must show — no more, no fewer. */
  shows: LayerControlId[];
}

const node = (
  path: string,
  type: string,
  attrs: Record<string, unknown>,
  children: SceneNode[] = []
): SceneNode =>
  ({ name: path.split('/').pop(), path, type, attrs, children }) as unknown as SceneNode;

const root = (...children: SceneNode[]): SceneNode => node('/', 'group', {}, children);

/** A layered leaf at the root, mounted with a real material of its type. */
function leafFixture(
  name: string,
  type: LeafSpec['type'],
  attrs: Record<string, unknown>,
  shows: LayerControlId[]
): Fixture {
  const all = { layer: true, type, ...attrs };
  return {
    name,
    layer: '/layer',
    graph: root(node('/layer', type, all)),
    leaves: [{ path: '/layer', type, attrs: all }],
    shows,
  };
}

const EMISSIVE: LayerControlId[] = ['displayRange', 'gamma', 'opacity', 'blend', 'layerOrder'];
const HOUSE_MESH: LayerControlId[] = [
  ...EMISSIVE,
  'ambient',
  'shadeExponent',
  'specular',
  'shininess',
  'alphaCutoff',
];
const VOCABULARY = { '1': 'nucleus', '2': 'membrane' };
const CUSTOM_LUT = Uint8Array.from({ length: 256 * 4 }, (_, i) => (i * 7) % 256);

const lodGroup = (path: string, displayType: string, leafType: LeafSpec['type']): SceneNode =>
  node(path, 'group', { kind: 'lod', display_type: displayType }, [
    node(`${path}/level_0`, leafType, { type: leafType }),
    node(`${path}/level_1`, leafType, { type: leafType }),
  ]);

const FIXTURES: Fixture[] = [
  leafFixture('points with scalars', 'points', { has_scalars: true, scalar_data_range: [0, 10] }, [
    ...EMISSIVE,
    'colormap',
  ]),
  leafFixture('lines', 'lines', {}, EMISSIVE),
  leafFixture('volumetric gsplats', 'gsplats', { blending_mode: 'volumetric' }, [
    ...EMISSIVE,
    'absorption',
    'colormap',
  ]),
  leafFixture('house mesh', 'mesh', { blending_mode: 'opaque', has_normals: true }, HOUSE_MESH),
  leafFixture('physical mesh', 'mesh', { material: 'physical', roughness: 0.4 }, [
    'displayRange',
    'opacity',
    'physical',
    'layerOrder',
  ]),
  {
    name: 'sound',
    layer: '/hum',
    graph: root(node('/hum', 'sound', { layer: true, gain: 0.5 })),
    leaves: [],
    shows: ['gain'],
  },
  {
    name: 'LOD group',
    layer: '/lod',
    graph: root({ ...lodGroup('/lod', 'points', 'points'), attrs: lodAttrs('points') }),
    leaves: [
      { path: '/lod/level_0', type: 'points' },
      { path: '/lod/level_1', type: 'points' },
    ],
    shows: [...EMISSIVE, 'activeLevel'],
  },
  {
    name: 'partition of LOD groups',
    layer: '/tiles',
    graph: root(
      node('/tiles', 'group', { layer: true, kind: 'partition', display_type: 'gsplats' }, [
        lodGroup('/tiles/part_0', 'gsplats', 'gsplats'),
      ])
    ),
    leaves: [
      { path: '/tiles/part_0/level_0', type: 'gsplats' },
      { path: '/tiles/part_0/level_1', type: 'gsplats' },
    ],
    shows: [...EMISSIVE, 'colormap', 'activeLevel'],
  },
  leafFixture('labelled gsplats', 'gsplats', { label_vocabulary: VOCABULARY }, [
    ...EMISSIVE,
    'colormap',
    'labels',
  ]),
  leafFixture('custom-LUT gsplats', 'gsplats', { colormap: 'custom', customLutBytes: CUSTOM_LUT }, [
    ...EMISSIVE,
    'colormap',
    'customColormap',
  ]),
];

function lodAttrs(displayType: string): Record<string, unknown> {
  return { layer: true, kind: 'lod', display_type: displayType };
}

function makeMaterial(spec: LeafSpec): THREE.Material {
  switch (spec.type) {
    case 'points':
      return new PointMaterial();
    case 'lines':
      return new LineMaterial();
    case 'gsplats':
      return new GSplatMaterial();
    case 'mesh':
      return spec.attrs?.material === 'physical'
        ? new PhysicalMeshMaterial({ roughness: 0.4 })
        : new MeshMaterial();
  }
}

/** Mount the fixture's THREE objects: one real-material mesh per leaf, groups by path. */
function buildScene(fixture: Fixture): THREE.Group {
  const sceneRoot = new THREE.Group();
  const objects = new Map<string, THREE.Object3D>([['/', sceneRoot]]);
  const visit = (n: SceneNode, parent: THREE.Object3D): void => {
    const leaf = fixture.leaves.find((l) => l.path === n.path);
    let obj: THREE.Object3D;
    if (leaf) {
      const geometry = new THREE.BufferGeometry();
      geometry.userData.hasScalars = leaf.attrs?.has_scalars === true;
      const mesh = new THREE.Mesh(geometry, makeMaterial(leaf));
      mesh.userData.nodeType = leaf.type;
      // The material IS the leaf's own, so the panel writes it in place.
      mesh.userData._layerMaterialCloned = true;
      obj = mesh;
    } else {
      obj = new THREE.Group();
    }
    obj.name = n.path;
    parent.add(obj);
    objects.set(n.path, obj);
    for (const child of n.children ?? []) visit(child, obj);
  };
  for (const child of fixture.graph.children ?? []) visit(child, sceneRoot);
  return sceneRoot;
}

// ---------------------------------------------------------------------------
// Observable state
// ---------------------------------------------------------------------------

/** Material props that move on ANY write (bookkeeping), not on an effect. */
const BOOKKEEPING = new Set(['version', 'uuid', 'id', 'needsUpdate']);

function serialize(value: unknown): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value !== 'object') return value;
  if ((value as THREE.Texture).isTexture) return `texture:${(value as THREE.Texture).uuid}`;
  if ('toArray' in value && typeof value.toArray === 'function') return value.toArray();
  if (ArrayBuffer.isView(value)) return Array.from(value as Float32Array);
  if (Array.isArray(value)) return value.map(serialize);
  return undefined;
}

function snapshotMaterial(material: THREE.Material | undefined): Record<string, unknown> {
  if (!material) return {};
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(material)) {
    if (BOOKKEEPING.has(key) || typeof value === 'function') continue;
    if (typeof value !== 'object' || value === null) out[key] = value;
    else if (key !== 'uniforms' && key !== 'userData') out[key] = serialize(value);
  }
  const uniforms = (material as THREE.ShaderMaterial).uniforms ?? {};
  for (const [key, u] of Object.entries(uniforms)) out[`u:${key}`] = serialize(u.value);
  out.defines = sortedJson((material as THREE.ShaderMaterial).defines ?? {});
  // `userData.gamma` RECORDS the composed input (clone/rebuild bookkeeping); the
  // render effect of gamma is its uniform/define. The physical family records it
  // and has no gamma term at all, so counting the record would hide a dead Gamma.
  out.userData = sortedJson({ ...material.userData, gamma: undefined });
  return out;
}

/**
 * Key order is insertion order, which a remove-then-re-add changes without effect;
 * a key set to null and an absent key both mean "unset".
 */
function sortedJson(record: Record<string, unknown>): string {
  const entries = Object.entries(record).filter(([, v]) => v !== null && v !== undefined);
  return JSON.stringify(entries.sort(([a], [b]) => a.localeCompare(b)));
}

interface Ctx {
  fixture: Fixture;
  panel: LayersPanel;
  container: HTMLElement;
  sceneRoot: THREE.Group;
  /** The audio graph's per-node gain, seeded from the authored attrs. */
  gains: Map<string, number>;
}

function observe(ctx: Ctx): unknown {
  const objects: Record<string, unknown> = {};
  ctx.sceneRoot.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    // Only a drawable reads render state; a group's userData is no effect.
    if (!mesh.material) return;
    const pick = obj.userData.pickNode as THREE.Mesh | undefined;
    objects[obj.name] = {
      visible: obj.visible,
      layerOrder: obj.userData.layerOrder,
      material: snapshotMaterial(mesh.material as THREE.Material | undefined),
      pick: snapshotMaterial(pick?.material as THREE.Material | undefined),
    };
  });
  return {
    objects,
    gains: Object.fromEntries(ctx.gains),
    // Every lod_group loads in `auto`, so an absent entry and `auto` are the same.
    selectorModes: JSON.stringify([...selectorModes].filter(([, mode]) => mode !== 'auto')),
  };
}

// ---------------------------------------------------------------------------
// Driving the real DOM
// ---------------------------------------------------------------------------

function controlEl(ctx: Ctx, id: LayerControlId): HTMLElement {
  const el = ctx.container.querySelector<HTMLElement>(`[data-control="${id}"]`);
  if (!el) throw new Error(`no [data-control="${id}"] in the panel`);
  return el;
}

function isShown(el: Element, stop: Element): boolean {
  if (!el.isConnected) return false;
  for (let e: Element | null = el; e && e !== stop; e = e.parentElement) {
    if ((e as HTMLElement).style?.display === 'none') return false;
  }
  return true;
}

/** Move a range input to whichever end of its track it is not already at. */
function dragToOtherEnd(input: HTMLInputElement): void {
  input.value = Number(input.value) >= Number(input.max) ? input.min : input.max;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

function pick(select: HTMLSelectElement, value: string): void {
  select.value = value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
}

function firstOtherOption(select: HTMLSelectElement): string {
  return Array.from(select.options).find((o) => o.value !== select.value && o.value !== '')!.value;
}

const slider = (id: LayerControlId) => ({
  drive: (ctx: Ctx) => dragToOtherEnd(controlEl(ctx, id).querySelector('input[type="range"]')!),
});

const selectIn = (id: LayerControlId, choose: (s: HTMLSelectElement) => string) => ({
  drive: (ctx: Ctx) => {
    const select = controlEl(ctx, id).querySelector('select')!;
    pick(select, choose(select));
  },
});

interface Probe {
  /** Put the control somewhere its effect can be seen (optional). */
  prepare?: (ctx: Ctx) => void;
  /** Set a non-default value through the real DOM element. */
  drive: (ctx: Ctx) => void;
}

const colormapSelect = (ctx: Ctx): HTMLSelectElement =>
  controlEl(ctx, 'colormap').querySelector('select')!;

const PROBES: Record<LayerControlId, Probe> = {
  displayRange: {
    drive: (ctx) => {
      const low = controlEl(ctx, 'displayRange').querySelector<HTMLInputElement>(
        '.luxar-range-slider__input'
      )!;
      low.value = String((Number(low.min) + Number(low.max)) / 2);
      low.dispatchEvent(new Event('input', { bubbles: true }));
    },
  },
  gamma: slider('gamma'),
  opacity: slider('opacity'),
  absorption: slider('absorption'),
  ambient: slider('ambient'),
  shadeExponent: slider('shadeExponent'),
  specular: slider('specular'),
  shininess: slider('shininess'),
  alphaCutoff: slider('alphaCutoff'),
  physical: slider('physical'),
  blend: selectIn('blend', (s) => (s.value === 'max' ? 'normal' : 'max')),
  layerOrder: {
    drive: (ctx) => {
      const input = controlEl(ctx, 'layerOrder').querySelector('input')!;
      input.value = '7';
      input.dispatchEvent(new Event('change', { bubbles: true }));
    },
  },
  colormap: selectIn('colormap', (s) => (s.value === 'viridis' ? 'plasma' : 'viridis')),
  customColormap: {
    // The authored LUT is already applied; move off it so choosing it shows.
    prepare: (ctx) => pick(colormapSelect(ctx), 'viridis'),
    drive: (ctx) => pick(colormapSelect(ctx), 'custom'),
  },
  labels: selectIn('labels', firstOtherOption),
  activeLevel: selectIn('activeLevel', (s) => (s.value === '0' ? '1' : '0')),
  gain: {
    drive: (ctx) => {
      const input = ctx.container.querySelector<HTMLInputElement>('[data-control="gain"]')!;
      input.value = '1.5';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    },
  },
};

function resetLayer(ctx: Ctx): void {
  const builder = ctx.panel as unknown as {
    buildRowMenuItems(layer: LayerInfo): ContextMenuItem[];
  };
  const items = builder.buildRowMenuItems(ctx.panel.layerState.getLayer(ctx.fixture.layer)!);
  items.find((item) => item.label === 'Reset this layer')!.action!();
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function animationController(): AnimationController {
  return {
    startAnimation: vi.fn(),
    addPerFrameCallback: vi.fn(),
    removePerFrameCallback: vi.fn(),
    hasPerFrameCallback: vi.fn(() => false),
  } as unknown as AnimationController;
}

let mounted: Ctx | null = null;

function mount(fixture: Fixture): Ctx {
  selectorModes.clear();
  const container = document.createElement('div');
  document.body.appendChild(container);
  const panel = new LayersPanel(container, animationController());
  const gains = new Map<string, number>();
  const visitSound = (n: SceneNode): void => {
    if (n.type === 'sound') gains.set(n.path, n.attrs.gain as number);
    for (const c of n.children ?? []) visitSound(c);
  };
  visitSound(fixture.graph);
  panel.setAudioPort({
    setNodeMuted: vi.fn(),
    setNodeGain: (path: string, gain: number) => gains.set(path, gain),
  });
  const sceneRoot = buildScene(fixture);
  panel.initFromScene(sceneRoot, fixture.graph);
  // The fixture's materials are bare constructions, so push the authored state
  // through the panel's own apply set once (what node-factory does at build time).
  panel.resetAllLayers();
  panel.show();
  panel.layerState.select(fixture.layer, 'single');
  mounted = { fixture, panel, container, sceneRoot, gains };
  return mounted;
}

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  mounted?.panel.dispose();
  mounted = null;
});

function shownControls(ctx: Ctx): LayerControlId[] {
  return LAYER_CONTROL_IDS.filter((id) => {
    const el = ctx.container.querySelector(`[data-control="${id}"]`);
    return el !== null && isShown(el, ctx.container);
  });
}

describe.each(FIXTURES)('Layers panel controls — $name', (fixture) => {
  it('shows exactly the declared controls, and exactly what LAYER_CONTROL_RULES allows', () => {
    const ctx = mount(fixture);
    const layer = ctx.panel.layerState.getLayer(fixture.layer)!;
    const allowed = LAYER_CONTROL_IDS.filter((id) => LAYER_CONTROL_RULES[id].visibleFor(layer));
    expect(shownControls(ctx)).toEqual(allowed);
    expect([...allowed].sort()).toEqual([...fixture.shows].sort());
  });

  it.each(LAYER_CONTROL_IDS)('%s, when shown, has an effect that Reset undoes', (id) => {
    const ctx = mount(fixture);
    // An invisible control asserts nothing; the test above pins which are shown.
    if (!shownControls(ctx).includes(id)) return;
    const initial = observe(ctx);
    PROBES[id].prepare?.(ctx);
    const before = observe(ctx);
    PROBES[id].drive(ctx);
    expect(observe(ctx), `${id} on ${fixture.name} changed nothing observable`).not.toEqual(before);
    resetLayer(ctx);
    // A subset match: switching a colormap off leaves its (now unread) uniforms
    // declared at neutral values, but every observed key of the initial state —
    // defines and userData included, compared whole — must be back.
    expect(observe(ctx), `Reset left ${id} on ${fixture.name} changed`).toMatchObject(
      initial as object
    );
  });
});

describe('LAYER_CONTROL_RULES completeness', () => {
  it('every data-control element the panel renders has a rule', () => {
    const rendered = new Set<string>();
    for (const fixture of FIXTURES) {
      const ctx = mount(fixture);
      for (const el of Array.from(ctx.container.querySelectorAll<HTMLElement>('[data-control]'))) {
        rendered.add(el.dataset.control!);
      }
      ctx.panel.dispose();
      mounted = null;
    }
    expect([...rendered].filter((id) => !(id in LAYER_CONTROL_RULES))).toEqual([]);
    // …and every rule's control is actually rendered somewhere.
    expect(LAYER_CONTROL_IDS.filter((id) => !rendered.has(id))).toEqual([]);
  });

  it('every control group in the appearance section carries a data-control id', () => {
    const ctx = mount(FIXTURES[0]);
    const untagged = Array.from(
      ctx.container.querySelectorAll<HTMLElement>(
        '.luxar-layers-panel__controls > .luxar-layers-panel__control-group'
      )
    ).filter((el) => !el.dataset.control);
    expect(untagged.map((el) => el.textContent)).toEqual([]);
  });

  it('every rule is shown (and so effect-checked) by at least one fixture', () => {
    const exercised = new Set(FIXTURES.flatMap((f) => f.shows));
    expect(LAYER_CONTROL_IDS.filter((id) => !exercised.has(id))).toEqual([]);
  });
});
