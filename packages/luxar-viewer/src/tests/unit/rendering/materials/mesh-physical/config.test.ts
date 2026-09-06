/**
 * The physical mesh material family: the one attr→property mapping, the data-driven
 * compositing rule, and the `LuxarMaterial` surface — on BOTH backends from one table.
 *
 * What is worth pinning is every case where the wrong answer renders something
 * plausible: a `sheen=1` that renders nothing (black sheen colour), a translucent shell
 * that writes depth and drops its own back faces, an opaque mesh the depth sorter
 * registers, a reset that overwrites the blend state through the generic fallback.
 */

import { describe, it, expect, beforeAll, vi } from 'vitest';
import * as THREE from 'three';
import { PhysicalMeshMaterial } from '../../../../../rendering/materials/mesh-physical/material-glsl';
import {
  PHYSICAL_MESH_DEFAULTS,
  PHYSICAL_MESH_KNOB_KEYS,
  applyPhysicalMeshConfig,
  derivePhysicalCompositing,
  isPhysicalMeshMaterial,
  physicalSetVertexAlpha,
  type PhysicalMeshHost,
  type PhysicalMeshMaterialConfig,
} from '../../../../../rendering/materials/mesh-physical/config';
import { loadTslMaterials } from '../../../../../rendering/tsl/load';
import { requireTslMaterials } from '../../../../../rendering/tsl/slot';
import { isLuxarMaterial } from '../../../../../ui/layers/luxar-material';
import { isCameraAwareMaterial } from '../../../../../rendering/materials/_shared/camera-aware-material';
import { MeshMaterial } from '../../../../../rendering/materials/mesh/material-glsl';

type PhysicalCtor = new (config?: PhysicalMeshMaterialConfig) => THREE.Material & PhysicalMeshHost;

beforeAll(async () => {
  await loadTslMaterials();
});

/** Both twins, resolved lazily so the TSL one is read after the registry loads. */
const BACKENDS: Array<[string, () => PhysicalCtor]> = [
  ['glsl', () => PhysicalMeshMaterial as unknown as PhysicalCtor],
  ['tsl', () => requireTslMaterials().materials.meshPhysical as unknown as PhysicalCtor],
];

describe('derivePhysicalCompositing — translucency is read off the data', () => {
  it.each([
    // opacity, vertexAlpha, alphaCutoff → transparent, depthWrite, alphaTest, stamp
    [1.0, false, undefined, false, true, 0, 'opaque', 'plain RGB mesh at full opacity'],
    [0.3, false, undefined, true, false, 0, undefined, 'opacity below 1'],
    [1.0, true, undefined, true, false, 0, undefined, 'vertex alpha with no cutoff'],
    [1.0, true, 0.4, false, true, 0.4, 'opaque', 'vertex alpha WITH a cutoff is a cutout'],
    [0.5, true, 0.4, true, false, 0, undefined, 'opacity wins over the cutoff'],
    [1.0, false, 0.4, false, true, 0.4, 'opaque', 'a cutoff on RGB is inert but harmless'],
  ] as const)(
    'opacity=%s vertexAlpha=%s cutoff=%s → transparent=%s depthWrite=%s alphaTest=%s stamp=%s (%s)',
    (opacity, vertexAlpha, alphaCutoff, transparent, depthWrite, alphaTest, stamp, _why) => {
      const d = derivePhysicalCompositing({ opacity, vertexAlpha, alphaCutoff });
      expect(d.transparent).toBe(transparent);
      expect(d.depthWrite).toBe(depthWrite);
      expect(d.alphaTest).toBe(alphaTest);
      expect(d.blendingMode).toBe(stamp);
    }
  );
});

describe.each(BACKENDS)('physical mesh material (%s)', (_name, ctor) => {
  it('maps every Phase 1 knob one-to-one and clamps like the house knobs', () => {
    const m = new (ctor())({
      roughness: 0.4,
      metalness: 1.0,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      iridescence: 0.7,
      sheen: 0.5,
      sheenColor: '#ff0000',
    });
    expect(m.roughness).toBe(0.4);
    expect(m.metalness).toBe(1.0);
    expect(m.clearcoat).toBe(1.0);
    expect(m.clearcoatRoughness).toBe(0.1);
    expect(m.iridescence).toBe(0.7);
    expect(m.sheen).toBe(0.5);
    // Hex is read as sRGB → linear working space: pure red survives exactly.
    expect(m.sheenColor.r).toBeCloseTo(1, 6);
    expect(m.sheenColor.g).toBe(0);
    expect(m.vertexColors).toBe(true);
    expect(m.toneMapped).toBe(false);

    const clamped = new (ctor())({ roughness: 5, metalness: -1, clearcoat: Number.NaN });
    expect(clamped.roughness).toBe(1);
    expect(clamped.metalness).toBe(0);
    // NaN routes to the DEFAULT, not a boundary — the sibling sanitizer policy.
    expect(clamped.clearcoat).toBe(PHYSICAL_MESH_DEFAULTS.clearcoat);
  });

  it("an absent knob is three's default — except sheen colour, which is WHITE", () => {
    const m = new (ctor())({ sheen: 1.0 });
    for (const key of PHYSICAL_MESH_KNOB_KEYS) {
      if (key === 'sheen') continue;
      const prop = key === 'clearcoat_roughness' ? 'clearcoatRoughness' : key;
      expect(m[prop as keyof PhysicalMeshHost]).toBe(
        PHYSICAL_MESH_DEFAULTS[prop as keyof typeof PHYSICAL_MESH_DEFAULTS]
      );
    }
    // Three's own default is black, under which `sheen=1` renders nothing.
    expect(m.sheenColor.getHex()).toBe(0xffffff);
    expect(new THREE.MeshPhysicalMaterial().sheenColor.getHex()).toBe(0x000000);
  });

  it('stamps the family marker and no blending mode when translucent', () => {
    const opaque = new (ctor())({});
    expect(isPhysicalMeshMaterial(opaque)).toBe(true);
    expect(opaque.userData.blendingMode).toBe('opaque');
    expect(opaque.transparent).toBe(false);
    expect(opaque.depthWrite).toBe(true);

    const translucent = new (ctor())({ opacity: 0.3 });
    expect(translucent.userData.blendingMode).toBeUndefined();
    expect(translucent.transparent).toBe(true);
    expect(translucent.depthWrite).toBe(false);
    expect(translucent.opacity).toBe(0.3);
  });

  it('flat shading is a property of the surface, not the house key', () => {
    expect(new (ctor())({ flatShading: true }).flatShading).toBe(true);
    expect(new (ctor())({}).flatShading).toBe(false);
  });

  it('exposes the LuxarMaterial surface with physical meanings, and no camera surface', () => {
    const m = new (ctor())({ intensity: 2.0, offset: 0.25, gamma: 1.8 }) as THREE.Material &
      PhysicalMeshHost & {
        updateOpacity(v: number): void;
        getOpacity(): number;
        updateIntensity(v: number): void;
        updateOffset(v: number): void;
        updateGamma(v: number): void;
        applyBlendingMode(mode: string): void;
      };
    expect(isLuxarMaterial(m)).toBe(true);
    expect(isCameraAwareMaterial(m)).toBe(false);

    // intensity → scalar base-colour gain; offset → emissive; gamma → recorded only.
    expect(m.color.r).toBe(2.0);
    expect(m.emissive.g).toBe(0.25);
    expect(m.userData.gamma).toBe(1.8);

    m.updateIntensity(0.5);
    expect(m.color.b).toBe(0.5);
    m.updateOffset(-1); // black-level subtraction has no physical counterpart
    expect(m.emissive.r).toBe(0);
    m.updateGamma(2.2);
    expect(m.userData.gamma).toBe(2.2);

    // Opacity re-derives the whole compositing decision, both ways.
    m.updateOpacity(0.4);
    expect(m.getOpacity()).toBe(0.4);
    expect(m.transparent).toBe(true);
    expect(m.userData.blendingMode).toBeUndefined();
    m.updateOpacity(1.0);
    expect(m.transparent).toBe(false);
    expect(m.userData.blendingMode).toBe('opaque');
  });

  it('applyBlendingMode is a no-op — the generic fallback must never reach this material', () => {
    const m = new (ctor())({ opacity: 0.3 }) as THREE.Material &
      PhysicalMeshHost & { applyBlendingMode(mode: string): void };
    const before = {
      transparent: m.transparent,
      depthWrite: m.depthWrite,
      blending: m.blending,
      stamp: m.userData.blendingMode,
    };
    m.applyBlendingMode('opaque');
    m.applyBlendingMode('additive');
    expect(m.transparent).toBe(before.transparent);
    expect(m.depthWrite).toBe(before.depthWrite);
    expect(m.blending).toBe(before.blending);
    expect(m.userData.blendingMode).toBe(before.stamp);
  });

  it('learns about vertex alpha at commit time and flips translucency once', () => {
    const m = new (ctor())({});
    expect(m.transparent).toBe(false);
    physicalSetVertexAlpha(m, true);
    expect(m.transparent).toBe(true);
    expect(m.depthWrite).toBe(false);
    expect(m.userData.blendingMode).toBeUndefined();
    // Idempotent: a second commit with the same answer costs no rebuild flag.
    m.needsUpdate = false;
    const version = (m as unknown as { version: number }).version;
    physicalSetVertexAlpha(m, true);
    expect((m as unknown as { version: number }).version).toBe(version);
    // With a cutoff, vertex alpha is a CUTOUT instead and the mesh stays opaque.
    const cut = new (ctor())({ alphaCutoff: 0.5 });
    physicalSetVertexAlpha(cut, true);
    expect(cut.transparent).toBe(false);
    expect(cut.alphaTest).toBe(0.5);
    expect(cut.userData.blendingMode).toBe('opaque');
  });
});

describe('applyPhysicalMeshConfig on a bare host (no three class involved)', () => {
  it('writes only the documented properties, so a stub is enough to audit the mapping', () => {
    const host: PhysicalMeshHost = {
      opacity: 1,
      transparent: false,
      depthWrite: true,
      alphaTest: 0,
      color: new THREE.Color(),
      emissive: new THREE.Color(),
      sheenColor: new THREE.Color(),
      roughness: 1,
      metalness: 0,
      clearcoat: 0,
      clearcoatRoughness: 0,
      iridescence: 0,
      sheen: 0,
      vertexColors: false,
      flatShading: false,
      toneMapped: true,
      needsUpdate: false,
      userData: {},
    };
    applyPhysicalMeshConfig(host, { metalness: 1, opacity: 0.5 });
    expect(host.metalness).toBe(1);
    expect(host.vertexColors).toBe(true);
    expect(host.transparent).toBe(true);
    expect(host.userData.material).toBe('physical');
    expect(isPhysicalMeshMaterial(host)).toBe(true);
    expect(isPhysicalMeshMaterial(new MeshMaterial({}))).toBe(false);
    expect(isPhysicalMeshMaterial(null)).toBe(false);
  });

  it('flags a rebuild only when a program-affecting property changed', () => {
    const m = new PhysicalMeshMaterial({});
    const spy = vi.spyOn(m, 'needsUpdate', 'set');
    m.updateOpacity(0.9); // false → true transparent: program changes
    expect(spy).toHaveBeenCalledTimes(1);
    m.updateOpacity(0.8); // still transparent: plain state write
    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });
});
