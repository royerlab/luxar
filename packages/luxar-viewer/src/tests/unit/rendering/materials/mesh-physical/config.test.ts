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
  PHYSICAL_MESH_KNOBS,
  applyPhysicalMeshConfig,
  clampPhysicalKnob,
  derivePhysicalCompositing,
  isPhysicalMeshMaterial,
  physicalKnobFromSlider,
  physicalKnobInertReason,
  physicalKnobToSlider,
  physicalSetVertexAlpha,
  type PhysicalMeshHost,
  type PhysicalMeshKnobKey,
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
    // opacity, vertexAlpha, alphaCutoff, transmission → transparent, depthWrite, alphaTest, stamp, first
    [1.0, false, undefined, 0, false, true, 0, 'opaque', false, 'plain RGB mesh at full opacity'],
    [0.3, false, undefined, 0, true, false, 0, undefined, false, 'opacity below 1'],
    [1.0, true, undefined, 0, true, false, 0, undefined, false, 'vertex alpha with no cutoff'],
    [
      1.0,
      true,
      0.4,
      0,
      false,
      true,
      0.4,
      'opaque',
      false,
      'vertex alpha WITH a cutoff is a cutout',
    ],
    [0.5, true, 0.4, 0, true, false, 0, undefined, false, 'opacity wins over the cutoff'],
    [
      1.0,
      false,
      0.4,
      0,
      false,
      true,
      0.4,
      'opaque',
      false,
      'a cutoff on RGB is inert but harmless',
    ],
    // Glass (spec §3.4): translucent, never writes depth, and draws before the
    // emissive data it shares a band with — the cutoff is moot on a see-through surface.
    [1.0, false, undefined, 1.0, true, false, 0, undefined, true, 'full transmission'],
    [1.0, true, 0.4, 0.2, true, false, 0, undefined, true, 'any transmission beats the cutout'],
  ] as const)(
    'opacity=%s vertexAlpha=%s cutoff=%s transmission=%s → transparent=%s depthWrite=%s alphaTest=%s stamp=%s first=%s (%s)',
    (
      opacity,
      vertexAlpha,
      alphaCutoff,
      transmission,
      transparent,
      depthWrite,
      alphaTest,
      stamp,
      first,
      _why
    ) => {
      const inputs = { opacity, vertexAlpha, alphaCutoff, transmission, refractData: false };
      const d = derivePhysicalCompositing(inputs);
      expect(d.transparent).toBe(transparent);
      expect(d.depthWrite).toBe(depthWrite);
      expect(d.alphaTest).toBe(alphaTest);
      expect(d.blendingMode).toBe(stamp);
      expect(d.drawBeforeEmissive).toBe(first);
      expect(d.drawAfterEmissive).toBe(false);
      // Phase 3: `refract_data` moves the SAME glass from first to last in its band
      // and changes nothing else about how it composites.
      const r = derivePhysicalCompositing({ ...inputs, refractData: true });
      expect(r.drawAfterEmissive).toBe(first);
      expect(r.drawBeforeEmissive).toBe(false);
      expect([r.transparent, r.depthWrite, r.alphaTest, r.blendingMode]).toEqual([
        d.transparent,
        d.depthWrite,
        d.alphaTest,
        d.blendingMode,
      ]);
    }
  );
});

describe('the knob table and its slider mapping', () => {
  it('lists the Phase 1 surface knobs then the Phase 2 glass family, in panel order', () => {
    expect(PHYSICAL_MESH_KNOB_KEYS).toEqual([
      'roughness',
      'metalness',
      'clearcoat',
      'clearcoat_roughness',
      'iridescence',
      'sheen',
      'transmission',
      'ior',
      'thickness',
      'attenuation_distance',
      'dispersion',
    ]);
    // Three's defaults, read off a bare three material so the table cannot drift.
    const three = new THREE.MeshPhysicalMaterial();
    for (const key of PHYSICAL_MESH_KNOB_KEYS) {
      const spec = PHYSICAL_MESH_KNOBS[key];
      expect(three[spec.prop], key).toBe(spec.default);
    }
  });

  it.each([
    ['roughness', 5, 1],
    ['roughness', -1, 0],
    ['roughness', Number.NaN, PHYSICAL_MESH_KNOBS.roughness.default],
    ['roughness', 'shiny', PHYSICAL_MESH_KNOBS.roughness.default],
    ['ior', 0.5, 1],
    ['ior', 10, 2.333],
    ['thickness', -2, 0],
    ['thickness', 42, 42], // unbounded above: past the slider is still a legal thickness
    ['thickness', Number.POSITIVE_INFINITY, 0], // …but an infinite one is corrupt, not chosen
    ['attenuation_distance', 0, 1e-3],
    ['attenuation_distance', Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    ['attenuation_distance', undefined, Number.POSITIVE_INFINITY],
    ['dispersion', 3, 3],
  ] as const)('clampPhysicalKnob(%s, %s) → %s', (key, value, expected) => {
    expect(clampPhysicalKnob(key, value)).toBe(expected);
  });

  it.each([
    // key, live values → reason (null = live). Measured on the reflections demo.
    ['clearcoat_roughness', { clearcoat: 0 }, /Clearcoat/],
    ['clearcoat_roughness', { clearcoat: 0.3 }, null],
    ['transmission', { metalness: 1 }, /metal/],
    ['transmission', { metalness: 0.5 }, null],
    ['ior', { metalness: 1 }, /metal/],
    ['ior', { metalness: 0 }, null],
    ['thickness', { transmission: 0 }, /Transmission/],
    ['thickness', { transmission: 1, metalness: 1 }, /Transmission/],
    ['thickness', { transmission: 1, metalness: 0 }, null],
    ['dispersion', { transmission: 0.2 }, null],
    ['attenuation_distance', { transmission: 1 }, /white/],
    ['attenuation_distance', { transmission: 1, attenuation_color: '#FFFFFF' }, /white/],
    ['attenuation_distance', { transmission: 1, attenuation_color: '#f6d148' }, null],
    ['attenuation_distance', { transmission: 0, attenuation_color: '#f6d148' }, /Transmission/],
    ['roughness', { metalness: 1, transmission: 0 }, null],
    ['sheen', {}, null],
  ] as const)('physicalKnobInertReason(%s, %j)', (key, values, expected) => {
    const reason = physicalKnobInertReason(key, values);
    if (expected === null) expect(reason).toBeNull();
    else expect(reason).toMatch(expected);
  });

  it('maps an infinite attenuation distance onto the slider top stop and back', () => {
    const top = PHYSICAL_MESH_KNOBS.attenuation_distance.sliderMax;
    expect(physicalKnobToSlider('attenuation_distance', Number.POSITIVE_INFINITY)).toBe(top);
    expect(physicalKnobFromSlider('attenuation_distance', top)).toBe(Number.POSITIVE_INFINITY);
    expect(physicalKnobFromSlider('attenuation_distance', 0.3)).toBe(0.3);
    // A log track's position 0 is an exact 0, which a strictly positive length is not.
    expect(physicalKnobToSlider('attenuation_distance', 0)).toBe(1e-3);
    // Bounded knobs round-trip unchanged; a thickness past the track sits on its top.
    expect(physicalKnobToSlider('roughness', 0.4)).toBe(0.4);
    expect(physicalKnobFromSlider('roughness', 0.4)).toBe(0.4);
    expect(physicalKnobToSlider('thickness', 42)).toBe(PHYSICAL_MESH_KNOBS.thickness.sliderMax);
    expect(physicalKnobFromSlider('thickness', 10)).toBe(10);
  });
});

describe.each(BACKENDS)('physical mesh material (%s)', (_name, ctor) => {
  it('maps every knob one-to-one and clamps like the house knobs', () => {
    const m = new (ctor())({
      roughness: 0.4,
      metalness: 1.0,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      iridescence: 0.7,
      sheen: 0.5,
      sheenColor: '#ff0000',
      transmission: 1.0,
      ior: 1.33,
      thickness: 0.4,
      attenuationDistance: 0.3,
      attenuationColor: '#00ff00',
      dispersion: 0.5,
    });
    expect(m.roughness).toBe(0.4);
    expect(m.metalness).toBe(1.0);
    expect(m.clearcoat).toBe(1.0);
    expect(m.clearcoatRoughness).toBe(0.1);
    expect(m.iridescence).toBe(0.7);
    expect(m.sheen).toBe(0.5);
    expect(m.transmission).toBe(1.0);
    expect(m.ior).toBe(1.33);
    expect(m.thickness).toBe(0.4);
    expect(m.attenuationDistance).toBe(0.3);
    expect(m.dispersion).toBe(0.5);
    // Hex is read as sRGB → linear working space: pure red / green survive exactly.
    expect(m.sheenColor.r).toBeCloseTo(1, 6);
    expect(m.sheenColor.g).toBe(0);
    expect(m.attenuationColor.g).toBeCloseTo(1, 6);
    expect(m.attenuationColor.r).toBe(0);
    expect(m.vertexColors).toBe(true);
    expect(m.toneMapped).toBe(false);

    const clamped = new (ctor())({
      roughness: 5,
      metalness: -1,
      clearcoat: Number.NaN,
      ior: 9,
      thickness: -1,
    });
    expect(clamped.roughness).toBe(1);
    expect(clamped.metalness).toBe(0);
    // NaN routes to the DEFAULT, not a boundary — the sibling sanitizer policy.
    expect(clamped.clearcoat).toBe(PHYSICAL_MESH_DEFAULTS.clearcoat);
    expect(clamped.ior).toBe(2.333);
    expect(clamped.thickness).toBe(0);
  });

  it("an absent knob is three's default — except the two colours, which are WHITE", () => {
    const m = new (ctor())({ sheen: 1.0 });
    for (const key of PHYSICAL_MESH_KNOB_KEYS) {
      if (key === 'sheen') continue;
      const spec = PHYSICAL_MESH_KNOBS[key];
      expect(m[spec.prop], key).toBe(spec.default);
    }
    expect(m.attenuationDistance).toBe(Number.POSITIVE_INFINITY);
    // Three's own sheen default is black, under which `sheen=1` renders nothing.
    expect(m.sheenColor.getHex()).toBe(0xffffff);
    expect(new THREE.MeshPhysicalMaterial().sheenColor.getHex()).toBe(0x000000);
    // Three's attenuation default IS white already; ours agrees.
    expect(m.attenuationColor.getHex()).toBe(0xffffff);
  });

  it('glass is translucent, never writes depth, and asks to draw before the emissive data', () => {
    const glass = new (ctor())({ transmission: 1.0 });
    expect(glass.transparent).toBe(true);
    expect(glass.depthWrite).toBe(false);
    expect(glass.userData.blendingMode).toBeUndefined();
    expect(glass.userData.drawBeforeEmissive).toBe(true);
    const metal = new (ctor())({ metalness: 1.0 });
    expect(metal.userData.drawBeforeEmissive).toBeUndefined();
    expect(metal.userData.drawAfterEmissive).toBeUndefined();
  });

  it('refract_data flips glass to draw AFTER the emissive data, and is inert without transmission', () => {
    const lens = new (ctor())({ transmission: 1.0, refractData: true });
    // Same compositing state as any glass — only the ordering stamp differs.
    expect(lens.transparent).toBe(true);
    expect(lens.depthWrite).toBe(false);
    expect(lens.blending).toBe(THREE.NormalBlending);
    expect(lens.userData.drawAfterEmissive).toBe(true);
    expect(lens.userData.drawBeforeEmissive).toBeUndefined();
    // Exactly one stamp at a time; a live toggle swaps them without a rebuild.
    const versionBefore = (lens as unknown as { version: number }).version;
    (lens as unknown as { updateRefractData(v: boolean): void }).updateRefractData(false);
    expect(lens.userData.drawBeforeEmissive).toBe(true);
    expect(lens.userData.drawAfterEmissive).toBeUndefined();
    expect((lens as unknown as { version: number }).version).toBe(versionBefore);
    (lens as unknown as { updateRefractData(v: boolean): void }).updateRefractData(true);
    expect(lens.userData.drawAfterEmissive).toBe(true);
    // The flag survives a transmission zero crossing and back (re-derived from the
    // stored inputs), and is inert while the surface transmits nothing.
    const knob = lens as unknown as { updatePhysicalKnob(k: PhysicalMeshKnobKey, v: number): void };
    knob.updatePhysicalKnob('transmission', 0);
    expect(lens.userData.drawAfterEmissive).toBeUndefined();
    expect(lens.userData.drawBeforeEmissive).toBeUndefined();
    knob.updatePhysicalKnob('transmission', 0.5);
    expect(lens.userData.drawAfterEmissive).toBe(true);
    // Without transmission the flag stamps nothing at all.
    const metal = new (ctor())({ metalness: 1.0, refractData: true });
    expect(metal.userData.drawAfterEmissive).toBeUndefined();
    expect(metal.userData.drawBeforeEmissive).toBeUndefined();
  });

  it('a live knob write clamps, re-derives compositing for transmission, and rebuilds on a zero crossing', () => {
    const m = new (ctor())({}) as THREE.Material &
      PhysicalMeshHost & { updatePhysicalKnob(key: PhysicalMeshKnobKey, v: number): void };
    const versionOf = (): number => (m as unknown as { version: number }).version;

    // A plain-state knob: no rebuild, whatever it crosses.
    let v = versionOf();
    m.updatePhysicalKnob('roughness', 0.2);
    m.updatePhysicalKnob('roughness', 7);
    expect(m.roughness).toBe(1);
    expect(versionOf()).toBe(v);

    // A program-affecting knob crossing zero: exactly one rebuild each way (the
    // compositing flip adds its own on transmission; either way the count is > 0).
    v = versionOf();
    m.updatePhysicalKnob('clearcoat', 0.5);
    expect(versionOf()).toBeGreaterThan(v);
    v = versionOf();
    m.updatePhysicalKnob('clearcoat', 0.9); // still > 0: no rebuild
    expect(versionOf()).toBe(v);
    m.updatePhysicalKnob('clearcoat', 0);
    expect(versionOf()).toBeGreaterThan(v);

    // Transmission also flips the compositing decision, both ways.
    expect(m.transparent).toBe(false);
    m.updatePhysicalKnob('transmission', 0.6);
    expect(m.transmission).toBe(0.6);
    expect(m.transparent).toBe(true);
    expect(m.depthWrite).toBe(false);
    expect(m.userData.drawBeforeEmissive).toBe(true);
    m.updatePhysicalKnob('transmission', 0);
    expect(m.transparent).toBe(false);
    expect(m.userData.drawBeforeEmissive).toBeUndefined();
    expect(m.userData.blendingMode).toBe('opaque');

    // The slider top stop reaches three's "no attenuation".
    m.updatePhysicalKnob(
      'attenuation_distance',
      physicalKnobFromSlider('attenuation_distance', 100)
    );
    expect(m.attenuationDistance).toBe(Number.POSITIVE_INFINITY);
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
      attenuationColor: new THREE.Color(),
      roughness: 1,
      metalness: 0,
      clearcoat: 0,
      clearcoatRoughness: 0,
      iridescence: 0,
      sheen: 0,
      transmission: 0,
      ior: 1.5,
      thickness: 0,
      attenuationDistance: Number.POSITIVE_INFINITY,
      dispersion: 0,
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
