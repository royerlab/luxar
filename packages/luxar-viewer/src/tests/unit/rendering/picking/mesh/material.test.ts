/**
 * The mesh pick wrapper PAIR (spec §6.5) — run against BOTH backends.
 *
 * Covers the three things it does that no sibling pick wrapper does — mode-derived
 * uniforms, `side` synced from the visual material, and the coverage inputs — plus
 * the clone fidelity every pick wrapper needs because `Material.clone()` would call
 * the constructor with no config.
 *
 * Table-driven over `[glsl, tsl]` following `materials/mesh/blending-mode.test.ts`,
 * because the two wrappers reimplement the same contract independently: each has its
 * own `setPickMode`, `setPickSide`, `clone`, and coverage setters. Testing only the
 * GLSL one would let the TSL twin drift — and a divergence there is invisible on a
 * WebGL-only suite, which is the failure mode this whole material pair exists to
 * guard against. (The TSL wrapper builds a real node graph in its constructor; that
 * works headless, as `pick-opacity-tail.test.ts` already relies on.)
 *
 * @module tests/unit/rendering/picking/mesh/material
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { MeshPickingMaterial } from '../../../../../rendering/picking/mesh/material';
import { MeshPickingTSLMaterial } from '../../../../../rendering/picking/mesh/material-tsl';
import { MESH_DEFAULTS } from '../../../../../rendering/materials/mesh/appearance';
import type { CameraAwareMaterial } from '../../../../../rendering/materials/_shared/camera-aware-material';
import { isCameraAwareMaterial } from '../../../../../rendering/materials/_shared/camera-aware-material';
import type { MeshPickingMaterialConfig } from '../../../../../rendering/picking/mesh/material';

/** The two wrappers, behind one constructor signature. */
type PickWrapper = MeshPickingMaterial | MeshPickingTSLMaterial;
const BACKENDS: ReadonlyArray<[string, (c: MeshPickingMaterialConfig) => PickWrapper]> = [
  ['glsl', (c) => new MeshPickingMaterial(c)],
  ['tsl', (c) => new MeshPickingTSLMaterial(c)],
];

describe.each(BACKENDS)('MeshPickingMaterial [%s] — construction', (_name, make) => {
  const build = (overrides = {}) => make({ nodeId: 7, ...overrides });

  it('starts in the `opaque` state, matching the mesh default blending mode', () => {
    // Not cosmetic: the first pick can precede any layers-panel interaction and any
    // pick render, so a material that started in the commutative state would answer
    // the first hover with brightness-as-depth on a depth-ordered surface.
    const m = build();
    expect(m.uniforms.uAlphaCutout.value).toBe(1);
    expect(m.uniforms.uSurfaceDepth.value).toBe(1);
    expect(m.uniforms.uAlphaCutoff.value).toBe(MESH_DEFAULTS.alphaCutoff);
  });

  it('starts on FrontSide, never DoubleSide', () => {
    // The safe start: FrontSide can never rasterize a face the visual material culls.
    // The three sibling pick wrappers pin DoubleSide; this one must not.
    expect(build().side).toBe(THREE.FrontSide);
  });

  it('keeps the opaque-ID-buffer contract', () => {
    const m = build();
    expect(m.transparent).toBe(false);
    expect(m.blending).toBe(THREE.NoBlending);
    expect(m.toneMapped).toBe(false);
    expect(m.depthTest).toBe(true);
    expect(m.depthWrite).toBe(true);
  });

  it('is camera-aware — for the near fade, and only for the near fade', () => {
    // The guard is what routes a pick material into the camera broadcast, and the
    // mesh pick pass has to reproduce the visual near fade or a surface fading out
    // of view stays fully pickable (#1431). It consumes only `nearCull`: there is
    // still no screen-space footprint to size from the resolution, the fade's ortho
    // test reads the camera being drawn with, and the two ignored arguments are the
    // honest shape of that.
    const m: unknown = build();
    expect(isCameraAwareMaterial(m as CameraAwareMaterial)).toBe(true);
    expect((m as PickWrapper).uniforms.uIsOrtho).toBeUndefined();
    expect((m as PickWrapper).uniforms.uNearCull.value).toBe(0.1);
    expect((m as PickWrapper).uniforms.uResolution).toBeUndefined();
  });

  it('writes the near-cull uniform from updateCameraParams', () => {
    const m = build();
    m.updateCameraParams(new THREE.Vector2(800, 600), true, 0.42);
    expect(m.uniforms.uIsOrtho).toBeUndefined();
    expect(m.uniforms.uNearCull.value).toBe(0.42);
    // An omitted nearCull must LEAVE the last one standing rather than resetting to
    // the default — the sibling materials all treat it as optional this way, and a
    // reset would fade a mid-session ortho toggle against the wrong plane.
    m.updateCameraParams(new THREE.Vector2(800, 600), false);
    expect(m.uniforms.uNearCull.value).toBe(0.42);
  });

  it('clamps the coverage inputs, which arrive from authored metadata', () => {
    // Both are reachable through `add_mesh(**attrs)`, so they can be any number at
    // all. An unclamped cutoff of 1e9 discards every fragment: the mesh becomes
    // unpickable with no diagnostic.
    expect(build({ alphaCutoff: 1e9 }).uniforms.uAlphaCutoff.value).toBe(1);
    expect(build({ alphaCutoff: -3 }).uniforms.uAlphaCutoff.value).toBe(0);
    expect(build({ opacity: 42 }).uniforms.uOpacity.value).toBe(1);
    expect(build({ alphaCutoff: Number.NaN }).uniforms.uAlphaCutoff.value).toBe(
      MESH_DEFAULTS.alphaCutoff
    );
  });
});

describe.each(BACKENDS)('MeshPickingMaterial [%s] — setPickMode', (_name, make) => {
  const build = (overrides = {}) => make({ nodeId: 7, ...overrides });

  it('drives both uniforms from the mode, together', () => {
    const m = build();
    m.setPickMode('additive');
    expect(m.uniforms.uAlphaCutout.value).toBe(0);
    expect(m.uniforms.uSurfaceDepth.value).toBe(0);

    m.setPickMode('normal');
    expect(m.uniforms.uAlphaCutout.value).toBe(0);
    expect(m.uniforms.uSurfaceDepth.value).toBe(1);

    m.setPickMode('opaque');
    expect(m.uniforms.uAlphaCutout.value).toBe(1);
    expect(m.uniforms.uSurfaceDepth.value).toBe(1);
  });

  it('does not recompile the program — both selectors are runtime uniforms', () => {
    // The point of making them uniforms rather than defines (§6.5): a layers-panel
    // mode switch must be a uniform write. `needsUpdate` is a write-only setter in
    // three (it just bumps `version`), so the observable is the version.
    const m = build();
    const before = m.version;
    m.setPickMode('additive');
    m.setPickMode('max');
    m.setPickMode('opaque');
    expect(m.version).toBe(before);
  });
});

describe.each(BACKENDS)('MeshPickingMaterial [%s] — setPickSide', (_name, make) => {
  const build = (overrides = {}) => make({ nodeId: 7, ...overrides });

  it('adopts the requested side and invalidates the program', () => {
    const m = build();
    const before = m.version;
    m.setPickSide(THREE.DoubleSide);
    expect(m.side).toBe(THREE.DoubleSide);
    expect(m.version).toBeGreaterThan(before);
  });

  it('is a no-op when the side already matches', () => {
    // Guarded on change because this is called on EVERY pick render: an unguarded
    // write would invalidate the program (GLSL) or rebuild the pipeline (WebGPU)
    // once per pick.
    const m = build();
    m.setPickSide(THREE.DoubleSide);
    const settled = m.version;
    m.setPickSide(THREE.DoubleSide);
    m.setPickSide(THREE.DoubleSide);
    expect(m.version).toBe(settled);
  });
});

describe.each(BACKENDS)('MeshPickingMaterial [%s] — clone', (_name, make) => {
  const build = (overrides = {}) => make({ nodeId: 7, ...overrides });

  it('carries the nodeId, the coverage inputs, the mode state and the side across', () => {
    // `Material.clone()` would call the constructor with no config, leaving nodeId
    // undefined — every pick wrapper overrides it for that reason. The mesh one also
    // has to carry `side`, since a clone taken on an undecidable frame would
    // otherwise revert to FrontSide and drop half the pickable surface.
    const m = build({ opacity: 0.8, alphaCutoff: 0.25 });
    m.setPickMode('additive');
    m.setPickSide(THREE.DoubleSide);
    m.updateCameraParams(new THREE.Vector2(800, 600), true, 0.42);

    const c = m.clone();
    expect(c).not.toBe(m);
    expect(c.uniforms.uNodeId.value).toBe(7);
    expect(c.uniforms.uOpacity.value).toBeCloseTo(0.8);
    expect(c.uniforms.uAlphaCutoff.value).toBeCloseTo(0.25);
    expect(c.uniforms.uAlphaCutout.value).toBe(0);
    expect(c.uniforms.uSurfaceDepth.value).toBe(0);
    expect(c.side).toBe(THREE.DoubleSide);
    // Camera state too: a clone that reverted to the 0.1 default would fade its
    // pick coverage against the wrong near plane.
    expect(c.uniforms.uNearCull.value).toBeCloseTo(0.42);
  });

  it('gives the clone its OWN uniforms object', () => {
    const m = build();
    const c = m.clone();
    c.setPickMode('additive');
    expect(m.uniforms.uAlphaCutout.value).toBe(1);
  });
});
