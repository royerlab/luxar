/**
 * Tests for MaterialManager
 *
 * This test suite verifies:
 * - Material creation and caching
 * - Camera parameter updates across all materials
 * - Material disposal and cleanup
 *
 * IMPORTANT: We test the REAL PointMaterial class (not mocked) to ensure
 * shader code generation, uniform initialization, and material behavior are correct.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  MaterialManager,
  resolveMaterialBackend,
  type PointMaterialProperties,
} from '../../../rendering/material-manager';
import { PointMaterial } from '../../../rendering/materials/point/material-glsl';
import type { RendererCapabilities } from '../../../rendering/renderer-capabilities';
import * as THREE from 'three';

// Mock only THREE.js (dependency), NOT PointMaterial (system under test)
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  // Mock ShaderMaterial to avoid WebGL dependencies
  const ShaderMaterial = vi.fn(function (this: any, params: any) {
    // Minimal EventDispatcher surface: MaterialManager subscribes to
    // the synchronous `dispose` event for automatic registry cleanup.
    const listeners: Record<string, ((e: unknown) => void)[]> = {};
    Object.assign(this, {
      uniforms: params.uniforms,
      vertexShader: params.vertexShader,
      fragmentShader: params.fragmentShader,
      vertexColors: params.vertexColors,
      transparent: params.transparent,
      depthWrite: params.depthWrite,
      toneMapped: params.toneMapped,
      blending: params.blending,
      userData: {},
      addEventListener: vi.fn((type: string, l: (e: unknown) => void) => {
        (listeners[type] ??= []).push(l);
      }),
      removeEventListener: vi.fn((type: string, l: (e: unknown) => void) => {
        if (listeners[type]) {
          listeners[type] = listeners[type].filter((x) => x !== l);
        }
      }),
      dispatchEvent: vi.fn((event: { type: string }) => {
        listeners[event.type]?.forEach((l) => l(event));
      }),
      dispose: vi.fn(function (this: any) {
        this.dispatchEvent?.({ type: 'dispose', target: this });
      }),
      clone: vi.fn(function (this: any) {
        // Simple clone for testing
        return {
          ...this,
          uniforms: { ...this.uniforms },
        };
      }),
    });
  });

  return {
    ...actual,
    ShaderMaterial: ShaderMaterial as any,
    Vector2: actual.Vector2,
    NormalBlending: 0,
    AdditiveBlending: 1,
    SubtractiveBlending: 2,
  };
});

describe('MaterialManager', () => {
  let manager: MaterialManager;

  beforeEach(() => {
    manager = new MaterialManager();
  });

  afterEach(() => {
    manager.dispose();
    vi.clearAllMocks();
  });

  // =========================================================================
  // MATERIAL CREATION
  // =========================================================================

  describe('getPointMaterial', () => {
    it('should create real PointMaterial with actual shaders', () => {
      const props: PointMaterialProperties = {
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      };

      const material = manager.getPointMaterial(props) as PointMaterial;

      // Verify it's a real PointMaterial instance (WebGL2 default path —
      // caps is unset in this test, so the dispatch picks GLSL).
      expect(material).toBeInstanceOf(PointMaterial);

      // Test REAL vertex shader content: the single per-instance
      // `aSortedIndex` attribute + the point data texture sampler
      // (per-point data lives in texels since the storage migration),
      // plus the per-vertex `aQuadCorner`.
      expect(material.vertexShader).toContain('in uint aSortedIndex');
      expect(material.vertexShader).toContain('uniform highp sampler2D uPointTex');
      expect(material.vertexShader).toContain('in vec2 aQuadCorner');
      expect(material.vertexShader).toContain('uniform float pointSizeFactor');
      expect(material.vertexShader).toContain('uniform float maxPointSize');
      expect(material.vertexShader).toContain('out mediump vec3 vColor');

      // Test REAL fragment shader content (GLSL ES 3.0 uses "out vec4 fragColor")
      expect(material.fragmentShader).toContain('uniform mediump float uOpacity');
      expect(material.fragmentShader).toContain('uniform mediump float uInvGamma');
      expect(material.fragmentShader).toContain('out vec4 fragColor');

      // [rendering.md/W2][P2] strengthened from toBeDefined() to specific
      // uniform-value assertions: a mutant that drops a uniform initialiser
      // (or returns an empty `{}` for `material.uniforms`) would previously
      // pass the toBeDefined() check on a stubbed object. Now we pin the
      // constructor-time values from material-glsl.ts:93-101.
      expect(material.uniforms.uOpacity.value).toBe(1.0); // props.opacity
      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0, 5); // 1/gamma=1
      // pointSizeFactor + maxPointSize are pre-computed from a default
      // resolution Y / FOV. They MUST be finite positives — a uniform
      // initialised to `null`/`undefined`/`NaN`/0 would surface here.
      expect(material.uniforms.pointSizeFactor.value).toBeGreaterThan(0);
      expect(Number.isFinite(material.uniforms.pointSizeFactor.value)).toBe(true);
      expect(material.uniforms.maxPointSize.value).toBeGreaterThan(0);
      expect(Number.isFinite(material.uniforms.maxPointSize.value)).toBe(true);
    });

    it('should respect custom opacity', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.uniforms.uOpacity.value).toBe(0.5);
    });

    it('should respect custom gamma', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 2.2,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.userData.gamma).toBe(2.2); // gamma stored in userData
      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
    });

    it('should set correct blending mode', () => {
      const additive = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const normal = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // additive uses classic THREE.AdditiveBlending (SrcAlpha, One)
      expect(additive.blending).toBe(THREE.AdditiveBlending);
      expect(normal.blending).toBe(THREE.NormalBlending);
    });

    it('should handle radius scale parameter', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
        radiusScale: 1.0 / 255.0, // For uint8 radii
      });

      expect(material.uniforms.radiusScale.value).toBeCloseTo(1.0 / 255.0, 5);
    });
  });

  // =========================================================================
  // PER-NODE IDENTITY (points are never cached)
  // =========================================================================

  describe('Per-node point materials', () => {
    it('returns a DISTINCT instance per call, even for identical props (per-node, LRU bypassed)', () => {
      // Point data lives in a per-node texture (uPointTex): two nodes can
      // never share a point material, or one node's commit would rebind
      // its texture onto the other's mesh. Mirrors getGSplatMaterial.
      const props: PointMaterialProperties = {
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      };

      const material1 = manager.getPointMaterial(props);
      const material2 = manager.getPointMaterial(props);

      expect(material1).not.toBe(material2);

      // The point cache map stays permanently empty.
      const stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(0);
      // Both per-node materials are registered for camera broadcast.
      expect(stats.totalRegistered).toBe(2);
    });

    it('per-node materials receive the current camera params immediately at creation', () => {
      const fov = Math.PI / 4;
      manager.updateCameraParams(fov, new THREE.Vector2(2560, 1440));
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      const expectedPointSizeFactor = (2.0 * 1440) / Math.tan(fov / 2);
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
    });

    it('should create different materials for different opacity', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 0.5, // Different!
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material1).not.toBe(material2);
    });

    it('should create different materials for different gamma', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 2.2, // Different!
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material1).not.toBe(material2);
    });

    it('should create different materials for different blending modes', () => {
      const additive = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const normal = manager.getPointMaterial({
        blendingMode: 'normal', // Different!
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(additive).not.toBe(normal);
    });

    it('should create different materials for different radius scales', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
        radiusScale: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
        radiusScale: 1.0 / 255.0, // Different!
      });

      expect(material1).not.toBe(material2);
    });
  });

  // =========================================================================
  // GLOBAL UPDATES
  // =========================================================================

  describe('Global Updates', () => {
    it('should update camera params for all materials', () => {
      // Create multiple materials
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
        intensity: 1.0,
        offset: 0.0,
      });

      // Update camera params globally
      const newFov = Math.PI / 3; // 60 degrees
      const newResolution = new THREE.Vector2(1920, 1080);

      manager.updateCameraParams(newFov, newResolution);

      // Both materials should be updated with pre-computed pointSizeFactor and maxPointSize
      const expectedPointSizeFactor = (2.0 * 1080) / Math.tan(newFov / 2);
      expect(material1.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material1.uniforms.maxPointSize.value).toBe(1080 * 0.5);

      expect(material2.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material2.uniforms.maxPointSize.value).toBe(1080 * 0.5);
    });

    it('should store current camera params for new materials', () => {
      // Update params before creating material
      const fov = Math.PI / 4; // 45 degrees
      const resolution = new THREE.Vector2(2560, 1440);

      manager.updateCameraParams(fov, resolution);

      // Create new material
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // Should have current params (with pre-computed pointSizeFactor and maxPointSize)
      const expectedPointSizeFactor = (2.0 * 1440) / Math.tan(fov / 2);
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material.uniforms.maxPointSize.value).toBe(1440 * 0.5);
    });
  });

  // =========================================================================
  // DISPOSAL
  // =========================================================================

  describe('dispose', () => {
    it('should dispose all cached materials', () => {
      // Create several materials
      const mat1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const mat2 = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
        intensity: 1.0,
        offset: 0.0,
      });

      // Dispose manager
      manager.dispose();

      // Materials should be disposed
      expect(mat1.dispose).toHaveBeenCalled();
      expect(mat2.dispose).toHaveBeenCalled();

      // Cache should be cleared
      const stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(0);
      expect(stats.ownedMaterials).toBe(0);
      expect(stats.totalRegistered).toBe(0);
    });

    it('should clear the registry on dispose (per-node point materials leave no cache)', () => {
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // Per-node: never cached, but registered for camera broadcast.
      expect(manager.getCacheStats().pointMaterials).toBe(0);
      expect(manager.getCacheStats().totalRegistered).toBe(1);

      manager.dispose();

      expect(manager.getCacheStats().pointMaterials).toBe(0);
      expect(manager.getCacheStats().totalRegistered).toBe(0);
    });

    it('should track and dispose registered non-cached material clones', () => {
      const perNode = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      const clone = new PointMaterial({ opacity: 0.5 });

      manager.register(clone);

      let stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(0); // per-node — no cache entry
      expect(stats.cachedMaterials).toBe(0);
      expect(stats.ownedMaterials).toBe(1);
      expect(stats.totalRegistered).toBe(2);

      manager.dispose();

      expect(perNode.dispose).toHaveBeenCalled();
      expect(clone.dispose).toHaveBeenCalled();
      stats = manager.getCacheStats();
      expect(stats.ownedMaterials).toBe(0);
      expect(stats.totalRegistered).toBe(0);
    });

    it('should unregister non-cached material clones without touching other registrations', () => {
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      const clone = new PointMaterial({ opacity: 0.5 });
      manager.register(clone);

      manager.unregister(clone);

      const stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(0); // per-node — no cache entry
      expect(stats.ownedMaterials).toBe(0);
      expect(stats.totalRegistered).toBe(1); // the per-node material stays registered
    });

    it('should drop a clone from the registry when the clone is disposed', () => {
      // Lock in the subscribeToDispose wiring — disposing a clone must
      // remove it from registeredMaterials, otherwise per-node clones
      // accumulate forever in the global update loop.
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      const clone = new PointMaterial({ opacity: 0.5 });
      manager.register(clone);
      expect(manager.getCacheStats().totalRegistered).toBe(2);

      clone.dispose();
      expect(manager.getCacheStats().totalRegistered).toBe(1);
      expect(manager.getCacheStats().ownedMaterials).toBe(0);
    });

    it('should not stack dispose listeners when registering the same material twice', () => {
      // Idempotency guard: re-registering a material that already has a
      // dispose listener must not stack another listener on the
      // EventDispatcher. Tested behaviorally via the mock's
      // `addEventListener` spy — a second `register()` for the same
      // material must NOT add another listener for the 'dispose' event.
      const mat = new PointMaterial({ opacity: 1.0 });
      // PointMaterial extends our mocked ShaderMaterial which exposes
      // addEventListener as a vi.fn. Pluck it to count calls.
      const addEventListener = (mat as unknown as { addEventListener: ReturnType<typeof vi.fn> })
        .addEventListener;
      addEventListener.mockClear();

      manager.register(mat);
      manager.register(mat); // second registration — should be a no-op

      const disposeCallCount = addEventListener.mock.calls.filter(
        (call) => (call as unknown[])[0] === 'dispose'
      ).length;
      expect(disposeCallCount).toBe(1);
      expect(manager.getCacheStats().totalRegistered).toBe(1);
    });
  });

  // =========================================================================
  // CACHE STATISTICS
  // =========================================================================

  describe('getCacheStats', () => {
    it('should return accurate statistics (per-node points registered, never cached)', () => {
      const stats1 = manager.getCacheStats();
      expect(stats1.pointMaterials).toBe(0);
      expect(stats1.totalRegistered).toBe(0);
      expect(stats1.keys).toEqual([]);

      // Create materials
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
        intensity: 1.0,
        offset: 0.0,
      });

      const stats2 = manager.getCacheStats();
      expect(stats2.pointMaterials).toBe(0); // per-node — no cache entries
      expect(stats2.totalRegistered).toBe(2); // but both are registered
      expect(stats2.keys.length).toBe(0); // no keys — nothing cached
    });

    it('should report no cache keys — all three material kinds are per-node (nothing cached)', () => {
      // The line-material LRU (the last cached kind) died with the lines
      // texture-storage migration: every getXMaterial call creates a
      // fresh per-node material, so the cache maps — and therefore
      // `keys` — stay permanently empty.
      manager.getLineMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      manager.getGSplatMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const stats = manager.getCacheStats();
      expect(stats.keys).toEqual([]);
      expect(stats.pointMaterials).toBe(0);
      expect(stats.lineMaterials).toBe(0);
      expect(stats.gsplatMaterials).toBe(0);
      expect(stats.cachedMaterials).toBe(0);
      expect(stats.evictions).toBe(0);
      // All three per-node materials are still registered for camera broadcast.
      expect(stats.totalRegistered).toBe(3);
    });
  });

  // =========================================================================
  // BLENDING MODE CONVERSION
  // =========================================================================

  describe('Blending Mode Conversion', () => {
    it('should convert normal blending mode', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.blending).toBe(THREE.NormalBlending);
    });

    it('should convert additive blending mode to THREE.AdditiveBlending', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // additive uses classic THREE.AdditiveBlending (SrcAlpha, One)
      expect(material.blending).toBe(THREE.AdditiveBlending);
    });

    it('should set depth write for opaque normal blending', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.99, // Opaque threshold
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.depthWrite).toBe(true);
    });

    it('should disable depth write for transparent materials', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5, // Transparent
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.depthWrite).toBe(false);
    });

    it('should disable depth write for additive blending', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0, // Even if opaque
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.depthWrite).toBe(false);
    });

    it('should configure max blending mode with custom blending', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'max',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.blending).toBe(THREE.CustomBlending);
      expect(material.depthWrite).toBe(false); // Max blending disables depth write
    });

    it('should configure opaque blending mode with depth write', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'opaque',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.blending).toBe(THREE.NormalBlending);
      expect(material.depthWrite).toBe(true); // Opaque writes to depth
      expect(material.transparent).toBe(false); // Not transparent
    });

    it('getGSplatMaterial threads absorption into the material (volumetric kappa seam)', () => {
      const material = manager.getGSplatMaterial({
        blendingMode: 'volumetric',
        opacity: 1.0,
        absorption: 2.5,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      // The factory seam must not drop kappa (props.absorption ?? 1.0).
      expect(material.uniforms.uAbsorption.value).toBe(2.5);
      expect(material.userData.blendingMode).toBe('volumetric');
      // Omitted -> identity default.
      const dflt = manager.getGSplatMaterial({
        blendingMode: 'volumetric',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });
      expect(dflt.uniforms.uAbsorption.value).toBe(1.0);
    });

    it('should configure luminous blending mode with AdditiveBlending and depthTest', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'luminous',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // Luminous uses same blending as additive (AdditiveBlending = SrcAlpha, One)
      // Only difference is depthTest: luminous=true (respects depth), additive=false (ignores depth)
      expect(material.blending).toBe(THREE.AdditiveBlending);
      expect(material.depthWrite).toBe(false); // Luminous doesn't write to depth
      expect(material.transparent).toBe(true); // Is transparent (for render order)
      expect(material.userData.depthTest).toBe(true); // Respects depth occlusion
    });

    it('should have depthTest true for luminous mode (respects depth occlusion)', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'luminous',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.userData.depthTest).toBe(true);
    });

    it('should have depthTest false for additive mode (ignores depth entirely)', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // 'additive' ignores depth entirely (renders on top of everything)
      expect(material.userData.depthTest).toBe(false);
    });

    it('should have depthTest true for opaque mode', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'opaque',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.userData.depthTest).toBe(true);
    });
  });

  // =========================================================================
  // SHADER VERIFICATION
  // =========================================================================

  describe('Shader Content Verification', () => {
    // These tests probe GLSL3 shader strings, which only the
    // `PointMaterial` (ShaderMaterial-backed) path exposes. The TSL
    // wrapper compiles its graph through Three.js's NodeBuilder and
    // doesn't surface a `vertexShader` / `fragmentShader` string — so
    // each test casts to the GLSL class, relying on the dispatch
    // default (caps unset → GLSL) inside the manager.
    it('should generate shaders with optimized world-space sizing', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      }) as PointMaterial;

      // Verify world-space sizing from VIEW-SPACE DEPTH (matches the
      // line + gsplat shaders) and the pre-computed pointSizeFactor.
      expect(material.vertexShader).toContain('normalizedRadius * pointSizeFactor * invDistance');
      expect(material.vertexShader).toContain('1.0 / max(-mvPosition.z, 1e-20)');
    });

    it('should generate shaders with the sharpness -> beta mapping (no size compensation)', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      }) as PointMaterial;

      // Sharpness maps to the super-Gaussian exponent; no size compensation.
      expect(material.vertexShader).not.toContain('sharpnessCompensation');
      expect(material.vertexShader).toContain('vBeta = exp2(6.0 * s - 2.0)');
    });

    it('should generate shaders with gamma correction', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 2.2,
        intensity: 1.0,
        offset: 0.0,
      }) as PointMaterial;

      // Verify GOG model in fragment shader
      expect(material.fragmentShader).toContain('vColor * uIntensity + uOffset');
      expect(material.fragmentShader).toContain('pow(adjusted, vec3(uInvGamma))');
      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
    });
  });

  // =========================================================================
  // MATERIAL UPDATES
  // =========================================================================

  describe('Material Updates', () => {
    it('should update newly created materials with current params', () => {
      // Set camera params first
      const fov = Math.PI / 2;
      const resolution = new THREE.Vector2(3840, 2160);
      manager.updateCameraParams(fov, resolution);

      // Create material after update
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      // Should have current params (with pre-computed pointSizeFactor and maxPointSize)
      const expectedPointSizeFactor = (2.0 * 2160) / Math.tan(fov / 2);
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material.uniforms.maxPointSize.value).toBe(2160 * 0.5);
    });

    it('should update existing materials when params change', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      const initialPointSizeFactor = material.uniforms.pointSizeFactor.value;

      // Update params
      const newFov = Math.PI / 4;
      manager.updateCameraParams(newFov, new THREE.Vector2(1280, 720));

      // Material should be updated with pre-computed pointSizeFactor
      const expectedPointSizeFactor = (2.0 * 720) / Math.tan(newFov / 2);
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material.uniforms.pointSizeFactor.value).not.toBe(initialPointSizeFactor);
    });
  });

  // =========================================================================
  // EDGE CASES
  // =========================================================================

  describe('Edge Cases', () => {
    it('should handle zero opacity', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.uniforms.uOpacity.value).toBe(0.0);
      expect(material.depthWrite).toBe(false); // Transparent
    });

    it('should handle maximum gamma', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 10.0, // Extreme value
        intensity: 1.0,
        offset: 0.0,
      });

      expect(material.userData.gamma).toBe(10.0); // gamma stored in userData
      expect(material.uniforms.uInvGamma.value).toBeCloseTo(0.1, 5);
    });

    it('should handle very small radius scale', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
        radiusScale: 0.00001,
      });

      expect(material.uniforms.radiusScale.value).toBeCloseTo(0.00001, 5);
    });
  });
});

describe('resolveMaterialBackend', () => {
  // Casts are stub-typed: only `api` is read by the helper; the rest of
  // RendererCapabilities is irrelevant to this dispatch decision.
  it('returns glsl when caps is null (pre-renderer default)', () => {
    expect(resolveMaterialBackend(null)).toBe('glsl');
  });

  it('returns glsl when caps reports the WebGL2 surface', () => {
    expect(resolveMaterialBackend({ apiSurface: 'webgl2' } as RendererCapabilities)).toBe('glsl');
  });

  it('returns tsl when caps reports the WebGPU surface', () => {
    expect(resolveMaterialBackend({ apiSurface: 'webgpu' } as RendererCapabilities)).toBe('tsl');
  });
});
