/**
 * Tests for viewer-state-capture: captureViewerState() function
 */

import { describe, it, expect, vi } from 'vitest';
import { captureViewerState } from '../../../config/zarr-bridge/viewer-state-capture';

// Mock ThemeManager
vi.mock('../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: () => ({
      getCurrentTheme: () => ({ id: 'dark', name: 'Dark Theme' }),
    }),
  },
}));

// Create mock objects. IMPORTANT: settings.{fov,near,far} are the
// source-of-truth (per viewer-state-capture.ts:55-58); camera.{fov,near,far}
// are intentionally set to DIFFERENT values so a mutant that swapped the
// source from settings → camera would be caught by the existing assertions.
function createMockSceneManager() {
  return {
    camera: {
      position: { x: 1, y: 2, z: 3 },
      up: { x: 0, y: 1, z: 0 },
      fov: 99, // distinct from settings.fov (47) — pinpoints source-of-truth
      near: 99, // distinct from settings.near (0.1)
      far: 99, // distinct from settings.far (1000)
    },
    scene: {
      background: {
        isColor: true,
        getHexString: () => '000000',
      },
    },
    controls: {
      getFocusTarget: () => ({ x: 0, y: 0, z: 0 }),
    },
  } as any;
}

function createMockRenderingControls() {
  return {
    settings: {
      fov: 47,
      fovPreset: '50mm Normal' as const,
      near: 0.1,
      far: 1000,
      bloomEnabled: true,
      bloomStrength: 0.5,
      bloomRadius: 1.0,
      bloomThreshold: 0.01,
      bloomLevels: 8,
      exposure: 1.0,
      globalOffset: 0.0,
      globalGamma: 1.0,
      toneMapping: 'ACES' as const,
      controlType: 'orbit' as const,
      autoRotate: false,
      autoRotateSpeed: 0.5,
      autoRotateAxis: 'horizontal' as const,
      cinematicMode: false,
      vignetteEnabled: false,
      vignetteDarkness: 0.5,
      vignetteOffset: 0.5,
      detectorNoiseEnabled: false,
      detectorNoiseReadoutSigma: 0.005,
      detectorNoisePhotonGain: 0.003,
      detectorNoiseFpnSigma: 0.001,
      fxaaEnabled: true,
      msaaEnabled: false,
      msaaSamples: 4,
      ssaaEnabled: false,
      ssaaMultiplier: 2,
      chromaticLensDistortionEnabled: false,
      chromaticLensDistortionX: 0,
      chromaticLensDistortionY: 0,
      chromaticLensDispersion: 0,
      chromaticLensPrincipalPointX: 0,
      chromaticLensPrincipalPointY: 0,
      chromaticLensFocalLengthX: 1,
      chromaticLensFocalLengthY: 1,
      chromaticLensSkew: 0,
      flyMovementSpeed: 1,
      flyRotationSpeed: 1,
      flyInertialMode: false,
      flyDamping: 0.9,
      flyRotationDamping: 0.9,
      dynamicClippingEnabled: true,
      adaptiveDPREnabled: true,
      allowHighDPR: true,
    },
  } as any;
}

function createMockSceneDimsManager(hasDims = true) {
  if (!hasDims) {
    return { getDims: () => null } as any;
  }
  return {
    getDims: () => ({
      ndim: 4,
      currentStep: [5, 0, 0, 0],
      displayed: [1, 2, 3],
      metadata: [],
    }),
  } as any;
}

describe('captureViewerState', () => {
  it('should capture camera state', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.camera).toBeDefined();
    expect(state.camera!.position).toEqual([1, 2, 3]);
    expect(state.camera!.target).toEqual([0, 0, 0]);
    expect(state.camera!.up).toEqual([0, 1, 0]);
    expect(state.camera!.fov).toBe(47);
    expect(state.camera!.fov_preset).toBe('50mm Normal');
    // near/far are omitted here because the mock has dynamic clipping ON — see
    // the dedicated cases below.
    expect(state.camera!.near).toBeUndefined();
    expect(state.camera!.far).toBeUndefined();
  });

  // `settings.near` / `settings.far` are live camera readouts whenever dynamic
  // clipping owns them (stamped in by ClippingDisplay's RAF loop and
  // syncCurrentState). Capturing those authored a zoomed-in pose's planes into
  // an exported viewer_config as if chosen deliberately — and because the same
  // capture emits `dynamic_clipping_enabled: true`, loading that export tripped
  // the "dynamic clipping will override these" warning, whose advice would pin
  // the pathological pair. Same rule as `stripDynamicClippingPlanes` applies to
  // localStorage.
  it('omits near/far while dynamic clipping owns them', () => {
    const controls = createMockRenderingControls();
    controls.settings.dynamicClippingEnabled = true;
    controls.settings.near = 1.05e-4; // a transient deep-zoom readout
    controls.settings.far = 61;

    const state = captureViewerState(
      createMockSceneManager() as never,
      controls as never,
      createMockSceneDimsManager() as never
    );

    expect(state.camera!.near).toBeUndefined();
    expect(state.camera!.far).toBeUndefined();
    // The rest of the camera block is unaffected — this is a targeted omission.
    expect(state.camera!.position).toEqual([1, 2, 3]);
    expect(state.camera!.fov).toBe(47);
  });

  it('captures near/far when the user owns them (dynamic clipping off)', () => {
    const controls = createMockRenderingControls();
    controls.settings.dynamicClippingEnabled = false;
    controls.settings.near = 0.25;
    controls.settings.far = 400;

    const state = captureViewerState(
      createMockSceneManager() as never,
      controls as never,
      createMockSceneDimsManager() as never
    );

    expect(state.camera!.near).toBe(0.25);
    expect(state.camera!.far).toBe(400);
  });

  it('should capture background color', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.background_color).toBe('#000000');
  });

  it('should capture rendering settings in snake_case', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.bloom_enabled).toBe(true);
    expect(state.bloom_strength).toBe(0.5);
    expect(state.exposure).toBe(1.0);
    expect(state.tone_mapping).toBe('ACES');
    expect(state.control_type).toBe('orbit');
    // Round-trips through RENDERING_SETTINGS_MAP. Without a map entry the
    // key is not merely absent — captureViewerState logs "dropping unknown
    // RenderingSettings key" and the authored axis is silently lost.
    expect(state.auto_rotate_axis).toBe('horizontal');
    expect(state.fxaa_enabled).toBe(true);
    expect(state.dynamic_clipping_enabled).toBe(true);
    expect(state.adaptive_dpr_enabled).toBe(true);
    expect(state.allow_high_dpr).toBe(true);
  });

  it('should capture theme', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.theme).toBe('dark');
  });

  it('should capture dimensions', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.dimensions).toBeDefined();
    expect(state.dimensions!.current_step).toEqual([5, 0, 0, 0]);
  });

  it('should handle missing dimensions gracefully', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager(false)
    );

    expect(state.dimensions).toBeUndefined();
    // The branch at viewer-state-capture.ts:81-86 gates the `animation` block
    // on truthy `dims`. With no dims, animation must also be undefined — a
    // regression that emitted `animation: []` would slip past a dimensions-only
    // check.
    expect(state.animation).toBeUndefined();
  });

  it('should handle missing background gracefully', () => {
    const sm = createMockSceneManager();
    sm.scene.background = null;

    const state = captureViewerState(
      sm,
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.background_color).toBeUndefined();
  });

  it('should produce JSON-serializable output', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    // Should not throw
    const json = JSON.stringify(state);
    const parsed = JSON.parse(json);
    expect(parsed.camera.position).toEqual([1, 2, 3]);
    expect(parsed.bloom_enabled).toBe(true);
  });

  // [G11][P5] Audit: source narrows on `'isColor' in bg && bg.isColor`
  // (viewer-state-capture.ts:58). If `scene.background` is a Texture
  // (no `isColor`), `result.background_color` must stay undefined.
  // Pre-audit only the `null` path was covered.
  it('leaves background_color undefined when scene.background is a non-Color (e.g. Texture)', () => {
    const sm = createMockSceneManager();
    // Mimic a Three.Texture-shaped object: no `isColor` flag at all.
    sm.scene.background = {
      isTexture: true,
      // Provide a getHexString() to prove the narrow is what gates the
      // call — without `isColor` the code must NOT invoke getHexString.
      getHexString: () => 'should-not-be-called',
    } as any;

    const state = captureViewerState(
      sm,
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.background_color).toBeUndefined();
  });

  it('leaves background_color undefined when scene.background has isColor=false', () => {
    // [P5] boundary: the predicate requires `bg.isColor === true`; an
    // object with the property set to `false` must also be rejected.
    const sm = createMockSceneManager();
    sm.scene.background = {
      isColor: false,
      getHexString: () => 'should-not-be-called',
    } as any;

    const state = captureViewerState(
      sm,
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.background_color).toBeUndefined();
  });
});

// [G10][P5,P8] Audit: the per-dimension `animation` block
// (viewer-state-capture.ts:89-111) was completely uncovered.
// We deliberately put these tests in their own `describe` so the
// mock fixtures don't pollute the smoke-path tests above.
describe('captureViewerState — animation block', () => {
  function createMockAnimationManager(states: Array<Record<string, unknown> | null>) {
    return {
      getState: (i: number) => states[i] ?? null,
    } as any;
  }

  function buildState(perDimStates: Array<Record<string, unknown> | null>, hasDims = true) {
    return captureViewerState(
      // Reuse the local helper bodies above
      {
        camera: {
          position: { x: 1, y: 2, z: 3 },
          up: { x: 0, y: 1, z: 0 },
          fov: 47,
          near: 0.1,
          far: 1000,
        },
        scene: { background: null },
        controls: { getFocusTarget: () => ({ x: 0, y: 0, z: 0 }) },
      } as any,
      { settings: {} } as any,
      hasDims
        ? ({
            getDims: () => ({
              ndim: perDimStates.length,
              currentStep: new Array(perDimStates.length).fill(0),
              displayed: [1, 2, 3],
              metadata: [],
            }),
          } as any)
        : ({ getDims: () => null } as any),
      createMockAnimationManager(perDimStates)
    );
  }

  it('captures per-dimension animation state when animationManager is provided', () => {
    const state = buildState([
      null,
      { isPlaying: true, targetFPS: 30, loopMode: 'loop', direction: 'forward' },
      null,
    ]);

    expect(state.animation).toBeDefined();
    expect(state.animation).toHaveLength(3);
    // Untouched dims must serialize as empty objects (per source line 104).
    expect(state.animation![0]).toEqual({});
    expect(state.animation![2]).toEqual({});
    expect(state.animation![1]).toEqual({
      playing: true,
      target_fps: 30,
      loop: 'loop',
      direction: 'forward',
    });
  });

  it('includes step_size only when the per-dimension override is set', () => {
    const state = buildState([
      { isPlaying: false, targetFPS: 10, loopMode: 'loop', direction: 'forward', stepSize: 0.25 },
      { isPlaying: false, targetFPS: 10, loopMode: 'loop', direction: 'forward' },
    ]);
    expect(state.animation![0]).toMatchObject({ step_size: 0.25 });
    // Auto (absent/null) must stay ABSENT — the exported shape for untouched
    // overrides is unchanged.
    expect('step_size' in state.animation![1]).toBe(false);
  });

  it('omits animation entirely when NO dimension has playing/queued state (hasAnyState=false)', () => {
    // [P5] symmetry: source uses an explicit `hasAnyState` flag — if NO
    // dimension reports state, the block must NOT emit an `animation: []`
    // or `[{}, {}, ...]` placeholder. Pre-audit this branch was untested.
    const state = buildState([null, null, null]);
    expect(state.animation).toBeUndefined();
  });

  it('omits animation when animationManager is provided but dims are missing', () => {
    // [G10] guard: even with an animationManager passed in, if dims === null,
    // the source's `if (animationManager && dims)` short-circuits and
    // animation must remain undefined.
    const state = buildState(
      [{ isPlaying: true, targetFPS: 30, loopMode: 'loop', direction: 'forward' }],
      false
    );
    expect(state.animation).toBeUndefined();
  });
});

// [W4][P3] / [G12][P5] Audit: the `vi.mock('../../themes/theme-manager')`
// at the top of this file short-circuits `ThemeManager.getInstance()` to
// a known fixture, which means the `try/catch` at lines 73-77 of source
// (the "ThemeManager not initialized" path) was completely untested.
//
// We can exercise the catch by passing an explicit `themeManager` whose
// `getCurrentTheme()` throws — that flows through the same try/catch
// without touching the module-level singleton (no `vi.doMock` gymnastics
// needed, no production-source edits).
describe('captureViewerState — ThemeManager catch branch', () => {
  it('omits result.theme when the supplied themeManager.getCurrentTheme() throws', () => {
    const throwingTM = {
      getCurrentTheme: () => {
        throw new Error('ThemeManager not initialized');
      },
    } as any;

    const state = captureViewerState(
      {
        camera: {
          position: { x: 0, y: 0, z: 0 },
          up: { x: 0, y: 1, z: 0 },
          fov: 47,
          near: 0.1,
          far: 1000,
        },
        scene: { background: null },
        controls: { getFocusTarget: () => ({ x: 0, y: 0, z: 0 }) },
      } as any,
      { settings: {} } as any,
      { getDims: () => null } as any,
      undefined,
      throwingTM
    );

    // The catch must SWALLOW the throw; capture must still produce a
    // ZarrViewerConfig, just without the `theme` field. A regression that
    // re-threw or set `theme` to a sentinel string would fail here.
    expect(state).toBeDefined();
    expect(state.theme).toBeUndefined();
    // Other required fields stay populated — the catch is local to
    // the theme block (line 72-78), not a global escape.
    expect(state.camera).toBeDefined();
  });
});
