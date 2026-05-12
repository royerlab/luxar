/**
 * Tests for viewer-state-capture: captureViewerState() function
 */

import { describe, it, expect, vi } from 'vitest';
import { captureViewerState } from '../../../config/viewer-state-capture';

// Mock ThemeManager
vi.mock('../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: () => ({
      getCurrentTheme: () => ({ id: 'dark', name: 'Dark Theme' }),
    }),
  },
}));

// Create mock objects
function createMockSceneManager() {
  return {
    camera: {
      position: { x: 1, y: 2, z: 3 },
      up: { x: 0, y: 1, z: 0 },
      fov: 47,
      near: 0.1,
      far: 1000,
    },
    scene: {
      background: {
        isColor: true,
        getHexString: () => '111111',
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
    expect(state.camera!.near).toBe(0.1);
    expect(state.camera!.far).toBe(1000);
  });

  it('should capture background color', () => {
    const state = captureViewerState(
      createMockSceneManager(),
      createMockRenderingControls(),
      createMockSceneDimsManager()
    );

    expect(state.background_color).toBe('#111111');
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
    expect(state.fxaa_enabled).toBe(true);
    expect(state.dynamic_clipping_enabled).toBe(true);
    expect(state.adaptive_dpr_enabled).toBe(true);
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
});
