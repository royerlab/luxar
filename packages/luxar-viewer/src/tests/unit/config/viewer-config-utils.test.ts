/**
 * Tests for viewer-config-utils: zarr viewer_config ↔ RenderingSettings conversion
 */

import { describe, it, expect } from 'vitest';
import {
  extractRenderingOverrides,
  extractCameraOverrides,
  extractBackgroundColor,
  renderingSettingsToZarr,
  RENDERING_SETTINGS_MAP,
} from '../../../config/zarr-bridge/viewer-config-utils';
import type { ZarrViewerConfig } from '../../../types/zarr';

describe('extractRenderingOverrides', () => {
  it('should return empty object for empty config', () => {
    const overrides = extractRenderingOverrides({});
    expect(Object.keys(overrides)).toHaveLength(0);
  });

  it('should map snake_case zarr keys to camelCase RenderingSettings keys', () => {
    const zarrConfig: ZarrViewerConfig = {
      bloom_enabled: true,
      bloom_strength: 0.5,
      exposure: 1.0,
      global_offset: 0.1,
      global_gamma: 1.5,
      tone_mapping: 'ACES',
      control_type: 'fly',
      auto_rotate: true,
      auto_rotate_speed: 0.3,
    };

    const overrides = extractRenderingOverrides(zarrConfig);

    expect(overrides.bloomEnabled).toBe(true);
    expect(overrides.bloomStrength).toBe(0.5);
    expect(overrides.exposure).toBe(1.0);
    expect(overrides.globalOffset).toBe(0.1);
    expect(overrides.globalGamma).toBe(1.5);
    expect(overrides.toneMapping).toBe('ACES');
    expect(overrides.controlType).toBe('fly');
    expect(overrides.autoRotate).toBe(true);
    expect(overrides.autoRotateSpeed).toBe(0.3);
  });

  it('should map cinematic mode and effects', () => {
    const zarrConfig: ZarrViewerConfig = {
      cinematic_mode: true,
      vignette_enabled: true,
      vignette_darkness: 0.6,
    };

    const overrides = extractRenderingOverrides(zarrConfig);

    expect(overrides.cinematicMode).toBe(true);
    expect(overrides.vignetteEnabled).toBe(true);
    expect(overrides.vignetteDarkness).toBe(0.6);
  });

  it('should map detector noise settings', () => {
    const zarrConfig: ZarrViewerConfig = {
      detector_noise_enabled: true,
      detector_noise_readout_sigma: 0.005,
      detector_noise_photon_gain: 0.003,
      detector_noise_fpn_sigma: 0.001,
    };

    const overrides = extractRenderingOverrides(zarrConfig);

    expect(overrides.detectorNoiseEnabled).toBe(true);
    expect(overrides.detectorNoiseReadoutSigma).toBe(0.005);
    expect(overrides.detectorNoisePhotonGain).toBe(0.003);
    expect(overrides.detectorNoiseFpnSigma).toBe(0.001);
  });

  it('should map anti-aliasing settings', () => {
    const overrides = extractRenderingOverrides({
      fxaa_enabled: true,
    });

    expect(overrides.fxaaEnabled).toBe(true);
  });

  it('should extract camera.fov into RenderingSettings.fov', () => {
    const overrides = extractRenderingOverrides({
      camera: { fov: 60 },
    });

    expect(overrides.fov).toBe(60);
  });

  it('should not extract camera position/target into RenderingSettings', () => {
    const overrides = extractRenderingOverrides({
      camera: { position: [1, 2, 3], target: [0, 0, 0] },
    });

    // position and target should NOT appear in RenderingSettings
    expect((overrides as any).position).toBeUndefined();
    expect((overrides as any).target).toBeUndefined();
  });

  it('should skip null and undefined values', () => {
    const zarrConfig: ZarrViewerConfig = {
      bloom_enabled: true,
      bloom_strength: undefined,
    };

    const overrides = extractRenderingOverrides(zarrConfig);

    expect(overrides.bloomEnabled).toBe(true);
    expect(overrides.bloomStrength).toBeUndefined();
  });

  it('should only include fields present in zarr config', () => {
    const overrides = extractRenderingOverrides({ bloom_enabled: true });

    expect(overrides.bloomEnabled).toBe(true);
    // All other fields should be absent
    expect(overrides.bloomStrength).toBeUndefined();
    expect(overrides.toneMapping).toBeUndefined();
    expect(overrides.fov).toBeUndefined();
  });

  // -- New field mapping tests --

  it('should map new bloom_levels field', () => {
    const overrides = extractRenderingOverrides({ bloom_levels: 8 });
    expect(overrides.bloomLevels).toBe(8);
  });

  it('should map vignette_offset', () => {
    const overrides = extractRenderingOverrides({
      vignette_offset: 0.3,
    });
    expect(overrides.vignetteOffset).toBe(0.3);
  });

  it('should map extended AA fields', () => {
    const overrides = extractRenderingOverrides({
      msaa_enabled: true,
      msaa_samples: 4,
      ssaa_enabled: false,
      ssaa_multiplier: 2.0,
    });
    expect(overrides.msaaEnabled).toBe(true);
    expect(overrides.msaaSamples).toBe(4);
    expect(overrides.ssaaEnabled).toBe(false);
    expect(overrides.ssaaMultiplier).toBe(2.0);
  });

  it('should map chromatic lens fields', () => {
    const overrides = extractRenderingOverrides({
      chromatic_lens_distortion_enabled: true,
      chromatic_lens_distortion_x: 0.1,
      chromatic_lens_distortion_y: 0.2,
      chromatic_lens_dispersion: 0.5,
    });
    expect(overrides.chromaticLensDistortionEnabled).toBe(true);
    expect(overrides.chromaticLensDistortionX).toBe(0.1);
  });

  it('should map fly control fields', () => {
    const overrides = extractRenderingOverrides({
      fly_movement_speed: 2.0,
      fly_rotation_speed: 0.5,
      fly_inertial_mode: true,
      fly_damping: 0.95,
      fly_rotation_damping: 0.9,
    });
    expect(overrides.flyMovementSpeed).toBe(2.0);
    expect(overrides.flyInertialMode).toBe(true);
  });

  it('should map clipping and adaptive fields', () => {
    const overrides = extractRenderingOverrides({
      dynamic_clipping_enabled: true,
      adaptive_dpr_enabled: true,
    });
    expect(overrides.dynamicClippingEnabled).toBe(true);
    expect(overrides.adaptiveDPREnabled).toBe(true);
  });

  it('should map camera.fov_preset and camera.near/far', () => {
    const overrides = extractRenderingOverrides({
      camera: {
        fov: 47,
        fov_preset: '50mm Normal',
        near: 0.1,
        far: 1000,
      },
    });
    expect(overrides.fov).toBe(47);
    expect(overrides.fovPreset).toBe('50mm Normal');
    expect(overrides.near).toBe(0.1);
    expect(overrides.far).toBe(1000);
  });
});

describe('extractCameraOverrides', () => {
  it('should return empty object for empty config', () => {
    const cam = extractCameraOverrides({});
    expect(cam.position).toBeUndefined();
    expect(cam.target).toBeUndefined();
    expect(cam.up).toBeUndefined();
  });

  it('should extract position', () => {
    const cam = extractCameraOverrides({
      camera: { position: [1, 2, 3] },
    });
    expect(cam.position).toEqual({ x: 1, y: 2, z: 3 });
  });

  it('should extract target', () => {
    const cam = extractCameraOverrides({
      camera: { target: [4, 5, 6] },
    });
    expect(cam.target).toEqual({ x: 4, y: 5, z: 6 });
  });

  it('should extract up', () => {
    const cam = extractCameraOverrides({
      camera: { up: [0, 1, 0] },
    });
    expect(cam.up).toEqual({ x: 0, y: 1, z: 0 });
  });

  it('should extract all camera fields when present', () => {
    const cam = extractCameraOverrides({
      camera: {
        position: [1, 2, 3],
        target: [0, 0, 0],
        up: [0, 1, 0],
      },
    });
    expect(cam.position).toEqual({ x: 1, y: 2, z: 3 });
    expect(cam.target).toEqual({ x: 0, y: 0, z: 0 });
    expect(cam.up).toEqual({ x: 0, y: 1, z: 0 });
  });

  it('should not include fov (fov goes to RenderingSettings)', () => {
    const cam = extractCameraOverrides({
      camera: { position: [1, 2, 3], fov: 60 },
    });
    expect(cam.position).toBeDefined();
    expect((cam as any).fov).toBeUndefined();
  });

  it('should extract target_node', () => {
    const cam = extractCameraOverrides({
      camera: { target_node: 'embryo' },
    });
    expect(cam.targetNode).toBe('embryo');
  });
});

describe('extractBackgroundColor', () => {
  it('should return undefined for empty config', () => {
    expect(extractBackgroundColor({})).toBeUndefined();
  });

  it('should return background color when set', () => {
    expect(extractBackgroundColor({ background_color: '#112233' })).toBe('#112233');
  });
});

describe('renderingSettingsToZarr', () => {
  it('should return empty object for empty settings', () => {
    const result = renderingSettingsToZarr({});
    expect(Object.keys(result)).toHaveLength(0);
  });

  it('should convert camelCase to snake_case', () => {
    const result = renderingSettingsToZarr({
      bloomEnabled: true,
      bloomStrength: 0.5,
      exposure: 1.0,
      globalOffset: 0.1,
      globalGamma: 1.5,
      toneMapping: 'ACES',
    });

    expect(result.bloom_enabled).toBe(true);
    expect(result.bloom_strength).toBe(0.5);
    expect(result.exposure).toBe(1.0);
    expect(result.global_offset).toBe(0.1);
    expect(result.global_gamma).toBe(1.5);
    expect(result.tone_mapping).toBe('ACES');
  });

  it('should skip undefined values', () => {
    const result = renderingSettingsToZarr({
      bloomEnabled: true,
      bloomStrength: undefined,
    });
    expect(result.bloom_enabled).toBe(true);
    expect(result.bloom_strength).toBeUndefined();
  });

  it('should convert new fields', () => {
    const result = renderingSettingsToZarr({
      bloomLevels: 8,
      dynamicClippingEnabled: true,
      adaptiveDPREnabled: false,
      flyMovementSpeed: 2.0,
    });
    expect(result.bloom_levels).toBe(8);
    expect(result.dynamic_clipping_enabled).toBe(true);
    expect(result.adaptive_dpr_enabled).toBe(false);
    expect(result.fly_movement_speed).toBe(2.0);
  });

  it('should not include fov/near/far (those go under camera)', () => {
    const result = renderingSettingsToZarr({
      fov: 47,
      near: 0.1,
      far: 1000,
    } as any);
    // fov, near, far are not in RENDERING_SETTINGS_MAP, so they should not appear
    expect(result).not.toHaveProperty('fov');
    expect(result).not.toHaveProperty('near');
    expect(result).not.toHaveProperty('far');
  });
});

describe('RENDERING_SETTINGS_MAP completeness', () => {
  it('should have an entry for every non-camera RenderingSettings field that maps to zarr', () => {
    // Verify the map has at least the critical fields
    const requiredEntries = [
      'bloom_enabled',
      'bloom_strength',
      'bloom_radius',
      'bloom_threshold',
      'bloom_levels',
      'exposure',
      'global_offset',
      'global_gamma',
      'tone_mapping',
      'control_type',
      'auto_rotate',
      'cinematic_mode',
      'vignette_enabled',
      'vignette_darkness',
      'vignette_offset',
      'detector_noise_enabled',
      'fxaa_enabled',
      'msaa_enabled',
      'msaa_samples',
      'ssaa_enabled',
      'ssaa_multiplier',
      'chromatic_lens_distortion_enabled',
      'fly_movement_speed',
      'fly_inertial_mode',
      'dynamic_clipping_enabled',
      'adaptive_dpr_enabled',
    ];

    for (const key of requiredEntries) {
      expect(RENDERING_SETTINGS_MAP).toHaveProperty(key);
    }
  });
});
