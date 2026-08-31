/**
 * Tests for viewer-config-utils: zarr viewer_config ↔ RenderingSettings conversion
 */

import { beforeEach, describe, it, expect, vi } from 'vitest';
import {
  _warnedFovPresetConflicts,
  extractRenderingOverrides,
  extractCameraOverrides,
  extractBackgroundColor,
  renderingSettingsToZarr,
  RENDERING_SETTINGS_MAP,
} from '../../../config/zarr-bridge/viewer-config-utils';
import { config } from '../../../config';
import { buildCinematicValues, CINEMATIC_SNAPSHOT_KEYS } from '../../../config/cinematic-preset';
import type { ZarrViewerConfig } from '../../../types/zarr';
import { log, Modules } from '../../../utils/log';

describe('extractRenderingOverrides', () => {
  beforeEach(() => {
    _warnedFovPresetConflicts.clear();
  });

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
      auto_rotate_axis: 'view',
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
    expect(overrides.autoRotateAxis).toBe('view');
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
    // [W1][P2] Audit: pre-strengthening this test only checked 3 specific
    // absent fields. A mutation that added spurious keys to the result
    // (e.g. defaulting unknown keys) would have passed. Assert the exact
    // result key set so any extra key kills the test.
    const overrides = extractRenderingOverrides({ bloom_enabled: true });

    expect(overrides.bloomEnabled).toBe(true);
    // All other fields should be absent — total key set must be exactly {bloomEnabled}
    expect(Object.keys(overrides)).toEqual(['bloomEnabled']);
    expect(Object.keys(overrides)).toHaveLength(1);
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

  it('resolves camera.fov_preset to its numeric FOV when camera.fov is absent', () => {
    const overrides = extractRenderingOverrides({
      camera: { fov_preset: '85mm Portrait' },
    });

    expect(overrides.fov).toBe(config.camera.fovPresets['85mm Portrait']);
    expect(overrides.fovPreset).toBe('85mm Portrait');
  });

  it('keeps an explicit camera.fov authoritative and warns once when the preset disagrees', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const configWithConflict = {
      camera: { fov: 90, fov_preset: '85mm Portrait' },
    } satisfies ZarrViewerConfig;

    const overrides = extractRenderingOverrides(configWithConflict);
    extractRenderingOverrides(configWithConflict);

    expect(overrides.fov).toBe(90);
    expect(overrides.fovPreset).toBe('85mm Portrait');
    expect(warnSpy).toHaveBeenCalledOnce();
    expect(warnSpy).toHaveBeenCalledWith(
      Modules.CONFIG,
      expect.stringMatching(/numeric FOV wins.*preset label will be re-derived/)
    );
    warnSpy.mockRestore();
  });

  it('does not warn when camera.fov remains within the preset label tolerance', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    const overrides = extractRenderingOverrides({
      camera: { fov: 29.2, fov_preset: '85mm Portrait' },
    });

    expect(overrides.fov).toBe(29.2);
    expect(overrides.fovPreset).toBe('85mm Portrait');
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('does not invent a numeric FOV for the Custom preset', () => {
    const overrides = extractRenderingOverrides({
      camera: { fov_preset: 'Custom' },
    });

    expect('fov' in overrides).toBe(false);
    expect(overrides.fovPreset).toBe('Custom');
  });

  it('does not invent a numeric FOV for an unknown preset name', () => {
    const overrides = extractRenderingOverrides({
      camera: { fov_preset: 'nonsense' },
    });

    expect('fov' in overrides).toBe(false);
    expect(overrides.fovPreset).toBe('nonsense');
  });
});

describe('extractRenderingOverrides — cinematic_mode preset expansion', () => {
  // `cinematic_mode` used to be a pass-through flag nothing acted on, so a
  // scene authoring it rendered with none of the preset's effects. The bridge
  // now expands the preset for every key the scene did NOT set explicitly.

  it('expands the whole cinematic preset when cinematic_mode is true', () => {
    const overrides = extractRenderingOverrides({ cinematic_mode: true });
    const preset = buildCinematicValues();

    // Assert against buildCinematicValues() itself so this test cannot drift
    // from the preset definition.
    for (const key of CINEMATIC_SNAPSHOT_KEYS) {
      expect(overrides[key]).toBe(preset[key]);
    }
    // The flag itself survives, so the panel checkbox still reads as on.
    expect(overrides.cinematicMode).toBe(true);
  });

  it('keeps author-set keys — the scene wins over the preset', () => {
    const overrides = extractRenderingOverrides({
      cinematic_mode: true,
      bloom_strength: 0.9,
      tone_mapping: 'Neutral',
      vignette_enabled: false,
    });
    const preset = buildCinematicValues();

    expect(overrides.bloomStrength).toBe(0.9);
    expect(overrides.toneMapping).toBe('Neutral');
    expect(overrides.vignetteEnabled).toBe(false);
    // Unset preset keys are still expanded around them.
    expect(overrides.bloomThreshold).toBe(preset.bloomThreshold);
    expect(overrides.detectorNoiseEnabled).toBe(preset.detectorNoiseEnabled);
    expect(overrides.chromaticLensDispersion).toBe(preset.chromaticLensDispersion);
  });

  // `fov` and `fovPreset` are a COUPLED pair: an author who set either half
  // owns the framing, so neither half is filled from the cinematic preset.
  // A recognized authored preset supplies its own numeric FOV upstream.
  it('leaves BOTH fov keys alone when the author set camera.fov', () => {
    const overrides = extractRenderingOverrides({
      cinematic_mode: true,
      camera: { fov: 90 },
    });

    expect(overrides.fov).toBe(90);
    expect('fovPreset' in overrides).toBe(false);
    // The rest of the preset still expands around the untouched pair.
    expect(overrides.toneMapping).toBe(buildCinematicValues().toneMapping);
  });

  it('fills fov from the preset table — not the cinematic lens — when the author set camera.fov_preset', () => {
    const overrides = extractRenderingOverrides({
      cinematic_mode: true,
      camera: { fov_preset: '85mm Portrait' },
    });

    expect(overrides.fovPreset).toBe('85mm Portrait');
    expect(overrides.fov).toBe(config.camera.fovPresets['85mm Portrait']);
    expect(overrides.toneMapping).toBe(buildCinematicValues().toneMapping);
  });

  it('expands both fov keys when the scene authors no camera framing', () => {
    const overrides = extractRenderingOverrides({ cinematic_mode: true });
    const preset = buildCinematicValues();

    expect(overrides.fov).toBe(preset.fov);
    expect(overrides.fovPreset).toBe(preset.fovPreset);
  });

  it('treats a null camera value as unset, not as author-set', () => {
    // The map walk skips null; the camera block must agree, otherwise a null
    // both blocks the preset key and leaks downstream to guards that only
    // test `!== undefined`.
    const overrides = extractRenderingOverrides({
      cinematic_mode: true,
      camera: { fov: null as unknown as number, fov_preset: null as unknown as string },
    });
    const preset = buildCinematicValues();

    expect(overrides.fov).toBe(preset.fov);
    expect(overrides.fovPreset).toBe(preset.fovPreset);
    expect('near' in overrides).toBe(false);
  });

  it('does not expand for cinematic_mode false, absent, or non-boolean truthy', () => {
    expect(extractRenderingOverrides({ cinematic_mode: false }).toneMapping).toBeUndefined();
    expect(extractRenderingOverrides({}).toneMapping).toBeUndefined();
    // Strictly `=== true`: a corrupt config must not silently restyle the scene.
    const truthy = extractRenderingOverrides({
      cinematic_mode: 1 as unknown as boolean,
    });
    expect(truthy.toneMapping).toBeUndefined();
    expect(truthy.bloomEnabled).toBeUndefined();
  });

  it('cinematic_mode false leaves the overrides to the explicit keys only', () => {
    const overrides = extractRenderingOverrides({
      cinematic_mode: false,
      bloom_enabled: true,
    });
    expect(Object.keys(overrides).sort()).toEqual(['bloomEnabled', 'cinematicMode']);
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
  // [W2][P2] Audit: the prior `requiredEntries` loop only asserted the
  // SNAKE key existed — never that it mapped to the EXPECTED camelCase
  // name. A mutation that mapped `bloom_strength → 'bloomRadius'` would
  // have passed. The bijection property test in
  // viewer-config-utils.property.test.ts covers the full 43-entry map
  // mechanically; here we keep explicit, named value-assertions for the
  // audit-called-out entries (naturalDrag, auto_rotate_speed,
  // chromatic_lens_*, fly_rotation_speed, ...) so they remain
  // discoverable when reading this file in isolation.
  const REQUIRED_PAIRS: Array<[string, string]> = [
    ['bloom_enabled', 'bloomEnabled'],
    ['bloom_strength', 'bloomStrength'],
    ['bloom_radius', 'bloomRadius'],
    ['bloom_threshold', 'bloomThreshold'],
    ['bloom_levels', 'bloomLevels'],
    ['exposure', 'exposure'],
    ['global_offset', 'globalOffset'],
    ['global_gamma', 'globalGamma'],
    ['tone_mapping', 'toneMapping'],
    ['control_type', 'controlType'],
    ['auto_rotate', 'autoRotate'],
    ['auto_rotate_speed', 'autoRotateSpeed'],
    ['auto_rotate_axis', 'autoRotateAxis'],
    ['auto_dolly', 'autoDolly'],
    ['auto_dolly_amplitude_percent', 'autoDollyAmplitudePercent'],
    ['auto_dolly_period', 'autoDollyPeriod'],
    ['natural_drag', 'naturalDrag'],
    ['cinematic_mode', 'cinematicMode'],
    ['vignette_enabled', 'vignetteEnabled'],
    ['vignette_darkness', 'vignetteDarkness'],
    ['vignette_offset', 'vignetteOffset'],
    ['detector_noise_enabled', 'detectorNoiseEnabled'],
    ['detector_noise_readout_sigma', 'detectorNoiseReadoutSigma'],
    ['detector_noise_photon_gain', 'detectorNoisePhotonGain'],
    ['detector_noise_fpn_sigma', 'detectorNoiseFpnSigma'],
    ['fxaa_enabled', 'fxaaEnabled'],
    ['msaa_enabled', 'msaaEnabled'],
    ['msaa_samples', 'msaaSamples'],
    ['ssaa_enabled', 'ssaaEnabled'],
    ['ssaa_multiplier', 'ssaaMultiplier'],
    ['chromatic_lens_distortion_enabled', 'chromaticLensDistortionEnabled'],
    ['chromatic_lens_distortion_x', 'chromaticLensDistortionX'],
    ['chromatic_lens_distortion_y', 'chromaticLensDistortionY'],
    ['chromatic_lens_dispersion', 'chromaticLensDispersion'],
    ['chromatic_lens_principal_point_x', 'chromaticLensPrincipalPointX'],
    ['chromatic_lens_principal_point_y', 'chromaticLensPrincipalPointY'],
    ['chromatic_lens_focal_length_x', 'chromaticLensFocalLengthX'],
    ['chromatic_lens_focal_length_y', 'chromaticLensFocalLengthY'],
    ['chromatic_lens_skew', 'chromaticLensSkew'],
    ['fly_movement_speed', 'flyMovementSpeed'],
    ['fly_rotation_speed', 'flyRotationSpeed'],
    ['fly_inertial_mode', 'flyInertialMode'],
    ['fly_damping', 'flyDamping'],
    ['fly_rotation_damping', 'flyRotationDamping'],
    ['dynamic_clipping_enabled', 'dynamicClippingEnabled'],
    ['adaptive_dpr_enabled', 'adaptiveDPREnabled'],
    ['allow_high_dpr', 'allowHighDPR'],
  ];

  it.each(REQUIRED_PAIRS)('maps snake `%s` → camel `%s`', (snake, camel) => {
    expect(RENDERING_SETTINGS_MAP[snake]).toBe(camel);
  });

  it('REQUIRED_PAIRS covers every entry in RENDERING_SETTINGS_MAP (no orphans)', () => {
    // Force a follow-up when a new entry is added to the source map.
    expect(REQUIRED_PAIRS.length).toBe(Object.keys(RENDERING_SETTINGS_MAP).length);
  });
});
