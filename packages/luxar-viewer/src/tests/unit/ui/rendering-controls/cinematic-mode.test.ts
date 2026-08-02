/**
 * Unit tests for CinematicModeController + the buildCinematicValues
 * helper extracted from rendering-controls.
 *
 * The controller has many manager dependencies but the logic worth
 * covering is pure: snapshot creation on enable, dirty-check on
 * disable, checkbox majority-vote, and the no-snapshot fallback path.
 *
 * The tests use plain stub objects for postProcessing / sceneManager
 * so we never spin up real WebGL.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  CinematicModeController,
  buildCinematicValues,
  type CinematicContext,
  type CinematicSnapshot,
} from '../../../../ui/rendering-controls/cinematic-mode';
import { config, type RenderingSettings } from '../../../../config';

function makeStubContext(overrides: Partial<RenderingSettings> = {}): {
  ctx: CinematicContext;
  settings: RenderingSettings;
  postProcessing: {
    setToneMapping: ReturnType<typeof vi.fn>;
    setBloomEnabled: ReturnType<typeof vi.fn>;
    setBloomLevels: ReturnType<typeof vi.fn>;
    setDetectorNoiseEnabled: ReturnType<typeof vi.fn>;
    setVignetteEnabled: ReturnType<typeof vi.fn>;
    setChromaticLensDistortionEnabled: ReturnType<typeof vi.fn>;
    startDeferRebuild: ReturnType<typeof vi.fn>;
    endDeferRebuild: ReturnType<typeof vi.fn>;
    withDeferredRebuild: ReturnType<typeof vi.fn>;
  };
  sceneManager: { currentFov: number; updateFOV: ReturnType<typeof vi.fn> };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  refreshAllControllers: ReturnType<typeof vi.fn>;
} {
  const settings = {
    ...config.renderingControls.defaults,
    ...overrides,
  } as RenderingSettings;

  const postProcessing = {
    setToneMapping: vi.fn(),
    setBloomEnabled: vi.fn(),
    setBloomLevels: vi.fn(),
    setDetectorNoiseEnabled: vi.fn(),
    setVignetteEnabled: vi.fn(),
    setChromaticLensDistortionEnabled: vi.fn(),
    startDeferRebuild: vi.fn(),
    endDeferRebuild: vi.fn(),
    // The production code calls `withDeferredRebuild(fn)`; mirror its
    // try/finally semantics so the closure runs and rebuild bookkeeping
    // happens on both success and throw paths.
    withDeferredRebuild: vi.fn((fn: () => void) => {
      postProcessing.startDeferRebuild();
      try {
        return fn();
      } finally {
        postProcessing.endDeferRebuild();
      }
    }),
  };
  const sceneManager = {
    currentFov: 50,
    updateFOV: vi.fn((delta: number) => {
      sceneManager.currentFov += delta * config.camera.fovSensitivity;
    }),
  };
  const saveSettings = vi.fn();
  const triggerAnimation = vi.fn();
  const refreshAllControllers = vi.fn();

  const ctx: CinematicContext = {
    settings,
    postProcessing: postProcessing as never,
    sceneManager: sceneManager as never,
    controllers: {} as never,
    saveSettings,
    triggerAnimation,
    refreshAllControllers,
  };

  return {
    ctx,
    settings,
    postProcessing,
    sceneManager,
    saveSettings,
    triggerAnimation,
    refreshAllControllers,
  };
}

describe('buildCinematicValues', () => {
  it('returns ACES tone mapping + detector noise + vignette + lens distortion enabled', () => {
    const v = buildCinematicValues();
    expect(v.toneMapping).toBe('ACES');
    expect(v.detectorNoiseEnabled).toBe(true);
    expect(v.vignetteEnabled).toBe(true);
    expect(v.chromaticLensDistortionEnabled).toBe(true);
  });

  it('enables bloom with the subtle wide-glow parameters', () => {
    const v = buildCinematicValues();
    expect(v.bloomEnabled).toBe(true);
    expect(v.bloomThreshold).toBe(0.01);
    expect(v.bloomStrength).toBe(0.05);
    expect(v.bloomRadius).toBe(1.0);
    expect(v.bloomLevels).toBe(8);
  });

  it('uses the 35mm FOV preset', () => {
    const v = buildCinematicValues();
    expect(v.fov).toBe(config.camera.fovPresets['35mm']);
    expect(v.fovPreset).toBe('35mm');
  });

  it('uses the 35mm lens distortion preset', () => {
    const v = buildCinematicValues();
    const lens35 = config.camera.lensDistortionPresets['35mm'];
    expect(v.chromaticLensDistortionX).toBe(lens35.distortionX);
    expect(v.chromaticLensDistortionY).toBe(lens35.distortionY);
    expect(v.chromaticLensDispersion).toBe(lens35.dispersion);
    expect(v.chromaticLensFocalLengthX).toBe(lens35.focalLengthX);
  });
});

describe('CinematicModeController.updateCheckbox', () => {
  it('flips cinematicMode true when ≥ 2 of the 4 signal effects are on', () => {
    const { ctx, settings } = makeStubContext({
      detectorNoiseEnabled: true,
      vignetteEnabled: true,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Linear',
    });
    new CinematicModeController(ctx).updateCheckbox();
    expect(settings.cinematicMode).toBe(true);
  });

  it('flips cinematicMode false when < 2 of the 4 signal effects are on', () => {
    const { ctx, settings } = makeStubContext({
      detectorNoiseEnabled: true,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Linear',
    });
    new CinematicModeController(ctx).updateCheckbox();
    expect(settings.cinematicMode).toBe(false);
  });

  it('treats ACES tone mapping as one of the four signal effects', () => {
    const { ctx, settings } = makeStubContext({
      detectorNoiseEnabled: false,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: true,
      toneMapping: 'ACES',
    });
    new CinematicModeController(ctx).updateCheckbox();
    expect(settings.cinematicMode).toBe(true);
  });
});

describe('CinematicModeController.toggle — enable path', () => {
  let stub: ReturnType<typeof makeStubContext>;
  let cm: CinematicModeController;

  beforeEach(() => {
    stub = makeStubContext({
      detectorNoiseEnabled: false,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Linear',
    });
    cm = new CinematicModeController(stub.ctx);
  });

  it('flips cinematic effects ON', () => {
    cm.toggle();
    expect(stub.settings.detectorNoiseEnabled).toBe(true);
    expect(stub.settings.vignetteEnabled).toBe(true);
    expect(stub.settings.chromaticLensDistortionEnabled).toBe(true);
    expect(stub.settings.toneMapping).toBe('ACES');
    expect(stub.settings.fovPreset).toBe('35mm');
  });

  it('turns bloom ON with the preset parameters and pushes them to post-processing', () => {
    cm.toggle();
    expect(stub.settings.bloomEnabled).toBe(true);
    expect(stub.settings.bloomThreshold).toBe(0.01);
    expect(stub.settings.bloomStrength).toBe(0.05);
    expect(stub.settings.bloomRadius).toBe(1.0);
    expect(stub.settings.bloomLevels).toBe(8);
    // setBloomEnabled(enabled, strength, radius, threshold)
    expect(stub.postProcessing.setBloomEnabled).toHaveBeenCalledWith(true, 0.05, 1.0, 0.01);
    expect(stub.postProcessing.setBloomLevels).toHaveBeenCalledWith(8);
  });

  it('wraps post-processing changes in startDeferRebuild / endDeferRebuild', () => {
    cm.toggle();
    expect(stub.postProcessing.startDeferRebuild).toHaveBeenCalled();
    expect(stub.postProcessing.endDeferRebuild).toHaveBeenCalled();
  });

  it('saves settings + triggers animation', () => {
    cm.toggle();
    expect(stub.saveSettings).toHaveBeenCalled();
    expect(stub.triggerAnimation).toHaveBeenCalled();
  });

  it('refreshes the GUI controllers', () => {
    cm.toggle();
    expect(stub.refreshAllControllers).toHaveBeenCalled();
  });
});

describe('CinematicModeController.toggle — disable / restore path', () => {
  it('restores user-customised values from snapshot when toggling back off', () => {
    const stub = makeStubContext({
      detectorNoiseEnabled: false,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Reinhard', // user picked Reinhard before turning cinematic on
    });
    const cm = new CinematicModeController(stub.ctx);

    cm.toggle(); // ON: snapshot taken, ACES applied
    expect(stub.settings.toneMapping).toBe('ACES');

    cm.toggle(); // OFF: dirty-check restore
    expect(stub.settings.toneMapping).toBe('Reinhard');
  });

  it('restores the pre-cinematic bloom state when toggling back off', () => {
    const stub = makeStubContext({
      detectorNoiseEnabled: false,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Linear',
      bloomEnabled: false,
      bloomStrength: 0.4,
    });
    const cm = new CinematicModeController(stub.ctx);

    cm.toggle(); // ON
    expect(stub.settings.bloomEnabled).toBe(true);
    expect(stub.settings.bloomStrength).toBe(0.05);

    cm.toggle(); // OFF
    expect(stub.settings.bloomEnabled).toBe(false);
    expect(stub.settings.bloomStrength).toBe(0.4);
  });

  it('keeps a bloom strength the user hand-edited while cinematic was on (dirty check)', () => {
    const stub = makeStubContext({
      detectorNoiseEnabled: false,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Linear',
      bloomEnabled: false,
      bloomStrength: 0.4,
    });
    const cm = new CinematicModeController(stub.ctx);

    cm.toggle(); // ON
    stub.settings.bloomStrength = 0.9; // user dials the glow up

    cm.toggle(); // OFF
    expect(stub.settings.bloomStrength).toBe(0.9);
  });

  it('does NOT restore values the user hand-edited while cinematic was on (dirty check)', () => {
    const stub = makeStubContext({
      vignetteEnabled: false,
      detectorNoiseEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Reinhard',
    });
    const cm = new CinematicModeController(stub.ctx);

    cm.toggle(); // ON, snapshot toneMapping=Reinhard, settings.toneMapping=ACES
    // Simulate user manually picking AgX while cinematic is on.
    stub.settings.toneMapping = 'AgX';

    cm.toggle(); // OFF — toneMapping doesn't equal cinematicValues.toneMapping anymore.
    // → user's choice is preserved.
    expect(stub.settings.toneMapping).toBe('AgX');
  });

  it('falls back to non-cinematic defaults when no snapshot exists (e.g. loaded with cinematic on)', () => {
    const stub = makeStubContext({
      detectorNoiseEnabled: true,
      vignetteEnabled: true,
      chromaticLensDistortionEnabled: true,
      toneMapping: 'ACES',
      fovPreset: '35mm',
      fov: config.camera.fovPresets['35mm'],
    });
    const cm = new CinematicModeController(stub.ctx);
    // No prior toggle call — snapshot is null.

    cm.toggle();

    // Because >=2 effects were on, this is the DISABLE branch with no
    // snapshot → falls back to the 50mm Normal defaults.
    expect(stub.settings.detectorNoiseEnabled).toBe(false);
    expect(stub.settings.vignetteEnabled).toBe(false);
    expect(stub.settings.chromaticLensDistortionEnabled).toBe(false);
    expect(stub.settings.toneMapping).toBe(config.renderingControls.defaults.toneMapping);
    expect(stub.settings.fovPreset).toBe('50mm Normal');
  });

  it('falls back to the default bloom settings when no snapshot exists', () => {
    const defaults = config.renderingControls.defaults;
    const stub = makeStubContext({
      detectorNoiseEnabled: true,
      vignetteEnabled: true,
      chromaticLensDistortionEnabled: true,
      toneMapping: 'ACES',
      bloomEnabled: true,
      bloomStrength: 0.05,
    });
    const cm = new CinematicModeController(stub.ctx);

    cm.toggle(); // DISABLE branch, snapshot is null

    expect(stub.settings.bloomEnabled).toBe(defaults.bloomEnabled);
    expect(stub.settings.bloomThreshold).toBe(defaults.bloomThreshold);
    expect(stub.settings.bloomStrength).toBe(defaults.bloomStrength);
    expect(stub.settings.bloomRadius).toBe(defaults.bloomRadius);
    expect(stub.settings.bloomLevels).toBe(defaults.bloomLevels);
  });
});

describe('CinematicModeController.clearSnapshot', () => {
  it('drops a captured snapshot so a subsequent disable falls through to defaults', () => {
    const stub = makeStubContext({
      detectorNoiseEnabled: false,
      vignetteEnabled: false,
      chromaticLensDistortionEnabled: false,
      toneMapping: 'Reinhard',
    });
    const cm = new CinematicModeController(stub.ctx);

    cm.toggle();
    cm.clearSnapshot();
    cm.toggle(); // disable path — snapshot is null → falls back to defaults

    // Falls back to default tone mapping, NOT the original Reinhard.
    expect(stub.settings.toneMapping).toBe(config.renderingControls.defaults.toneMapping);
  });

  it('safe to call when no snapshot has ever been captured', () => {
    const { ctx } = makeStubContext();
    const cm = new CinematicModeController(ctx);
    expect(() => cm.clearSnapshot()).not.toThrow();
  });
});

describe('CinematicSnapshot type — round trip', () => {
  it('snapshot type covers exactly the keys mutated by buildCinematicValues', () => {
    const v: CinematicSnapshot = buildCinematicValues();
    // Smoke check that all expected keys are present.
    const keys: (keyof CinematicSnapshot)[] = [
      'toneMapping',
      'bloomEnabled',
      'bloomThreshold',
      'bloomStrength',
      'bloomRadius',
      'bloomLevels',
      'detectorNoiseEnabled',
      'detectorNoiseReadoutSigma',
      'detectorNoisePhotonGain',
      'detectorNoiseFpnSigma',
      'vignetteEnabled',
      'chromaticLensDistortionEnabled',
      'chromaticLensDistortionX',
      'chromaticLensDistortionY',
      'chromaticLensDispersion',
      'chromaticLensPrincipalPointX',
      'chromaticLensPrincipalPointY',
      'chromaticLensFocalLengthX',
      'chromaticLensFocalLengthY',
      'chromaticLensSkew',
      'fov',
      'fovPreset',
    ];
    for (const k of keys) {
      expect(v[k]).toBeDefined();
    }
  });
});
