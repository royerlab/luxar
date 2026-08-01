/**
 * Unit tests for `applyRenderingSettings` — pure helper that pushes
 * the current `RenderingSettings` object into the post-processing
 * pipeline and scene manager.
 *
 * The helper is a flat list of delegating calls so we just stub
 * postProcessing / sceneManager / animationController and assert
 * each method gets the expected arguments.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { applyRenderingSettings } from '../../../../ui/rendering-controls/apply-settings';
import { TONE_MAPPING_BY_NAME } from '../../../../rendering/post-processing/tone-mapping';
import { config, type RenderingSettings } from '../../../../config';

function makeStubs() {
  const settings: RenderingSettings = {
    ...config.renderingControls.defaults,
    detectorNoiseEnabled: false,
  };
  const postProcessing = {
    setBloomEnabled: vi.fn(),
    setBloomLevels: vi.fn(),
    setSSAAEnabled: vi.fn(),
    setSSAAMultiplier: vi.fn(),
    setFXAAEnabled: vi.fn(),
    setMSAAEnabled: vi.fn(),
    setMSAASamples: vi.fn(),
    setToneMapping: vi.fn(),
    setDetectorNoiseEnabled: vi.fn(),
    setVignetteEnabled: vi.fn(),
    setChromaticLensDistortionEnabled: vi.fn(),
  };
  const sceneManager = {
    updateExposure: vi.fn(),
    updateGlobalOffset: vi.fn(),
    updateGlobalGamma: vi.fn(),
    setDynamicClipping: vi.fn(),
  };
  const animationController = { startAnimation: vi.fn() };
  const updateClippingControlsState = vi.fn();
  const triggerAnimation = vi.fn();

  return {
    settings,
    postProcessing,
    sceneManager,
    animationController,
    updateClippingControlsState,
    triggerAnimation,
  };
}

let stubs: ReturnType<typeof makeStubs>;

beforeEach(() => {
  stubs = makeStubs();
});

function run() {
  applyRenderingSettings({
    settings: stubs.settings,
    postProcessing: stubs.postProcessing as never,
    sceneManager: stubs.sceneManager as never,
    animationController: stubs.animationController as never,
    updateClippingControlsState: stubs.updateClippingControlsState,
    triggerAnimation: stubs.triggerAnimation,
  });
}

describe('applyRenderingSettings — bloom', () => {
  it('forwards bloom enabled + the four scalar parameters', () => {
    stubs.settings.bloomEnabled = true;
    stubs.settings.bloomStrength = 1.5;
    stubs.settings.bloomRadius = 0.4;
    stubs.settings.bloomThreshold = 0.8;
    stubs.settings.bloomLevels = 5;
    run();
    expect(stubs.postProcessing.setBloomEnabled).toHaveBeenCalledWith(true, 1.5, 0.4, 0.8);
    expect(stubs.postProcessing.setBloomLevels).toHaveBeenCalledWith(5);
  });
});

describe('applyRenderingSettings — global EOG', () => {
  it('routes exposure / offset / gamma through the scene manager', () => {
    stubs.settings.exposure = 1.2;
    stubs.settings.globalOffset = -0.05;
    stubs.settings.globalGamma = 2.2;
    run();
    expect(stubs.sceneManager.updateExposure).toHaveBeenCalledWith(1.2);
    expect(stubs.sceneManager.updateGlobalOffset).toHaveBeenCalledWith(-0.05);
    expect(stubs.sceneManager.updateGlobalGamma).toHaveBeenCalledWith(2.2);
  });
});

describe('applyRenderingSettings — anti-aliasing', () => {
  it('applies SSAA / FXAA / MSAA in order', () => {
    stubs.settings.ssaaEnabled = true;
    stubs.settings.ssaaMultiplier = 2;
    stubs.settings.fxaaEnabled = true;
    stubs.settings.msaaEnabled = true;
    stubs.settings.msaaSamples = 4;
    run();
    expect(stubs.postProcessing.setSSAAEnabled).toHaveBeenCalledWith(true);
    expect(stubs.postProcessing.setSSAAMultiplier).toHaveBeenCalledWith(2);
    expect(stubs.postProcessing.setFXAAEnabled).toHaveBeenCalledWith(true);
    expect(stubs.postProcessing.setMSAAEnabled).toHaveBeenCalledWith(true);
    expect(stubs.postProcessing.setMSAASamples).toHaveBeenCalledWith(4);
  });
});

describe('applyRenderingSettings — tone mapping', () => {
  it('maps the string tone-mapping name to the THREE constant', () => {
    stubs.settings.toneMapping = 'ACES';
    run();
    expect(stubs.postProcessing.setToneMapping).toHaveBeenCalledWith(TONE_MAPPING_BY_NAME['ACES']);
  });
});

describe('applyRenderingSettings — detector noise', () => {
  it('forwards the four detector-noise parameters', () => {
    stubs.settings.detectorNoiseEnabled = true;
    stubs.settings.detectorNoiseReadoutSigma = 0.001;
    stubs.settings.detectorNoisePhotonGain = 0.002;
    stubs.settings.detectorNoiseFpnSigma = 0.003;
    run();
    expect(stubs.postProcessing.setDetectorNoiseEnabled).toHaveBeenCalledWith(
      true,
      0.001,
      0.002,
      0.003
    );
  });

  it('starts the animation when detector noise is enabled (continuous render needed)', () => {
    stubs.settings.detectorNoiseEnabled = true;
    run();
    expect(stubs.animationController.startAnimation).toHaveBeenCalled();
  });

  it('does NOT start the animation when detector noise is disabled', () => {
    stubs.settings.detectorNoiseEnabled = false;
    run();
    expect(stubs.animationController.startAnimation).not.toHaveBeenCalled();
  });
});

describe('applyRenderingSettings — vignette', () => {
  it('forwards vignette enabled + darkness + offset', () => {
    stubs.settings.vignetteEnabled = true;
    stubs.settings.vignetteDarkness = 0.4;
    stubs.settings.vignetteOffset = 1.2;
    run();
    expect(stubs.postProcessing.setVignetteEnabled).toHaveBeenCalledWith(true, 0.4, 1.2);
  });
});

describe('applyRenderingSettings — chromatic lens distortion', () => {
  it('forwards all nine lens parameters', () => {
    stubs.settings.chromaticLensDistortionEnabled = true;
    stubs.settings.chromaticLensDistortionX = 0.1;
    stubs.settings.chromaticLensDistortionY = 0.05;
    stubs.settings.chromaticLensDispersion = 0.5;
    stubs.settings.chromaticLensPrincipalPointX = 0.5;
    stubs.settings.chromaticLensPrincipalPointY = 0.5;
    stubs.settings.chromaticLensFocalLengthX = 1.0;
    stubs.settings.chromaticLensFocalLengthY = 1.0;
    stubs.settings.chromaticLensSkew = 0.0;
    run();
    expect(stubs.postProcessing.setChromaticLensDistortionEnabled).toHaveBeenCalledWith(
      true,
      0.1,
      0.05,
      0.5,
      0.5,
      0.5,
      1.0,
      1.0,
      0.0
    );
  });
});

describe('applyRenderingSettings — dynamic clipping', () => {
  it('applies enabled state to scene manager AND notifies controls', () => {
    stubs.settings.dynamicClippingEnabled = true;
    run();
    expect(stubs.sceneManager.setDynamicClipping).toHaveBeenCalledWith(true);
    expect(stubs.updateClippingControlsState).toHaveBeenCalledWith(true);
  });
});

describe('applyRenderingSettings — finalize', () => {
  it('calls triggerAnimation as the last step', () => {
    run();
    expect(stubs.triggerAnimation).toHaveBeenCalledTimes(1);
  });

  it('does not crash when animationController is undefined', () => {
    expect(() =>
      applyRenderingSettings({
        settings: stubs.settings,
        postProcessing: stubs.postProcessing as never,
        sceneManager: stubs.sceneManager as never,
        animationController: undefined,
        updateClippingControlsState: stubs.updateClippingControlsState,
        triggerAnimation: stubs.triggerAnimation,
      })
    ).not.toThrow();
  });
});
