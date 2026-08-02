/**
 * Pure helper that pushes the current `RenderingSettings` into the
 * post-processing pipeline and the scene manager.
 *
 * Called whenever settings are mutated wholesale (load from storage,
 * apply zarr defaults, reset to defaults). Has no opinion about
 * source — only about how each field maps to a manager call.
 */

import type { RenderingSettings } from '../../config';
import type { PostProcessingManager } from '../../rendering';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import { toneMappingFromName } from '../../rendering/post-processing/tone-mapping';

export interface ApplySettingsContext {
  settings: RenderingSettings;
  postProcessing: PostProcessingManager;
  sceneManager: SceneManager;
  animationController?: AnimationController;
  updateClippingControlsState: (dynamicEnabled: boolean) => void;
  triggerAnimation: () => void;
}

export function applyRenderingSettings(context: ApplySettingsContext): void {
  const {
    settings,
    postProcessing,
    sceneManager,
    animationController,
    updateClippingControlsState,
    triggerAnimation,
  } = context;

  // Bloom (enabled + parameters)
  postProcessing.setBloomEnabled(
    settings.bloomEnabled,
    settings.bloomStrength,
    settings.bloomRadius,
    settings.bloomThreshold
  );
  postProcessing.setBloomLevels(settings.bloomLevels);

  // Global EOG (Exposure / Offset / Gamma) — routed through scene manager.
  sceneManager.updateExposure(settings.exposure);
  sceneManager.updateGlobalOffset(settings.globalOffset);
  sceneManager.updateGlobalGamma(settings.globalGamma);

  // Anti-aliasing: SSAA, FXAA, MSAA.
  postProcessing.setSSAAEnabled(settings.ssaaEnabled);
  postProcessing.setSSAAMultiplier(settings.ssaaMultiplier);
  postProcessing.setFXAAEnabled(settings.fxaaEnabled);
  postProcessing.setMSAAEnabled(settings.msaaEnabled);
  postProcessing.setMSAASamples(settings.msaaSamples);

  postProcessing.setToneMapping(toneMappingFromName(settings.toneMapping));

  postProcessing.setDetectorNoiseEnabled(
    settings.detectorNoiseEnabled,
    settings.detectorNoiseReadoutSigma,
    settings.detectorNoisePhotonGain,
    settings.detectorNoiseFpnSigma
  );

  // Detector noise needs continuous rendering — kick the animation loop.
  if (settings.detectorNoiseEnabled) {
    animationController?.startAnimation();
  }

  postProcessing.setVignetteEnabled(
    settings.vignetteEnabled,
    settings.vignetteDarkness,
    settings.vignetteOffset
  );

  postProcessing.setChromaticLensDistortionEnabled(
    settings.chromaticLensDistortionEnabled,
    settings.chromaticLensDistortionX,
    settings.chromaticLensDistortionY,
    settings.chromaticLensDispersion,
    settings.chromaticLensPrincipalPointX,
    settings.chromaticLensPrincipalPointY,
    settings.chromaticLensFocalLengthX,
    settings.chromaticLensFocalLengthY,
    settings.chromaticLensSkew
  );

  sceneManager.setDynamicClipping(settings.dynamicClippingEnabled);

  updateClippingControlsState(settings.dynamicClippingEnabled);

  triggerAnimation();
}
