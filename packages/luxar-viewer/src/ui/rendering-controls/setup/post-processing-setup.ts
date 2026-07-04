/**
 * Post-processing effects controls setup for rendering controls UI.
 *
 * Creates controls for the visual effects the mega-shader pipeline
 * supports:
 * - Bloom (glow/light bleeding, separate pre-pass + mix in mega-shader)
 * - Detector Noise (physics-based: shot, readout, FPN)
 * - Chromatic Lens Distortion (barrel/pincushion + chromatic aberration
 *   via per-channel sampling at distorted UVs)
 * - Vignette (multiplicative edge darkening)
 */

import type { SetupContext, SetupResult } from '../types';
import { FOLDER_ICONS } from '../folder-icons';

/**
 * Set up post-processing effects controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @param _controllersRef - Reference to controllers object (needed for FOV preset lens distortion sync)
 * @returns Setup result with controller references
 */
export function setupPostProcessingControls(
  context: SetupContext,
  _controllersRef: SetupResult['controllers']
): SetupResult {
  const { gui, settings, postProcessing, animationController, saveSettings, triggerAnimation } =
    context;

  const controllers: SetupResult['controllers'] = {};

  // Post-Processing Effects folder
  const effectsFolder = gui.addFolder('Post-Processing', FOLDER_ICONS.postProcessing);
  effectsFolder.close(); // Closed by default

  effectsFolder.domElement?.setAttribute(
    'title',
    'Post-Processing: Visual effects applied after the scene is rendered\n\n' +
      'Each effect is independent and can be toggled on/off:\n' +
      '• Bloom — glow around bright areas (great for HDR data)\n' +
      '• Detector Noise — physics-based noise (shot, readout, FPN)\n' +
      '• Vignette — darkened edges for cinematic framing\n' +
      '• Chromatic Lens Distortion — barrel/pincushion + color fringing\n\n' +
      'Tip: Enable multiple effects for cinematic results, or use\n' +
      'Cinematic Mode (C key) to enable a curated preset.'
  );

  // Bloom subfolder - moved here from top level
  const bloomFolder = effectsFolder.addFolder('Bloom');
  bloomFolder.close(); // Closed by default like all other effects

  bloomFolder.domElement?.setAttribute(
    'title',
    'Bloom: Simulates light bleeding from bright areas\n\n' +
      'Creates a soft glow halo around pixels above the brightness threshold.\n' +
      'Works best with HDR color data where some points are much brighter.\n' +
      '• Threshold — which brightness level starts glowing\n' +
      '• Strength — how intense the glow appears\n' +
      '• Radius — how far the glow spreads\n' +
      '• Mipmap Levels — quality of the blur (more = smoother)'
  );

  const bloomEnabledControl = bloomFolder
    .add(settings, 'bloomEnabled')
    .name('Enabled')
    .onChange((value: boolean) => {
      postProcessing.setBloomEnabled(
        value,
        settings.bloomStrength,
        settings.bloomRadius,
        settings.bloomThreshold
      );
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for bloom enabled
  bloomEnabledControl.domElement.setAttribute(
    'title',
    'Bloom Effect: HDR glow/light bleeding\n' +
      '• Creates realistic light halo around bright areas\n' +
      '• Works best with HDR colors\n' +
      '• Disable for performance boost'
  );

  const bloomThresholdControl = bloomFolder
    .add(settings, 'bloomThreshold', 0, 1, 0.01)
    .name('Threshold')
    .onChange((value: number) => {
      postProcessing.updateBloomSettings(undefined, undefined, value);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip on the DOM element
  bloomThresholdControl.domElement.setAttribute(
    'title',
    'Bloom Threshold: Minimum brightness for bloom\n' +
      '• Only pixels brighter than this value will bloom\n' +
      '• 0 = everything blooms, 1 = only brightest areas bloom\n' +
      '• Use with HDR intensity for best results'
  );

  const bloomStrengthControl = bloomFolder
    .add(settings, 'bloomStrength', 0, 2, 0.01)
    .name('Strength')
    .onChange((value: number) => {
      postProcessing.updateBloomSettings(value, undefined, undefined);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip on the DOM element
  bloomStrengthControl.domElement.setAttribute(
    'title',
    'Bloom Strength: Intensity of the glow effect\n' +
      '• 0 = no bloom, 1 = normal, 2 = intense glow\n' +
      '• Creates realistic light bleeding from bright areas'
  );

  const bloomRadiusControl = bloomFolder
    .add(settings, 'bloomRadius', 0, 1, 0.01)
    .name('Radius')
    .onChange((value: number) => {
      postProcessing.updateBloomSettings(undefined, value, undefined);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip on the DOM element
  bloomRadiusControl.domElement.setAttribute(
    'title',
    'Bloom Radius: Size of the glow spread\n' +
      '• 0 = tight glow, 1 = wide spread\n' +
      '• Larger radius = softer, more diffuse glow\n' +
      '• Affects computational cost'
  );

  // Bloom levels control (mipmap blur levels)
  const bloomLevelsControl = bloomFolder
    .add(settings, 'bloomLevels', 1, 12, 1)
    .name('Mipmap Levels')
    .onChange((value: number) => {
      postProcessing.setBloomLevels(Math.round(value));
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for bloom levels
  bloomLevelsControl.domElement.setAttribute(
    'title',
    'Bloom Mipmap Levels: Quality vs Performance\n' +
      '• 1-3 = Coarse bloom (fastest)\n' +
      '• 4-6 = Balanced quality\n' +
      '• 7-9 = Smooth bloom (default 8)\n' +
      '• 10-12 = Very smooth (slowest)'
  );

  // Detector Noise subfolder (physics-based: Shot + Readout + FPN)
  const detectorNoiseFolder = effectsFolder.addFolder('Detector Noise');
  detectorNoiseFolder.close();

  detectorNoiseFolder.domElement?.setAttribute(
    'title',
    'Detector Noise: Physics-based noise simulating a real camera sensor\n\n' +
      'Three independent noise sources, each modeling a real physical effect:\n' +
      '• Shot Noise — Poisson noise from photon statistics (signal-dependent)\n' +
      '• Readout Noise — Gaussian noise from electronics (temporal, per-frame)\n' +
      '• Fixed Pattern Noise — static per-pixel offset (sensor non-uniformity)\n\n' +
      'Useful for scientific visualization aesthetics or testing denoising.'
  );

  const detectorNoiseEnabledControl = detectorNoiseFolder
    .add(settings, 'detectorNoiseEnabled')
    .name('Enabled')
    .onChange((value: boolean) => {
      postProcessing.setDetectorNoiseEnabled(
        value,
        settings.detectorNoiseReadoutSigma,
        settings.detectorNoisePhotonGain,
        settings.detectorNoiseFpnSigma
      );
      saveSettings();
      // Keep animation running when detector noise is enabled
      if (value) {
        animationController?.startAnimation();
      }
      triggerAnimation();
    });

  // Set tooltip for detector noise enabled
  detectorNoiseEnabledControl.domElement.setAttribute(
    'title',
    'Detector Noise (Physics-Based):\n' +
      '• Shot noise: Poisson noise from photon statistics\n' +
      '• Readout noise: Temporal Gaussian noise from electronics\n' +
      '• Fixed Pattern Noise: Static per-pixel offset\n' +
      '• Ideal for scientific imaging aesthetics'
  );

  const detectorNoiseReadoutSigmaControl = detectorNoiseFolder
    .add(settings, 'detectorNoiseReadoutSigma', 0, 0.1, 0.001)
    .name('Readout Noise')
    .onChange((value: number) => {
      postProcessing.updateDetectorNoiseSettings({ readoutSigma: value });
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for readout sigma
  detectorNoiseReadoutSigmaControl.domElement.setAttribute(
    'title',
    'Readout Noise (temporal, varies each frame):\n' +
      '• Signal-independent electronic noise\n' +
      '• 0 = No readout noise\n' +
      '• 0.01 = Subtle (default)\n' +
      '• 0.05 = Moderate (old detector)\n' +
      '• 0.1 = High (uncooled sensor)'
  );

  const detectorNoisePhotonGainControl = detectorNoiseFolder
    .add(settings, 'detectorNoisePhotonGain', 0.0001, 0.1, 0.0001)
    .name('Shot Noise')
    .onChange((value: number) => {
      postProcessing.updateDetectorNoiseSettings({ photonGain: value });
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for photon gain
  detectorNoisePhotonGainControl.domElement.setAttribute(
    'title',
    'Shot Noise (Poisson, signal-dependent):\n' +
      '• Higher = more visible shot noise (fewer photons)\n' +
      '• 0.001 = Bright illumination (minimal shot noise)\n' +
      '• 0.01 = Normal conditions (default)\n' +
      '• 0.05 = Low light (visible shot noise)\n' +
      '• 0.1 = Very low light (strong shot noise)'
  );

  const detectorNoiseFpnControl = detectorNoiseFolder
    .add(settings, 'detectorNoiseFpnSigma', 0, 0.05, 0.001)
    .name('Fixed Pattern')
    .onChange((value: number) => {
      postProcessing.updateDetectorNoiseSettings({ fpnSigma: value });
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for FPN
  detectorNoiseFpnControl.domElement.setAttribute(
    'title',
    'Fixed Pattern Noise (static, constant per pixel):\n' +
      '• Per-pixel offset from detector non-uniformities\n' +
      '• 0 = No FPN (perfect detector)\n' +
      '• 0.005 = Subtle (default, good detector)\n' +
      '• 0.02 = Moderate (older detector)\n' +
      '• 0.05 = High (uncalibrated sensor)'
  );

  // Vignette subfolder
  const vignetteFolder = effectsFolder.addFolder('Vignette');
  vignetteFolder.close();

  vignetteFolder.domElement?.setAttribute(
    'title',
    'Vignette: Gradually darkens the edges and corners of the image\n\n' +
      'A classic photographic effect that draws the eye toward the center.\n' +
      '• Darkness — how dark the edges become\n' +
      '• Offset — how far from center the darkening starts\n' +
      '• Very low performance cost'
  );

  const vignetteEnabledControl = vignetteFolder
    .add(settings, 'vignetteEnabled')
    .name('Enabled')
    .onChange((value: boolean) => {
      postProcessing.setVignetteEnabled(value, settings.vignetteDarkness, settings.vignetteOffset);
      saveSettings();
      triggerAnimation();
    });

  vignetteEnabledControl.domElement.setAttribute(
    'title',
    'Vignette: Darkens edges of the screen\n' +
      '• Draws focus to center of view\n' +
      '• Creates cinematic look\n' +
      '• Minimal performance impact'
  );

  const vignetteDarknessControl = vignetteFolder
    .add(settings, 'vignetteDarkness', 0, 1, 0.01)
    .name('Darkness')
    .onChange((value: number) => {
      if (settings.vignetteEnabled) {
        postProcessing.setVignetteEnabled(true, value, settings.vignetteOffset);
        saveSettings();
        triggerAnimation();
      }
    });

  vignetteDarknessControl.domElement.setAttribute(
    'title',
    'Vignette Darkness:\n' +
      '• 0 = No darkening\n' +
      '• 0.5 = Moderate darkness (default)\n' +
      '• 1.0 = Maximum darkness'
  );

  const vignetteOffsetControl = vignetteFolder
    .add(settings, 'vignetteOffset', 0, 1, 0.01)
    .name('Offset')
    .onChange((value: number) => {
      if (settings.vignetteEnabled) {
        postProcessing.setVignetteEnabled(true, settings.vignetteDarkness, value);
        saveSettings();
        triggerAnimation();
      }
    });

  vignetteOffsetControl.domElement.setAttribute(
    'title',
    'Vignette Offset:\n' +
      '• 0 = Effect starts at center\n' +
      '• 0.5 = Effect starts mid-way (default)\n' +
      '• 1.0 = Effect only at very edges'
  );

  // Chromatic Lens Distortion subfolder (combined optical distortion effect)
  const chromaticLensDistortionFolder = effectsFolder.addFolder('Chromatic Lens Distortion');
  chromaticLensDistortionFolder.close();

  chromaticLensDistortionFolder.domElement?.setAttribute(
    'title',
    'Chromatic Lens Distortion: Simulates real camera lens imperfections\n\n' +
      'Combines two optical effects in a single efficient pass:\n' +
      '• Lens Distortion — barrel (wide-angle) or pincushion (telephoto) warping\n' +
      '• Chromatic Aberration — color fringing where R/G/B refract differently\n\n' +
      'FOV presets auto-apply matching distortion when this effect is enabled.\n' +
      'Advanced controls (principal point, focal length, skew) model a full\n' +
      'pinhole camera for physically accurate results.'
  );

  const chromaticLensDistortionEnabledControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensDistortionEnabled')
    .name('Enabled')
    .onChange((value: boolean) => {
      postProcessing.setChromaticLensDistortionEnabled(
        value,
        settings.chromaticLensDistortionX,
        settings.chromaticLensDistortionY,
        settings.chromaticLensDispersion,
        settings.chromaticLensPrincipalPointX,
        settings.chromaticLensPrincipalPointY,
        settings.chromaticLensFocalLengthX,
        settings.chromaticLensFocalLengthY,
        settings.chromaticLensSkew
      );
      saveSettings();
      triggerAnimation();
    });

  chromaticLensDistortionEnabledControl.domElement.setAttribute(
    'title',
    'Chromatic Lens Distortion: Combined effect simulating optical dispersion\n' +
      '• Physically accurate wavelength-dependent distortion\n' +
      '• Combines lens distortion + chromatic aberration\n' +
      '• More efficient than separate effects\n' +
      '• Color fringing follows lens geometry'
  );

  const chromaticLensDispersionControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensDispersion', 0, 0.5, 0.001)
    .name('Dispersion')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ dispersion: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensDispersionControl.domElement.setAttribute(
    'title',
    'Chromatic Dispersion: Wavelength-dependent distortion strength\n' +
      '• 0 = No chromatic effect (pure lens distortion)\n' +
      '• 0.03 = Subtle, realistic (default)\n' +
      '• 0.1 = Noticeable color fringing\n' +
      '• 0.2-0.5 = Strong stylized effect'
  );

  const chromaticLensDistortionXControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensDistortionX', -1, 1, 0.001)
    .name('Distortion X')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ distortionX: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensDistortionXControl.domElement.setAttribute(
    'title',
    'Radial Distortion X:\n' +
      '• 0 = No distortion\n' +
      '• Negative = Barrel distortion (wide angle)\n' +
      '• Positive = Pincushion distortion (telephoto)\n' +
      '• Chromatic fringing scales with distortion'
  );

  const chromaticLensDistortionYControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensDistortionY', -1, 1, 0.001)
    .name('Distortion Y')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ distortionY: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensDistortionYControl.domElement.setAttribute(
    'title',
    'Radial Distortion Y:\n' +
      '• 0 = No distortion\n' +
      '• Negative = Barrel distortion (wide angle)\n' +
      '• Positive = Pincushion distortion (telephoto)\n' +
      '• Usually same as Distortion X'
  );

  const chromaticLensPrincipalPointXControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensPrincipalPointX', -1, 1, 0.001)
    .name('Principal Point X')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ principalPointX: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensPrincipalPointXControl.domElement.setAttribute(
    'title',
    'Principal Point X offset:\n' +
      '• 0 = Centered (default)\n' +
      '• Shifts optical center horizontally\n' +
      '• Affects distortion center'
  );

  const chromaticLensPrincipalPointYControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensPrincipalPointY', -1, 1, 0.001)
    .name('Principal Point Y')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ principalPointY: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensPrincipalPointYControl.domElement.setAttribute(
    'title',
    'Principal Point Y offset:\n' +
      '• 0 = Centered (default)\n' +
      '• Shifts optical center vertically\n' +
      '• Affects distortion center'
  );

  const chromaticLensFocalLengthXControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensFocalLengthX', 0.1, 3, 0.001)
    .name('Focal Length X')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ focalLengthX: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensFocalLengthXControl.domElement.setAttribute(
    'title',
    'Focal Length X:\n' +
      '• 1 = Normal (default)\n' +
      '• < 1 = Wide angle effect\n' +
      '• > 1 = Telephoto effect'
  );

  const chromaticLensFocalLengthYControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensFocalLengthY', 0.1, 3, 0.001)
    .name('Focal Length Y')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ focalLengthY: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensFocalLengthYControl.domElement.setAttribute(
    'title',
    'Focal Length Y:\n' +
      '• 1 = Normal (default)\n' +
      '• < 1 = Wide angle effect\n' +
      '• > 1 = Telephoto effect'
  );

  const chromaticLensSkewControl = chromaticLensDistortionFolder
    .add(settings, 'chromaticLensSkew', -0.1, 0.1, 0.001)
    .name('Skew')
    .onChange((value: number) => {
      postProcessing.updateChromaticLensDistortion({ skew: value });
      saveSettings();
      triggerAnimation();
    });

  chromaticLensSkewControl.domElement.setAttribute(
    'title',
    'Lens Skew (radians):\n' +
      '• 0 = No skew (default)\n' +
      '• Corrects for non-square pixels\n' +
      '• Rare in modern systems'
  );

  // Store chromatic lens distortion controller references for FOV preset synchronization
  controllers.chromaticLensDistortionX = chromaticLensDistortionXControl;
  controllers.chromaticLensDistortionY = chromaticLensDistortionYControl;
  controllers.chromaticLensDispersion = chromaticLensDispersionControl;
  controllers.chromaticLensPrincipalPointX = chromaticLensPrincipalPointXControl;
  controllers.chromaticLensPrincipalPointY = chromaticLensPrincipalPointYControl;
  controllers.chromaticLensFocalLengthX = chromaticLensFocalLengthXControl;
  controllers.chromaticLensFocalLengthY = chromaticLensFocalLengthYControl;
  controllers.chromaticLensSkew = chromaticLensSkewControl;

  return {
    controllers,
  };
}
