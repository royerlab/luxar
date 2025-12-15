/**
 * Post-processing effects controls setup for rendering controls UI.
 *
 * Creates controls for all visual effects:
 * - Bloom (glow/light bleeding)
 * - Detector Noise (physics-based: shot, readout, FPN)
 * - Depth of Field (bokeh blur)
 * - Chromatic Aberration (lens color fringing)
 * - Ambient Occlusion (contact shadows)
 * - Vignette (edge darkening)
 * - Lens Distortion (barrel/pincushion with full camera model)
 */

import type { SetupContext, SetupResult } from './types';

/**
 * Set up post-processing effects controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @param controllersRef - Reference to controllers object (needed for FOV preset lens distortion sync)
 * @returns Setup result with controller references
 */
export function setupPostProcessingControls(
  context: SetupContext,
  controllersRef: SetupResult['controllers']
): SetupResult {
  const { gui, settings, postProcessing, animationController, saveSettings, triggerAnimation } =
    context;

  const controllers: SetupResult['controllers'] = {};

    // Post-Processing Effects folder
    const effectsFolder = gui.addFolder('Post-Processing Effects');
    effectsFolder.close(); // Closed by default

    // Bloom subfolder - moved here from top level
    const bloomFolder = effectsFolder.addFolder('Bloom');
    bloomFolder.close(); // Closed by default like all other effects

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

    // Depth of Field subfolder
    const dofFolder = effectsFolder.addFolder('Depth of Field');
    dofFolder.close();

    const dofEnabledControl = dofFolder
      .add(settings, 'dofEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        postProcessing.setDOF(value, settings.dofFocus, settings.dofStrength);
        saveSettings();
        triggerAnimation();
      });

    // Set tooltip for DOF enabled
    dofEnabledControl.domElement.setAttribute(
      'title',
      'Depth of Field: Simulates camera focus\n' +
        '• Blurs objects outside the focal distance\n' +
        '• Creates cinematic depth effect\n' +
        '• Performance impact when enabled'
    );

    const dofFocusControl = dofFolder
      .add(settings, 'dofFocus', 0.1, 100, 0.1)
      .name('Focus Distance')
      .onChange((value: number) => {
        // Always update and trigger animation so user can see changes immediately
        postProcessing.updateDOF({ focus: value });
        saveSettings();
        triggerAnimation();
      });

    // Set tooltip for DOF focus
    dofFocusControl.domElement.setAttribute(
      'title',
      'Focus Distance: Distance to the sharp focal plane\n' +
        '• Objects at this distance will be sharp\n' +
        '• Objects closer or farther will be blurred\n' +
        '• Value in world units (adjust based on scene scale)'
    );

    const dofStrengthControl = dofFolder
      .add(settings, 'dofStrength', 0, 1, 0.01)
      .name('Blur Strength')
      .onChange((value: number) => {
        // Always update and trigger animation so user can see changes immediately
        postProcessing.updateDOF({ strength: value });
        saveSettings();
        triggerAnimation();
      });

    // Set tooltip for DOF strength
    dofStrengthControl.domElement.setAttribute(
      'title',
      'Blur Strength: Amount of out-of-focus blur\n' +
        '• 0 = No blur (everything in focus)\n' +
        '• 0.5 = Moderate blur\n' +
        '• 1.0 = Maximum blur\n' +
        '• Higher values create stronger bokeh effect'
    );

    // Chromatic Aberration subfolder
    const chromaticFolder = effectsFolder.addFolder('Chromatic Aberration');
    chromaticFolder.close();

    const chromaticEnabledControl = chromaticFolder
      .add(settings, 'chromaticAberrationEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        postProcessing.setChromaticAberration(
          value,
          settings.chromaticAberrationStrength
        );
        saveSettings();
        triggerAnimation();
      });

    // Set tooltip for chromatic aberration enabled
    chromaticEnabledControl.domElement.setAttribute(
      'title',
      'Chromatic Aberration: Simulates lens color fringing\n' +
        '• Separates RGB channels slightly\n' +
        '• Creates rainbow edges on high contrast areas\n' +
        '• Adds cinematic/stylistic effect'
    );

    const chromaticStrengthControl = chromaticFolder
      .add(settings, 'chromaticAberrationStrength', 0, 1, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        // Always update the uniform, even if disabled (so it's ready when enabled)
        postProcessing.updateChromaticAberration(value);
        saveSettings();
        triggerAnimation();
      });

    // Set tooltip for chromatic aberration strength
    chromaticStrengthControl.domElement.setAttribute(
      'title',
      'Chromatic Aberration Strength\n' +
        '• 0 = No color separation\n' +
        '• 0.15 = Subtle effect (default)\n' +
        '• 0.5 = Moderate color fringing\n' +
        '• 1.0 = Strong rainbow edges'
    );

    // Ambient Occlusion subfolder
    const aoFolder = effectsFolder.addFolder('Ambient Occlusion');
    aoFolder.close();

    const aoEnabledControl = aoFolder
      .add(settings, 'aoEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        postProcessing.setAOEnabled(value, settings.aoQuality);
        saveSettings();
        triggerAnimation();
      });

    aoEnabledControl.domElement.setAttribute(
      'title',
      'Ambient Occlusion: Darkens corners and crevices\n' +
        '• Adds depth and realism to scene\n' +
        '• Simulates indirect shadows\n' +
        '• Performance impact scales with quality'
    );

    const aoQualityControl = aoFolder
      .add(settings, 'aoQuality', ['low', 'medium', 'high', 'ultra'])
      .name('Quality')
      .onChange((value: 'low' | 'medium' | 'high' | 'ultra') => {
        if (settings.aoEnabled) {
          postProcessing.setAOEnabled(true, value);
          saveSettings();
          triggerAnimation();
        }
      });

    aoQualityControl.domElement.setAttribute(
      'title',
      'AO Quality Level:\n' +
        '• Low: Fast, lower quality (4 samples)\n' +
        '• Medium: Balanced (8 samples)\n' +
        '• High: Better quality (16 samples)\n' +
        '• Ultra: Best quality, slower (32 samples)'
    );

    // Vignette subfolder
    const vignetteFolder = effectsFolder.addFolder('Vignette');
    vignetteFolder.close();

    const vignetteEnabledControl = vignetteFolder
      .add(settings, 'vignetteEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        postProcessing.setVignetteEnabled(
          value,
          settings.vignetteDarkness,
          settings.vignetteOffset
        );
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

    // Lens Distortion subfolder
    const lensDistortionFolder = effectsFolder.addFolder('Lens Distortion');
    lensDistortionFolder.close();

    const lensDistortionEnabledControl = lensDistortionFolder
      .add(settings, 'lensDistortionEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        postProcessing.setLensDistortionEnabled(
          value,
          settings.lensDistortionX,
          settings.lensDistortionY,
          settings.lensPrincipalPointX,
          settings.lensPrincipalPointY,
          settings.lensFocalLengthX,
          settings.lensFocalLengthY,
          settings.lensSkew
        );
        saveSettings();
        triggerAnimation();
      });

    lensDistortionEnabledControl.domElement.setAttribute(
      'title',
      'Lens Distortion: Simulates camera lens imperfections\n' +
        '• Barrel/pincushion distortion effects\n' +
        '• Principal point and focal length adjustment\n' +
        '• Skew correction for non-square pixels'
    );

    const lensDistortionXControl = lensDistortionFolder
      .add(settings, 'lensDistortionX', -1, 1, 0.001)
      .name('Distortion X')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ distortionX: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensDistortionX = lensDistortionXControl;

    lensDistortionXControl.domElement.setAttribute(
      'title',
      'Radial Distortion X:\n' +
        '• 0 = No distortion\n' +
        '• Negative = Barrel distortion (fish-eye)\n' +
        '• Positive = Pincushion distortion'
    );

    const lensDistortionYControl = lensDistortionFolder
      .add(settings, 'lensDistortionY', -1, 1, 0.001)
      .name('Distortion Y')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ distortionY: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensDistortionY = lensDistortionYControl;

    lensDistortionYControl.domElement.setAttribute(
      'title',
      'Radial Distortion Y:\n' +
        '• 0 = No distortion\n' +
        '• Negative = Barrel distortion (fish-eye)\n' +
        '• Positive = Pincushion distortion'
    );

    const lensPrincipalPointXControl = lensDistortionFolder
      .add(settings, 'lensPrincipalPointX', -1, 1, 0.001)
      .name('Principal Point X')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ principalPointX: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensPrincipalPointX = lensPrincipalPointXControl;

    lensPrincipalPointXControl.domElement.setAttribute(
      'title',
      'Principal Point X offset:\n' +
        '• 0 = Centered (default)\n' +
        '• Negative = Shift distortion center left\n' +
        '• Positive = Shift distortion center right'
    );

    const lensPrincipalPointYControl = lensDistortionFolder
      .add(settings, 'lensPrincipalPointY', -1, 1, 0.001)
      .name('Principal Point Y')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ principalPointY: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensPrincipalPointY = lensPrincipalPointYControl;

    lensPrincipalPointYControl.domElement.setAttribute(
      'title',
      'Principal Point Y offset:\n' +
        '• 0 = Centered (default)\n' +
        '• Negative = Shift distortion center up\n' +
        '• Positive = Shift distortion center down'
    );

    const lensFocalLengthXControl = lensDistortionFolder
      .add(settings, 'lensFocalLengthX', 0.1, 3, 0.001)
      .name('Focal Length X')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ focalLengthX: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensFocalLengthX = lensFocalLengthXControl;

    lensFocalLengthXControl.domElement.setAttribute(
      'title',
      'Focal Length X:\n' +
        '• 1 = Normal (default)\n' +
        '• < 1 = Wide angle effect\n' +
        '• > 1 = Telephoto effect'
    );

    const lensFocalLengthYControl = lensDistortionFolder
      .add(settings, 'lensFocalLengthY', 0.1, 3, 0.001)
      .name('Focal Length Y')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ focalLengthY: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensFocalLengthY = lensFocalLengthYControl;

    lensFocalLengthYControl.domElement.setAttribute(
      'title',
      'Focal Length Y:\n' +
        '• 1 = Normal (default)\n' +
        '• < 1 = Wide angle effect\n' +
        '• > 1 = Telephoto effect'
    );

    const lensSkewControl = lensDistortionFolder
      .add(settings, 'lensSkew', -0.1, 0.1, 0.001)
      .name('Skew')
      .onChange((value: number) => {
        postProcessing.updateLensDistortion({ skew: value });
        saveSettings();
        triggerAnimation();
      });

    // Store reference for updates
    controllersRef.lensSkew = lensSkewControl;

    lensSkewControl.domElement.setAttribute(
      'title',
      'Lens Skew (radians):\n' +
        '• 0 = No skew (default)\n' +
        '• Corrects for non-square pixels\n' +
        '• Usually very small values'
    );


  // Store all lens distortion controller references for FOV preset synchronization
  controllers.lensDistortionX = lensDistortionXControl;
  controllers.lensDistortionY = lensDistortionYControl;
  controllers.lensPrincipalPointX = lensPrincipalPointXControl;
  controllers.lensPrincipalPointY = lensPrincipalPointYControl;
  controllers.lensFocalLengthX = lensFocalLengthXControl;
  controllers.lensFocalLengthY = lensFocalLengthYControl;
  controllers.lensSkew = lensSkewControl;

  return {
    controllers,
  };
}
