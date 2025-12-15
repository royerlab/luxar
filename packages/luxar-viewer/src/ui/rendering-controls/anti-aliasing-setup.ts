/**
 * Anti-aliasing controls setup for rendering controls UI.
 *
 * Creates controls for various AA techniques:
 * - SSAA (Supersampling) with resolution multiplier
 * - FXAA (Fast Approximate AA)
 * - MSAA (Multisample AA) with sample count ⚠️
 * - SMAA (Subpixel Morphological AA)
 */

import type { SetupContext, SetupResult } from './types';

/**
 * Set up anti-aliasing controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @returns Setup result with controller references
 */
export function setupAntiAliasingControls(context: SetupContext): SetupResult {
  const { gui, settings, postProcessing, saveSettings, triggerAnimation } = context;

  const controllers: SetupResult['controllers'] = {};

  // Anti-aliasing folder
  const aaFolder = gui.addFolder('Anti-Aliasing');
  aaFolder.close(); // Collapsed by default

  // SSAA settings (collapsible) - First because it's the highest quality
  const ssaaFolder = aaFolder.addFolder('SSAA Settings (Supersampling)');

  aaFolder
    .add(settings, 'ssaaEnabled')
    .name('SSAA Enabled')
    .onChange((value: boolean) => {
      postProcessing.setSSAAEnabled(value);
      saveSettings();
      triggerAnimation();
      // Show/hide SSAA settings folder
      if (value) {
        ssaaFolder.show();
        ssaaFolder.open();
      } else {
        ssaaFolder.close();
        ssaaFolder.hide();
      }
    });

  ssaaFolder
    .add(settings, 'ssaaMultiplier', [1.5, 2.0, 3.0, 4.0])
    .name('Resolution Multiplier')
    .onChange((value: number) => {
      postProcessing.setSSAAMultiplier(value);
      saveSettings();
      triggerAnimation();
    });

  // FXAA toggle
  const fxaaControl = aaFolder
    .add(settings, 'fxaaEnabled')
    .name('FXAA Enabled')
    .onChange((value: boolean) => {
      postProcessing.setFXAAEnabled(value);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for FXAA
  fxaaControl.domElement.setAttribute(
    'title',
    'FXAA (Fast Approximate Anti-Aliasing)\n' +
      '• Fast post-process anti-aliasing\n' +
      '• Good performance, decent quality\n' +
      '• May slightly blur the image\n' +
      '• Works well with additive blending'
  );

  // MSAA settings (collapsible)
  const msaaFolder = aaFolder.addFolder('MSAA Settings ⚠️');

  const msaaControl = aaFolder
    .add(settings, 'msaaEnabled')
    .name('MSAA Enabled')
    .onChange((value: boolean) => {
      postProcessing.setMSAAEnabled(value);
      saveSettings();
      triggerAnimation();
      // Show/hide MSAA settings folder
      if (value) {
        msaaFolder.show();
        msaaFolder.open();
      } else {
        msaaFolder.close();
        msaaFolder.hide();
      }
    });

  // Set tooltip for MSAA with warning
  msaaControl.domElement.setAttribute(
    'title',
    'MSAA (Multisample Anti-Aliasing) ⚠️\n' +
      '• Hardware-accelerated anti-aliasing\n' +
      '• WARNING: Causes brightness issues with additive blending\n' +
      '• Points will appear brighter with more samples\n' +
      '• Consider using FXAA or SMAA instead'
  );

  msaaFolder
    .add(settings, 'msaaSamples', [2, 4, 8])
    .name('Sample Count')
    .onChange((value: number) => {
      postProcessing.setMSAASamples(value);
      saveSettings();
      triggerAnimation();
    });

  // SMAA anti-aliasing (no subfolder needed - only on/off toggle)
  const smaaControl = aaFolder
    .add(settings, 'smaaEnabled')
    .name('SMAA Enabled')
    .onChange((value: boolean) => {
      postProcessing.setSMAAEnabled(value);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for SMAA
  smaaControl.domElement.setAttribute(
    'title',
    'SMAA (Subpixel Morphological Anti-Aliasing)\n' +
      '• Advanced edge detection anti-aliasing\n' +
      '• Better quality than FXAA, faster than SSAA\n' +
      '• Preserves sharpness while smoothing edges\n' +
      '• Good balance of quality and performance\n' +
      '• Uses HIGH preset (optimal quality/performance balance)'
  );

  // Note: SMAA threshold and search steps controls removed because pmndrs/postprocessing
  // SMAAEffect only supports preset-based configuration (LOW/MEDIUM/HIGH/ULTRA).
  // Fine-grained control is not available in the underlying library.

  // Initially show/hide folders based on settings
  if (settings.ssaaEnabled) {
    ssaaFolder.show();
    ssaaFolder.open();
  } else {
    ssaaFolder.hide();
  }

  if (!settings.msaaEnabled) {
    msaaFolder.hide();
  }

  return {
    controllers,
  };
}
