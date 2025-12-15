/**
 * HDR controls setup for rendering controls UI.
 *
 * Creates controls for HDR and tone mapping:
 * - HDR intensity (logarithmic slider for perceptual linearity)
 * - Tone mapping selector (None, Linear, Reinhard, Cineon, ACES, AgX, Neutral)
 */

import * as THREE from 'three';
import type { SetupContext, SetupResult } from './types';

/**
 * Set up HDR controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @param hdrLogValue - Shadow object for logarithmic HDR intensity slider
 * @returns Setup result with controller references and shadow object
 */
export function setupHDRControls(context: SetupContext, hdrLogValue: { log: number }): SetupResult {
  const { gui, settings, sceneManager, postProcessing, saveSettings, triggerAnimation } = context;

  const controllers: SetupResult['controllers'] = {};

  // HDR folder
  const hdrFolder = gui.addFolder('HDR');
  hdrFolder.open();

  // Logarithmic HDR intensity slider
  // Using shadow property pattern: slider controls log10(value), giving equal
  // distance for equal perceptual change (orders of magnitude)
  const logMin = Math.log10(0.01); // -2
  const logMax = Math.log10(100); // +2

  // Initialize shadow log value from current settings
  hdrLogValue.log = Math.log10(settings.hdrMultiplier);

  // Helper to format the actual intensity value for display
  const formatIntensity = (logValue: number): string => {
    const actual = Math.pow(10, logValue);
    if (actual >= 10) return actual.toFixed(0);
    if (actual >= 1) return actual.toFixed(1);
    if (actual >= 0.1) return actual.toFixed(2);
    return actual.toFixed(3);
  };

  const hdrControl = hdrFolder
    .add(hdrLogValue, 'log', logMin, logMax, 0.01)
    .name('Intensity')
    .onChange((logValue: number) => {
      // Convert log to actual value
      const actualValue = Math.pow(10, logValue);
      settings.hdrMultiplier = actualValue;
      // Update shader config and trigger material updates
      // HDR multiplier is now handled through material manager
      sceneManager.updateHDRMultiplier(actualValue);
      saveSettings();
      triggerAnimation();
    });

  // Override updateDisplay to show actual intensity value instead of log value
  // This is necessary because lil-gui doesn't support custom value formatters
  const originalUpdateDisplay = hdrControl.updateDisplay.bind(hdrControl);

  (hdrControl as any).updateDisplay = () => {
    originalUpdateDisplay();
    // After lil-gui updates the display, override the input value with formatted intensity

    const input = (hdrControl as any).$input as HTMLInputElement | undefined;
    if (input) {
      input.value = formatIntensity(hdrLogValue.log);
    }
    return hdrControl;
  };

  // Store controller reference for sync updates
  controllers.hdrMultiplier = hdrControl;

  // Initial display update to show actual value
  hdrControl.updateDisplay();

  // Set tooltip on the DOM element
  hdrControl.domElement.setAttribute(
    'title',
    'Intensity: Multiplies point brightness (logarithmic slider)\n' +
      '• Range: 0.01× (dim) to 100× (bright)\n' +
      '• 1.0 = neutral (no change)\n' +
      '• Higher values increase glow/bloom effects\n' +
      '• Logarithmic scale: equal slider distance = equal perceived change'
  );

  // Tone Mapping selector - moved to HDR folder
  const toneMappingControl = hdrFolder
    .add(settings, 'toneMapping', [
      'None',
      'Linear',
      'Reinhard',
      'Cineon',
      'ACES',
      'AgX',
      'Neutral',
    ])
    .name('Tone Mapping')
    .onChange((value: string) => {
      const toneMappingMap: { [key: string]: THREE.ToneMapping } = {
        None: THREE.NoToneMapping,
        Linear: THREE.LinearToneMapping,
        Reinhard: THREE.ReinhardToneMapping,
        Cineon: THREE.CineonToneMapping,
        ACES: THREE.ACESFilmicToneMapping,
        AgX: THREE.AgXToneMapping,
        Neutral: THREE.NeutralToneMapping,
      };
      postProcessing.setToneMapping(toneMappingMap[value]);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for tone mapping
  toneMappingControl.domElement.setAttribute(
    'title',
    'Tone Mapping: Converts HDR colors to display range\n' +
      '• None: No tone mapping (may clip bright values)\n' +
      '• Linear: Simple linear mapping\n' +
      '• Reinhard: Classic tone mapping operator\n' +
      '• Cineon: Film-like response curve\n' +
      '• ACES: Academy Color Encoding (film industry standard)\n' +
      '• AgX: Modern filmic mapping with good color preservation\n' +
      '• Neutral: Minimal color shift tone mapping'
  );

  return {
    controllers,
    shadowObjects: {
      hdrLogValue,
    },
  };
}
