/**
 * HDR controls setup for rendering controls UI.
 *
 * Creates controls for global Exposure-Offset-Gamma (EOG) and tone mapping:
 * - Exposure (log2 stops): global brightness in photography-standard units
 * - Offset: global additive brightness shift (lift/lower composited image)
 * - Gamma: global midtone curve adjustment
 * - Tone mapping selector (None, Linear, Reinhard, Cineon, ACES, AgX, Neutral)
 */

import * as THREE from 'three';
import type { SetupContext, SetupResult } from '../types';
import { FOLDER_ICONS } from '../folder-icons';

/**
 * Set up HDR controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @returns Setup result with controller references
 */
export function setupHDRControls(context: SetupContext): SetupResult {
  const { gui, settings, sceneManager, postProcessing, saveSettings, triggerAnimation } = context;

  const controllers: SetupResult['controllers'] = {};

  // HDR folder
  const hdrFolder = gui.addFolder('HDR', FOLDER_ICONS.hdr);
  hdrFolder.open();

  hdrFolder.domElement?.setAttribute(
    'title',
    'HDR (High Dynamic Range): Global color and brightness controls\n\n' +
      'These controls adjust how the final image looks, applied in order:\n' +
      '1. Exposure — scales brightness like a camera (in log2 stops)\n' +
      '2. Offset — adds/subtracts a flat brightness value\n' +
      '3. Gamma — reshapes the midtone curve (contrast)\n' +
      '4. Tone Mapping — compresses HDR values to fit the display'
  );

  // Exposure: log2 stops (-10 to +10)
  // 0 = neutral, +1 = 2x brighter, -1 = half brightness
  const exposureControl = hdrFolder
    .add(settings, 'exposure', -10.0, 10.0, 0.01)
    .name('Exposure')
    .onChange((value: number) => {
      sceneManager.updateExposure(value);
      saveSettings();
      triggerAnimation();
    });

  exposureControl.domElement.setAttribute(
    'title',
    'Exposure: Global brightness in log2 stops (photography standard)\n' +
      '• Range: -10 (very dim) to +10 (very bright)\n' +
      '• 0 = neutral (no change)\n' +
      '• +1 = 2× brighter, -1 = half brightness\n' +
      '• Applied before tone mapping in a single shader pass'
  );
  controllers.exposure = exposureControl;

  // Offset: additive shift (-1 to +1)
  const offsetControl = hdrFolder
    .add(settings, 'globalOffset', -1.0, 1.0, 0.001)
    .name('Offset')
    .onChange((value: number) => {
      sceneManager.updateGlobalOffset(value);
      saveSettings();
      triggerAnimation();
    });

  offsetControl.domElement.setAttribute(
    'title',
    'Offset: Global additive brightness shift\n' +
      '• Range: -1.0 to +1.0\n' +
      '• 0 = neutral (no change)\n' +
      '• Negative: darken the entire image\n' +
      '• Positive: brighten the entire image'
  );
  controllers.globalOffset = offsetControl;

  // Gamma: midtone curve (0.1 to 10.0)
  const gammaControl = hdrFolder
    .add(settings, 'globalGamma', 0.1, 10.0, 0.01)
    .name('Gamma')
    .onChange((value: number) => {
      sceneManager.updateGlobalGamma(value);
      saveSettings();
      triggerAnimation();
    });

  gammaControl.domElement.setAttribute(
    'title',
    'Gamma: Global midtone curve adjustment\n' +
      '• Range: 0.1 to 10.0\n' +
      '• 1.0 = linear (no change)\n' +
      '• < 1.0 = brighten midtones\n' +
      '• > 1.0 = darken midtones'
  );
  controllers.globalGamma = gammaControl;

  // Tone Mapping selector
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
      '• ACES: Academy Color Encoding, film industry standard (default; shifts hues)\n' +
      '• AgX: Modern filmic mapping with good color preservation\n' +
      '• Neutral: Minimal color shift — best for exact colormap-LUT fidelity'
  );

  return {
    controllers,
  };
}
