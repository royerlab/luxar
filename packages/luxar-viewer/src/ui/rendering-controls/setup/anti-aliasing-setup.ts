/**
 * Anti-aliasing controls setup for rendering controls UI.
 *
 * Creates controls for the three AA techniques the post-processing
 * pipeline supports:
 * - SSAA (Supersampling) with resolution multiplier
 * - FXAA (Fast Approximate AA)
 * - MSAA (Multisample AA) with sample count
 */

import type { SetupContext, SetupResult } from '../types';
import { FOLDER_ICONS } from '../folder-icons';

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
  const aaFolder = gui.addFolder('Anti-Aliasing', FOLDER_ICONS.antiAliasing);
  aaFolder.close(); // Collapsed by default

  // Set tooltip on the folder title to guide users on AA choices
  aaFolder.domElement?.setAttribute(
    'title',
    'Anti-Aliasing: Smooths jagged edges in the rendered image\n\n' +
      'Different scenes benefit from different AA methods:\n' +
      '• Point clouds with fine detail → FXAA (fast, preserves detail)\n' +
      '• Dense scenes with overlapping points → MSAA (hardware-accelerated, sharp)\n' +
      '• Final renders or screenshots → SSAA (best quality, highest cost)\n\n' +
      'You can combine methods (e.g., MSAA + FXAA) but watch for diminishing\n' +
      'returns and increased GPU cost. Start with one and add more only if needed.'
  );

  let ssaaFolder: ReturnType<typeof aaFolder.addFolder>;
  const ssaaControl = aaFolder
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

  // SSAA settings (collapsible) - First because it's the highest quality
  ssaaFolder = aaFolder.addFolder('SSAA Settings (Supersampling)');

  ssaaFolder.domElement?.setAttribute(
    'title',
    'SSAA Settings: Configure the supersampling resolution multiplier\n' +
      '• Higher multiplier = better quality but heavier on the GPU'
  );

  // Set tooltip for SSAA
  ssaaControl.domElement.setAttribute(
    'title',
    'SSAA (Supersampling Anti-Aliasing)\n' +
      '• Renders the scene at a higher resolution, then downscales\n' +
      '• Highest quality AA — smooths edges, textures, and shading\n' +
      '• Significant performance cost (scales with multiplier)\n' +
      '• Works with all blending modes including additive\n' +
      '• Best used for final renders or screenshots'
  );

  const ssaaMultiplierControl = ssaaFolder
    .add(settings, 'ssaaMultiplier', [1.5, 2.0, 3.0, 4.0])
    .name('Resolution Multiplier')
    .onChange((value: number) => {
      postProcessing.setSSAAMultiplier(value);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for SSAA multiplier
  ssaaMultiplierControl.domElement.setAttribute(
    'title',
    'SSAA Resolution Multiplier: How many times larger to render\n' +
      '• 1.5× = Mild improvement, moderate cost\n' +
      '• 2.0× = Good quality (renders 4× more pixels)\n' +
      '• 3.0× = High quality (renders 9× more pixels)\n' +
      '• 4.0× = Maximum quality (renders 16× more pixels)\n' +
      '• Higher values use proportionally more GPU memory and time'
  );

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

  let msaaFolder: ReturnType<typeof aaFolder.addFolder>;
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

  // MSAA settings (collapsible)
  msaaFolder = aaFolder.addFolder('MSAA Settings');

  msaaFolder.domElement?.setAttribute(
    'title',
    'MSAA Settings: Configure the multisample count\n' +
      '• More samples = smoother edges but more GPU work'
  );

  // Set tooltip for MSAA
  msaaControl.domElement.setAttribute(
    'title',
    'MSAA (Multisample Anti-Aliasing)\n' +
      '• Hardware-accelerated anti-aliasing (runs on the GPU)\n' +
      '• Fast and sharp — great default choice for most scenes\n' +
      '• Smooths geometric edges without blurring the image\n' +
      '• Works well with additive blending'
  );

  const msaaSamplesControl = msaaFolder
    .add(settings, 'msaaSamples', [2, 4, 8])
    .name('Sample Count')
    .onChange((value: number) => {
      postProcessing.setMSAASamples(value);
      saveSettings();
      triggerAnimation();
    });

  // Set tooltip for MSAA sample count
  msaaSamplesControl.domElement.setAttribute(
    'title',
    'MSAA Sample Count: Number of samples per pixel\n' +
      '• 2 = Minimal smoothing, lowest cost\n' +
      '• 4 = Good balance (default for most GPUs)\n' +
      '• 8 = Best quality, highest cost\n' +
      '• More samples = smoother edges but more GPU work'
  );

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
