// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI from 'lil-gui';
import * as THREE from 'three';
import { PostProcessingManager } from '../rendering/post-processing';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { config, type RenderingSettings } from '../config';
import { SHADER_CONFIG } from '../rendering/shader-manager';

/**
 * RenderingControls manages the advanced rendering parameters GUI
 *
 * Features:
 * - Real-time control of bloom parameters
 * - HDR exposure and intensity controls
 * - Settings persistence per scene
 * - Clean, collapsible UI using lil-gui
 */
export class RenderingControls {
  /** The lil-gui instance */
  private gui: GUI;

  /** Current rendering settings */
  private settings: RenderingSettings;

  /** Scene identifier for settings persistence */
  private sceneId: string = '';

  /** Reference to post-processing manager */
  private postProcessing: PostProcessingManager;

  /** Reference to scene manager */
  private sceneManager: SceneManager;

  /** Reference to animation controller for triggering re-renders */
  private animationController?: AnimationController;

  /** Visibility state */
  private visible: boolean = false;

  constructor(postProcessing: PostProcessingManager, sceneManager: SceneManager) {
    this.postProcessing = postProcessing;
    this.sceneManager = sceneManager;
    this.settings = { ...config.renderingControls.defaults };

    // Initialize GUI
    this.gui = new GUI({
      title: 'Rendering Controls',
      width: 300,
      closeFolders: false,
    });

    // Position on the left side with standard margins
    this.gui.domElement.style.position = 'fixed';
    this.gui.domElement.style.top = '20px'; // Standard 20px margin
    this.gui.domElement.style.left = '20px'; // Standard 20px margin
    this.gui.domElement.style.zIndex = '1999'; // Below performance monitor (2000)

    // Start hidden
    this.gui.hide();

    // Apply custom styling to match other panels
    this.applyCustomStyling();

    this.setupControls();
  }

  /**
   * Setup all GUI controls
   */
  private setupControls(): void {
    // Bloom folder
    const bloomFolder = this.gui.addFolder('Bloom Effects');
    bloomFolder.open();

    const bloomThresholdControl = bloomFolder
      .add(this.settings, 'bloomThreshold', 0, 1, 0.01)
      .name('Threshold')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, undefined, value);
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'bloomStrength', 0, 2, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(value, undefined, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    bloomStrengthControl.domElement.setAttribute(
      'title',
      'Bloom Strength: Intensity of the glow effect\n' +
        '• 0 = no bloom, 1 = normal, 2 = intense glow\n' +
        '• Creates realistic light bleeding from bright areas'
    );

    const bloomRadiusControl = bloomFolder
      .add(this.settings, 'bloomRadius', 0, 1, 0.01)
      .name('Radius')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, value, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    bloomRadiusControl.domElement.setAttribute(
      'title',
      'Bloom Radius: Size of the glow spread\n' +
        '• 0 = tight glow, 1 = wide spread\n' +
        '• Larger radius = softer, more diffuse glow\n' +
        '• Affects computational cost'
    );

    // HDR/Exposure folder
    const hdrFolder = this.gui.addFolder('HDR & Exposure');
    hdrFolder.open();

    const exposureControl = hdrFolder
      .add(this.settings, 'exposure', 0.1, 3, 0.01)
      .name('Exposure')
      .onChange((value: number) => {
        this.postProcessing.updateExposure(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    exposureControl.domElement.setAttribute(
      'title',
      'Exposure: Controls overall image brightness (post-process)\n' +
        '• Acts like a camera exposure setting\n' +
        '• Applied AFTER HDR rendering during tone mapping\n' +
        '• 1.0 = neutral, <1.0 = darker, >1.0 = brighter\n' +
        '• Affects the entire image uniformly'
    );

    const hdrControl = hdrFolder
      .add(this.settings, 'hdrMultiplier', 1, 100, 0.1)
      .name('HDR Intensity')
      .onChange((value: number) => {
        // Update shader config and trigger material updates
        (SHADER_CONFIG.POINTS as any).hdrMultiplier = value;
        this.sceneManager.updateHDRMultiplier(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    hdrControl.domElement.setAttribute(
      'title',
      'HDR Intensity: Multiplies point light emission (pre-process)\n' +
        '• Controls how bright points can be in HDR space\n' +
        '• Applied DURING rendering before tone mapping\n' +
        '• Higher values = stronger glow/bloom effects\n' +
        '• Can create values >1.0 for realistic bright sources'
    );

    // Tone Mapping selector - moved to HDR & Exposure folder
    const toneMappingControl = hdrFolder
      .add(this.settings, 'toneMapping', ['None', 'Linear', 'Reinhard', 'Cineon', 'ACES', 'AgX', 'Neutral'])
      .name('Tone Mapping')
      .onChange((value: string) => {
        const toneMappingMap: { [key: string]: THREE.ToneMapping } = {
          'None': THREE.NoToneMapping,
          'Linear': THREE.LinearToneMapping,
          'Reinhard': THREE.ReinhardToneMapping,
          'Cineon': THREE.CineonToneMapping,
          'ACES': THREE.ACESFilmicToneMapping,
          'AgX': THREE.AgXToneMapping,
          'Neutral': THREE.NeutralToneMapping,
        };
        this.postProcessing.setToneMapping(toneMappingMap[value]);
        this.saveSettings();
        this.triggerAnimation();
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

    // Anti-aliasing folder
    const aaFolder = this.gui.addFolder('Anti-Aliasing');
    aaFolder.close(); // Collapsed by default

    // SSAA settings (collapsible) - First because it's the highest quality
    const ssaaFolder = aaFolder.addFolder('SSAA Settings (Supersampling)');

    aaFolder
      .add(this.settings, 'ssaaEnabled')
      .name('SSAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setSSAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'ssaaMultiplier', [1.5, 2.0, 3.0, 4.0])
      .name('Resolution Multiplier')
      .onChange((value: number) => {
        this.postProcessing.setSSAAMultiplier(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // FXAA toggle
    const fxaaControl = aaFolder
      .add(this.settings, 'fxaaEnabled')
      .name('FXAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setFXAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'msaaEnabled')
      .name('MSAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setMSAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'msaaSamples', [2, 4, 8])
      .name('Sample Count')
      .onChange((value: number) => {
        this.postProcessing.setMSAASamples(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // SMAA settings (collapsible)
    const smaaFolder = aaFolder.addFolder('SMAA Settings');

    const smaaControl = aaFolder
      .add(this.settings, 'smaaEnabled')
      .name('SMAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setSMAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
        // Show/hide SMAA settings folder
        if (value) {
          smaaFolder.show();
          smaaFolder.open();
        } else {
          smaaFolder.close();
          smaaFolder.hide();
        }
      });
    
    // Set tooltip for SMAA
    smaaControl.domElement.setAttribute(
      'title',
      'SMAA (Subpixel Morphological Anti-Aliasing)\n' +
        '• Advanced edge detection anti-aliasing\n' +
        '• Better quality than FXAA, faster than SSAA\n' +
        '• Preserves sharpness while smoothing edges\n' +
        '• Good balance of quality and performance'
    );

    smaaFolder
      .add(this.settings, 'smaaThreshold', 0.05, 0.2, 0.01)
      .name('Edge Threshold')
      .onChange((value: number) => {
        this.postProcessing.updateSMAASettings(value, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    smaaFolder
      .add(this.settings, 'smaaSearchSteps', [4, 8, 16, 32])
      .name('Search Steps')
      .onChange((value: number) => {
        this.postProcessing.updateSMAASettings(undefined, value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Initially show/hide folders based on settings
    if (this.settings.ssaaEnabled) {
      ssaaFolder.show();
      ssaaFolder.open();
    } else {
      ssaaFolder.hide();
    }

    if (!this.settings.msaaEnabled) {
      msaaFolder.hide();
    }

    if (!this.settings.smaaEnabled) {
      smaaFolder.hide();
    }

    // Post-Processing Effects folder
    const effectsFolder = this.gui.addFolder('Post-Processing Effects');
    effectsFolder.close(); // Closed by default

    // Depth of Field subfolder
    const dofFolder = effectsFolder.addFolder('Depth of Field');
    dofFolder.close();

    const dofEnabledControl = dofFolder
      .add(this.settings, 'dofEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setDOF(value, this.settings.dofFocus, this.settings.dofStrength);
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'dofFocus', 0.1, 100, 0.1)
      .name('Focus Distance')
      .onChange((value: number) => {
        // Always update and trigger animation so user can see changes immediately
        this.postProcessing.updateDOF({ focus: value });
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'dofStrength', 0, 1, 0.01)
      .name('Blur Strength')
      .onChange((value: number) => {
        // Always update and trigger animation so user can see changes immediately
        this.postProcessing.updateDOF({ strength: value });
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'chromaticAberrationEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setChromaticAberration(value, this.settings.chromaticAberrationStrength);
        this.saveSettings();
        this.triggerAnimation();
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
      .add(this.settings, 'chromaticAberrationStrength', 0, 1, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        // Always update the uniform, even if disabled (so it's ready when enabled)
        this.postProcessing.updateChromaticAberration(value);
        this.saveSettings();
        this.triggerAnimation();
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

    // Vignetting removed - effect was lame
  }

  /**
   * Apply custom styling to match dimension sliders and help panel
   */
  private applyCustomStyling(): void {
    const root = this.gui.domElement;

    // Style the main container
    root.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
    root.style.borderRadius = '8px';
    root.style.backdropFilter = 'blur(10px)';
    root.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
    root.style.fontFamily =
      '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
    root.style.fontSize = '12px';

    // Style the title
    const title = root.querySelector('.title') as HTMLElement;
    if (title) {
      title.style.backgroundColor = 'transparent';
      title.style.color = '#e0e0e0';
      title.style.fontSize = '14px';
      title.style.fontWeight = 'bold';
      title.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';
      title.style.paddingBottom = '6px';
      title.style.marginBottom = '10px';
    }

    // Override lil-gui's default styles with CSS
    const style = document.createElement('style');
    style.textContent = `
      .lil-gui {
        --background-color: rgba(30, 30, 30, 0.9) !important;
        --title-background-color: transparent !important;
        --title-text-color: #e0e0e0 !important;
        --widget-color: rgba(255, 255, 255, 0.1) !important;
        --hover-color: rgba(255, 255, 255, 0.15) !important;
        --focus-color: #4CAF50 !important;
        --number-color: #4CAF50 !important;
        --string-color: #4CAF50 !important;
        --font-size: 12px !important;
        --input-font-size: 12px !important;
        --folder-border-color: rgba(255, 255, 255, 0.2) !important;
        --checkbox-border-radius: 4px !important;
        color: #e0e0e0 !important;
      }
      
      .lil-gui .controller {
        border-radius: 4px !important;
        margin: 2px 0 !important;
      }
      
      .lil-gui .controller:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
      }
      
      .lil-gui .title {
        padding: 5px !important;
        border-radius: 4px 4px 0 0 !important;
      }
      
      .lil-gui button {
        border-radius: 4px !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
      }
      
      .lil-gui button:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
      }
      
      .lil-gui input[type="number"],
      .lil-gui input[type="text"] {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: none !important;
        border-radius: 4px !important;
        color: #4CAF50 !important;
        padding: 2px 6px !important;
      }
      
      .lil-gui select {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: none !important;
        border-radius: 4px !important;
        color: #e0e0e0 !important;
        padding: 2px 6px !important;
      }
      
      /* Remove borders from sliders too */
      .lil-gui .widget {
        border: none !important;
      }
      
      .lil-gui .controller.number .slider {
        background-color: rgba(255, 255, 255, 0.1) !important;
      }
      
      /* Remove all controller borders and outlines */
      .lil-gui .controller {
        border: none !important;
        outline: none !important;
      }
      
      .lil-gui .controller.number {
        border: none !important;
      }
      
      /* Clean folder styling */
      .lil-gui .children {
        border: none !important;
        margin-left: 20px !important;
      }
      
      .lil-gui .folder {
        border: none !important;
        margin-bottom: 2px !important;
      }
      
      /* Remove all borders from folder titles */
      .lil-gui .title {
        border: none !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
      }
      
      .lil-gui > .title {
        background-color: transparent !important;
        border: none !important;
      }
      
      .lil-gui .folder > .title {
        border: none !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
      }
    `;

    // Only add style once
    if (!document.getElementById('lil-gui-custom-styles')) {
      style.id = 'lil-gui-custom-styles';
      document.head.appendChild(style);
    }
  }

  /**
   * Set the animation controller reference
   * @param animationController - The animation controller instance
   */
  setAnimationController(animationController: AnimationController): void {
    this.animationController = animationController;
  }

  /**
   * Set the scene identifier for settings persistence
   * @param zarrUrl - URL of the zarr store
   * @param sceneName - Name of the scene
   */
  setSceneId(zarrUrl: string, sceneName?: string): void {
    // Generate a unique ID from URL and scene name
    const baseId = zarrUrl.replace(/[^a-zA-Z0-9]/g, '_');
    this.sceneId = sceneName ? `${baseId}_${sceneName}` : baseId;

    // Load settings for this scene
    this.loadSettings();
  }

  /**
   * Trigger animation when parameters change
   */
  private triggerAnimation(): void {
    // Start animation to see changes immediately
    this.animationController?.startAnimation();
  }

  /**
   * Save current settings to localStorage
   */
  private saveSettings(): void {
    if (!this.sceneId) return;

    const key = `luxar-rendering-settings-${this.sceneId}`;
    localStorage.setItem(key, JSON.stringify(this.settings));
  }

  /**
   * Load settings from localStorage
   */
  private loadSettings(): void {
    if (!this.sceneId) return;

    const key = `luxar-rendering-settings-${this.sceneId}`;
    const stored = localStorage.getItem(key);

    if (stored) {
      try {
        const loadedSettings = JSON.parse(stored) as Partial<RenderingSettings>;

        // Merge with defaults to handle missing properties
        this.settings = { ...config.renderingControls.defaults, ...loadedSettings };

        // Apply loaded settings
        this.applySettings();

        // Update GUI to reflect loaded values
        this.gui.controllersRecursive().forEach((controller) => {
          controller.updateDisplay();
        });

        console.log(`Loaded rendering settings for scene: ${this.sceneId}`);
      } catch (e) {
        console.warn('Failed to load rendering settings:', e);
      }
    }
  }

  /**
   * Apply current settings to rendering pipeline
   */
  private applySettings(): void {
    // Apply bloom settings
    this.postProcessing.updateBloomSettings(
      this.settings.bloomStrength,
      this.settings.bloomRadius,
      this.settings.bloomThreshold
    );

    // Apply exposure
    this.postProcessing.updateExposure(this.settings.exposure);

    // Apply HDR multiplier
    (SHADER_CONFIG.POINTS as any).hdrMultiplier = this.settings.hdrMultiplier;
    this.sceneManager.updateHDRMultiplier(this.settings.hdrMultiplier);

    // Apply SSAA settings
    this.postProcessing.setSSAAEnabled(this.settings.ssaaEnabled);
    this.postProcessing.setSSAAMultiplier(this.settings.ssaaMultiplier);

    // Apply FXAA setting
    this.postProcessing.setFXAAEnabled(this.settings.fxaaEnabled);

    // Apply MSAA settings
    this.postProcessing.setMSAAEnabled(this.settings.msaaEnabled);
    this.postProcessing.setMSAASamples(this.settings.msaaSamples);

    // Apply SMAA settings
    this.postProcessing.setSMAAEnabled(this.settings.smaaEnabled);
    if (this.settings.smaaEnabled) {
      this.postProcessing.updateSMAASettings(
        this.settings.smaaThreshold,
        this.settings.smaaSearchSteps
      );
    }

    // Apply tone mapping
    const toneMappingMap: { [key: string]: THREE.ToneMapping } = {
      'None': THREE.NoToneMapping,
      'Linear': THREE.LinearToneMapping,
      'Reinhard': THREE.ReinhardToneMapping,
      'Cineon': THREE.CineonToneMapping,
      'ACES': THREE.ACESFilmicToneMapping,
      'AgX': THREE.AgXToneMapping,
      'Neutral': THREE.NeutralToneMapping,
    };
    this.postProcessing.setToneMapping(toneMappingMap[this.settings.toneMapping]);

    // Apply DOF settings
    this.postProcessing.setDOF(
      this.settings.dofEnabled,
      this.settings.dofFocus,
      this.settings.dofStrength
    );

    // Apply chromatic aberration
    this.postProcessing.setChromaticAberration(
      this.settings.chromaticAberrationEnabled,
      this.settings.chromaticAberrationStrength
    );

    // Vignetting removed - effect was lame
  }

  /**
   * Show the controls panel
   */
  show(): void {
    this.gui.show();
    this.visible = true;
  }

  /**
   * Hide the controls panel
   */
  hide(): void {
    this.gui.hide();
    this.visible = false;
  }

  /**
   * Toggle visibility
   */
  toggle(): void {
    if (this.visible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.gui.destroy();
  }
}
