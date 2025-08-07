// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI from 'lil-gui';
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
    this.gui.domElement.style.top = '20px';  // Standard 20px margin
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

    bloomFolder
      .add(this.settings, 'bloomThreshold', 0, 1, 0.01)
      .name('Threshold')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, undefined, value);
        this.saveSettings();
        this.triggerAnimation();
      });

    bloomFolder
      .add(this.settings, 'bloomStrength', 0, 2, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(value, undefined, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    bloomFolder
      .add(this.settings, 'bloomRadius', 0, 1, 0.01)
      .name('Radius')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, value, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    // HDR/Exposure folder
    const hdrFolder = this.gui.addFolder('HDR & Exposure');
    hdrFolder.open();

    hdrFolder
      .add(this.settings, 'exposure', 0.1, 3, 0.01)
      .name('Exposure')
      .onChange((value: number) => {
        this.postProcessing.updateExposure(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    hdrFolder
      .add(this.settings, 'hdrMultiplier', 1, 20, 0.1)
      .name('HDR Intensity')
      .onChange((value: number) => {
        // Update shader config and trigger material updates
        (SHADER_CONFIG.POINTS as any).hdrMultiplier = value;
        this.sceneManager.updateHDRMultiplier(value);
        this.saveSettings();
        this.triggerAnimation();
      });

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
    aaFolder
      .add(this.settings, 'fxaaEnabled')
      .name('FXAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setFXAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // MSAA settings (collapsible)
    const msaaFolder = aaFolder.addFolder('MSAA Settings ⚠️');

    aaFolder
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

    aaFolder
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
    root.style.fontFamily = '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
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

        console.log(`✓ Loaded rendering settings for scene: ${this.sceneId}`);
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
