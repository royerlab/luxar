// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI from 'lil-gui';
import { PostProcessingManager } from './post-processing';
import { SceneManager } from './scene-manager';
import { AnimationController } from './animation-controller';
import { config, type RenderingSettings } from './config';
import { SHADER_CONFIG } from './shader-manager';

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
    
    // Position on the left side, below performance monitor
    this.gui.domElement.style.position = 'fixed';
    this.gui.domElement.style.top = '100px';
    this.gui.domElement.style.left = '10px';
    this.gui.domElement.style.zIndex = '1999'; // Below performance monitor (2000)
    
    // Start hidden
    this.gui.hide();
    
    this.setupControls();
  }
  
  /**
   * Setup all GUI controls
   */
  private setupControls(): void {
    // Bloom folder
    const bloomFolder = this.gui.addFolder('Bloom Effects');
    bloomFolder.open();
    
    bloomFolder.add(this.settings, 'bloomThreshold', 0, 1, 0.01)
      .name('Threshold')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, undefined, value);
        this.saveSettings();
        this.triggerAnimation();
      });
    
    bloomFolder.add(this.settings, 'bloomStrength', 0, 2, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(value, undefined, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });
    
    bloomFolder.add(this.settings, 'bloomRadius', 0, 1, 0.01)
      .name('Radius')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, value, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });
    
    // HDR/Exposure folder
    const hdrFolder = this.gui.addFolder('HDR & Exposure');
    hdrFolder.open();
    
    hdrFolder.add(this.settings, 'exposure', 0.1, 3, 0.01)
      .name('Tone Mapping Exposure')
      .onChange((value: number) => {
        this.postProcessing.updateExposure(value);
        this.saveSettings();
        this.triggerAnimation();
      });
    
    hdrFolder.add(this.settings, 'hdrMultiplier', 1, 20, 0.1)
      .name('HDR Intensity')
      .onChange((value: number) => {
        // Update shader config and trigger material updates
        (SHADER_CONFIG.POINTS as any).hdrMultiplier = value;
        this.sceneManager.updateHDRMultiplier(value);
        this.saveSettings();
        this.triggerAnimation();
      });
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
        this.gui.controllersRecursive().forEach(controller => {
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