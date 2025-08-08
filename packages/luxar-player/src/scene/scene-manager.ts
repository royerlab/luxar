// Scene, renderer, and camera management for the Luxar scene player
//
// This module handles all Three.js setup and 3D graphics configuration:
// - WebGL renderer initialization with optimal settings
// - Camera setup with proper projection and positioning
// - ArcballControls for intuitive 3D navigation
// - Scene graph management and Zarr data loading
// - Resource disposal for memory management

import * as THREE from 'three';
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls';
import { loadScene } from '../data/zarr-loader';
import { showLoadingIndicator, hideLoadingIndicator, showError } from '../ui/helpers';
import { config } from '../config';
import { PostProcessingManager } from '../rendering/post-processing';
import { ShaderValidator } from '../rendering/shader-manager';
import { materialManager } from '../rendering/material-manager';

/**
 * SceneManager orchestrates all Three.js components for 3D rendering
 *
 * Responsibilities:
 * - WebGL renderer setup with HDR capabilities
 * - Camera configuration for optimal 3D viewing
 * - Control system for user interaction (rotation, zoom, pan)
 * - Scene graph management for 3D objects
 * - HDR post-processing pipeline with bloom effects
 * - Advanced shader-based point cloud rendering
 * - Dynamic loading of point cloud data from Zarr sources
 * - Resource cleanup to prevent memory leaks
 *
 * Technical Details:
 * - Uses perspective camera for realistic 3D projection
 * - ArcballControls provide constraint-based camera movement
 * - HDR post-processing with ACES tone mapping and bloom
 * - Custom Gaussian point shaders for enhanced visual quality
 * - Automatic canvas resizing for responsive design
 */
export class SceneManager {
  /** Three.js WebGL renderer - handles all GPU-accelerated rendering */
  public renderer!: THREE.WebGLRenderer;

  /** Three.js scene graph - container for all 3D objects and lights */
  public scene!: THREE.Scene;

  /** Perspective camera - provides realistic 3D viewing with depth */
  public camera!: THREE.PerspectiveCamera;

  /** ArcballControls - handles mouse/touch input for camera manipulation */
  public controls!: ArcballControls;

  /** HDR post-processing manager for bloom and tone mapping effects */
  public postProcessing!: PostProcessingManager;

  /** The HTML canvas element where 3D rendering occurs */
  private canvasElement!: HTMLCanvasElement;

  /** Track whether we're centered on bounding box or origin */
  private isCenteredOnBoundingBox: boolean = false;

  /**
   * Initialize the complete 3D scene setup
   */
  async init(): Promise<void> {
    this.setupCanvas();
    this.setupRenderer();
    this.setupScene();
    this.setupCamera();
    this.setupControls();
    this.setupPostProcessing();
  }

  /**
   * Get and validate the canvas element
   */
  private setupCanvas(): void {
    const element = document.getElementById(config.canvasId) as HTMLCanvasElement;
    if (!element) {
      showError(
        `Canvas element with id '${config.canvasId}' not found. Please check the HTML structure.`
      );
      throw new Error('Required canvas element not found');
    }
    this.canvasElement = element;
  }

  /**
   * Initialize Three.js WebGL renderer with optimal settings
   *
   * This creates the WebGL rendering context that will handle all GPU operations.
   * Key configurations:
   * - Antialiasing for smooth edges (MSAA)
   * - Custom canvas element for precise DOM control
   * - High DPI display support via pixel ratio
   * - Accessibility attributes for screen readers
   * - Fullscreen immersive experience
   */
  private setupRenderer(): void {
    // Create WebGL renderer with antialiasing enabled
    // Antialiasing uses MSAA (Multisample Anti-Aliasing) to smooth jagged edges
    // This is especially important for point clouds and wireframe objects
    this.renderer = new THREE.WebGLRenderer({
      antialias: true, // Enable MSAA for smoother rendering
      canvas: this.canvasElement, // Use our pre-existing canvas element
    });

    // Remove browser default focus outline
    this.renderer.domElement.style.outline = 'none';

    // Configure page for immersive fullscreen 3D experience
    // Remove default margins to eliminate whitespace around canvas
    document.body.style.margin = '0';

    // Hide scrollbars since 3D scene uses entire viewport
    document.body.style.overflow = 'hidden';

    // NOTE: We don't append renderer.domElement because we're using the existing HTML canvas
    // This allows for better integration into complex HTML pages

    // Configure renderer dimensions and high-DPI support
    this.updateRendererSize();
  }

  /**
   * Initialize the Three.js scene
   */
  private setupScene(): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(config.scene.backgroundColor);
  }

  /**
   * Initialize perspective camera with optimal 3D viewing parameters
   *
   * Perspective camera provides realistic 3D projection with depth perception.
   * Key parameters:
   * - FOV: Field of view angle (wider = more visible, narrower = more focused)
   * - Aspect ratio: Width/height ratio matching canvas dimensions
   * - Near/far planes: Define visible depth range (Z-clipping)
   * - Initial position: Starting camera location in 3D space
   */
  private setupCamera(): void {
    // Get canvas dimensions for proper aspect ratio
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    // Create perspective camera with realistic 3D projection
    // FOV of 60° provides natural human-like viewing angle
    this.camera = new THREE.PerspectiveCamera(
      config.camera.fov, // Field of view (60 degrees)
      width / height, // Aspect ratio (canvas width/height)
      config.camera.near, // Near clipping plane (0.1 units)
      config.camera.far // Far clipping plane (1000 units)
    );

    // Position camera at initial viewing location
    // Z=8 provides good overview of typical point cloud scenes
    // X=0, Y=0 centers the view on the origin
    this.camera.position.set(
      config.camera.initialPosition.x, // X position (0 = centered)
      config.camera.initialPosition.y, // Y position (0 = centered)
      config.camera.initialPosition.z // Z position (8 = pulled back for overview)
    );
  }

  /**
   * Initialize ArcballControls for intuitive 3D camera manipulation
   *
   * ArcballControls provide constraint-based camera movement that feels natural:
   * - Mouse drag: Rotate camera around target point (arcball rotation)
   * - Mouse wheel: Zoom in/out while maintaining focus point
   * - Right drag: Pan camera horizontally and vertically
   * - Automatic damping for smooth motion cessation
   *
   * This control scheme is ideal for examining 3D objects and point clouds
   * as it maintains spatial orientation and provides predictable movement.
   */
  private setupControls(): void {
    // ArcballControls bind to camera and DOM element for mouse/touch input
    // The renderer's canvas element captures all mouse/touch events
    // IMPORTANT: Pass scene as third parameter to enable proper gizmo management
    // Gizmos control the rotation center and prevent jumps during zoom
    this.controls = new ArcballControls(this.camera, this.renderer.domElement, this.scene);

    // Hide the gizmos - we want their functionality but not their visual representation
    // The gizmos still work internally to manage the rotation center properly
    (this.controls as any).setGizmoVisible?.(false);

    // Set default target to origin for predictable zooming behavior
    // Note: ArcballControls doesn't have full TypeScript definitions, so we use type assertion
    (this.controls as any).target.set(0, 0, 0);
    this.controls.update();

    // Save initial state so reset() works properly
    this.controls.saveState();
  }

  /**
   * Reset controls to default state
   * This is needed when loading a new dataset to prevent accumulated transformations
   */
  private resetControls(): void {
    // Reset camera to default position
    this.camera.position.set(
      config.camera.initialPosition.x,
      config.camera.initialPosition.y,
      config.camera.initialPosition.z
    );

    // Reset camera rotation to look at origin
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);

    // Reset controls target to origin
    (this.controls as any).target.set(0, 0, 0);

    // Clear any internal state by calling update multiple times
    // This ensures the controls fully sync with the new camera state
    this.controls.update();
    this.controls.update();

    // Save this configuration as the new default state
    // This is critical - it must happen AFTER all updates
    this.controls.saveState();

    console.log('✓ Controls reset to default state');
  }

  /**
   * Initialize HDR post-processing pipeline for advanced visual effects
   *
   * This creates a sophisticated rendering chain:
   * 1. Scene renders to HDR buffer (16-bit float precision)
   * 2. UnrealBloomPass creates glow effects on bright areas
   * 3. OutputPass applies ACES tone mapping and sRGB conversion
   *
   * The result is professional-quality rendering with realistic bloom
   * effects and proper color management for accurate display.
   */
  private setupPostProcessing(): void {
    // Get canvas dimensions for proper HDR render target sizing
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    // Create post-processing manager with actual canvas dimensions
    this.postProcessing = new PostProcessingManager(
      this.renderer,
      this.scene,
      this.camera,
      width,
      height
    );

    // Log shader configuration for debugging
    ShaderValidator.logShaderConfig();

    console.log('✓ HDR post-processing pipeline initialized');
  }

  /**
   * Load scene data from Zarr source
   */
  async loadSceneData(src: string): Promise<void> {
    showLoadingIndicator();

    try {
      // Clear existing scene content (keep lights and background)
      this.clearSceneContent();

      // Reset controls to default state before loading new content
      this.resetControls();

      const root = await loadScene(src);
      hideLoadingIndicator();
      this.scene.add(root);

      // Don't automatically center - let the scene designer's positioning take precedence
      // User can press 'C' to center on bounding box if desired
      console.log('Scene loaded. Press C to toggle centering on bounding box.');
    } catch (error) {
      hideLoadingIndicator();
      console.error('Failed to load scene:', error);
      showError(`Failed to load scene from "${src}". Please check the path and try again.`);
      throw error;
    }
  }

  /**
   * Clear all loaded content from the scene, keeping lights and background
   */
  private clearSceneContent(): void {
    // Helper function to recursively dispose of objects
    const disposeObject = (obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Points) {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      }

      // Recursively dispose children
      while (obj.children.length > 0) {
        disposeObject(obj.children[0]);
        obj.remove(obj.children[0]);
      }
    };

    // Find all objects to remove (direct children of scene)
    const objectsToRemove: THREE.Object3D[] = [];

    for (let i = this.scene.children.length - 1; i >= 0; i--) {
      const child = this.scene.children[i];

      // Keep lights and any background/environment objects
      if (child instanceof THREE.Light) continue;
      if (child.userData?.isBackground) continue;

      // Mark everything else for removal
      objectsToRemove.push(child);
    }

    // Remove and dispose marked objects
    for (const obj of objectsToRemove) {
      disposeObject(obj);
      this.scene.remove(obj);
    }

    console.log(`Cleared ${objectsToRemove.length} objects from scene`);
  }

  /**
   * Center camera on the bounding box of all visible objects
   */
  public centerCameraOnScene(): void {
    // Ensure world matrices are up to date before computing bounds
    this.scene.updateMatrixWorld(true);

    // Create a bounding box that encompasses all visible objects
    const box = new THREE.Box3();
    let totalPointCount = 0;

    // Traverse the scene and expand the box to include all geometries
    this.scene.traverse((object) => {
      if (object instanceof THREE.Points) {
        const geometry = object.geometry;

        // For point clouds, compute bounding box from position attribute
        const positions = geometry.attributes.position;
        if (positions && positions.count > 0) {
          totalPointCount += positions.count;

          // First, compute the bounding box
          if (!geometry.boundingBox) {
            geometry.computeBoundingBox();
          }

          if (geometry.boundingBox) {
            const tempBox = geometry.boundingBox.clone();

            // Apply object's world transform
            tempBox.applyMatrix4(object.matrixWorld);

            // Only include if box has valid size (not empty)
            if (!tempBox.isEmpty()) {
              box.union(tempBox);
            }
          }
        }
      }
    });

    // Only center camera if we have a reasonable scene
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);

      // Don't center if bounding box is too small or too few points
      // This prevents awkward camera positioning on edge cases
      if (maxDim < 1.0 || totalPointCount < 100) {
        // Keep default camera position for better user experience
        console.warn(
          `Scene too small for auto-centering (size: ${maxDim.toFixed(2)}, points: ${totalPointCount})`
        );
        return;
      }

      const center = box.getCenter(new THREE.Vector3());

      // Position camera to see the entire scene
      const distance = maxDim * 1.2; // Closer for better visibility
      this.camera.position.set(center.x, center.y, center.z + distance);

      // Point camera at the center
      this.camera.lookAt(center);
      this.camera.updateMatrixWorld(true);

      // Update controls to orbit around the center
      // Note: ArcballControls doesn't have full TypeScript definitions, so we use type assertion
      const controlsAny = this.controls as any;

      // Set the new target position
      controlsAny.target.copy(center);

      // CRITICAL: Force the controls to recalculate internal state after target change
      // ArcballControls maintains internal gizmos that need to be synchronized
      this.controls.update();

      // Reset the saved state to the current configuration
      // This prevents the "jump" on first zoom interaction by ensuring
      // the saved state matches the actual current state
      this.controls.reset();
      this.controls.saveState();

      console.log('✓ Controls target updated and state saved');

      console.log(
        `✓ Camera centered on scene (center: [${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}], distance: ${distance.toFixed(2)})`
      );
    } else {
      console.warn('No visible geometry found to center camera on');
    }
  }

  /**
   * Toggle between centering on origin (native) and bounding box center
   */
  public toggleCentering(): void {
    if (this.isCenteredOnBoundingBox) {
      // Switch to origin (native center)
      this.centerOnOrigin();
      this.isCenteredOnBoundingBox = false;
      console.log('✓ Centered on origin (native center)');
    } else {
      // Switch to bounding box center
      this.centerCameraOnScene();
      this.isCenteredOnBoundingBox = true;
      console.log('✓ Centered on bounding box');
    }
  }

  /**
   * Center camera and controls on the origin
   */
  private centerOnOrigin(): void {
    // Get current camera distance from target
    const currentDistance = this.camera.position.distanceTo((this.controls as any).target);

    // Reset target to origin
    const origin = new THREE.Vector3(0, 0, 0);

    // Position camera at same distance from origin
    this.camera.position.set(0, 0, currentDistance);
    this.camera.lookAt(origin);
    this.camera.updateMatrixWorld(true);

    // Update controls target
    (this.controls as any).target.copy(origin);
    this.controls.update();

    // Reset and save state to prevent jumps
    this.controls.reset();
    this.controls.saveState();

    console.log(`✓ Camera reset to origin with distance: ${currentDistance.toFixed(2)}`);
  }

  /**
   * Update renderer and camera for window resize
   */
  updateSize(): void {
    // In fullscreen, use screen dimensions; otherwise use canvas client dimensions
    let width: number;
    let height: number;

    if (document.fullscreenElement) {
      // Force fullscreen dimensions
      width = screen.width;
      height = screen.height;
      console.log(`✓ Using fullscreen dimensions: ${width}x${height}`);
    } else {
      // Use window dimensions for windowed mode - more reliable than canvas client dimensions
      width = window.innerWidth;
      height = window.innerHeight;
      console.log(`✓ Using windowed dimensions: ${width}x${height}`);
    }

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.updateRendererSize(width, height);

    // Update post-processing pipeline for new dimensions
    this.postProcessing.resize(width, height);
  }

  /**
   * Update renderer size and pixel ratio
   */
  private updateRendererSize(width?: number, height?: number): void {
    // Use provided dimensions or fall back to canvas client dimensions
    const canvas = this.renderer.domElement;
    const w = width || canvas.clientWidth || window.innerWidth;
    const h = height || canvas.clientHeight || window.innerHeight;

    // Let Three.js handle both WebGL buffer and canvas dimensions properly
    // This ensures coordinate system remains correct for mouse interactions
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    console.log(`✓ Renderer resized: ${w}x${h}`);
  }

  /**
   * Update camera FOV with bounds checking
   */
  updateFOV(deltaY: number): void {
    const fovChange = deltaY * config.camera.fovSensitivity;
    this.camera.fov = THREE.MathUtils.clamp(
      this.camera.fov + fovChange,
      config.camera.fovMin,
      config.camera.fovMax
    );
    this.camera.updateProjectionMatrix();
  }

  /**
   * Update HDR multiplier for all point materials in the scene
   *
   * @param multiplier - New HDR multiplier value (1.0 to 20.0)
   */
  updateHDRMultiplier(multiplier: number): void {
    // Update all materials in the material manager
    materialManager.updateHDRMultiplier(multiplier);

    // Also update any legacy materials not managed by the material manager
    this.scene.traverse((object) => {
      if (object instanceof THREE.Points) {
        const material = object.material as THREE.ShaderMaterial;
        if (
          material.uniforms &&
          material.uniforms.hdrMultiplier &&
          !material.userData.managedByMaterialManager
        ) {
          material.uniforms.hdrMultiplier.value = multiplier;
        }
      }
    });

    console.log(`✓ HDR multiplier updated for all point materials: ${multiplier}`);
  }

  /**
   * Clean up all Three.js resources to prevent memory leaks
   *
   * WebGL resources (textures, buffers, shaders) are not automatically
   * garbage collected and must be explicitly disposed. This method ensures
   * proper cleanup of all GPU resources:
   *
   * 1. Post-processing: Dispose HDR render targets and effect composer
   * 2. Controls: Remove event listeners and internal references
   * 3. Renderer: Clean up WebGL context and associated resources
   * 4. Scene objects: Dispose geometry buffers and material shaders
   * 5. Materials: Free texture memory and shader programs
   *
   * Critical for preventing memory leaks in long-running applications.
   */
  dispose(): void {
    // Dispose post-processing resources first
    // This includes HDR render targets, effect composer, and all passes
    this.postProcessing.dispose();

    // Dispose controls - removes all event listeners and internal references
    // This prevents memory leaks from mouse/touch event handlers
    this.controls.dispose();

    // Dispose material manager - cleans up all cached materials
    materialManager.dispose();

    // Dispose renderer - cleans up WebGL context and associated GPU resources
    // This frees vertex buffers, textures, and shader programs
    this.renderer.dispose();

    // Traverse scene graph and dispose all geometry and material resources
    // This is critical because WebGL resources are not garbage collected
    this.scene.traverse((object) => {
      if ('geometry' in object && 'material' in object) {
        const mesh = object as THREE.Mesh;

        // Dispose geometry - frees vertex and index buffers on GPU
        mesh.geometry.dispose();

        // Handle both single materials and material arrays
        if (mesh.material instanceof THREE.Material) {
          // Single material - dispose textures and shader programs
          mesh.material.dispose();
        } else if (Array.isArray(mesh.material)) {
          // Multiple materials - dispose each one individually
          mesh.material.forEach((material) => material.dispose());
        }
      }
    });
  }
}
