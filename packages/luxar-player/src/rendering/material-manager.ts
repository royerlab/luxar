/**
 * Material Manager for Luxar
 *
 * This module manages all materials in the scene, providing caching,
 * global uniform updates, and support for multiple material types.
 * Currently supports point materials, with future support for lines, meshes, etc.
 */

import * as THREE from 'three';
import { PointMaterial } from './point-material';
import { log, Modules } from '../utils/log';

// Supported blending modes
export type BlendingMode = 'normal' | 'additive' | 'subtractive' | 'minimum' | 'maximum';

// Point material properties
export interface PointMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
}

/**
 * Manages all materials in the scene with caching and global updates.
 * Future: Will handle line, mesh, volume materials in addition to points.
 */
export class MaterialManager {
  private pointMaterialCache = new Map<string, PointMaterial>();
  private registeredMaterials = new Set<THREE.Material>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians
  private currentResolution = new THREE.Vector2(1, 1); // Minimal default

  /**
   * Get or create a point material with caching
   */
  getPointMaterial(props: PointMaterialProperties): PointMaterial {
    // Create cache key from properties
    const key = `point_${props.blendingMode}_${props.opacity.toFixed(2)}_${props.gamma.toFixed(2)}`;

    // Check cache first
    let material = this.pointMaterialCache.get(key);
    if (material) {
      return material;
    }

    // Create new PointMaterial instance
    material = new PointMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      blending: this.getThreeBlending(props.blendingMode),
      depthWrite: props.blendingMode === 'normal' && props.opacity >= 0.99,
    });

    // Set render order (renderOrder is a property of Object3D, not Material)
    // Store it in userData for later application to the Points object
    material.userData.renderOrder = this.getRenderOrder(props.blendingMode, props.opacity);

    // Mark this material as managed by MaterialManager
    material.userData.managedByMaterialManager = true;

    // Register for global updates
    this.registeredMaterials.add(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution);

    // Cache it
    this.pointMaterialCache.set(key, material);

    log.info(Modules.RENDERER, `Created point material: ${key}`);
    return material;
  }

  /**
   * Convert our blending mode to Three.js blending constant
   */
  private getThreeBlending(mode: BlendingMode): THREE.Blending {
    switch (mode) {
      case 'normal':
        return THREE.NormalBlending;
      case 'additive':
        return THREE.AdditiveBlending;
      case 'subtractive':
        return THREE.SubtractiveBlending;
      case 'minimum':
        // Three.js doesn't have minimum blending, use subtractive as approximation
        return THREE.SubtractiveBlending;
      case 'maximum':
        // Three.js doesn't have maximum blending, use additive as approximation
        return THREE.AdditiveBlending;
      default:
        log.warning(Modules.RENDERER, `Unknown blending mode: ${mode}, using normal`);
        return THREE.NormalBlending;
    }
  }

  /**
   * Determine render order based on blending mode and opacity
   */
  private getRenderOrder(mode: BlendingMode, opacity: number): number {
    // Opaque objects render first (order 0)
    if (mode === 'normal' && opacity >= 0.99) {
      return 0;
    }

    // Transparent and special blend modes render later
    // Higher values render later
    switch (mode) {
      case 'normal':
        return 100; // Transparent normal blending
      case 'subtractive':
        return 200; // Subtractive needs to see what's behind
      case 'additive':
        return 300; // Additive on top
      case 'minimum':
      case 'maximum':
        return 400; // Special modes last
      default:
        return 100;
    }
  }

  /**
   * Update HDR multiplier globally
   */
  updateHDRMultiplier(multiplier: number): void {
    this.registeredMaterials.forEach((material) => {
      if (material instanceof PointMaterial) {
        material.updateHDRMultiplier(multiplier);
      }
      // Future: handle other material types
    });
  }

  /**
   * Update camera parameters for all registered materials
   */
  updateCameraParams(fov: number, resolution: THREE.Vector2): void {
    // Store current values for future material creation
    this.currentFov = fov;
    this.currentResolution.copy(resolution);

    // Update all registered materials
    this.registeredMaterials.forEach((material) => {
      if (material instanceof PointMaterial) {
        material.updateCameraParams(fov, resolution);
      }
      // Future: handle other material types
    });
  }

  /**
   * Dispose all cached materials
   */
  dispose(): void {
    this.registeredMaterials.forEach((material) => {
      material.dispose();
    });
    this.pointMaterialCache.clear();
    this.registeredMaterials.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { pointMaterials: number; totalRegistered: number; keys: string[] } {
    return {
      pointMaterials: this.pointMaterialCache.size,
      totalRegistered: this.registeredMaterials.size,
      keys: Array.from(this.pointMaterialCache.keys()),
    };
  }
}

// Global material manager instance
export const materialManager = new MaterialManager();
