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
export type BlendingMode = 'normal' | 'additive';

// Point material properties
export interface PointMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  radiusScale?: number; // Scale factor for radius normalization (e.g., 1/255 for uint8)
  sharpnessScale?: number; // Scale factor for sharpness normalization (e.g., 1/255 for uint8)
}

/**
 * Manages all materials in the scene with caching and global updates.
 * Future: Will handle line, mesh, volume materials in addition to points.
 */
export class MaterialManager {
  private pointMaterialCache = new Map<string, PointMaterial>();
  private registeredMaterials = new Set<THREE.Material>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default

  /**
   * Get or create a point material with caching
   */
  getPointMaterial(props: PointMaterialProperties): PointMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    // This prevents floating-point precision issues while still grouping similar values
    // Clamp values to valid ranges to handle edge cases gracefully
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100); // 0-100 range
    const gammaBucket = Math.round(Math.max(0, Math.min(3, props.gamma)) * 10); // 0-30 range
    const radiusBucket = props.radiusScale ? Math.round(Math.max(0, props.radiusScale) * 1000) : 1000;
    const sharpnessBucket = props.sharpnessScale ? Math.round(Math.max(0, props.sharpnessScale) * 1000) : 1000;

    const key = `point_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_r${radiusBucket}_s${sharpnessBucket}`;

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
      radiusScale: props.radiusScale,
      sharpnessScale: props.sharpnessScale,
    });

    // Register for global updates
    this.registeredMaterials.add(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution);

    // Debug log the camera params being set
    log.info(
      Modules.RENDERER,
      `Material camera params: FOV=${((this.currentFov * 180) / Math.PI).toFixed(1)}°, ` +
        `Resolution=${this.currentResolution.x}x${this.currentResolution.y}`
    );

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
      default:
        log.warning(Modules.RENDERER, `Unknown blending mode: ${mode}, using normal`);
        return THREE.NormalBlending;
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
      // Check if this material has updateCameraParams method
      if (
        'updateCameraParams' in material &&
        typeof (material as any).updateCameraParams === 'function'
      ) {
        (material as any).updateCameraParams(fov, resolution);
      }
    });
  }

  /**
   * Unregister a material from global updates
   * This should be called when a material is disposed to prevent memory leaks
   */
  unregister(material: THREE.Material): void {
    this.registeredMaterials.delete(material);

    // Also remove from cache if it's a point material
    if (material instanceof PointMaterial) {
      // Find and remove from cache
      for (const [key, cachedMaterial] of this.pointMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.pointMaterialCache.delete(key);
          break;
        }
      }
    }
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
