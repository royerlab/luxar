/**
 * Material Manager for Luxar
 *
 * This module manages all materials in the scene, providing caching,
 * global uniform updates, and support for multiple material types.
 * Supports point materials and line materials.
 */

import * as THREE from 'three';
import { PointMaterial } from './point-material';
import { LineMaterial } from './line-material';
import { log, Modules } from '../utils/log';
import { config } from '../config';

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

// Line material properties
export interface LineMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  hdrMultiplier?: number;
}

/**
 * Manages all materials in the scene with caching and global updates.
 * Supports points, lines, and future material types.
 */
export class MaterialManager {
  private pointMaterialCache = new Map<string, PointMaterial>();
  private lineMaterialCache = new Map<string, LineMaterial>();
  private registeredMaterials = new Set<THREE.Material>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default
  private currentHdrMultiplier = config.shader.points.hdrMultiplier; // Current HDR multiplier (default 1.0)

  /**
   * Get or create a point material with caching
   */
  getPointMaterial(props: PointMaterialProperties): PointMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    // This prevents floating-point precision issues while still grouping similar values
    // Clamp values to valid ranges to handle edge cases gracefully
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100); // 0-100 range
    const gammaBucket = Math.round(Math.max(0, Math.min(3, props.gamma)) * 10); // 0-30 range
    const radiusBucket = props.radiusScale
      ? Math.round(Math.max(0, props.radiusScale) * 1000)
      : 1000;
    const sharpnessBucket = props.sharpnessScale
      ? Math.round(Math.max(0, props.sharpnessScale) * 1000)
      : 1000;

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

    // Update with current HDR multiplier (critical for settings loaded before material creation)
    material.updateHDRMultiplier(this.currentHdrMultiplier);

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
   * Get or create a line material with caching
   */
  getLineMaterial(props: LineMaterialProperties): LineMaterial {
    // Create cache key
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100);
    const hdrBucket = props.hdrMultiplier ? Math.round(Math.max(0, props.hdrMultiplier) * 10) : 160; // Default 16.0 → 160

    const key = `line_${props.blendingMode}_o${opacityBucket}_h${hdrBucket}`;

    // Check cache first
    let material = this.lineMaterialCache.get(key);
    if (material) {
      return material;
    }

    // Create new LineMaterial instance
    material = new LineMaterial({
      opacity: props.opacity,
      blendingMode: props.blendingMode,
      hdrMultiplier: props.hdrMultiplier ?? this.currentHdrMultiplier, // Use current if not provided
    });

    // Register for global updates
    this.registeredMaterials.add(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution);

    // Cache it
    this.lineMaterialCache.set(key, material);

    log.info(Modules.RENDERER, `Created line material: ${key}`);
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
   * Stores the value so new materials created after this call will use the updated multiplier
   */
  updateHDRMultiplier(multiplier: number): void {
    // Store current value for future material creation
    this.currentHdrMultiplier = multiplier;

    // Update all existing materials
    this.registeredMaterials.forEach((material) => {
      if (material instanceof PointMaterial) {
        material.updateHDRMultiplier(multiplier);
      } else if (material instanceof LineMaterial) {
        material.updateHDRMultiplier(multiplier);
      }
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

    // Also remove from cache based on material type
    if (material instanceof PointMaterial) {
      for (const [key, cachedMaterial] of this.pointMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.pointMaterialCache.delete(key);
          break;
        }
      }
    } else if (material instanceof LineMaterial) {
      for (const [key, cachedMaterial] of this.lineMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.lineMaterialCache.delete(key);
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
    this.lineMaterialCache.clear();
    this.registeredMaterials.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    pointMaterials: number;
    lineMaterials: number;
    totalRegistered: number;
    keys: string[];
    } {
    return {
      pointMaterials: this.pointMaterialCache.size,
      lineMaterials: this.lineMaterialCache.size,
      totalRegistered: this.registeredMaterials.size,
      keys: [
        ...Array.from(this.pointMaterialCache.keys()),
        ...Array.from(this.lineMaterialCache.keys()),
      ],
    };
  }
}

// Global material manager instance
export const materialManager = new MaterialManager();
