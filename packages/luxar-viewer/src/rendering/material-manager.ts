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
import { GSplatMaterial } from './gsplat-material';
import { log, Modules } from '../utils/log';

/**
 * Supported blending modes for materials.
 *
 * - 'normal': Standard alpha blending (semi-transparent)
 * - 'additive': Classic additive blending, ignores depth (renders on top of everything)
 * - 'max': Maximum of source and destination (brightest wins)
 * - 'opaque': Solid rendering with depth write (closest object wins)
 * - 'luminous': Same as additive visually, but respects depth occlusion (occluded by closer objects)
 *
 * Depth behavior:
 * - 'additive': depthTest=false, depthWrite=false (ignores depth entirely)
 * - 'luminous': depthTest=true, depthWrite=false (respects occlusion, doesn't occlude others)
 */
export type BlendingMode = 'normal' | 'additive' | 'max' | 'opaque' | 'luminous';

// Point material properties
export interface PointMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
  radiusScale?: number; // Scale factor for radius normalization (e.g., 1/255 for uint8)
  sharpnessScale?: number; // Scale factor for sharpness normalization (e.g., 1/255 for uint8)
}

// Line material properties
export interface LineMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
}

// GSplat material properties
export interface GSplatMaterialProperties {
  blendingMode: BlendingMode;
  opacity: number;
  gamma: number;
  intensity: number;
  offset: number;
  truncationRadius?: number; // Default 3.0
}

/**
 * Manages all materials in the scene with caching and global updates.
 * Supports points, lines, and future material types.
 */
export class MaterialManager {
  private pointMaterialCache = new Map<string, PointMaterial>();
  private lineMaterialCache = new Map<string, LineMaterial>();
  private gsplatMaterialCache = new Map<string, GSplatMaterial>();
  private registeredMaterials = new Set<THREE.Material>();
  private currentFov = (60 * Math.PI) / 180; // Current FOV in radians
  private currentResolution = new THREE.Vector2(1920, 1080); // Use reasonable default
  // Note: Global HDR multiplier has been replaced by exposure/offset/gamma in post-processing

  /**
   * Get or create a point material with caching
   */
  getPointMaterial(props: PointMaterialProperties): PointMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    // This prevents floating-point precision issues while still grouping similar values
    // Clamp values to valid ranges to handle edge cases gracefully
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100); // 0-100 range
    const gammaBucket = Math.round(Math.max(0, Math.min(10, props.gamma)) * 10); // 0-100 range
    const intensityBucket = Math.round(Math.max(0, Math.min(100, props.intensity)) * 10); // 0-1000 range
    const offsetBucket = Math.round((Math.max(-10, Math.min(10, props.offset)) + 10) * 10); // 0-200 range
    const radiusBucket = props.radiusScale
      ? Math.round(Math.max(0, props.radiusScale) * 1000)
      : 1000;
    const sharpnessBucket = props.sharpnessScale
      ? Math.round(Math.max(0, props.sharpnessScale) * 1000)
      : 1000;

    const key = `point_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_r${radiusBucket}_s${sharpnessBucket}`;

    // Check cache first
    let material = this.pointMaterialCache.get(key);
    if (material) {
      return material;
    }

    // Determine material properties based on mode
    const isOpaque = this.isOpaqueMode(props.blendingMode);
    const isAdditive = props.blendingMode === 'additive';

    // Create new PointMaterial instance
    material = new PointMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blending: this.getThreeBlending(props.blendingMode),
      // Opaque: write depth. All others: no depth write
      depthWrite: isOpaque || (props.blendingMode === 'normal' && props.opacity >= 0.99),
      // Additive ignores depth entirely (renders on top of everything)
      depthTest: !isAdditive,
      transparent: !isOpaque,
      radiusScale: props.radiusScale,
      sharpnessScale: props.sharpnessScale,
    });

    // Configure custom blending for max mode
    if (props.blendingMode === 'max') {
      material.blendEquation = THREE.MaxEquation; // Max(source, destination)
      material.blendSrc = THREE.OneFactor;
      material.blendDst = THREE.OneFactor;
    }

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
   * Get or create a line material with caching
   */
  getLineMaterial(props: LineMaterialProperties): LineMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100);
    const gammaBucket = Math.round(Math.max(0, Math.min(10, props.gamma)) * 10); // 0-100 range
    const intensityBucket = Math.round(Math.max(0, Math.min(100, props.intensity)) * 10);
    const offsetBucket = Math.round((Math.max(-10, Math.min(10, props.offset)) + 10) * 10);

    const key = `line_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}`;

    // Check cache first
    let material = this.lineMaterialCache.get(key);
    if (material) {
      return material;
    }

    // Create new LineMaterial instance
    material = new LineMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
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
   * Get or create a gsplat material with caching
   */
  getGSplatMaterial(props: GSplatMaterialProperties): GSplatMaterial {
    // Create cache key using integer bucketing for predictable caching behavior
    const opacityBucket = Math.round(Math.max(0, Math.min(1, props.opacity)) * 100);
    const gammaBucket = Math.round(Math.max(0, Math.min(10, props.gamma)) * 10); // 0-100 range
    const intensityBucket = Math.round(Math.max(0, Math.min(100, props.intensity)) * 10);
    const offsetBucket = Math.round((Math.max(-10, Math.min(10, props.offset)) + 10) * 10);
    const truncBucket = Math.round((props.truncationRadius ?? 3.0) * 10);

    const key = `gsplat_${props.blendingMode}_o${opacityBucket}_g${gammaBucket}_i${intensityBucket}_f${offsetBucket}_t${truncBucket}`;

    // Check cache first
    let material = this.gsplatMaterialCache.get(key);
    if (material) {
      return material;
    }

    // Create new GSplatMaterial instance
    material = new GSplatMaterial({
      opacity: props.opacity,
      gamma: props.gamma,
      intensity: props.intensity,
      offset: props.offset,
      blendingMode: props.blendingMode,
      truncationRadius: props.truncationRadius ?? 3.0,
    });

    // Register for global updates
    this.registeredMaterials.add(material);

    // Update with current camera params
    material.updateCameraParams(this.currentFov, this.currentResolution);

    // Cache it
    this.gsplatMaterialCache.set(key, material);

    log.info(Modules.RENDERER, `Created gsplat material: ${key}`);
    return material;
  }

  /**
   * Convert our blending mode to Three.js blending constant
   */
  private getThreeBlending(mode: BlendingMode): THREE.Blending {
    switch (mode) {
      case 'opaque':
        return THREE.NormalBlending; // Solid rendering
      case 'normal':
        return THREE.NormalBlending; // Semi-transparent alpha blending
      case 'additive':
      case 'luminous':
        // Both use AdditiveBlending (SrcAlpha, One) - same visual output
        // Difference is only in depthTest (additive=false, luminous=true)
        return THREE.AdditiveBlending;
      case 'max':
        return THREE.CustomBlending; // Max blending uses CustomBlending with MaxEquation
      default:
        log.warning(Modules.RENDERER, `Unknown blending mode: ${mode}, using normal`);
        return THREE.NormalBlending;
    }
  }

  /**
   * Check if a blending mode is opaque (solid rendering)
   */
  private isOpaqueMode(mode: BlendingMode): boolean {
    return mode === 'opaque';
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
    } else if (material instanceof GSplatMaterial) {
      for (const [key, cachedMaterial] of this.gsplatMaterialCache.entries()) {
        if (cachedMaterial === material) {
          this.gsplatMaterialCache.delete(key);
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
    this.gsplatMaterialCache.clear();
    this.registeredMaterials.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    pointMaterials: number;
    lineMaterials: number;
    gsplatMaterials: number;
    totalRegistered: number;
    keys: string[];
  } {
    return {
      pointMaterials: this.pointMaterialCache.size,
      lineMaterials: this.lineMaterialCache.size,
      gsplatMaterials: this.gsplatMaterialCache.size,
      totalRegistered: this.registeredMaterials.size,
      keys: [
        ...Array.from(this.pointMaterialCache.keys()),
        ...Array.from(this.lineMaterialCache.keys()),
        ...Array.from(this.gsplatMaterialCache.keys()),
      ],
    };
  }
}

// Global material manager instance
export const materialManager = new MaterialManager();
