/**
 * Type definitions for pmndrs/postprocessing library
 * 
 * These interfaces provide proper typing for the postprocessing effects
 * to avoid unsafe 'any' casting and improve type safety.
 */

import { BloomEffect, ToneMappingEffect, DepthOfFieldEffect, ChromaticAberrationEffect, VignetteEffect } from 'postprocessing';
import * as THREE from 'three';

/**
 * Extended BloomEffect type with proper typing for internal properties
 * Note: BloomEffect from pmndrs has a specific structure:
 * - intensity is a direct property
 * - radius is accessed via mipmapBlurPass.radius
 * - luminanceThreshold is accessed via luminanceMaterial.threshold
 * 
 * We use a type alias instead of interface to avoid strict structural typing
 */
export type BloomEffectTyped = BloomEffect & {
  /** Direct access to intensity property */
  intensity: number;
}

/**
 * ToneMapping uniforms interface
 */
export interface ToneMappingUniforms {
  whitePoint?: { value: number };
  middleGrey?: { value: number };
  averageLuminance?: { value: number };
}

/**
 * Extended ToneMappingEffect interface with uniforms
 * We use intersection type to add our uniforms without conflicting with base type
 */
export type ToneMappingEffectTyped = ToneMappingEffect & {
  uniforms?: ToneMappingUniforms;
};

/**
 * DOF material uniforms interface
 */
export interface DOFUniforms {
  focusDistance?: { value: number };
  focalLength?: { value: number };
  bokehScale?: { value: number };
}

/**
 * Extended DepthOfFieldEffect interface
 * We use intersection type to add our properties without conflicting with base type
 */
export type DepthOfFieldEffectTyped = DepthOfFieldEffect & {
  bokehScale: number;
  circleOfConfusionMaterial?: {
    uniforms?: DOFUniforms;
  };
};

/**
 * Extended ChromaticAberrationEffect interface
 */
export interface ChromaticAberrationEffectTyped extends ChromaticAberrationEffect {
  offset: THREE.Vector2;
}

/**
 * Extended VignetteEffect interface
 */
export interface VignetteEffectTyped extends VignetteEffect {
  darkness: number;
  offset: number;
}

/**
 * Depth mapping utilities for perspective cameras
 */
export class PerspectiveDepthMapper {
  /**
   * Converts world-space distance to normalized depth value (0-1)
   * using proper perspective projection mathematics.
   * 
   * @param distance - World-space distance from camera
   * @param near - Camera near plane
   * @param far - Camera far plane
   * @returns Normalized depth value between 0 and 1 where 0=near, 1=far
   */
  static worldToNormalizedDepth(distance: number, near: number, far: number): number {
    // Clamp distance to valid range
    const clampedDistance = Math.max(near, Math.min(far, distance));
    
    // Use inverse depth mapping for better precision distribution
    // This gives more precision to near objects (as GPU depth buffers do)
    // We map so that near=0 and far=1 for the normalized output
    const invNear = 1.0 / near;
    const invFar = 1.0 / far;
    const invDistance = 1.0 / clampedDistance;
    
    // Map inverse depth linearly between near and far
    // Note: invNear > invFar, so we reverse the mapping
    const normalizedInverseDepth = (invNear - invDistance) / (invNear - invFar);
    
    return normalizedInverseDepth;
  }
  
  /**
   * Converts normalized depth value (0-1) back to world-space distance
   * 
   * @param normalizedDepth - Normalized depth value (0-1) where 0=near, 1=far
   * @param near - Camera near plane
   * @param far - Camera far plane
   * @returns World-space distance from camera
   */
  static normalizedDepthToWorld(normalizedDepth: number, near: number, far: number): number {
    // Clamp normalized depth to valid range
    const clamped = Math.max(0, Math.min(1, normalizedDepth));
    
    // Reverse the inverse depth mapping
    const invNear = 1.0 / near;
    const invFar = 1.0 / far;
    
    // Calculate inverse distance (reversing the normalization)
    const invDistance = invNear - clamped * (invNear - invFar);
    
    // Return world distance
    return 1.0 / invDistance;
  }
  
  /**
   * Alternative logarithmic depth mapping for better precision
   * This provides more uniform precision across the depth range.
   * 
   * @param distance - World-space distance from camera
   * @param near - Camera near plane
   * @param far - Camera far plane
   * @returns Normalized logarithmic depth value
   */
  static worldToLogDepth(distance: number, near: number, far: number): number {
    const clampedDistance = Math.max(near, Math.min(far, distance));
    
    // Logarithmic mapping
    const logNear = Math.log(near);
    const logFar = Math.log(far);
    const logDistance = Math.log(clampedDistance);
    
    // Normalize to 0-1 range
    return (logDistance - logNear) / (logFar - logNear);
  }
  
  /**
   * Convert logarithmic depth back to world distance
   * 
   * @param logDepth - Normalized logarithmic depth (0-1)
   * @param near - Camera near plane
   * @param far - Camera far plane
   * @returns World-space distance
   */
  static logDepthToWorld(logDepth: number, near: number, far: number): number {
    const clamped = Math.max(0, Math.min(1, logDepth));
    
    const logNear = Math.log(near);
    const logFar = Math.log(far);
    
    // Inverse logarithmic mapping
    const logDistance = logNear + clamped * (logFar - logNear);
    
    return Math.exp(logDistance);
  }
}

/**
 * Type guard functions for safe type checking
 */
export function isBloomEffectTyped(effect: any): effect is BloomEffectTyped {
  // The BloomEffect from pmndrs has specific structure:
  // - intensity is a direct property
  // - radius is on mipmapBlurPass.radius
  // - threshold is on luminanceMaterial.threshold
  if (!effect) return false;
  
  try {
    // Check for the key identifying properties of BloomEffect
    const hasIntensity = typeof effect.intensity === 'number' || effect.intensity !== undefined;
    
    // Check for the sub-objects that contain other properties
    const hasMipmapBlurPass = effect.mipmapBlurPass !== undefined;
    const hasLuminanceMaterial = effect.luminanceMaterial !== undefined;
    
    // A valid bloom effect should have intensity and at least one of the sub-objects
    return hasIntensity && (hasMipmapBlurPass || hasLuminanceMaterial);
  } catch {
    // If accessing properties throws, it's not a valid bloom effect
    return false;
  }
}

export function isToneMappingEffectTyped(effect: any): effect is ToneMappingEffectTyped {
  return effect && 'mode' in effect;
}

export function isDepthOfFieldEffectTyped(effect: any): effect is DepthOfFieldEffectTyped {
  return effect && 'bokehScale' in effect;
}

export function isChromaticAberrationEffectTyped(effect: any): effect is ChromaticAberrationEffectTyped {
  return effect && 'offset' in effect;
}

export function isVignetteEffectTyped(effect: any): effect is VignetteEffectTyped {
  return effect && 'darkness' in effect && 'offset' in effect;
}