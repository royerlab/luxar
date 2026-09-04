/**
 * ViewState Manager - Centralized ViewState initialization and validation
 *
 * This module handles all complex logic for initializing ViewState from scene dimensions,
 * including tolerance calculation, slice positioning, and dimension validation.
 *
 * Extracted from SceneLoader to improve testability and maintainability.
 */

import { ViewState } from './data-loader-types';
import type { DimensionMetadata } from '../types/dims';
import { log, Modules } from '../utils/log';
import { config } from '../config';

/**
 * Scene dimensions structure
 */
export interface SceneDimensions {
  dimensions: DimensionMetadata[];
}

// Re-export DimensionMetadata for convenience
export type { DimensionMetadata };

/**
 * Validation result
 */
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * ViewState manager - handles initialization and validation
 */
export class ViewStateManager {
  /**
   * Initialize ViewState from scene dimensions
   */
  static initializeFromDimensions(sceneDims: SceneDimensions): ViewState {
    // Validate input - cannot initialize with empty dimensions
    if (!sceneDims.dimensions || sceneDims.dimensions.length === 0) {
      throw new Error('[ViewState] Cannot initialize: dimensions array is empty or undefined');
    }

    const metadata = this.extractMetadata(sceneDims);
    const ndim = metadata.length;
    const displayed = this.findDisplayedDimensions(metadata);
    const slicePosition = this.calculateInitialSlice(metadata, displayed);
    const tolerance = this.buildToleranceArray(metadata, displayed);

    void ndim;
    return {
      displayDims: displayed,
      slicePosition,
      tolerance,
      dimensions: metadata,
    };
  }

  /**
   * Validate scene dimensions for consistency and correctness
   */
  static validateDimensions(dimensions: DimensionMetadata[]): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check for duplicate dimension names
    this.checkDuplicateNames(dimensions, warnings);

    // Check displayed dimension count
    this.checkDisplayCount(dimensions, warnings);

    // Validate each dimension
    dimensions.forEach((dim, index) => this.validateDimension(dim, index, errors, warnings));

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }

  // ========================================================================
  // PRIVATE HELPER METHODS
  // ========================================================================

  /**
   * Extract and normalize metadata from scene dimensions
   */
  private static extractMetadata(sceneDims: SceneDimensions): DimensionMetadata[] {
    type RawDim = Partial<DimensionMetadata> & { range?: unknown };
    return sceneDims.dimensions.map((dim: RawDim) => ({
      name: dim.name as string,
      unit: dim.unit as string,
      scale: dim.scale || 1.0, // Required, default to 1.0
      range: dim.range as [number, number] | undefined,
      display: dim.display,
      discrete: dim.discrete,
      // Same default as `SceneDimsManager.initFromScene` (`dim.step || 1.0`):
      // every consumer already treats a missing step as 1, and the two
      // builders must agree so `viewStatesEqual` / the S-cache key see the
      // eager load's state and the first slider state as the SAME state —
      // otherwise the post-load `updateAllNDNodes` re-streams every ladder.
      step: dim.step || 1.0,
      spatial: dim.spatial,
      cyclic: dim.cyclic,
      categories: dim.categories,
      description: dim.description,
    }));
  }

  /**
   * Find which dimensions should be displayed (max 3)
   */
  private static findDisplayedDimensions(metadata: DimensionMetadata[]): number[] {
    const displayed: number[] = [];

    for (let i = 0; i < metadata.length && displayed.length < 3; i++) {
      if (metadata[i].display === true) {
        displayed.push(i);
      }
    }

    return displayed;
  }

  /**
   * Calculate initial slice position for all dimensions.
   *
   * Policy (aligned with SceneDimsManager.initFromScene):
   * - Discrete/categorical non-displayed: start at minimum (first frame/category)
   * - Continuous non-displayed: start at center (no natural "first" position)
   * - Displayed: start at 0 (camera-controlled)
   */
  private static calculateInitialSlice(
    metadata: DimensionMetadata[],
    displayed: number[]
  ): number[] {
    const ndim = metadata.length;
    const currentStep = new Array(ndim).fill(0);

    for (let i = 0; i < ndim; i++) {
      const range = metadata[i].range;
      if (!displayed.includes(i) && range) {
        const [rangeMin, rangeMax] = range;

        if (metadata[i].discrete || metadata[i].categories) {
          // Discrete/categorical: start at minimum (first frame, first category)
          currentStep[i] = rangeMin;
        } else {
          // Continuous: start at center (no natural "first" position)
          currentStep[i] = (rangeMin + rangeMax) / 2;
        }
      }
    }

    return currentStep;
  }

  /**
   * Build tolerance array based on dimension types
   */
  private static buildToleranceArray(metadata: DimensionMetadata[], displayed: number[]): number[] {
    const ndim = metadata.length;
    const tolerance = new Array(ndim);

    for (let i = 0; i < ndim; i++) {
      if (displayed.includes(i)) {
        // Displayed dimensions don't need tolerance
        tolerance[i] = 0;
      } else if (metadata[i].discrete && metadata[i].spatial !== true) {
        // Non-spatial discrete axis (time, channel): exact matching. The slider
        // builder (`simpleDimsToViewState`) uses 0.5 here; `viewStatesEqual` and
        // the S-cache key both treat that pair as the same query. A discrete
        // SPATIAL axis falls through to the radius below, again matching the
        // slider builder's `maxRadius`, so the eager load and the first slider
        // update describe the same query and the post-load
        // `updateAllNDNodes` can be skipped instead of re-streaming every node.
        tolerance[i] = 0;
      } else {
        // Continuous non-displayed dimensions get default tolerance.
        // NOTE: this names a different constant than the navigation-time builder
        // (`simpleDimsToViewState`, which uses `maxRadius` here). No QUERY consumer
        // reads the continuous ride-along any more (issue #1183):
        // `fallbackQueryTolerance` used to take `tolerance[d] ?? maxRadius` and now
        // always uses the node's own `max_radius`, while
        // gsplats/lines/points-with-config recompute via `computeTolerance`. But the
        // MAGNITUDE is still read for a continuous dim by two non-query consumers —
        // `buildSliceViewSig` keys on the raw value, and `viewStatesEqual` skips a
        // tolerance difference only for a discrete non-spatial dim — so the two
        // builders are inert today only because they agree NUMERICALLY:
        // `scene.userData.maxRadius` is never set in production, so
        // `zarr-loader.ts` hands that builder
        // `config.dataLoading.spatial.defaultMaxRadius`, and both config values are
        // 0.1. A future divergence would cost SliceCache misses and
        // progressive-loader resets, not wrong data.
        tolerance[i] = config.dataLoading.spatial.defaultTolerance;
      }
    }

    return tolerance;
  }

  /**
   * Check for duplicate dimension names
   */
  private static checkDuplicateNames(dimensions: DimensionMetadata[], warnings: string[]): void {
    const names = dimensions.map((d) => d.name);
    const uniqueNames = new Set(names);

    if (names.length !== uniqueNames.size) {
      const duplicates = names.filter((name, index) => names.indexOf(name) !== index);
      warnings.push(
        `Duplicate dimension names found: ${duplicates.join(', ')}. This may cause unexpected behavior.`
      );
    }
  }

  /**
   * Check displayed dimension count
   */
  private static checkDisplayCount(dimensions: DimensionMetadata[], warnings: string[]): void {
    const displayedCount = dimensions.filter((d) => d.display === true).length;

    if (displayedCount > 3) {
      warnings.push(
        `Scene has ${displayedCount} displayed dimensions, but viewer can only show 3. ` +
          'Only the first 3 will be displayed.'
      );
    } else if (displayedCount === 0) {
      warnings.push(
        'Scene has no displayed dimensions. At least one dimension should be displayed.'
      );
    }
  }

  /**
   * Validate a single dimension
   */
  private static validateDimension(
    dim: DimensionMetadata,
    index: number,
    _errors: string[],
    warnings: string[]
  ): void {
    // Check required fields
    if (!dim.name) {
      warnings.push(`Dimension ${index} missing name field`);
    }

    // Validate range if present
    if (dim.range) {
      if (!Array.isArray(dim.range) || dim.range.length !== 2) {
        warnings.push(
          `Dimension '${dim.name}' has invalid range format: ${JSON.stringify(dim.range)}`
        );
      } else if ((dim.categories && dim.categories.length > 0) || dim.discrete) {
        // Discrete dims (a single index) and categorical dims (a single category)
        // may have a zero-width range [0, 0]: their auto-range collapses to [0, 0]
        // (mirrors Python dimensions.py). Only warn if the range is inverted.
        if (dim.range[0] > dim.range[1]) {
          warnings.push(
            `Dimension '${dim.name}' has an inverted range [${dim.range[0]}, ${dim.range[1]}]. Min should not be greater than max.`
          );
        }
      } else if (dim.range[0] >= dim.range[1]) {
        warnings.push(
          `Dimension '${dim.name}' has invalid range [${dim.range[0]}, ${dim.range[1]}]. Min should be < max.`
        );
      }
    }

    // Validate step if present
    if (dim.step !== undefined && dim.step !== null && dim.step <= 0) {
      warnings.push(`Dimension '${dim.name}' has invalid step: ${dim.step}. Step must be > 0.`);
    }

    // Warn about discrete dimensions without range
    if (dim.discrete === true && !dim.range) {
      warnings.push(
        `Discrete dimension '${dim.name}' should have a defined range for proper navigation.`
      );
    }
  }

  /**
   * Log validation results
   */
  static logValidationResults(
    validation: ValidationResult,
    ndim: number,
    displayedCount: number
  ): void {
    // Log errors
    validation.errors.forEach((error) => log.error(Modules.SCENE_LOADER, error));

    // Log warnings
    validation.warnings.forEach((warning) => log.warning(Modules.SCENE_LOADER, warning));

    // Log success if no errors
    if (validation.isValid) {
      log.info(
        Modules.SCENE_LOADER,
        `Scene dimensions validated: ${ndim} dimensions, ${displayedCount} displayed`
      );
    }
  }
}
