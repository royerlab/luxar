/**
 * GeometryUpdateManager - Handles GPU buffer management and geometry updates.
 *
 * This module is responsible for:
 * - Creating and updating THREE.js geometries for Points, Lines, and GSplats
 * - GPU buffer pool management for zero-allocation updates
 * - Worker-based projection offloading
 * - Data validation and transform application
 *
 * Extracted from SceneLoader to reduce its complexity (~500 lines extracted).
 *
 * @module data/geometry-update-manager
 */

import * as THREE from 'three';
import { LoadedPointsData } from './data-loader-types';
import type { LoadedLinesData, ProcessedLinesData } from '../types/lines';
import { isLinesUserData } from '../types/lines';
import type { LoadedGSplatsData, GSplatsViewState, GSplatsUserData } from '../types/gsplats';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { materialManager, BlendingMode } from '../rendering/material-manager';
import { updateInstancedLinesMesh } from '../rendering/line-material';
import { buildInstanceBuffers } from './lines-spatial-index-loader';
import { computeLinesTolerance } from './lines-chunk-spatial-index';
import { processGSplats } from './gsplats-processor';
import { updateInstancedGSplatsMesh, packCholeskyForShader } from '../rendering/gsplat-material';
import { getWorkerPool } from '../workers/worker-pool';
import { log, Modules } from '../utils/log';
import { config as appConfig } from '../config';
import type { UpdateSession } from '../profiling/update-profiler';
import type { PointsMetadata } from '../types/points';
import { isPointsUserData } from '../types/points';

/**
 * Configuration options for GeometryUpdateManager.
 * Allows dependency injection for better testability.
 */
export interface GeometryUpdateManagerConfig {
  /** Whether to use web workers for projection (default: from appConfig) */
  useWebWorkers?: boolean;
  /** Threshold for worker offload (default: 1000) */
  workerThreshold?: number;
}

/**
 * Manages GPU geometry creation, updates, and buffer pool operations.
 *
 * This class extracts geometry-related responsibilities from SceneLoader,
 * providing a focused interface for GPU memory management and geometry updates.
 *
 * Key features:
 * - GPU buffer pool integration for zero-allocation updates
 * - Worker-based projection for large datasets
 * - Data validation and error handling
 * - Support for Points, Lines, and GSplats geometry types
 *
 * @example
 * ```typescript
 * const manager = new GeometryUpdateManager(gpuBufferPool);
 *
 * // Update points geometry with new data
 * manager.updatePointsGeometry(rootGroup, '/points/cloud1', pointsData);
 *
 * // Update lines geometry with worker projection
 * await manager.updateLinesGeometry(rootGroup, '/lines/edges', linesData, viewState);
 * ```
 */
export class GeometryUpdateManager {
  private _gpuBufferPool: GPUBufferPool | null;
  private _config: GeometryUpdateManagerConfig;

  constructor(
    gpuBufferPool: GPUBufferPool | null = null,
    config: GeometryUpdateManagerConfig = {}
  ) {
    this._gpuBufferPool = gpuBufferPool;
    this._config = config;
  }

  /**
   * Check if web workers should be used for projection.
   * Uses injected config if provided, otherwise falls back to appConfig.
   */
  private shouldUseWorkers(): boolean {
    return this._config.useWebWorkers ?? appConfig.dataLoading.performance.useWebWorkers;
  }

  /**
   * Get the threshold for worker offload.
   */
  private getWorkerThreshold(): number {
    return this._config.workerThreshold ?? 1000;
  }

  /**
   * Get the GPU buffer pool instance (if enabled).
   */
  get gpuBufferPool(): GPUBufferPool | null {
    return this._gpuBufferPool;
  }

  /**
   * Dispose of resources.
   */
  dispose(): void {
    if (this._gpuBufferPool) {
      this._gpuBufferPool.dispose();
      this._gpuBufferPool = null;
    }
  }

  // ============================================================================
  // Points Geometry
  // ============================================================================

  /**
   * Create THREE.js geometry from points data.
   *
   * @param data - Points data with positions, colors, radii, sharpness
   * @param maxRadius - Maximum radius from node attributes for scaling uint8 radii
   * @param maxSharpness - Maximum sharpness from node attributes for scaling uint8 sharpness
   * @returns Configured BufferGeometry with all attributes
   */
  createPointsGeometry(
    data: LoadedPointsData,
    maxRadius: number = 1.0,
    maxSharpness: number = 31.0
  ): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();

    // VALIDATION: Check for edge cases and log detailed diagnostics
    this.validateLoadedPointsData(data);

    // Set positions (handle Float16Array conversion if needed)
    if (
      typeof (globalThis as any).Float16Array !== 'undefined' &&
      data.positions instanceof (globalThis as any).Float16Array
    ) {
      // Convert Float16Array to Float32Array for THREE.js compatibility
      const float32Positions = new Float32Array(data.positions);
      geometry.setAttribute('position', new THREE.BufferAttribute(float32Positions, 3));
    } else {
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(data.positions as Float32Array, 3)
      );
    }

    // Set colors if available
    if (data.colors) {
      // Validate color mode consistency
      this.validateColorMode(data.colors, data.metadata as any);

      // Check if colors need normalization (for uint8/uint16 arrays)
      const needsNormalization =
        data.colors instanceof Uint8Array || data.colors instanceof Uint16Array;

      geometry.setAttribute('color', new THREE.BufferAttribute(data.colors, 3, needsNormalization));
    }

    // Set radii if available, or use default
    let radiusScale = 1.0; // Default scale for float32 radii

    if (data.radii) {
      // Check if radii need normalization or conversion
      if (
        typeof (globalThis as any).Float16Array !== 'undefined' &&
        data.radii instanceof (globalThis as any).Float16Array
      ) {
        // Convert Float16Array to Float32Array for THREE.js
        const float32Radii = new Float32Array(data.radii);
        geometry.setAttribute('radius', new THREE.BufferAttribute(float32Radii, 1));
        // Float16 values are already in world units, no scaling needed
        radiusScale = 1.0;
      } else if (data.radii instanceof Uint8Array) {
        // Uint8 radii need scaling from 0-255 to 0-1 (or world units)
        // Use the normalization flag for proper GPU upload
        geometry.setAttribute(
          'radius',
          new THREE.BufferAttribute(data.radii, 1, true) // true = normalize on GPU
        );
        // GPU normalizes uint8 [0, 255] to [0, 1]
        // Python encodes radii with bounded_scalar_uint8: value in [0, max_radius]
        // After GPU normalization we get normalized values in [0, 1]
        // Multiply by maxRadius to get world-space radius
        radiusScale = maxRadius;
      } else {
        // Float32 radii - no normalization or scaling needed
        geometry.setAttribute(
          'radius',
          new THREE.BufferAttribute(data.radii as Float32Array, 1, false)
        );
        radiusScale = 1.0;
      }
    } else {
      // Create default radius array with value 0.5 for all points
      const numPoints = data.positions.length / 3;
      const defaultRadii = new Float32Array(numPoints).fill(0.5);
      geometry.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
      radiusScale = 1.0;
    }

    // Set sharpness if available, or use default
    let sharpnessScale = 1.0; // Default scale for float32 sharpness

    if (data.sharpness) {
      // Check if sharpness needs normalization or conversion
      if (
        typeof (globalThis as any).Float16Array !== 'undefined' &&
        data.sharpness instanceof (globalThis as any).Float16Array
      ) {
        // Convert Float16Array to Float32Array for THREE.js
        const float32Sharpness = new Float32Array(data.sharpness);
        geometry.setAttribute('sharpness', new THREE.BufferAttribute(float32Sharpness, 1));
        // Float16 values are already in world units, no scaling needed
        sharpnessScale = 1.0;
      } else if (data.sharpness instanceof Uint8Array) {
        // Uint8 sharpness needs scaling - check metadata for range
        // Use the normalization flag for proper GPU upload
        geometry.setAttribute(
          'sharpness',
          new THREE.BufferAttribute(data.sharpness, 1, true) // true = normalize on GPU
        );

        // GPU normalizes uint8 to [0,1], then scale to sharpness range
        // Use max_sharpness passed from node attributes
        sharpnessScale = maxSharpness;
      } else {
        // Float32 sharpness - no normalization or scaling needed
        geometry.setAttribute(
          'sharpness',
          new THREE.BufferAttribute(data.sharpness as Float32Array, 1, false)
        );
        sharpnessScale = 1.0;
      }
    } else {
      // Create default sharpness array with value 2.0 for all points
      const numPoints = data.positions.length / 3;
      const defaultSharpness = new Float32Array(numPoints).fill(2.0);
      geometry.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
      sharpnessScale = 1.0;
    }

    // Compute bounding box
    geometry.boundingBox = data.metadata.bounds.clone();

    // Store radius and sharpness scales as user data for material creation
    if (!geometry.userData) {
      geometry.userData = {};
    }
    geometry.userData.radiusScale = radiusScale;
    geometry.userData.sharpnessScale = sharpnessScale;

    return geometry;
  }

  /**
   * Create material for points.
   *
   * @param attrs - Node attributes (opacity, gamma, blending_mode)
   * @param radiusScale - Scale factor for radii
   * @param sharpnessScale - Scale factor for sharpness
   * @returns Configured ShaderMaterial
   */
  createPointsMaterial(
    attrs: PointsMetadata,
    radiusScale: number = 1.0,
    sharpnessScale: number = 1.0
  ): THREE.ShaderMaterial {
    return materialManager.getPointMaterial({
      opacity: attrs.opacity ?? 1.0,
      gamma: attrs.gamma ?? 1.0,
      intensity: attrs.intensity ?? 1.0,
      offset: attrs.offset ?? 0.0,
      blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
      radiusScale: radiusScale,
      sharpnessScale: sharpnessScale,
    });
  }

  /**
   * Update geometry for a specific points object.
   *
   * Uses GPU buffer pool if enabled for zero-allocation updates.
   *
   * @param rootGroup - The root THREE.Group containing the scene
   * @param path - Path/name of the points object
   * @param data - New points data
   * @param session - Optional profiler session
   */
  updatePointsGeometry(
    rootGroup: THREE.Group,
    path: string,
    data: LoadedPointsData,
    session?: UpdateSession
  ): void {
    // Find the points object
    const points = rootGroup.getObjectByName(path) as THREE.Points;
    if (!points) return;

    // Log if updating to empty geometry (clearing points)
    if (data.pointCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing points for ${path} (no visible points at current slice)`
      );
    }

    // Update visible point count in userData (following Lines/GSplats pattern)
    if (isPointsUserData(points.userData)) {
      points.userData.visiblePointCount = data.pointCount;
    }

    // Phase 4: Use GPU buffer pool if enabled (now supports all TypedArray types!)
    const bufferSession = session?.begin('Update Buffers');
    try {
      if (this._gpuBufferPool) {
        // Acquire geometry from pool (type-aware: matches capacity AND attribute types)
        const geometry = this._gpuBufferPool.acquirePointsGeometry(path, data, data.pointCount);

        // Update attributes in place (zero GPU allocations on reuse)
        this._gpuBufferPool.updatePointsGeometry(geometry, data, data.pointCount);

        // Update bounding box
        if (data.metadata.bounds) {
          geometry.boundingBox = data.metadata.bounds.clone();
        }

        // Assign to mesh (might be same geometry, reused)
        points.geometry = geometry;
      } else {
        // Fallback: GPU buffer pool disabled
        const oldGeometry = points.geometry;
        const oldPositionAttr = oldGeometry?.getAttribute(
          'position'
        ) as THREE.BufferAttribute | null;
        const oldCount = oldPositionAttr ? oldPositionAttr.count : 0;

        if (oldCount === data.pointCount && data.pointCount > 0) {
          // Same size: update in place (zero GPU allocation)
          // Use typed array .set() which handles implicit type conversion safely
          const posArr = oldPositionAttr!.array as Float32Array;
          posArr.set(
            data.positions instanceof Float32Array
              ? data.positions
              : new Float32Array(data.positions)
          );
          oldPositionAttr!.needsUpdate = true;

          const colorAttr = oldGeometry.getAttribute('color') as THREE.BufferAttribute;
          if (colorAttr && data.colors) {
            (colorAttr.array as Float32Array).set(data.colors);
            colorAttr.needsUpdate = true;
          }

          const radiiAttr = oldGeometry.getAttribute('radius') as THREE.BufferAttribute;
          if (radiiAttr && data.radii) {
            (radiiAttr.array as Float32Array).set(data.radii);
            radiiAttr.needsUpdate = true;
          }

          const sharpAttr = oldGeometry.getAttribute('sharpness') as THREE.BufferAttribute;
          if (sharpAttr && data.sharpness) {
            (sharpAttr.array as Float32Array).set(data.sharpness);
            sharpAttr.needsUpdate = true;
          }

          oldGeometry.computeBoundingBox();
          oldGeometry.computeBoundingSphere();
        } else {
          // Different size: dispose + create (handles complex dtype logic)
          if (oldGeometry) {
            oldGeometry.dispose();
          }
          points.geometry = this.createPointsGeometry(data);
        }
      }
    } finally {
      bufferSession?.end();
    }
  }

  // ============================================================================
  // Lines Geometry
  // ============================================================================

  /**
   * Update lines geometry for a specific path.
   *
   * Handles:
   * - Building instance buffers with visibility filtering
   * - Worker-based projection for large datasets
   * - GPU buffer pool integration
   * - extend_to_all tolerance handling
   *
   * @param rootGroup - The root THREE.Group containing the scene
   * @param path - Path/name of the lines object
   * @param data - Loaded lines data
   * @param viewState - Current view state
   * @param session - Optional profiler session
   */
  async updateLinesGeometry(
    rootGroup: THREE.Group,
    path: string,
    data: LoadedLinesData,
    viewState: { displayDims: number[]; slicePosition: number[]; dimensions?: any[] },
    session?: UpdateSession
  ): Promise<void> {
    const mesh = rootGroup.getObjectByName(path) as THREE.Mesh;
    if (!mesh || !isLinesUserData(mesh.userData)) return;

    // Build new instance buffers
    const ndim = data.ndim;
    let tolerance = viewState.dimensions
      ? computeLinesTolerance(viewState.dimensions, viewState.displayDims)
      : new Array(ndim).fill(0).map((_, i) => (viewState.displayDims.includes(i) ? 1e10 : 0));

    // CRITICAL: For extend_to_all dimensions, set tolerance to infinity
    const attrs = mesh.userData.attrs as { extend_to_all?: string[] };
    const extendDims: string[] = attrs.extend_to_all || [];
    if (extendDims.length > 0 && viewState.dimensions) {
      tolerance = [...tolerance]; // Make a copy to avoid mutating shared array
      for (const dimName of extendDims) {
        const dimIndex = viewState.dimensions.findIndex(
          (d: { name?: string }) => d.name === dimName
        );
        if (dimIndex >= 0 && dimIndex < tolerance.length) {
          tolerance[dimIndex] = 1e10; // Effectively infinite tolerance
        }
      }
    }

    // Build instance buffers with timing
    // Strategy: use worker for larger datasets when enabled
    const useWorkerProjection =
      this.shouldUseWorkers() && data.segmentCount > this.getWorkerThreshold();

    let processed: ProcessedLinesData;
    if (session) {
      const buildSession = session.begin('Project to 3D');
      try {
        if (useWorkerProjection) {
          processed = await this.projectLinesTo3DUsingWorker(data, viewState, tolerance);
        } else {
          processed = buildInstanceBuffers(
            data,
            viewState.slicePosition,
            tolerance,
            viewState.displayDims
          );
        }
      } finally {
        buildSession.end();
      }
    } else {
      if (useWorkerProjection) {
        processed = await this.projectLinesTo3DUsingWorker(data, viewState, tolerance);
      } else {
        processed = buildInstanceBuffers(
          data,
          viewState.slicePosition,
          tolerance,
          viewState.displayDims
        );
      }
    }

    // Phase 4: Use GPU buffer pool if enabled
    const bufferSession = session?.begin('Update Buffers');
    try {
      if (this._gpuBufferPool) {
        // Acquire geometry from pool (may reuse existing)
        const geometry = this._gpuBufferPool.acquireLinesGeometry(path, processed.segmentCount);

        // Update attributes in place (zero GPU allocations on reuse)
        this._gpuBufferPool.updateLinesGeometry(geometry, processed, processed.segmentCount);

        // Assign to mesh (might be same geometry, reused)
        mesh.geometry = geometry;
        mesh.count = processed.segmentCount;
      } else {
        // Fallback: In-place update (mirrors gsplats updateInstancedGSplatsMesh pattern)
        updateInstancedLinesMesh(mesh, processed);
        mesh.count = processed.segmentCount;
      }
    } finally {
      bufferSession?.end();
    }

    // Track visible segment count in mesh userData for monitor reporting
    if (isLinesUserData(mesh.userData)) {
      mesh.userData.visibleSegmentCount = processed.segmentCount;
    }

    if (data.segmentCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing lines for ${path} (no visible segments at current slice)`
      );
    }
  }

  /**
   * Project lines to 3D using a web worker.
   *
   * Offloads CPU-intensive segment clipping and interpolation to a worker thread.
   * Uses Comlink.transfer() for zero-copy ArrayBuffer transfer.
   */
  private async projectLinesTo3DUsingWorker(
    data: LoadedLinesData,
    viewState: { displayDims: number[]; slicePosition: number[] },
    tolerance: number[]
  ): Promise<ProcessedLinesData> {
    try {
      const worker = await getWorkerPool().getWorker();

      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.segmentCount} line segments to 3D using worker`
      );

      const workerResult = await worker.projectLinesTo3D({
        positions: data.positions,
        segments: data.segments,
        widths: data.widths,
        colors: data.colors,
        sharpness: data.sharpness,
        slicePosition: viewState.slicePosition,
        tolerance,
        displayDims: viewState.displayDims,
        ndim: data.ndim,
        segmentCount: data.segmentCount,
      });

      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleSegmentCount}/${data.segmentCount} visible segments`
      );

      // Return processed data using the worker result
      return {
        startPositions: workerResult.startPositions,
        endPositions: workerResult.endPositions,
        startColors: workerResult.startColors,
        endColors: workerResult.endColors,
        startWidths: workerResult.startWidths,
        endWidths: workerResult.endWidths,
        startSharpness: workerResult.startSharpness,
        endSharpness: workerResult.endSharpness,
        segmentLengths: workerResult.segmentLengths,
        startClipped: workerResult.startClipped,
        endClipped: workerResult.endClipped,
        segmentCount: workerResult.visibleSegmentCount,
      };
    } catch (error) {
      // Fallback to main thread on worker failure
      log.warning(
        Modules.SCENE_LOADER,
        'Worker projection failed, falling back to main thread:',
        error
      );
      return buildInstanceBuffers(data, viewState.slicePosition, tolerance, viewState.displayDims);
    }
  }

  // ============================================================================
  // GSplats Geometry
  // ============================================================================

  /**
   * Update gsplats geometry for a specific path.
   *
   * Handles:
   * - Processing nD GSplats to 3D
   * - Worker-based projection for large datasets
   * - GPU buffer pool integration
   * - Cholesky factor packing for shader
   *
   * @param rootGroup - The root THREE.Group containing the scene
   * @param path - Path/name of the gsplats object
   * @param data - Loaded gsplats data
   * @param viewState - Current GSplats view state
   * @param session - Optional profiler session
   */
  async updateGSplatsGeometry(
    rootGroup: THREE.Group,
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<void> {
    const mesh = rootGroup.getObjectByName(path) as THREE.Mesh;
    if (!mesh || mesh.userData?.nodeType !== 'gsplats') return;

    // Strategy: use worker for larger datasets when enabled (nD only, not 3D)
    const useWorkerProjection =
      this.shouldUseWorkers() && data.splatCount > this.getWorkerThreshold() && data.ndim > 3; // Only worth offloading for nD processing

    // Process nD data to 3D for rendering (with timing)
    let processed: ReturnType<typeof processGSplats>;
    let cholesky01: Float32Array;
    let cholesky23: Float32Array;
    let cholesky45: Float32Array;

    if (session) {
      const projectSession = session.begin('Project to 3D');
      try {
        if (useWorkerProjection) {
          processed = await this.projectGSplatsTo3DUsingWorker(data, viewState);
        } else {
          processed = processGSplats(data, viewState);
        }
        // Pack Cholesky factors for shader
        const packed = packCholeskyForShader(processed.choleskyFactors3D, processed.splatCount);
        cholesky01 = packed.cholesky01;
        cholesky23 = packed.cholesky23;
        cholesky45 = packed.cholesky45;
      } finally {
        projectSession.end();
      }
    } else {
      if (useWorkerProjection) {
        processed = await this.projectGSplatsTo3DUsingWorker(data, viewState);
      } else {
        processed = processGSplats(data, viewState);
      }
      // Pack Cholesky factors for shader
      const packed = packCholeskyForShader(processed.choleskyFactors3D, processed.splatCount);
      cholesky01 = packed.cholesky01;
      cholesky23 = packed.cholesky23;
      cholesky45 = packed.cholesky45;
    }

    // Phase 4: Use GPU buffer pool if enabled
    const bufferSession = session?.begin('Update Buffers');
    try {
      if (this._gpuBufferPool) {
        // Acquire geometry from pool (may reuse existing)
        const geometry = this._gpuBufferPool.acquireGSplatsGeometry(path, processed.splatCount);

        // Update attributes in place (zero GPU allocations on reuse)
        this._gpuBufferPool.updateGSplatsGeometry(
          geometry,
          {
            centers3D: processed.centers3D,
            amplitudes: processed.amplitudes,
            cholesky01,
            cholesky23,
            cholesky45,
            colors: processed.colors,
            splatCount: processed.splatCount,
          },
          processed.splatCount
        );

        // Assign to mesh (might be same geometry, reused)
        mesh.geometry = geometry;
      } else {
        // Fallback: Original path (updateInstancedGSplatsMesh)
        updateInstancedGSplatsMesh(mesh, {
          centers: processed.centers3D,
          cholesky01,
          cholesky23,
          cholesky45,
          amplitudes: processed.amplitudes,
          colors: processed.colors,
          splatCount: processed.splatCount,
        });
      }
    } finally {
      bufferSession?.end();
    }

    // Track visible splat count in mesh userData
    if (mesh.userData) {
      (mesh.userData as GSplatsUserData).visibleSplatCount = processed.splatCount;
    }

    if (processed.splatCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing gsplats for ${path} (no visible splats at current slice)`
      );
    }
  }

  /**
   * Project GSplats to 3D using a web worker.
   *
   * Offloads CPU-intensive Mahalanobis distance calculation and Cholesky
   * submatrix extraction to a worker thread.
   * Uses Comlink.transfer() for zero-copy ArrayBuffer transfer.
   */
  private async projectGSplatsTo3DUsingWorker(
    data: LoadedGSplatsData,
    viewState: GSplatsViewState
  ): Promise<ReturnType<typeof processGSplats>> {
    try {
      const worker = await getWorkerPool().getWorker();

      log.info(
        Modules.SCENE_LOADER,
        `Projecting ${data.splatCount} gsplats to 3D using worker (ndim=${data.ndim})`
      );

      // Extract discrete dimension info and extend_to_all dims for worker.
      const discreteDims: number[] = [];
      const discreteSteps: Record<number, number> = {};
      const extendToAllDims: number[] = [];
      if (viewState.dimensions) {
        for (let d = 0; d < viewState.dimensions.length; d++) {
          if (viewState.displayDims.includes(d)) continue;
          if (viewState.tolerance[d] >= 1e9) {
            extendToAllDims.push(d);
          } else if (viewState.dimensions[d]?.discrete) {
            discreteDims.push(d);
            discreteSteps[d] = viewState.dimensions[d].step ?? 1.0;
          }
        }
      }

      const workerResult = await worker.projectGSplatsTo3D({
        positions: data.positions,
        choleskyFactors: data.choleskyFactors,
        amplitudes: data.amplitudes,
        colors: data.colors,
        sharpness: null,
        displayDims: viewState.displayDims,
        slicePosition: viewState.slicePosition,
        ndim: data.ndim,
        splatCount: data.splatCount,
        discreteDims,
        discreteSteps,
        extendToAllDims,
      });

      log.info(
        Modules.SCENE_LOADER,
        `Worker projection complete: ${workerResult.visibleCount}/${data.splatCount} visible splats`
      );

      return {
        centers3D: workerResult.centers3D,
        choleskyFactors3D: workerResult.choleskyFactors3D,
        amplitudes: workerResult.amplitudes,
        colors: workerResult.colors,
        splatCount: workerResult.visibleCount,
      };
    } catch (error) {
      // Fallback to main thread on worker failure
      log.warning(
        Modules.SCENE_LOADER,
        'Worker GSplats projection failed, falling back to main thread:',
        error
      );
      return processGSplats(data, viewState);
    }
  }

  // ============================================================================
  // Transform Utilities
  // ============================================================================

  /**
   * Apply transformation matrix to a THREE.js object.
   *
   * Decomposes the 4x4 matrix into position, quaternion, and scale components.
   *
   * @param object - The THREE.Object3D to transform
   * @param transform - 16-element array (column-major for THREE.js)
   */
  applyTransform(object: THREE.Object3D, transform: number[]): void {
    if (transform.length !== 16) {
      log.warning(Modules.SCENE_LOADER, `Invalid transform length: ${transform.length}`);
      return;
    }

    // Validate transform format (detect common mistakes)
    this.validateTransformFormat(transform);

    const matrix = new THREE.Matrix4().fromArray(transform);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();

    matrix.decompose(position, quaternion, scale);

    object.position.copy(position);
    object.quaternion.copy(quaternion);
    object.scale.copy(scale);
  }

  /**
   * Validate transform matrix format (detect row-major vs column-major).
   *
   * THREE.js expects column-major (OpenGL-style) where translation is at indices [12, 13, 14].
   * NumPy uses row-major (C-style) where translation is at indices [3, 7, 11].
   *
   * Python should transpose before writing: `matrix.T.ravel().tolist()`
   *
   * @param transform - 16-element transform array
   * @returns true if format appears correct, false if row-major detected
   */
  validateTransformFormat(transform: number[]): boolean {
    // Check if translation components look suspicious
    // In column-major (correct for THREE.js): [12]=tx, [13]=ty, [14]=tz
    // In row-major (wrong for THREE.js): [3]=tx, [7]=ty, [11]=tz

    const colMajorTranslation = [transform[12], transform[13], transform[14]];
    const rowMajorTranslation = [transform[3], transform[7], transform[11]];

    const colMajorNonZero = colMajorTranslation.some((v) => Math.abs(v) > 0.001);
    const rowMajorNonZero = rowMajorTranslation.some((v) => Math.abs(v) > 0.001);

    // If row-major positions are non-zero but column-major are zero, likely wrong format
    if (rowMajorNonZero && !colMajorNonZero) {
      log.warning(
        Modules.SCENE_LOADER,
        'Transform matrix appears to be in row-major (NumPy) format instead of column-major (THREE.js). ' +
          'Translation detected at wrong indices [3,7,11] instead of [12,13,14]. ' +
          'Python should transpose before storing: matrix.T.ravel().tolist()'
      );
      return false;
    }

    return true;
  }

  // ============================================================================
  // Validation Helpers
  // ============================================================================

  /**
   * Validate points data for edge cases and malformed data.
   *
   * Logs detailed diagnostics to browser console for debugging.
   *
   * @param data - Points data to validate
   * @throws Error if positions array is malformed (not divisible by 3)
   */
  validateLoadedPointsData(data: LoadedPointsData): void {
    const pointCount = data.positions.length / 3;

    // Log data summary for debugging
    log.info(Modules.SCENE_LOADER, 'Points Data Validation:', {
      pointCount,
      positionsLength: data.positions.length,
      positionsType: data.positions.constructor.name,
      hasColors: !!data.colors,
      colorsType: data.colors?.constructor.name,
      colorsLength: data.colors?.length,
      hasRadii: !!data.radii,
      radiiType: data.radii?.constructor.name,
      radiiLength: data.radii?.length,
      hasSharpness: !!data.sharpness,
      sharpnessType: data.sharpness?.constructor.name,
      sharpnessLength: data.sharpness?.length,
    });

    // EDGE CASE: Empty dataset
    if (pointCount === 0) {
      log.warning(Modules.SCENE_LOADER, 'Empty dataset detected - no points to render');
      return;
    }

    // EDGE CASE: Malformed positions (not multiple of 3)
    if (data.positions.length % 3 !== 0) {
      const error = `Malformed positions array: length ${data.positions.length} is not divisible by 3`;
      log.error(Modules.SCENE_LOADER, error);
      throw new Error(error);
    }

    // VALIDATION: Colors length consistency
    if (data.colors && data.colors.length !== data.positions.length) {
      const expected = data.positions.length;
      const actual = data.colors.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Colors length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    // VALIDATION: Radii length consistency
    if (data.radii && data.radii.length !== pointCount) {
      const expected = pointCount;
      const actual = data.radii.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Radii length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    // VALIDATION: Sharpness length consistency
    if (data.sharpness && data.sharpness.length !== pointCount) {
      const expected = pointCount;
      const actual = data.sharpness.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Sharpness length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    // Log successful validation
    log.success(Modules.SCENE_LOADER, `Points data validated: ${pointCount} points`);
  }

  /**
   * Validate color mode consistency.
   *
   * Ensures color array type matches expected encoding:
   * - Float32Array for HDR colors (values > 1.0)
   * - Uint8Array for SDR colors (values [0, 1])
   *
   * @param colors - Color data array
   * @param nodeMetadata - Node metadata with color_mode info
   */
  validateColorMode(colors: Uint8Array | Uint16Array | Float32Array, nodeMetadata: any): void {
    const isHDR = colors instanceof Float32Array;
    const isSDR = colors instanceof Uint8Array || colors instanceof Uint16Array;

    // Check for suspicious patterns
    if (isSDR && nodeMetadata?.color_mode === 'hdr') {
      log.warning(
        Modules.SCENE_LOADER,
        `Node metadata indicates HDR colors but array is ${colors.constructor.name}. ` +
          'HDR colors should use Float32Array. This may indicate incorrect encoding.'
      );
    }

    if (isHDR) {
      // For float32 colors, check if any values exceed 1.0 (HDR range)
      const hasHDRValues = Array.from(colors).some((v) => v > 1.0);
      if (!hasHDRValues && nodeMetadata?.color_mode === 'hdr') {
        log.info(
          Modules.SCENE_LOADER,
          'HDR color mode specified but all values in [0, 1] range. Consider using SDR mode for better compression.'
        );
      }
    }

    // Log color mode for debugging
    const colorType = colors.constructor.name;
    const colorMode = isHDR ? 'HDR (float32)' : 'SDR (normalized integer)';
    log.info(Modules.SCENE_LOADER, `Colors: ${colorType} - ${colorMode}`);
  }
}
