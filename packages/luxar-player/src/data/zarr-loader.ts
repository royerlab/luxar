/**
 * Zarr-based nD point cloud data loader with advanced slicing capabilities.
 * 
 * This module provides the core data loading infrastructure for Luxar's nD visualization
 * system. It handles loading point cloud data from Zarr stores, processes dimension
 * metadata, and performs initial slicing operations for nD datasets.
 * 
 * Key capabilities:
 * - Hierarchical scene loading from Zarr group structures
 * - nD point cloud data loading with automatic slicing
 * - Scene-level dimension metadata extraction and propagation
 * - Radius-based hypersphere slicing for smooth navigation
 * - Optional attribute loading (colors, radii, sharpness)
 * - GPU-optimized data format conversion
 * - Rendering parameter inheritance in nested scenes
 * 
 * Data pipeline:
 * 1. Load Zarr store with consolidated metadata
 * 2. Extract scene-level dimension configuration
 * 3. Enumerate and load point cloud groups hierarchically
 * 4. Perform initial nD slicing based on dimension state
 * 5. Convert data formats for GPU compatibility
 * 6. Create THREE.js geometry with appropriate materials
 * 
 * Performance considerations:
 * - Uses streaming Zarr loading for large datasets
 * - Minimizes memory usage through on-demand slicing
 * - Batch-processes attribute arrays for efficiency
 * - Provides fallbacks for missing optional attributes
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import * as THREE from 'three';
import { materialManager, BlendingMode } from '../rendering/material-manager';
import { SimpleDims, DimensionMetadata } from '../types/dims';
import {
  slicePoints,
  extractDisplayDimensions,
  sliceColors,
  sliceScalarAttribute,
  computeEffectiveRadii,
} from '../utils/slicing';

/* ------------------------------------------------------------------ utils */

/**
 * Normalizes a path string to a valid URL for Zarr store access.
 * 
 * @param path - Local path or full URL to Zarr store
 * @returns Normalized URL with trailing slash
 */
function toURL(path: string) {
  const abs = path.startsWith('http')
    ? path
    : new URL(path.replace(/^\/?/, '/'), window.location.origin).toString();
  return abs.endsWith('/') ? abs : abs + '/';
}

/**
 * Implements rendering attribute inheritance in hierarchical scenes.
 * 
 * Child objects inherit rendering properties (opacity, gamma, blending) from their
 * parents unless explicitly overridden. This allows for consistent styling across
 * scene hierarchies while enabling local customization.
 * 
 * @param attrs - Current group's attributes
 * @param parentAttrs - Parent group's attributes for inheritance
 * @returns Merged attributes with inheritance applied
 */
function inheritRenderingAttributes(
  attrs: ZarrGroupAttrs,
  parentAttrs?: ZarrGroupAttrs
): ZarrGroupAttrs {
  if (!parentAttrs) return attrs;

  return {
    ...attrs,
    opacity: attrs.opacity ?? parentAttrs.opacity,
    gamma: attrs.gamma ?? parentAttrs.gamma,
    blending_mode: attrs.blending_mode ?? parentAttrs.blending_mode,
  };
}

/** Internal record for tracking THREE.js objects and their Zarr metadata */
type ObjRecord = { obj: THREE.Object3D; path: string; attrs?: ZarrGroupAttrs };

/**
 * Zarr group attributes defining rendering and dimensional properties.
 * 
 * These attributes are stored in Zarr group metadata and control how
 * point clouds are rendered and how their dimensions are interpreted.
 * 
 * @interface ZarrGroupAttrs
 */
interface ZarrGroupAttrs {
  /** Object type identifier (e.g., 'points') */
  type?: string;
  
  /** 4x4 transformation matrix as 16-element array */
  transform?: number[];
  
  /** Metadata describing each dimension's properties */
  dimension_metadata?: DimensionMetadata[];
  
  /** Total number of points in the dataset */
  num_points?: number;
  
  /** Rendering opacity (0.0 - 1.0) */
  opacity?: number;
  
  /** Gamma correction factor for color */
  gamma?: number;
  
  /** Blending mode for compositing */
  blending_mode?: BlendingMode;
  
  /** Scene-level dimension configuration */
  scene_dimensions?: any;
}

/* ------------------------------------------------------------------ main */

/**
 * Loads an nD scene from a Zarr store, creating a complete THREE.js scene graph.
 * 
 * This is the main entry point for loading Luxar scene data. It handles the complete
 * pipeline from Zarr store access to THREE.js scene construction, including:
 * 
 * - Scene-level dimension metadata extraction and initialization
 * - Hierarchical loading of nested groups and point clouds
 * - Automatic nD slicing for high-dimensional datasets
 * - Rendering attribute inheritance through the scene hierarchy
 * - GPU-optimized geometry creation with appropriate materials
 * 
 * The loading process is designed to handle large scientific datasets efficiently
 * while providing immediate visual feedback and smooth navigation capabilities.
 * 
 * Scene structure:
 * - Root group contains scene-level dimension metadata
 * - Child groups can contain point clouds or nested groups
 * - Each point cloud can have its own transformation and rendering properties
 * - Dimension metadata is propagated down the hierarchy for consistency
 * 
 * @param src - URL or path to the Zarr store containing the scene data
 * @returns Promise resolving to a THREE.Group containing the complete scene
 */
export async function loadScene(src: string): Promise<THREE.Group> {
  try {
    // Phase 1: Initialize Zarr store with optimized metadata access
    const rawStore = new zarr.FetchStore(toURL(src));
    const store = await zarr.tryWithConsolidated(rawStore); // Use consolidated metadata when available

    // Phase 2: Discover scene structure
    const listing = await store.contents(); // Enumerate all groups and arrays

    const rootLoc = zarr.root(store);
    const rootThree = new THREE.Group();

    // Phase 3: Load scene-level configuration and dimension metadata
    const rootGroup = await zarr.open(rootLoc, { kind: 'group' });
    const sceneAttrs = rootGroup.attrs as ZarrGroupAttrs;
  
    const lookup = new Map<string, ObjRecord>([
      ['/', { obj: rootThree, path: '/', attrs: sceneAttrs }],
    ]);

    // Phase 4: Initialize scene-level dimension system
    let sceneDims: SimpleDims | undefined;
    if (sceneAttrs?.scene_dimensions) {
    // Store raw metadata in THREE.js userData for scene manager access
      rootThree.userData.sceneDimensions = sceneAttrs.scene_dimensions;
    
      // Parse and normalize dimension metadata
      const metadata = sceneAttrs.scene_dimensions.dimensions.map((dim: any) => ({
        name: dim.name,
        unit: dim.unit,
        scale: dim.scale || 1.0,
        range: dim.range ? [dim.range[0], dim.range[1]] : undefined,
        display: dim.display,
        discrete: dim.discrete || false,
        step: dim.step || 1.0,
      }));
    
      const ndim = metadata.length;
      const displayed: number[] = [];
    
      // Identify which dimensions should be displayed in 3D
      for (let i = 0; i < ndim; i++) {
        if (metadata[i].display === true && displayed.length < 3) {
          displayed.push(i);
        }
      }
    
      // Initialize positions: non-displayed dims start at minimum for predictable slicing
      const currentStep = new Array(ndim).fill(0);
      for (let i = 0; i < ndim; i++) {
        if (!displayed.includes(i) && metadata[i].range) {
          currentStep[i] = metadata[i].range[0];
        }
      }
    
      // Create dimension state object for initial slicing
      sceneDims = {
        ndim,
        currentStep,
        displayed,
        metadata
      };
    }

    /* 3. create Three.js objects in path-depth order */
    const groups = listing
      .filter((e: { kind: string; path: string }) => e.kind === 'group' && e.path !== '/')
      .sort(
        (a: { path: string }, b: { path: string }) =>
          a.path.split('/').length - b.path.split('/').length
      );

    for (const entry of groups) {
      try {
        const loc = rootLoc.resolve(entry.path.slice(1)); // drop leading "/"
        const grp = await zarr.open(loc, { kind: 'group' });
        let attrs = grp.attrs as ZarrGroupAttrs;

        // Get parent attributes for inheritance
        const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
        const parentRecord = lookup.get(parentPath);
        const parentAttrs = parentRecord?.attrs;

        // Inherit rendering attributes from parent
        attrs = inheritRenderingAttributes(attrs, parentAttrs);

        // Pass scene dimensions down through attrs
        if (rootThree.userData.sceneDimensions && !attrs.scene_dimensions) {
          attrs.scene_dimensions = rootThree.userData.sceneDimensions;
        }

        /* build renderable */
        let obj: THREE.Object3D;
        if (attrs?.type === 'points') {
          obj = await buildPoints(loc, attrs, sceneDims);
        } else {
          obj = new THREE.Group();
          // Store rendering attrs on groups for child inheritance
          obj.userData.opacity = attrs.opacity;
          obj.userData.gamma = attrs.gamma;
          obj.userData.blendingMode = attrs.blending_mode;
        }

        if (Array.isArray(attrs?.transform) && attrs.transform.length === 16) {
          obj.applyMatrix4(new THREE.Matrix4().fromArray(attrs.transform));
        }

    /* attach to parent in Three.js graph */
    lookup.get(parentPath)!.obj.add(obj);
    lookup.set(entry.path, { obj, path: entry.path, attrs });
      } catch (error) {
        console.error(`Failed to load group ${entry.path}:`, error);
      // Continue loading other groups instead of failing completely
      }
    }

    return rootThree;
  } catch (error) {
    console.error('Failed to load scene from Zarr store:', error);
    throw new Error(`Unable to load scene from ${src}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/* ---------------------------------------------------------------- geometry */

/**
 * Constructs a THREE.js Points object from nD point cloud data in a Zarr group.
 * 
 * This function is the heart of the nD visualization system, responsible for:
 * - Loading nD position data and determining dimensionality
 * - Performing initial radius-based hypersphere slicing for nD datasets
 * - Loading and processing optional attributes (colors, radii, sharpness)
 * - Converting data formats for GPU compatibility
 * - Creating optimized THREE.js geometry with appropriate materials
 * - Handling edge cases like empty slices and missing attributes
 * 
 * The function automatically detects whether slicing is needed based on the
 * dimensionality of the data relative to the display dimensions, ensuring
 * optimal performance for both 3D and nD datasets.
 * 
 * Data processing pipeline:
 * 1. Load position array and infer dimensionality
 * 2. Load optional radii array for hypersphere slicing
 * 3. Perform nD slicing if data exceeds 3D
 * 4. Load and slice optional attributes (colors, sharpness)
 * 5. Compute effective radii for sliced hyperspheres
 * 6. Convert data formats for GPU (uint8→float32 for colors)
 * 7. Create THREE.js geometry with all attributes
 * 8. Apply materials and store metadata for runtime updates
 * 
 * @param loc - Zarr location containing the point cloud arrays
 * @param attrs - Group attributes with rendering and metadata properties
 * @param sceneDims - Scene-level dimension state for consistent slicing
 * @returns Promise resolving to a THREE.Points object ready for rendering
 */
async function buildPoints(
  loc: zarr.Location<zarr.Readable>,
  attrs: ZarrGroupAttrs,
  sceneDims?: SimpleDims
): Promise<THREE.Points> {
  // Phase 1: Load core position data
  const posArr = await zarr.open(loc.resolve('positions'), { kind: 'array' });
  const posData = (await get(posArr)).data as Float32Array;

  // Determine dataset size - explicit count takes precedence over inference
  const numPoints = attrs.num_points || posData.length / 3;

  // Infer dimensionality from data structure
  const ndim = posData.length / numPoints;
  if (!Number.isInteger(ndim)) {
    throw new Error(`Invalid positions array: ${posData.length} elements for ${numPoints} points`);
  }
  
  // Use scene-level dimensions for consistency, or create default for standalone data
  const dims = sceneDims || {
    ndim,
    currentStep: new Array(ndim).fill(0),
    displayed: ndim <= 3 ? Array.from({ length: ndim }, (_, i) => i) : [ndim - 3, ndim - 2, ndim - 1],
    metadata: undefined
  };
  
  // Phase 2: Load radii for hypersphere slicing (critical for nD navigation)
  let radiiData: Float32Array | undefined;
  try {
    const radiiArr = await zarr.open(loc.resolve('radii'), { kind: 'array' });
    radiiData = (await get(radiiArr)).data as Float32Array;
  } catch (error) {
    console.debug('No radii array found:', error);
    // Radii are optional but highly recommended for nD datasets
  }

  // Phase 3: Perform nD slicing if needed
  let pos: Float32Array;
  let visibleIndices: Uint32Array | null = null;

  if (dims.ndim > 3) {
    // Apply radius-based hypersphere slicing for nD visualization
    visibleIndices = slicePoints(posData, dims, numPoints, radiiData);
    pos = extractDisplayDimensions(posData, visibleIndices, dims);

    // Handle edge case: empty slice (user navigated to region with no data)
    if (visibleIndices.length === 0) {
      console.warn('No points visible at current slice position!');
      // Provide dummy geometry to prevent GPU errors
      pos = new Float32Array([0, 0, 0]);
      visibleIndices = new Uint32Array([0]);
    }
  } else {
    // 3D or lower dimensional data requires no slicing
    pos = posData;
  }

  // Load optional attributes
  let colData: Uint8Array | undefined;
  let col: Uint8Array | undefined;
  try {
    const colArr = await zarr.open(loc.resolve('colors'), { kind: 'array' });
    colData = (await get(colArr)).data as Uint8Array;
    col = visibleIndices ? sliceColors(colData, visibleIndices) || undefined : colData;
  } catch (error) {
    console.debug('Optional array not found:', error);
    /* optional */
  }

  // Process radii for visible points
  let radii: Float32Array | undefined;
  if (radiiData) {
    if (visibleIndices && dims.ndim > 3) {
      // Compute effective radii based on slice intersection
      radii = computeEffectiveRadii(posData, visibleIndices, dims, radiiData);
    } else {
      radii = radiiData;
    }
  }

  let sharpnessData: Float32Array | undefined;
  let sharpness: Float32Array | undefined;
  try {
    const sharpnessArr = await zarr.open(loc.resolve('sharpness'), { kind: 'array' });
    sharpnessData = (await get(sharpnessArr)).data as Float32Array;
    sharpness = visibleIndices
      ? sliceScalarAttribute(sharpnessData, visibleIndices) || undefined
      : sharpnessData;
  } catch (error) {
    console.debug('Optional array not found:', error);
    /* optional */
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  if (col) {
    // Convert Uint8Array colors (0-255) to Float32Array (0.0-1.0) for shader compatibility
    const floatColors = new Float32Array(col.length);
    for (let i = 0; i < col.length; i++) {
      floatColors[i] = col[i] / 255.0;
    }
    geom.setAttribute('color', new THREE.BufferAttribute(floatColors, 3));
  } else {
    // Provide default white colors if no color data is available
    // This ensures the vertex shader always has color data to work with
    const numVertices = pos.length / 3;
    const defaultColors = new Float32Array(numVertices * 3);
    defaultColors.fill(1.0); // All white (RGB = 1.0, 1.0, 1.0)
    geom.setAttribute('color', new THREE.BufferAttribute(defaultColors, 3));
  }

  if (radii) {
    // Add radius attribute for custom shaders
    geom.setAttribute('radius', new THREE.BufferAttribute(radii, 1));
  } else {
    // Provide default radii if not specified
    const numVertices = pos.length / 3;
    const defaultRadii = new Float32Array(numVertices);
    defaultRadii.fill(1.0); // Default radius - increased for better visibility
    geom.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
  }

  if (sharpness) {
    // Add sharpness attribute for custom shaders
    geom.setAttribute('sharpness', new THREE.BufferAttribute(sharpness, 1));
  } else {
    // Provide default sharpness if not specified
    const numVertices = pos.length / 3;
    const defaultSharpness = new Float32Array(numVertices);
    defaultSharpness.fill(2.0); // Default sharpness (quadratic falloff)
    geom.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
  }

  // Get rendering properties from attributes, with defaults
  const opacity = attrs.opacity ?? 1.0;
  const gamma = attrs.gamma ?? 1.0;
  const blendingMode = attrs.blending_mode ?? 'additive';

  // Get or create material from the manager
  const material = materialManager.getMaterial({
    blendingMode: blendingMode as BlendingMode,
    opacity,
    gamma,
  });

  const points = new THREE.Points(geom, material);

  // Apply render order from material
  if (material.userData.renderOrder !== undefined) {
    points.renderOrder = material.userData.renderOrder;
  }

  // Store rendering properties in userData for runtime updates
  points.userData.opacity = opacity;
  points.userData.gamma = gamma;
  points.userData.blendingMode = blendingMode;

  // Store original data in userData for later access (but NOT dims!)
  if (dims.ndim > 3) {
    points.userData.originalNumPoints = numPoints;
    points.userData.originalPositions = posData;
    points.userData.originalColors = colData;
    points.userData.originalRadii = radiiData;
    points.userData.originalSharpness = sharpnessData;
  }

  return points;
}
