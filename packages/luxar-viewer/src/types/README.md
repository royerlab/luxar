# Types Package

TypeScript type definitions for high-dimensional data visualization in Luxar. This package provides the fundamental data structures and interfaces for managing nD points and lines data, dimension metadata, and coordinate system definitions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Core Types](#core-types)
- [Dimension Metadata](#dimension-metadata)
- [SimpleDims Interface](#simpledims-interface)
- [Utility Functions](#utility-functions)
- [Zarr Types](#zarr-types)
- [Animation Types](#animation-types)
- [Float16Array Type Declaration](#float16array-type-declaration)
- [Usage Examples](#usage-examples)
- [Type Safety](#type-safety)
- [Best Practices](#best-practices)

## Overview

The types package defines the foundational type system for Luxar's nD visualization capabilities. It provides type-safe interfaces for representing high-dimensional datasets, managing dimension states, and handling coordinate system transformations.

**Core Philosophy**: Provide precise, well-documented types that capture the mathematical and conceptual structure of high-dimensional data visualization while ensuring compile-time safety and runtime reliability.

## Key Features

- **nD Data Structures**: Complete type definitions for high-dimensional points and lines
- **Lines Types**: Type definitions for line segments with nD clipping support
- **GSplats Types**: Type definitions for Gaussian splats with anisotropic covariance
- **Dimension Metadata**: Rich semantic information for dataset dimensions
- **Navigation State**: Type-safe dimension slicing and display configuration
- **Initialization Utilities**: Functions for creating properly structured dimension objects
- **Range Computation**: Mathematical utilities for analyzing dimension bounds
- **Type Safety**: Comprehensive interfaces preventing common visualization errors

## Core Types

### DimensionMetadata Interface

Describes the semantic properties and behavior of a single dimension in an nD dataset:

```typescript
interface DimensionMetadata {
  /** Human-readable name (e.g., "Time", "X", "Channel") */
  name: string;

  /** Physical unit of measurement (e.g., "μm", "s", "nm") */
  unit: string;

  /** Scale factor for converting indices to real-world units */
  scale: number;

  /** Optional min/max bounds in real-world units */
  range?: [number, number];

  /** Whether this dimension should be displayed by default */
  display?: boolean;

  /** Whether values are discrete (integers) vs continuous */
  discrete?: boolean;

  /** Whether dimension wraps around (for angles, periodic states) */
  cyclic?: boolean;

  /** Step size for navigation in this dimension */
  step?: number;

  /** Whether points extend through this dimension (true for spatial dims) */
  spatial?: boolean;

  /** Optional category labels for categorical dimensions */
  categories?: string[];

  /** Optional human-readable description for UI tooltips */
  description?: string;
}
```

**Use Cases**:

- **Scientific Data**: Physical dimensions with units and scales
- **Time Series**: Temporal dimensions with discrete time points
- **Categorical Data**: Discrete dimensions like channels (using `categories` field)
- **Spatial Coordinates**: X, Y, Z with physical units and continuous navigation
- **Periodic Dimensions**: Angles or cyclic time (using `cyclic` field)
- **Spatial vs Categorical**: Use `spatial` field to distinguish physical extent

### SimpleDims Interface

The core state object for nD visualization, tracking current position and display configuration:

```typescript
interface SimpleDims {
  /** Total number of dimensions in the dataset */
  ndim: number;

  /** Current position/slice in each dimension */
  currentStep: number[];

  /** Indices of dimensions currently displayed (max 3) */
  displayed: number[];

  /** Optional metadata for each dimension */
  metadata?: DimensionMetadata[];
}
```

**State Management**:

- **Navigation**: `currentStep` tracks position in nD space
- **Visualization**: `displayed` determines which dimensions are visualized
- **Semantics**: `metadata` provides human-readable context

## Dimension Metadata

### Physical Units and Scales

Dimension metadata enables proper handling of physical coordinate systems:

```typescript
const dimensionMeta: DimensionMetadata = {
  name: 'X',
  unit: 'μm',
  scale: 0.1, // 0.1 μm per array index
  range: [0, 100], // 0-100 μm physical range
  display: true, // Show in 3D visualization
  discrete: false, // Continuous spatial dimension
  step: 1.0, // 1 μm navigation steps
};
```

### Discrete vs Continuous Dimensions

The type system distinguishes between discrete and continuous dimensions for proper navigation behavior:

```typescript
// Continuous dimension (spatial coordinates)
const spatialDim: DimensionMetadata = {
  name: 'Y',
  unit: 'nm',
  scale: 1.0,
  discrete: false, // Supports interpolation and smooth navigation
  step: 10.0, // Fine-grained steps
};

// Discrete dimension (time frames)
const timeDim: DimensionMetadata = {
  name: 'Time',
  unit: 'frame',
  scale: 1.0,
  discrete: true, // Exact integer matching required
  step: 1.0, // Integer frame steps
  spatial: false, // Points exist at discrete time points
};

// Cyclic dimension (angles)
const angleDim: DimensionMetadata = {
  name: 'Theta',
  unit: 'deg',
  scale: 1.0,
  range: [0, 360],
  cyclic: true, // Wraps from 360° to 0°
  discrete: false,
};

// Categorical dimension (with labels)
const channelDim: DimensionMetadata = {
  name: 'Channel',
  unit: '',
  scale: 1.0,
  range: [0, 2],
  discrete: true,
  categories: ['DAPI', 'GFP', 'RFP'], // Human-readable labels
  spatial: false, // Points don't extend through channels
  description: 'Fluorescence imaging channel',
};
```

### Display Configuration

Metadata controls which dimensions are visualized by default:

```typescript
// Typical 5D scientific dataset (T, Z, C, Y, X)
const metadata: DimensionMetadata[] = [
  { name: 'Time', unit: 's', scale: 0.1, discrete: true, display: false },
  { name: 'Z', unit: 'μm', scale: 0.2, discrete: false, display: false },
  { name: 'Channel', unit: '', scale: 1, discrete: true, display: false },
  { name: 'Y', unit: 'μm', scale: 0.1, discrete: false, display: true }, // Spatial
  { name: 'X', unit: 'μm', scale: 0.1, discrete: false, display: true }, // Spatial
];
```

## SimpleDims Interface

### Navigation State

The `SimpleDims` interface tracks the complete state of nD navigation:

```typescript
const dims: SimpleDims = {
  ndim: 5, // 5D dataset
  currentStep: [0, 2.5, 1, 0, 0], // Position in each dimension
  displayed: [3, 4], // Show Y and X dimensions
  metadata: dimensionMetadata, // Semantic information
};
```

**State Interpretation**:

- `currentStep[0] = 0`: Time frame 0
- `currentStep[1] = 2.5`: Z-slice at 2.5 μm
- `currentStep[2] = 1`: Channel 1
- `currentStep[3]` and `currentStep[4]`: Camera-controlled (displayed dimensions)

### Display Dimension Management

The `displayed` array determines which dimensions are visualized:

```typescript
// 2D visualization (common for image data)
displayed: [3, 4]; // Show Y, X

// 3D visualization (common for volume data)
displayed: [2, 3, 4]; // Show Z, Y, X

// 1D visualization (for line plots or profiles)
displayed: [4]; // Show only X
```

**Constraints**:

- Maximum 3 displayed dimensions (hardware limitation)
- Displayed dimensions are camera-controlled
- Non-displayed dimensions use slice navigation

## Utility Functions

### initializeDims()

Creates a properly initialized `SimpleDims` object from dataset properties:

```typescript
export function initializeDims(
  numPoints: number,
  totalElements: number,
  metadata?: DimensionMetadata[]
): SimpleDims;
```

**Initialization Logic**:

1. **Dimension Count**: Calculate `ndim` from array structure
2. **Default Position**: Initialize all dimensions at position 0
3. **Display Selection**: Use metadata preferences or default to last 3 dimensions
4. **Validation**: Ensure consistent array structure

**Usage Example**:

```typescript
// Initialize from points data
const dims = initializeDims(100000, 500000, metadata);
// Result: 5D dataset with 100k points, last 3 dims displayed
```

### getDimensionRanges()

Analyzes actual data values to determine dimension bounds:

```typescript
export function getDimensionRanges(
  positions: Float32Array,
  ndim: number,
  numPoints: number
): Array<[number, number]>;
```

**Analysis Process**:

1. **Scan Data**: Examine all point positions across all dimensions
2. **Min/Max Calculation**: Find actual data bounds for each dimension
3. **Return Ranges**: Array of `[min, max]` tuples for navigation setup

**Usage Example**:

```typescript
const ranges = getDimensionRanges(positions, 5, 100000);
// Result: [[0, 10], [0, 5.2], [0, 2], [-50, 50], [-30, 30]]
//         Time    Z      Chan   Y        X
```

## Points Types

The package includes type definitions for point cloud visualization in `points.ts`:

### PointsMetadata

Metadata for points nodes from zarr `.zattrs`:

```typescript
interface PointsMetadata {
  type: 'points';
  n_points: number; // Total point count
  ndim: number; // Position dimensionality
  max_radius?: number; // Maximum point radius
  has_colors?: boolean; // Whether colors array is present
  has_radii?: boolean; // Whether radii array is present
  has_sharpness?: boolean; // Whether sharpness array is present
  // ... additional properties
}
```

### Type Guards

```typescript
import { isPointsMetadata, isPointsUserData } from '../types/points';

if (isPointsMetadata(attrs)) {
  console.log(`Found ${attrs.n_points} points`);
}
```

See `points.ts` for complete interface definitions including `PointsChunkSpatialIndex`, `PointsViewState`, `PointsDataLoader`, and `PointsUserData`.

## Lines Types

The package includes type definitions for line segment visualization in `lines.ts`:

### LinesMetadata

Metadata for lines nodes from zarr `.zattrs`:

```typescript
interface LinesMetadata {
  type: 'lines';
  line_type: LineType; // 'segments' | 'polyline' | 'loop' | 'indexed'
  n_vertices: number; // Total vertex count
  n_segments: number; // Total segment count
  ndim: number; // Position dimensionality
  has_colors?: boolean; // Whether colors array is present
  has_widths?: boolean; // Whether widths array is present
  // ... additional properties
}
```

### Type Guards

```typescript
import { isLinesMetadata, isLinesUserData, isValidLineType } from '../types/lines';

if (isLinesMetadata(attrs)) {
  console.log(`Found ${attrs.n_segments} segments`);
}
```

See `lines.ts` for complete interface definitions including `OrderingMetadata`, `LinesChunkSpatialIndex`, `SegmentRange`, `LoadedLinesData`, `ProcessedLinesData`, `ClippedSegment`, and `LinesViewState`.

## GSplats Types

The package includes complete type definitions for Gaussian Splats visualization in `gsplats.ts`:

### GSplatsMetadata

Metadata for gsplats nodes from zarr `.zattrs`:

```typescript
interface GSplatsMetadata {
  type: 'gsplats';
  n_splats: number; // Total splat count
  ndim: number; // Position dimensionality
  has_colors: boolean; // Whether colors array is present
  has_sharpness: boolean; // Whether sharpness array is present
  chunk_size: number; // Elements per chunk
  amplitude_range: ValueRange; // Amplitude value range
  sharpness_bounds: ValueRange; // Sharpness value bounds
  center_bounds: CoordinateBounds; // Center coordinate bounds
  ordering: 'morton' | 'hilbert' | 'none'; // Spatial ordering method
  extend_to_all?: string[]; // Dimensions to extend visibility across
  // ... additional properties
}
```

### LoadedGSplatsData

Raw gsplats data loaded from zarr before nD projection:

```typescript
interface LoadedGSplatsData {
  positions: Float32Array; // Splat positions (N * ndim)
  amplitudes: Float32Array; // Splat amplitudes (N,)
  choleskyFactors: Float32Array; // Packed Cholesky (N * k) where k = ndim*(ndim+1)/2
  colors: Float32Array | Uint8Array | Uint16Array | null; // RGB colors
  sharpness: Float32Array | null; // Sharpness values (defaults to 2.0)
  splatCount: number;
  ndim: number;
}
```

### Type Guards

```typescript
import {
  isGSplatsMetadata,
  isGSplatsUserData,
  choleskyPackedSize,
  CHOLESKY_SIZES,
} from '../types/gsplats';

// Check if zarr attrs is for gsplats
if (isGSplatsMetadata(attrs)) {
  console.log(`Found ${attrs.n_splats} splats`);
}

// Check if THREE.Object3D is gsplats
if (isGSplatsUserData(mesh.userData)) {
  console.log(`Visible: ${mesh.userData.visibleSplatCount}`);
}

// Compute packed Cholesky factor count for a given dimensionality
const packed = choleskyPackedSize(3); // 6 = 3*(3+1)/2

// Pre-computed sizes for common dimensions (1D-7D)
console.log(CHOLESKY_SIZES); // [1, 3, 6, 10, 15, 21, 28]
```

See `gsplats.ts` for complete interface definitions including `GSplatsChunkSpatialIndex`, `ProcessedGSplatsData`, `GSplatsViewState`, and `GSplatsUserData`.

## Zarr Types

Type definitions in `zarr.ts` for Zarr store attributes and scene graph metadata. These provide proper typing for Zarr `.zattrs` data, eliminating `as any` assertions.

### Key Types

- **`ZarrViewerConfig`** -- Viewer configuration stored in the zarr root `.zattrs` by the Python API. All fields are optional and use `snake_case`. Covers camera, tone mapping, bloom, controls, post-processing, UI panel visibility, dimension navigation state, and animation state. This is also the format exported by Ctrl+Shift+S in the viewer, enabling round-trip Python-to-viewer-to-Python workflows.
- **`ZarrSceneAttrs`** -- Root scene group attributes: format version, scene dimensions, units, position bounds, and an optional `viewer_config`.
- **`ZarrNodeAttrs`** -- Per-node attributes in the scene graph: node type, 4x4 transform, nD transform, rendering properties (opacity, gamma, blending mode, etc.), position bounds, and `extend_to_all`.
- **`SceneDimensionAttrs`** -- Scene-level dimension array mirroring Python's `luxar.core.Dimension` class.
- **`PositionBounds`** -- nD bounding box with `min` and `max` arrays (one entry per dimension).

### nD Transform Types

Per-dimension transforms applied to non-displayed dimensions (see `docs/guides/specs/ND_TRANSFORMS_SPEC.md`):

- **`NdTransformAffine`** -- Affine transform for continuous/discrete dimensions: `{ scale?, offset? }`. Applied as `effective = scale * value + offset`.
- **`NdTransformPermutation`** -- Permutation for categorical dimensions: `{ permutation: number[] }`. Maps old category index to new index.
- **`NdTransformEntry`** -- Union of `NdTransformAffine | NdTransformPermutation`.
- **`NdTransformMap`** -- `Record<string, NdTransformEntry>` mapping dimension name to its transform.

### Type Guards

```typescript
import {
  hasContentsMethod,
  hasTransform,
  hasNdTransform,
  hasSceneDimensions,
  isPermutation,
} from '../types/zarr';
```

- `hasContentsMethod(store)` -- checks if a zarr store supports `contents()`.
- `hasTransform(attrs)` -- checks for a 16-element transform array.
- `hasNdTransform(attrs)` -- checks for an `nd_transform` object.
- `hasSceneDimensions(attrs)` -- checks for `scene_dimensions` with a `dimensions` array.
- `isPermutation(entry)` -- distinguishes permutation entries from affine entries.

## Animation Types

Type definitions in `animation.ts` for FPS-based dimension animation. These types are **not re-exported** from `index.ts` -- import them directly from `../types/animation`.

```typescript
import type {
  LoopMode,
  AnimationDirection,
  DimensionAnimationState,
  DimensionAnimationEvents,
} from '../types/animation';
```

- **`LoopMode`** -- `'once' | 'loop' | 'bounce'`. Controls behavior when animation reaches a dimension boundary.
- **`AnimationDirection`** -- `'forward' | 'backward'`. Current playback direction.
- **`DimensionAnimationState`** -- Full state for a single dimension's animation: `isPlaying`, `targetFPS`, `loopMode`, `direction`, FPS measurement fields (`actualFPS`, `frameCount`, timestamps).
- **`DimensionAnimationEvents`** -- Event map emitted by `DimensionAnimationManager`: `play`, `pause`, `complete`, `speedChange`, `loopModeChange`, `directionChange`, `fpsWarning`.

## Float16Array Type Declaration

The file `float16array.d.ts` provides TypeScript type declarations for `Float16Array`, which is supported in modern browsers (Chrome 122+, Firefox 127+, Safari 17+) but lacks built-in TypeScript definitions. This allows the viewer to handle Float16-encoded zarr arrays without type errors.

## Usage Examples

### Scientific Dataset Setup

```typescript
import { initializeDims, DimensionMetadata } from '../types/dims';

// 4D microscopy data: Time, Z, Y, X
const metadata: DimensionMetadata[] = [
  {
    name: 'Time',
    unit: 's',
    scale: 0.5, // 0.5 seconds per frame
    discrete: true, // Discrete time points
    display: false, // Navigate via slicing
    step: 1,
  },
  {
    name: 'Z',
    unit: 'μm',
    scale: 0.2, // 200 nm Z-steps
    discrete: false, // Continuous space
    display: true, // Show as 3D depth
    step: 0.2,
  },
  {
    name: 'Y',
    unit: 'μm',
    scale: 0.065, // 65 nm pixels
    discrete: false,
    display: true,
    step: 0.065,
  },
  {
    name: 'X',
    unit: 'μm',
    scale: 0.065,
    discrete: false,
    display: true,
    step: 0.065,
  },
];

// Initialize dimension state
const dims = initializeDims(numPoints, totalElements, metadata);
```

### Multi-condition Experiment

```typescript
// 6D dataset: Condition, Time, Channel, Z, Y, X
const experimentMetadata: DimensionMetadata[] = [
  {
    name: 'Condition',
    unit: '',
    scale: 1,
    discrete: true, // Control vs Treatment
    display: false,
    step: 1,
  },
  {
    name: 'Timepoint',
    unit: 'h',
    scale: 2, // 2-hour intervals
    discrete: true,
    display: false,
    step: 1,
  },
  {
    name: 'Channel',
    unit: '',
    scale: 1, // Fluorescence channels
    discrete: true,
    display: false,
    step: 1,
  },
  // ... spatial dimensions (Z, Y, X) with continuous navigation
];
```

### Navigation Through Dimensions

**Note**: Navigation utilities are implemented in the `input` package. See `../input/README.md` for complete navigation API documentation.

```typescript
// Navigation is handled by the InputHandler class in the input package
import { InputHandler } from '../input/input-handler';

// InputHandler provides dimension navigation methods:
// - navigateDimension(dimIndex, direction): Navigate forward/backward
// - setDimensionPosition(dimIndex, position): Jump to specific position
// - getNonDisplayedDimensions(): Get list of navigable dimensions

// Example: Create input handler for dimension navigation
const inputHandler = new InputHandler(sceneManager, config);

// Navigate forward in time dimension (typically first non-displayed dimension)
const success = inputHandler.navigateDimension(0, 1);

// Check which dimensions can be keyboard-navigated
const navigableDims = inputHandler.getNonDisplayedDimensions();
// Returns indices of dimensions not currently displayed (e.g., [0, 1, 4])
```

## Type Safety

### Compile-time Validation

TypeScript interfaces prevent common errors at compile time:

```typescript
// ✅ Type-safe dimension access
const currentZ = dims.currentStep[zDimension];

// ❌ Compile error: dimension index must be number
const invalidDim = dims.currentStep['z'];

// ✅ Type-safe metadata access
const zUnit = dims.metadata?.[2]?.unit ?? 'unknown';

// ❌ Compile error: display property is boolean
dims.metadata[0].display = 'yes'; // Should be boolean
```

### Runtime Validation

Utility functions include runtime validation for data consistency:

```typescript
// Validates array structure consistency
const dims = initializeDims(numPoints, totalElements, metadata);
// Throws: "Invalid positions array: 500001 elements for 100000 points"
// if totalElements is not evenly divisible by numPoints
```

### Optional Properties

Metadata properties are carefully designed with optional fields:

```typescript
interface DimensionMetadata {
  name: string; // Required
  unit: string; // Required
  scale: number; // Required
  range?: [number, number]; // Optional: calculated from data if missing
  display?: boolean; // Optional: defaults to false
  discrete?: boolean; // Optional: defaults to false (continuous)
  step?: number; // Optional: calculated from range if missing
}
```

## Best Practices

### Dimension Ordering

Follow consistent conventions for dimension ordering:

```typescript
// ✅ Good: Standard scientific convention (T, Z, C, Y, X)
const standardOrder = ['Time', 'Z', 'Channel', 'Y', 'X'];

// ✅ Good: Physics convention (T, X, Y, Z)
const physicsOrder = ['Time', 'X', 'Y', 'Z'];

// ❌ Avoid: Inconsistent or unclear ordering
const confusingOrder = ['X', 'Time', 'Y', 'Channel', 'Z'];
```

### Metadata Completeness

Provide complete metadata for better user experience:

```typescript
// ✅ Good: Complete semantic information
const completeMetadata: DimensionMetadata = {
  name: 'Time', // Clear, descriptive name
  unit: 'min', // Standard unit with clear meaning
  scale: 2.5, // Explicit conversion factor
  discrete: true, // Navigation behavior specification
  display: false, // Clear display intent
  step: 1, // Appropriate step size
  range: [0, 120], // Expected data bounds
};

// ❌ Avoid: Minimal or unclear metadata
const poorMetadata = {
  name: 'D0', // Generic name
  unit: '', // Missing unit information
  scale: 1, // No additional context
};
```

### Type Guards and Validation

Use type guards for safe metadata access:

```typescript
// ✅ Good: Safe metadata access
function getDimensionName(dims: SimpleDims, index: number): string {
  if (index < 0 || index >= dims.ndim) {
    return `D${index}`;
  }

  return dims.metadata?.[index]?.name ?? `D${index}`;
}

// ✅ Good: Validate dimension state consistency
function validateDims(dims: SimpleDims): boolean {
  // Check displayed dimensions are within bounds
  return (
    dims.displayed.every((d) => d >= 0 && d < dims.ndim) &&
    dims.currentStep.length === dims.ndim &&
    dims.displayed.length <= 3
  );
}
```

### Performance Considerations

```typescript
// ✅ Good: Cache dimension lookups
const displayedSet = new Set(dims.displayed);
const isDisplayed = displayedSet.has(dimIndex); // O(1)

// ❌ Avoid: Linear search for each check
const isDisplayed = dims.displayed.includes(dimIndex); // O(n)

// ✅ Good: Reuse range calculations
const ranges = getDimensionRanges(positions, dims.ndim, numPoints);
// Use ranges for multiple navigation operations
```

The types package provides the type-safe foundation for all nD visualization operations in Luxar, ensuring data consistency and enabling rich semantic interpretation of high-dimensional datasets.
