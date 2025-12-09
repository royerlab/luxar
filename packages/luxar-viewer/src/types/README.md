# Types Package

TypeScript type definitions for high-dimensional data visualization in Luxar. This package provides the fundamental data structures and interfaces for managing nD points data, dimension metadata, and coordinate system definitions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Core Types](#core-types)
- [Dimension Metadata](#dimension-metadata)
- [SimpleDims Interface](#simpledims-interface)
- [Utility Functions](#utility-functions)
- [Usage Examples](#usage-examples)
- [Type Safety](#type-safety)
- [Best Practices](#best-practices)

## Overview

The types package defines the foundational type system for Luxar's nD visualization capabilities. It provides type-safe interfaces for representing high-dimensional datasets, managing dimension states, and handling coordinate system transformations.

**Core Philosophy**: Provide precise, well-documented types that capture the mathematical and conceptual structure of high-dimensional data visualization while ensuring compile-time safety and runtime reliability.

## Key Features

- **nD Data Structures**: Complete type definitions for high-dimensional points
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

```typescript
import { stepDimension, jumpToDimension } from '../utils/dims-navigation';

// Step through time dimension
const timeIndex = 1;
const didChange = stepDimension(dims, timeIndex, 1, ranges, {
  stepSize: 0.1, // 10% of time range per step
  wrap: true, // Loop back to start
});

// Jump to specific Z position (50% through range)
const zIndex = 2;
jumpToDimension(dims, zIndex, 0.5, ranges);

// Check which dimensions can be keyboard-navigated
const [primaryDim, secondaryDim] = getNavigableDimensions(dims);
// Typically returns first two non-displayed dimensions
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
