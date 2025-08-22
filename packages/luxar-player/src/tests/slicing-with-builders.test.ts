/**
 * Slicing tests using test data builders
 *
 * This demonstrates how test data builders improve test readability
 * and maintainability compared to manually creating test data.
 */

import { describe, it, expect } from 'vitest';
import {
  slicePoints,
  extractDisplayDimensions,
  sliceColorsFloat32,
  computeEffectiveRadii,
  getDimsLabel,
} from '../utils/slicing';
import { PointCloudBuilder, DimensionsBuilder } from './builders/test-data-builders';

describe('slicing with builders', () => {
  describe('slicePoints with builders', () => {
    it('should filter 4D points based on time dimension', () => {
      // Build test data with clear intent
      const pointCloud = new PointCloudBuilder()
        .withPoints(4)
        .withDimensions(4)
        .withPositions([
          0,
          0,
          0,
          0.0, // Point at time=0.0
          1,
          1,
          1,
          0.4, // Point at time=0.4
          2,
          2,
          2,
          0.6, // Point at time=0.6
          3,
          3,
          3,
          1.5, // Point at time=1.5
        ])
        .build();

      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withSpatialDimensions()
        .withTimeDimension([0, 2], 0.1)
        .withDisplayed(0, 1, 2)
        .withCurrentPosition(0, 0, 0, 0.5) // Slice at time=0.5
        .build();

      const result = slicePoints(
        pointCloud.positions,
        dims,
        pointCloud.numPoints,
        undefined,
        0.2 // tolerance
      );

      // Points at time 0.4 and 0.6 are within tolerance of 0.5
      expect(Array.from(result)).toEqual([1, 2]);
    });

    it('should handle discrete dimensions', () => {
      const pointCloud = new PointCloudBuilder()
        .withPoints(4)
        .withDimensions(4)
        .withPositions([
          0,
          0,
          0,
          0, // Channel 0
          1,
          1,
          1,
          1, // Channel 1
          2,
          2,
          2,
          1, // Channel 1
          3,
          3,
          3,
          2, // Channel 2
        ])
        .build();

      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withSpatialDimensions()
        .withDimension(3, 'channel', '', [0, 2], {
          discrete: true,
          step: 1,
          display: false,
        })
        .withDisplayed(0, 1, 2)
        .withCurrentPosition(0, 0, 0, 1) // Select channel 1
        .build();

      const result = slicePoints(pointCloud.positions, dims, 4);

      // Only points with channel=1 are selected
      expect(Array.from(result)).toEqual([1, 2]);
    });

    it('should use per-point radii for slicing', () => {
      const pointCloud = new PointCloudBuilder()
        .withPoints(3)
        .withDimensions(4)
        .withPositions([
          0,
          0,
          0,
          0.0, // Point at time=0.0
          1,
          1,
          1,
          0.3, // Point at time=0.3
          2,
          2,
          2,
          0.8, // Point at time=0.8
        ])
        .withVaryingRadii(0.1, 0.5)
        .build();

      // Set specific radii for testing
      pointCloud.radii![0] = 0.1; // Small radius
      pointCloud.radii![1] = 0.3; // Medium radius
      pointCloud.radii![2] = 0.5; // Large radius

      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withSpatialDimensions()
        .withTimeDimension()
        .withDisplayed(0, 1, 2)
        .withCurrentPosition(0, 0, 0, 0.5)
        .build();

      const result = slicePoints(pointCloud.positions, dims, 3, pointCloud.radii!);

      // Point 0: |0.0 - 0.5| = 0.5 > 0.1 (excluded)
      // Point 1: |0.3 - 0.5| = 0.2 < 0.3 (included)
      // Point 2: |0.8 - 0.5| = 0.3 < 0.5 (included)
      expect(Array.from(result)).toEqual([1, 2]);
    });
  });

  describe('extractDisplayDimensions with builders', () => {
    it('should extract selected dimensions from nD data', () => {
      const pointCloud = new PointCloudBuilder()
        .withPoints(2)
        .withDimensions(5)
        .withPositions([
          1,
          2,
          3,
          4,
          5, // Point 0
          6,
          7,
          8,
          9,
          10, // Point 1
        ])
        .build();

      const dims = new DimensionsBuilder()
        .withNDimensions(5)
        .withDisplayed(0, 2, 4) // Display dimensions 0, 2, 4
        .build();

      const indices = new Uint32Array([0, 1]);
      const result = extractDisplayDimensions(pointCloud.positions, indices, dims);

      expect(Array.from(result)).toEqual([
        1,
        3,
        5, // Point 0: dims 0, 2, 4
        6,
        8,
        10, // Point 1: dims 0, 2, 4
      ]);
    });

    it('should pad when fewer than 3 dimensions displayed', () => {
      const pointCloud = new PointCloudBuilder()
        .withPoints(2)
        .withDimensions(4)
        .withPositions([1, 2, 3, 4, 5, 6, 7, 8])
        .build();

      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withDisplayed(1, 3) // Only 2 dimensions
        .build();

      const indices = new Uint32Array([0, 1]);
      const result = extractDisplayDimensions(pointCloud.positions, indices, dims);

      expect(Array.from(result)).toEqual([
        2,
        4,
        0, // Point 0: dims 1, 3, padded
        6,
        8,
        0, // Point 1: dims 1, 3, padded
      ]);
    });
  });

  describe('computeEffectiveRadii with builders', () => {
    it('should compute effective radii for sliced hyperspheres', () => {
      const pointCloud = new PointCloudBuilder()
        .withPoints(3)
        .withDimensions(4)
        .withPositions([
          0,
          0,
          0,
          0, // At slice plane
          1,
          1,
          1,
          0.3, // Offset from slice
          2,
          2,
          2,
          0.4, // Offset from slice
        ])
        .withRadii(1.0) // Uniform radius initially
        .build();

      // Set varying radii
      pointCloud.radii![2] = 0.5;

      const dims = new DimensionsBuilder()
        .withNDimensions(4)
        .withSpatialDimensions()
        .withTimeDimension()
        .withDisplayed(0, 1, 2)
        .withCurrentPosition(0, 0, 0, 0) // Slice at time=0
        .build();

      const indices = new Uint32Array([0, 1, 2]);
      const result = computeEffectiveRadii(pointCloud.positions, indices, dims, pointCloud.radii!);

      // Point 0: at slice, full radius
      expect(result[0]).toBeCloseTo(1.0);

      // Point 1: distance 0.3, effective = √(1² - 0.3²)
      expect(result[1]).toBeCloseTo(Math.sqrt(1 - 0.09));

      // Point 2: distance 0.4, effective = √(0.5² - 0.4²) = 0.3
      expect(result[2]).toBeCloseTo(0.3);
    });
  });

  describe('sliceColorsFloat32 with builders', () => {
    it('should preserve HDR colors during slicing', () => {
      const pointCloud = new PointCloudBuilder()
        .withPoints(4)
        .withDimensions(3)
        .withRandomPositions()
        .withColors([
          1.0,
          0.0,
          0.0, // Normal red
          0.0,
          2.5,
          0.0, // HDR green
          0.0,
          0.0,
          0.5, // Dark blue
          10.0,
          10.0,
          10.0, // HDR white
        ])
        .build();

      const indices = new Uint32Array([1, 3]); // Select HDR colors
      const result = sliceColorsFloat32(pointCloud.colors, indices);

      expect(result).not.toBeNull();
      expect(Array.from(result!)).toEqual([
        0.0,
        2.5,
        0.0, // HDR green preserved
        10.0,
        10.0,
        10.0, // HDR white preserved
      ]);
    });
  });

  describe('getDimsLabel with builders', () => {
    it('should generate human-readable labels', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(5)
        .withSpatialDimensions('μm', [0, 100])
        .withTimeDimension([0, 10], 0.1)
        .withChannelDimension(4)
        .withDisplayed(0, 1, 2)
        .withCurrentPosition(0, 0, 0, 5.25, 2.0)
        .build();

      const label = getDimsLabel(dims);

      expect(label).toBe('time=5.25s, channel=2.00');
    });

    it('should handle complex dimension configurations', () => {
      const dims = new DimensionsBuilder()
        .withNDimensions(7)
        .withDimension(0, 'x', 'μm', [0, 100], { display: true })
        .withDimension(1, 'y', 'μm', [0, 100], { display: true })
        .withDimension(2, 'z', 'μm', [0, 50], { display: true })
        .withDimension(3, 'time', 's', [0, 10], { display: false })
        .withDimension(4, 'channel', '', [0, 3], { display: false, discrete: true })
        .withDimension(5, 'angle', '°', [0, 360], { display: false })
        .withDimension(6, 'depth', 'mm', [0, 10], { display: false })
        .withDisplayed(0, 1, 2)
        .withCurrentPosition(0, 0, 0, 2.5, 1, 45.0, 3.75)
        .build();

      const label = getDimsLabel(dims);

      expect(label).toContain('time=2.50s');
      expect(label).toContain('channel=1.00');
      expect(label).toContain('angle=45.00°');
      expect(label).toContain('depth=3.75mm');
    });
  });
});
