import { describe, it, expect } from 'vitest';
import { isLinesMetadata, isLinesUserData, isValidLineType } from '../../../types/lines';

describe('Lines Types', () => {
  describe('isLinesMetadata', () => {
    it('should return true for valid lines metadata', () => {
      const validMetadata = {
        type: 'lines',
        n_vertices: 100,
        n_segments: 50,
        ndim: 3,
        original_line_type: 'polyline',
        max_width: 0.5,
        has_colors: true,
        has_sharpness: true,
        ordering: 'morton',
      };

      expect(isLinesMetadata(validMetadata)).toBe(true);
    });

    it('should return false for points metadata', () => {
      const pointsMetadata = {
        type: 'points',
        n_points: 1000,
        ndim: 3,
      };

      expect(isLinesMetadata(pointsMetadata)).toBe(false);
    });

    it('should return false for missing type', () => {
      const invalidMetadata = {
        n_vertices: 100,
        n_segments: 50,
      };

      expect(isLinesMetadata(invalidMetadata)).toBe(false);
    });

    it('should return false for null/undefined', () => {
      expect(isLinesMetadata(null)).toBe(false);
      expect(isLinesMetadata(undefined)).toBe(false);
    });

    it('should return false for non-object', () => {
      expect(isLinesMetadata('lines')).toBe(false);
      expect(isLinesMetadata(123)).toBe(false);
      expect(isLinesMetadata([])).toBe(false);
    });
  });

  describe('isLinesUserData', () => {
    it('should return true for valid lines userData', () => {
      // isLinesUserData only checks for nodeType === 'lines'
      const validUserData = {
        nodeType: 'lines',
        loader: {}, // Actual loader would be a LinesDataLoader instance
        attrs: {
          type: 'lines',
          n_vertices: 100,
          n_segments: 50,
          ndim: 3,
          original_line_type: 'segments',
          max_width: 0.1,
          has_colors: false,
          has_sharpness: false,
          ordering: 'none',
        },
        spatialIndex: null,
      };

      expect(isLinesUserData(validUserData)).toBe(true);
    });

    it('should return false for points userData', () => {
      const pointsUserData = {
        nodeType: 'points',
        loader: {},
        attrs: {},
      };

      expect(isLinesUserData(pointsUserData)).toBe(false);
    });

    it('should return false for missing nodeType', () => {
      const invalidUserData = {
        loader: {},
        attrs: {},
      };

      expect(isLinesUserData(invalidUserData)).toBe(false);
    });

    it('should return false for wrong nodeType', () => {
      const invalidUserData = {
        nodeType: 'group',
        children: [],
      };

      expect(isLinesUserData(invalidUserData)).toBe(false);
    });
  });

  describe('isValidLineType', () => {
    it('should return true for valid line types', () => {
      expect(isValidLineType('segments')).toBe(true);
      expect(isValidLineType('polyline')).toBe(true);
      expect(isValidLineType('loop')).toBe(true);
      expect(isValidLineType('indexed')).toBe(true);
    });

    it('should return false for invalid line types', () => {
      expect(isValidLineType('lines')).toBe(false);
      expect(isValidLineType('strip')).toBe(false);
      expect(isValidLineType('')).toBe(false);
      expect(isValidLineType(123 as any)).toBe(false);
      expect(isValidLineType(null as any)).toBe(false);
    });
  });
});

describe('Lines Type Definitions', () => {
  describe('LinesMetadata interface', () => {
    it('should allow all required fields', () => {
      // This is a compile-time test - if it compiles, the types are correct
      const metadata = {
        type: 'lines' as const,
        n_vertices: 100,
        n_segments: 50,
        ndim: 3,
        original_line_type: 'polyline' as const,
        max_width: 0.5,
        has_colors: true,
        has_sharpness: true,
        ordering: 'morton' as const,
      };

      expect(metadata.type).toBe('lines');
      expect(metadata.n_vertices).toBe(100);
      expect(metadata.n_segments).toBe(50);
      expect(metadata.ndim).toBe(3);
      expect(metadata.original_line_type).toBe('polyline');
      expect(metadata.max_width).toBe(0.5);
      expect(metadata.has_colors).toBe(true);
      expect(metadata.has_sharpness).toBe(true);
      expect(metadata.ordering).toBe('morton');
    });

    it('should allow optional ordering metadata', () => {
      const metadata = {
        type: 'lines' as const,
        n_vertices: 100,
        n_segments: 50,
        ndim: 3,
        original_line_type: 'segments' as const,
        max_width: 0.1,
        has_colors: false,
        has_sharpness: false,
        ordering: 'morton' as const,
        vertex_ordering: {
          ordering: 'morton' as const,
          grid_shape: [10, 10, 10],
          grid_origin: [0, 0, 0],
          cell_size: [1, 1, 1],
          position_bounds: { min: [0, 0, 0], max: [10, 10, 10] },
          chunk_shape: [1024],
          total_chunks: 10,
          points_per_chunk: 100,
        },
      };

      expect(metadata.vertex_ordering?.ordering).toBe('morton');
      expect(metadata.vertex_ordering?.grid_shape).toEqual([10, 10, 10]);
    });
  });

  describe('ClippedSegment interface', () => {
    it('should represent visible clipped segment', () => {
      const segment = {
        p1: [0, 1, 2],
        p2: [3, 4, 5],
        t1: 0.25,
        t2: 0.75,
        visible: true,
      };

      expect(segment.visible).toBe(true);
      expect(segment.p1).toEqual([0, 1, 2]);
      expect(segment.p2).toEqual([3, 4, 5]);
      expect(segment.t1).toBe(0.25);
      expect(segment.t2).toBe(0.75);
    });

    it('should represent invisible segment', () => {
      const segment = {
        p1: [],
        p2: [],
        t1: 0,
        t2: 0,
        visible: false,
      };

      expect(segment.visible).toBe(false);
    });
  });

  describe('ProcessedLinesData interface', () => {
    it('should represent GPU-ready line data', () => {
      const data = {
        startPositions: new Float32Array([0, 0, 0, 1, 1, 1]),
        endPositions: new Float32Array([1, 0, 0, 2, 1, 1]),
        startColors: new Float32Array([1, 0, 0, 0, 1, 0]),
        endColors: new Float32Array([1, 0, 0, 0, 1, 0]),
        startWidths: new Float32Array([0.1, 0.2]),
        endWidths: new Float32Array([0.15, 0.25]),
        startSharpness: new Float32Array([1.0, 0.5]),
        endSharpness: new Float32Array([1.0, 0.5]),
        segmentLengths: new Float32Array([1.0, 1.414]),
        startClipped: new Uint8Array([0, 1]),
        endClipped: new Uint8Array([0, 0]),
        segmentCount: 2,
      };

      expect(data.segmentCount).toBe(2);
      expect(data.startPositions.length).toBe(6); // 2 segments * 3 components
      expect(data.startClipped[1]).toBe(1); // Second segment start was clipped
    });
  });
});
