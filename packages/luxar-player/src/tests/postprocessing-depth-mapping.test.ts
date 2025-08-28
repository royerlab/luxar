/**
 * Tests for perspective depth mapping utilities
 */

import { describe, it, expect } from 'vitest';
import { PerspectiveDepthMapper } from '../rendering/postprocessing-types';

describe('PerspectiveDepthMapper', () => {
  const near = 0.1;
  const far = 1000;

  describe('worldToNormalizedDepth', () => {
    it('should map near plane to 0', () => {
      const result = PerspectiveDepthMapper.worldToNormalizedDepth(near, near, far);
      expect(result).toBeCloseTo(0, 5);
    });

    it('should map far plane to 1', () => {
      const result = PerspectiveDepthMapper.worldToNormalizedDepth(far, near, far);
      expect(result).toBeCloseTo(1, 5);
    });

    it('should clamp values outside range', () => {
      const belowNear = PerspectiveDepthMapper.worldToNormalizedDepth(0.01, near, far);
      expect(belowNear).toBeCloseTo(0, 5);

      const beyondFar = PerspectiveDepthMapper.worldToNormalizedDepth(2000, near, far);
      expect(beyondFar).toBeCloseTo(1, 5);
    });

    it('should provide non-linear mapping for perspective projection', () => {
      // Test that depth mapping is non-linear (inverse relationship)
      const d1 = 1.0;
      const d2 = 10.0;
      const d3 = 100.0;
      
      const n1 = PerspectiveDepthMapper.worldToNormalizedDepth(d1, near, far);
      const n2 = PerspectiveDepthMapper.worldToNormalizedDepth(d2, near, far);
      const n3 = PerspectiveDepthMapper.worldToNormalizedDepth(d3, near, far);
      
      // The difference between n1 and n2 should be larger than between n2 and n3
      // This is because perspective projection gives more precision to near objects
      const diff1 = n2 - n1;
      const diff2 = n3 - n2;
      
      expect(diff1).toBeGreaterThan(diff2);
    });

    it('should handle edge cases correctly', () => {
      // Test with very small near plane
      const tinyNear = 0.001;
      const result = PerspectiveDepthMapper.worldToNormalizedDepth(0.01, tinyNear, far);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThanOrEqual(1);

      // Test with very large far plane
      const hugeFar = 100000;
      const result2 = PerspectiveDepthMapper.worldToNormalizedDepth(5000, near, hugeFar);
      expect(result2).toBeGreaterThanOrEqual(0);
      expect(result2).toBeLessThanOrEqual(1);
    });
  });

  describe('normalizedDepthToWorld', () => {
    it('should correctly reverse the depth mapping', () => {
      const distances = [0.5, 1, 5, 10, 50, 100, 500];
      
      for (const distance of distances) {
        const normalized = PerspectiveDepthMapper.worldToNormalizedDepth(distance, near, far);
        const reversed = PerspectiveDepthMapper.normalizedDepthToWorld(normalized, near, far);
        
        expect(reversed).toBeCloseTo(distance, 3);
      }
    });

    it('should map 0 to near plane', () => {
      const result = PerspectiveDepthMapper.normalizedDepthToWorld(0, near, far);
      expect(result).toBeCloseTo(near, 5);
    });

    it('should map 1 to far plane', () => {
      const result = PerspectiveDepthMapper.normalizedDepthToWorld(1, near, far);
      expect(result).toBeCloseTo(far, 5);
    });
  });

  describe('worldToLogDepth', () => {
    it('should provide logarithmic distribution', () => {
      const d1 = 1.0;
      const d2 = 10.0;
      const d3 = 100.0;
      
      const n1 = PerspectiveDepthMapper.worldToLogDepth(d1, near, far);
      const n2 = PerspectiveDepthMapper.worldToLogDepth(d2, near, far);
      const n3 = PerspectiveDepthMapper.worldToLogDepth(d3, near, far);
      
      // For logarithmic mapping, the ratios should be more uniform
      // Ratio d2/d1 = 10, Ratio d3/d2 = 10
      // Since ratios are equal, normalized differences should be similar
      const diff1 = n2 - n1;
      const diff2 = n3 - n2;
      
      expect(Math.abs(diff1 - diff2)).toBeLessThan(0.1);
    });

    it('should map near and far correctly', () => {
      const nearResult = PerspectiveDepthMapper.worldToLogDepth(near, near, far);
      expect(nearResult).toBeCloseTo(0, 5);

      const farResult = PerspectiveDepthMapper.worldToLogDepth(far, near, far);
      expect(farResult).toBeCloseTo(1, 5);
    });
  });

  describe('logDepthToWorld', () => {
    it('should correctly reverse logarithmic depth', () => {
      const distances = [0.5, 1, 5, 10, 50, 100, 500];
      
      for (const distance of distances) {
        const logDepth = PerspectiveDepthMapper.worldToLogDepth(distance, near, far);
        const reversed = PerspectiveDepthMapper.logDepthToWorld(logDepth, near, far);
        
        expect(reversed).toBeCloseTo(distance, 3);
      }
    });
  });

  describe('Comparison of mapping methods', () => {
    it('should show different distributions for linear vs log mapping', () => {
      const testDistance = 10.0;
      
      const inverseDepth = PerspectiveDepthMapper.worldToNormalizedDepth(testDistance, near, far);
      const logDepth = PerspectiveDepthMapper.worldToLogDepth(testDistance, near, far);
      
      // These should be different values due to different mapping functions
      expect(Math.abs(inverseDepth - logDepth)).toBeGreaterThan(0.01);
    });

    it('should maintain monotonic increasing property', () => {
      const distances = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];
      
      let prevInverse = -1;
      let prevLog = -1;
      
      for (const d of distances) {
        const inverse = PerspectiveDepthMapper.worldToNormalizedDepth(d, near, far);
        const log = PerspectiveDepthMapper.worldToLogDepth(d, near, far);
        
        // Both mappings should be monotonically increasing
        expect(inverse).toBeGreaterThan(prevInverse);
        expect(log).toBeGreaterThan(prevLog);
        
        prevInverse = inverse;
        prevLog = log;
      }
    });
  });
});