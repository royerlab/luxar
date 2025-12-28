/**
 * Unit tests for Points types and utilities.
 *
 * Tests type guards and utility functions for Points data structures.
 */

import { describe, it, expect } from 'vitest';
import { isPointsUserData } from '../../../types/points';

describe('Points Types', () => {
  describe('isPointsUserData', () => {
    it('should return true for valid PointsUserData', () => {
      const validData = {
        nodeType: 'points' as const,
        attrs: {
          n_points: 100,
          max_radius: 1.0,
        },
      };

      expect(isPointsUserData(validData)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isPointsUserData(null)).toBe(false);
    });

    it('should return false for undefined', () => {
      expect(isPointsUserData(undefined)).toBe(false);
    });

    it('should return false for non-object values', () => {
      expect(isPointsUserData('string')).toBe(false);
      expect(isPointsUserData(123)).toBe(false);
      expect(isPointsUserData(true)).toBe(false);
    });

    it('should return false for wrong nodeType', () => {
      const wrongType = {
        nodeType: 'lines',
        attrs: {},
      };

      expect(isPointsUserData(wrongType)).toBe(false);
    });

    it('should return false for missing nodeType', () => {
      const missingNodeType = {
        attrs: {},
      };

      expect(isPointsUserData(missingNodeType)).toBe(false);
    });

    it('should return true with optional visiblePointCount', () => {
      const withCount = {
        nodeType: 'points' as const,
        attrs: {
          n_points: 100,
        },
        visiblePointCount: 50,
      };

      expect(isPointsUserData(withCount)).toBe(true);
    });
  });
});
