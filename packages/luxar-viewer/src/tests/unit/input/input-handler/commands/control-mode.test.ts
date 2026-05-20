/**
 * Unit tests for the camera control-mode cycle helper.
 */

import { describe, it, expect } from 'vitest';
import { nextControlType } from '../../../../../input/input-handler/commands/control-mode';

describe('nextControlType', () => {
  it('cycles orbit → fly → ortho → orbit', () => {
    expect(nextControlType('orbit')).toBe('fly');
    expect(nextControlType('fly')).toBe('ortho');
    expect(nextControlType('ortho')).toBe('orbit');
  });

  it('completes a full cycle in three steps', () => {
    let current = 'orbit' as const;
    const sequence: string[] = [current];
    for (let i = 0; i < 3; i++) {
      const next = nextControlType(current);
      sequence.push(next);
      current = next as typeof current;
    }
    expect(sequence).toEqual(['orbit', 'fly', 'ortho', 'orbit']);
  });

  it('falls back to orbit for unknown control types', () => {
    expect(nextControlType('unknown')).toBe('orbit');
    expect(nextControlType('')).toBe('orbit');
  });
});
