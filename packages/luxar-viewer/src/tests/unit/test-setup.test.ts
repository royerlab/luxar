import { describe, expect, it } from 'vitest';

describe('test setup', () => {
  it('preserves native Reflect.construct semantics for non-constructible functions', () => {
    expect(() => Reflect.construct(() => ({}), [])).toThrow(TypeError);
  });
});
