import { describe, expect, it } from 'vitest';

import {
  MAX_WIRE_DEPTH,
  encodeBlobForWire,
  sanitizeForWire,
} from '../../../../../core/app/control/wire-values';

describe('sanitizeForWire', () => {
  it('normalizes non-JSON scalar values', () => {
    expect(sanitizeForWire(new Error('boom'))).toEqual({ name: 'Error', message: 'boom' });
    expect(sanitizeForWire(Number.NaN)).toBeNull();
    expect(sanitizeForWire(Number.POSITIVE_INFINITY)).toBeNull();
    expect(sanitizeForWire(12n)).toBe('12');
    expect(sanitizeForWire(() => undefined)).toBeUndefined();
  });

  it('preserves array positions while dropping object functions', () => {
    const sparse: Array<number | undefined> = [1, undefined, 3];
    delete sparse[1];
    const sanitized = sanitizeForWire({ sparse, keep: true, drop: () => undefined }) as {
      sparse: unknown[];
      keep: boolean;
      drop?: unknown;
    };

    expect(sanitized.sparse).toHaveLength(3);
    expect(1 in sanitized.sparse).toBe(false);
    expect(sanitized.keep).toBe(true);
    expect(sanitized).not.toHaveProperty('drop');
  });

  it('copies typed arrays and caps excessive depth', () => {
    expect(sanitizeForWire(new Float32Array([1, 2.5]))).toEqual([1, 2.5]);

    let nested: Record<string, unknown> = {};
    const root = nested;
    for (let index = 0; index <= MAX_WIRE_DEPTH; index += 1) {
      nested.next = {};
      nested = nested.next as Record<string, unknown>;
    }
    expect(JSON.stringify(sanitizeForWire(root))).toContain('"next":null');
  });
});

describe('encodeBlobForWire', () => {
  it('round-trips a multi-chunk blob without spreading the whole payload', async () => {
    const bytes = Uint8Array.from({ length: 0x8000 * 2 + 17 }, (_, index) => index % 251);
    const encoded = await encodeBlobForWire(new Blob([bytes], { type: 'image/png' }));
    const decoded = Uint8Array.from(atob(encoded.base64), (character) => character.charCodeAt(0));

    expect(encoded.mime).toBe('image/png');
    expect(decoded).toEqual(bytes);
  });
});
