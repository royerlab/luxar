import { describe, expect, it } from 'vitest';
import { detectMimeType } from '../../../utils/image-mime';

describe('detectMimeType', () => {
  it.each([
    [new Uint8Array([0xff, 0xd8]), 'image/jpeg'],
    [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), 'image/png'],
    [
      new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]),
      'image/webp',
    ],
  ])('detects image magic bytes', (bytes, expected) => {
    expect(detectMimeType(bytes)).toBe(expected);
  });

  it('falls back to PNG for unknown or truncated payloads', () => {
    expect(detectMimeType(new Uint8Array())).toBe('image/png');
    expect(detectMimeType(new Uint8Array([0x89, 0x50, 0x4e, 0x47]))).toBe('image/png');
  });
});
