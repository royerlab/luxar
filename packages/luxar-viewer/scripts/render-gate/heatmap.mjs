/**
 * Write a per-pixel ULP16 difference map as a PNG: black where the builds
 * agree exactly, blue for drift (0 < d <= FLIP_ULP, brighter = larger), red for
 * flips (d > FLIP_ULP). Rows are written as captured (top-down).
 *
 * A dependency-free PNG encoder (`node:zlib` deflate + crc32), because the
 * gate should not add a package for one debug image.
 *
 * @module scripts/render-gate/heatmap
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { crc32, deflateSync } from 'node:zlib';

import { FLIP_ULP } from './exactness.mjs';

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body) >>> 0);
  return Buffer.concat([len, body, crc]);
}

/**
 * Encode an RGB8 image as PNG bytes.
 *
 * @param {Uint8Array} rgb Packed RGB, row-major, top-down.
 * @param {number} width Image width.
 * @param {number} height Image height.
 * @returns {Buffer} PNG file bytes.
 */
export function encodePng(rgb, width, height) {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0; // filter: none
    Buffer.from(rgb.buffer, rgb.byteOffset + y * width * 3, width * 3).copy(
      raw,
      y * (width * 3 + 1) + 1
    );
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type RGB
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  return Buffer.concat([
    signature,
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/**
 * @param {string} path Output PNG path (parent directories are created).
 * @param {Float64Array} perPixelUlp ULP16 distance per pixel.
 * @param {number} width Capture width.
 * @param {number} height Capture height.
 */
export function writeUlpHeatmap(path, perPixelUlp, width, height) {
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const d = perPixelUlp[i];
    if (d === 0) continue;
    if (d > FLIP_ULP) {
      rgb[i * 3] = 255;
      rgb[i * 3 + 1] = Math.min(255, Math.round(Math.log2(Math.min(d, 1e6)) * 12));
    } else {
      rgb[i * 3 + 2] = Math.round(80 + (175 * d) / FLIP_ULP);
    }
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, encodePng(rgb, width, height));
}
