import { beforeEach, describe, expect, it, vi } from 'vitest';
import { perfCounters } from '../../../profiling/perf-counters';
import {
  installUploadCounters,
  wrapWebGLUploads,
  wrapWebGPUUploads,
} from '../../../rendering/upload-counters';

type Upload = (...args: unknown[]) => unknown;
const upload = () => vi.fn<Upload>();

function fakeGL() {
  return {
    bufferData: vi.fn<Upload>(() => 'bd'),
    bufferSubData: upload(),
    texImage2D: upload(),
    texImage3D: upload(),
    texSubImage2D: upload(),
    texSubImage3D: upload(),
    compressedTexImage2D: upload(),
    compressedTexImage3D: upload(),
    compressedTexSubImage2D: upload(),
    compressedTexSubImage3D: upload(),
  };
}

function fakeQueue() {
  return {
    writeBuffer: upload(),
    writeTexture: upload(),
    copyExternalImageToTexture: upload(),
  };
}

const bytes = () => perfCounters.get('gpu.uploadBytes');
const calls = () => perfCounters.get('gpu.uploadCalls');
const bufferBytes = () => perfCounters.get('gpu.uploadBytes.buffer');
const textureBytes = () => perfCounters.get('gpu.uploadBytes.texture');

describe('upload-counters', () => {
  beforeEach(() => perfCounters.reset());

  describe('WebGL', () => {
    it('forwards the exact arguments and result', () => {
      const gl = fakeGL();
      const original = gl.bufferData;
      wrapWebGLUploads(gl);
      const data = new Float32Array(4);
      expect(gl.bufferData(1, data, 2)).toBe('bd');
      expect(original).toHaveBeenCalledWith(1, data, 2);
      expect(original.mock.calls[0]).toHaveLength(3);
    });

    it('counts bufferData by source bytes, numeric size, and WebGL2 srcOffset/length', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      gl.bufferData(1, new Float32Array(10), 2); // 40 B
      gl.bufferData(1, 100, 2); // 100 B allocation
      gl.bufferData(1, new Float32Array(10), 2, 4); // elements 4..9 → 24 B
      gl.bufferData(1, new Float32Array(10), 2, 2, 3); // 3 elements → 12 B
      expect(bufferBytes()).toBe(40 + 100 + 24 + 12);
      expect(calls()).toBe(4);
    });

    it('counts bufferSubData by the data slice it reads', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      gl.bufferSubData(1, 0, new Uint16Array(8)); // 16 B
      gl.bufferSubData(1, 0, new Uint16Array(8), 2, 3); // 6 B
      expect(bufferBytes()).toBe(22);
      expect(textureBytes()).toBe(0);
    });

    it('counts texture uploads from views, image sources, and PBO offsets', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      // Sized form with a view: its bytes.
      gl.texImage2D(1, 0, 2, 4, 4, 0, 3, 4, new Uint8Array(64));
      // 6-arg form with an image-like source: w*h*4.
      gl.texImage2D(1, 0, 2, 3, 4, { width: 8, height: 2 });
      // Sized WebGL2 form with an image source: explicit dims*4.
      gl.texImage2D(1, 0, 2, 5, 5, 0, 3, 4, { width: 99, height: 99 });
      // PBO offset: counted call, zero bytes.
      gl.texImage2D(1, 0, 2, 4, 4, 0, 3, 4, 0);
      // 3D view.
      gl.texImage3D(1, 0, 2, 2, 2, 2, 0, 3, 4, new Float32Array(8));
      // 7-arg sub-image with an ImageBitmap-like source.
      gl.texSubImage2D(1, 0, 0, 0, 3, 4, { width: 2, height: 2 });
      // Sized sub-image with a view and srcOffset (elements).
      gl.texSubImage2D(1, 0, 0, 0, 2, 2, 3, 4, new Uint8Array(20), 4);
      // 3D sub-image with a view.
      gl.texSubImage3D(1, 0, 0, 0, 0, 1, 1, 2, 3, 4, new Uint8Array(8));
      expect(textureBytes()).toBe(64 + 64 + 100 + 0 + 32 + 16 + 16 + 8);
      expect(bytes()).toBe(textureBytes());
      expect(calls()).toBe(8);
    });

    it('counts a known-format typed-array upload by the region it writes, not the view', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      const RGBA = 0x1908;
      const FLOAT = 0x1406;
      const UNSIGNED_BYTE = 0x1401;
      const whole = new Float32Array(8 * 4 * 4); // an 8x4 RGBA32F image, 512 B
      // three's classic ranged path: one row, the WHOLE image.data, row picked
      // by UNPACK_SKIP_ROWS — uploads 8 px * 16 B, not the 512-B view.
      gl.texSubImage2D(1, 0, 0, 2, 8, 1, RGBA, FLOAT, whole);
      expect(textureBytes()).toBe(8 * 16);
      // texImage2D / 3D sized forms with a known format too.
      gl.texImage2D(1, 0, 2, 4, 4, 0, RGBA, UNSIGNED_BYTE, new Uint8Array(1000));
      gl.texImage3D(1, 0, 2, 2, 2, 2, 0, RGBA, FLOAT, new Float32Array(1000));
      gl.texSubImage3D(1, 0, 0, 0, 0, 1, 1, 2, 0x1903, FLOAT, new Float32Array(1000));
      expect(textureBytes()).toBe(8 * 16 + 64 + 128 + 8);
      // A view SHORTER than the region still counts only what it holds.
      perfCounters.reset();
      gl.texSubImage2D(1, 0, 0, 0, 8, 4, RGBA, FLOAT, new Float32Array(4));
      expect(textureBytes()).toBe(16);
    });

    it('counts an undeterminable image source as a zero-byte call', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      gl.texImage2D(1, 0, 2, 3, 4, {});
      expect(calls()).toBe(1);
      expect(bytes()).toBe(0);
    });

    it.fails('counts compressed texture uploads (KTX2 on WebGL) by the block bytes they read', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      // compressedTexImage2D(target, level, fmt, w, h, border, data[, srcOffset, length])
      gl.compressedTexImage2D(1, 0, 2, 8, 8, 0, new Uint8Array(32)); // 32 B
      gl.compressedTexImage2D(1, 0, 2, 8, 8, 0, new Uint8Array(32), 8, 16); // 16 B
      gl.compressedTexImage2D(1, 0, 2, 8, 8, 0, 32, 0); // PBO form: 0 B
      // compressedTexImage3D(target, level, fmt, w, h, d, border, data, …)
      gl.compressedTexImage3D(1, 0, 2, 4, 4, 2, 0, new Uint8Array(64)); // 64 B
      // compressedTexSubImage2D(target, level, x, y, w, h, fmt, data, …)
      gl.compressedTexSubImage2D(1, 0, 0, 0, 4, 4, 2, new Uint8Array(16)); // 16 B
      // compressedTexSubImage3D(target, level, x, y, z, w, h, d, fmt, data, …)
      gl.compressedTexSubImage3D(1, 0, 0, 0, 0, 4, 4, 1, 2, new Uint8Array(16), 0, 8); // 8 B
      expect(textureBytes()).toBe(32 + 16 + 64 + 16 + 8);
      expect(bufferBytes()).toBe(0);
      expect(calls()).toBe(6);
    });

    it('wraps only once', () => {
      const gl = fakeGL();
      expect(wrapWebGLUploads(gl)).toBe(true);
      expect(wrapWebGLUploads(gl)).toBe(false);
      gl.bufferData(1, new Uint8Array(5), 2);
      expect(bytes()).toBe(5);
      expect(calls()).toBe(1);
    });
  });

  describe('WebGPU', () => {
    it('counts writeBuffer by size arg (elements for a view) or data bytes', () => {
      const q = fakeQueue();
      wrapWebGPUUploads(q);
      q.writeBuffer({}, 0, new Float32Array(16)); // 64 B
      q.writeBuffer({}, 0, new Float32Array(16), 4, 2); // 8 B
      q.writeBuffer({}, 0, new ArrayBuffer(32), 8); // 24 B
      expect(bufferBytes()).toBe(64 + 8 + 24);
    });

    it('counts writeTexture by data bytes past the layout offset', () => {
      const q = fakeQueue();
      wrapWebGPUUploads(q);
      q.writeTexture({}, new Uint8Array(100), { offset: 20 }, [5, 4]);
      expect(textureBytes()).toBe(80);
    });

    it('caps writeTexture at the bytesPerRow * rows region the layout describes', () => {
      const q = fakeQueue();
      wrapWebGPUUploads(q);
      // Two 64-B rows written out of a 1000-B buffer.
      q.writeTexture({}, new Uint8Array(1000), { offset: 128, bytesPerRow: 64 }, [16, 2]);
      q.writeTexture({}, new Uint8Array(1000), { bytesPerRow: 64 }, { width: 16, height: 3 });
      expect(textureBytes()).toBe(128 + 192);
    });

    it('counts copyExternalImageToTexture as texels*4', () => {
      const q = fakeQueue();
      wrapWebGPUUploads(q);
      q.copyExternalImageToTexture({}, {}, { width: 4, height: 2 });
      q.copyExternalImageToTexture({}, {}, [3, 3]);
      expect(textureBytes()).toBe(32 + 36);
      expect(calls()).toBe(2);
    });
  });

  describe('installUploadCounters', () => {
    it('wraps a WebGLRenderer context', () => {
      const gl = fakeGL();
      expect(installUploadCounters({ getContext: () => gl })).toBe('webgl');
      gl.bufferData(1, 8, 2);
      expect(bytes()).toBe(8);
    });

    it('wraps a WebGPURenderer device queue', () => {
      const queue = fakeQueue();
      expect(installUploadCounters({ backend: { device: { queue } } })).toBe('webgpu');
      queue.writeBuffer({}, 0, new Uint8Array(3));
      expect(bytes()).toBe(3);
    });

    it('wraps the WebGL2 context of the WebGPURenderer fallback backend', () => {
      const gl = fakeGL();
      expect(installUploadCounters({ backend: { gl } })).toBe('webgl');
      gl.bufferSubData(1, 0, new Uint8Array(7));
      expect(bytes()).toBe(7);
    });

    it('returns null and does not throw for an unrecognised renderer', () => {
      expect(installUploadCounters({})).toBeNull();
      expect(installUploadCounters(null)).toBeNull();
    });
  });
});
