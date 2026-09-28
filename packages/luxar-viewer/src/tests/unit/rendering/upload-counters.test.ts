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

    it('counts an undeterminable image source as a zero-byte call', () => {
      const gl = fakeGL();
      wrapWebGLUploads(gl);
      gl.texImage2D(1, 0, 2, 3, 4, {});
      expect(calls()).toBe(1);
      expect(bytes()).toBe(0);
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
