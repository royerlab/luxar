/**
 * Rows-only element-texture uploads on three's WebGPU backends (#2944).
 *
 * The fakes below model a backend's GPU texture as a Float32Array mirror
 * that the fake `queue.writeTexture` / `gl.texSubImage2D` really write
 * into, so every test can assert the strong property — after each update
 * the GPU mirror is byte-identical to the CPU image, i.e. to what a full
 * upload would have produced — alongside which bytes were written.
 * `renderUpload` replays three's `Textures.updateTexture` sequence:
 * `createTexture` on first use, `backend.updateTexture`, then `onUpdate`.
 */
import * as THREE from 'three';
import { beforeEach, describe, expect, it } from 'vitest';
import { perfCounters } from '../../../profiling/perf-counters';
import {
  attachElementStorage,
  markElementTextureFullDirty,
  registerElementTexelDirtyRange,
} from '../../../rendering/element-storage';
import {
  SPLAT_FLOATS_PER_SPLAT,
  SPLAT_TEXTURE_LAYOUT,
  resetElementTextureLayoutForTests,
} from '../../../rendering/element-texture-layout';
import {
  installElementTextureRowUploads,
  planDirtyRows,
} from '../../../rendering/element-texture-row-upload';
import { wrapWebGLUploads, wrapWebGPUUploads } from '../../../rendering/upload-counters';

const ROWS = 20;

interface GpuTex {
  mirror: Float32Array;
  width: number;
}

interface Write {
  firstRow: number;
  rows: number;
  bytes: number;
}

interface UploadOptions {
  image: THREE.DataTexture['image'];
  width?: number;
  height?: number;
  depth?: number;
}

type AnyBackend = Record<string, unknown> & {
  updateTexture: (t: THREE.DataTexture, o: UploadOptions) => void;
  get: (o: object) => Record<string, unknown>;
  writes: Write[];
};

/** Native-WebGPU-shaped backend whose original updateTexture writes the whole image. */
function fakeWebGPUBackend(): AnyBackend {
  const records = new WeakMap<object, Record<string, unknown>>();
  const writes: Write[] = [];
  const queue = {
    writeTexture(
      dst: { texture: GpuTex; origin?: { y?: number } },
      data: Float32Array,
      layout: { offset?: number; bytesPerRow: number },
      size: { width: number; height: number }
    ) {
      const y = dst.origin?.y ?? 0;
      const rowFloats = layout.bytesPerRow / 4;
      const src = data.subarray(
        (layout.offset ?? 0) / 4,
        (layout.offset ?? 0) / 4 + size.height * rowFloats
      );
      dst.texture.mirror.set(src, y * rowFloats);
      writes.push({ firstRow: y, rows: size.height, bytes: src.byteLength });
    },
  };
  const backend: AnyBackend = {
    isWebGPUBackend: true,
    device: { queue },
    writes,
    get(o: object) {
      let r = records.get(o);
      if (!r) records.set(o, (r = {}));
      return r;
    },
    createTexture(texture: THREE.DataTexture) {
      const { width, height } = texture.image;
      Object.assign(backend.get(texture), {
        texture: { mirror: new Float32Array(width * height * 4), width } satisfies GpuTex,
        textureDescriptorGPU: {
          format: 'rgba32float',
          size: { width, height, depthOrArrayLayers: 1 },
          mipLevelCount: 1,
          sampleCount: 1,
        },
      });
    },
    // three's WebGPUTextureUtils.updateTexture for a DataTexture: one full writeTexture.
    updateTexture(texture: THREE.DataTexture, options: { image: THREE.DataTexture['image'] }) {
      const r = backend.get(texture) as { texture: GpuTex };
      const { width, height, data } = options.image;
      queue.writeTexture(
        { texture: r.texture },
        data as Float32Array,
        { offset: 0, bytesPerRow: width * 16 },
        { width, height }
      );
    },
  };
  return backend;
}

const GL = { TEXTURE_2D: 0x0de1, RGBA: 0x1908, FLOAT: 0x1406 };

/** WebGL2-fallback-shaped backend (texSubImage2D full-image original). */
function fakeWebGLFallbackBackend(): AnyBackend {
  const records = new WeakMap<object, Record<string, unknown>>();
  const writes: Write[] = [];
  let bound: GpuTex | null = null;
  const gl = {
    ...GL,
    texSubImage2D(
      _t: number,
      _l: number,
      _x: number,
      y: number,
      w: number,
      h: number,
      _f: number,
      _ty: number,
      data: Float32Array
    ) {
      if (!bound) throw new Error('no texture bound');
      bound.mirror.set(data.subarray(0, w * h * 4), y * w * 4);
      writes.push({ firstRow: y, rows: h, bytes: data.byteLength });
    },
  };
  const backend: AnyBackend = {
    isWebGLBackend: true,
    gl,
    writes,
    state: { bindTexture: (_t: number, tex: GpuTex) => (bound = tex) },
    textureUtils: { setTextureParameters: () => undefined },
    get(o: object) {
      let r = records.get(o);
      if (!r) records.set(o, (r = {}));
      return r;
    },
    createTexture(texture: THREE.DataTexture) {
      const { width, height } = texture.image;
      Object.assign(backend.get(texture), {
        textureGPU: { mirror: new Float32Array(width * height * 4), width } satisfies GpuTex,
        glTextureType: GL.TEXTURE_2D,
        glFormat: GL.RGBA,
        glType: GL.FLOAT,
      });
    },
    updateTexture(texture: THREE.DataTexture, options: { image: THREE.DataTexture['image'] }) {
      const r = backend.get(texture) as { textureGPU: GpuTex };
      const { width, height, data } = options.image;
      bound = r.textureGPU;
      gl.texSubImage2D(
        GL.TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        GL.RGBA,
        GL.FLOAT,
        data as Float32Array
      );
    },
  };
  return backend;
}

/** Replays three's Textures.updateTexture for one needsUpdate. */
function renderUpload(backend: AnyBackend, texture: THREE.DataTexture): void {
  const record = backend.get(texture);
  if (!record.created) {
    (backend.createTexture as (t: THREE.Texture) => void)(texture);
    record.created = true;
  }
  const { width, height } = texture.image;
  backend.updateTexture(texture, { image: texture.image, width, height, depth: 1 });
  texture.onUpdate?.(texture);
}

function gpuMirror(backend: AnyBackend, texture: THREE.Texture): Float32Array {
  const r = backend.get(texture);
  return ((r.texture ?? r.textureGPU) as GpuTex).mirror;
}

function makeTexture(): THREE.DataTexture {
  const perRow = 4096 / SPLAT_TEXTURE_LAYOUT.texelsPerElement;
  return attachElementStorage(
    new THREE.InstancedBufferGeometry(),
    perRow * ROWS,
    SPLAT_TEXTURE_LAYOUT
  );
}

/** Write elements [from, to) with a recognisable value and register them. */
function writeElements(texture: THREE.DataTexture, from: number, to: number, value: number): void {
  const data = texture.image.data as Float32Array;
  data.fill(value, from * SPLAT_FLOATS_PER_SPLAT, to * SPLAT_FLOATS_PER_SPLAT);
  registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, from, to);
}

const PER_ROW = 4096 / 4;
const ROW_BYTES = 4096 * 16;

describe.each([
  ['webgpu', fakeWebGPUBackend],
  ['webgl-fallback', fakeWebGLFallbackBackend],
] as const)('element-texture-row-upload (%s)', (kind, makeBackend) => {
  let backend: AnyBackend;
  let texture: THREE.DataTexture;

  beforeEach(() => {
    resetElementTextureLayoutForTests();
    backend = makeBackend();
    expect(installElementTextureRowUploads({ backend })).toBe(kind);
    texture = makeTexture();
  });

  it('writes the whole texture on a first upload with nothing ranged', () => {
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('a never-uploaded texture uploads only its written rows into zeroed GPU storage', () => {
    // Growth headroom: 2 of 20 rows written before the first draw.
    writeElements(texture, 0, PER_ROW * 2, 1);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: 2, bytes: 2 * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
    expect(texture.updateRanges).toHaveLength(0);
  });

  it('a ranged first upload leaves the texture in sync, so the next commit is rows-only', () => {
    // First progressive rung: 2 of 20 rows, uploaded ranged.
    writeElements(texture, 0, PER_ROW * 2, 1);
    renderUpload(backend, texture);
    backend.writes.length = 0;
    // Second rung: one more row — must not re-upload the full capacity.
    writeElements(texture, PER_ROW * 2, PER_ROW * 3, 2);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 2, rows: 1, bytes: ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('a texture another renderer already uploaded gets a full first upload here', () => {
    // Some other renderer consumed an upload (and its ranges): this backend's
    // fresh resource must not trust the ranges alone.
    writeElements(texture, 0, PER_ROW * 2, 1);
    texture.clearUpdateRanges();
    texture.onUpdate?.(texture);
    writeElements(texture, PER_ROW * 4, PER_ROW * 5, 2);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('writes only the dirty rows of a later partial update, byte-identical to a full upload', () => {
    renderUpload(backend, texture);
    backend.writes.length = 0;
    // Elements straddling rows 3..5 (partial first and last row).
    writeElements(texture, PER_ROW * 3 + 10, PER_ROW * 5 + 7, 2);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 3, rows: 3, bytes: 3 * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
    expect(texture.updateRanges).toHaveLength(0);
  });

  it('consumes ranges so the next commit uploads only its own rows', () => {
    renderUpload(backend, texture);
    writeElements(texture, 0, PER_ROW * 2, 3);
    renderUpload(backend, texture);
    backend.writes.length = 0;
    writeElements(texture, PER_ROW * 2, PER_ROW * 3, 4); // an append
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 2, rows: 1, bytes: ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('accumulated commits between uploads are all uploaded', () => {
    renderUpload(backend, texture);
    backend.writes.length = 0;
    writeElements(texture, PER_ROW, PER_ROW * 2, 5);
    writeElements(texture, PER_ROW * 2, PER_ROW * 4, 6);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 1, rows: 3, bytes: 3 * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('a full-dirty mark writes the whole texture', () => {
    renderUpload(backend, texture);
    backend.writes.length = 0;
    (texture.image.data as Float32Array).fill(7);
    markElementTextureFullDirty(texture);
    writeElements(texture, 0, 3, 8); // a ranged write cannot downgrade it
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
    // …and once flushed, ranged writes are partial again.
    backend.writes.length = 0;
    writeElements(texture, 0, 3, 9);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: 1, bytes: ROW_BYTES }]);
  });

  it('a >=75%-dirty write takes the full upload', () => {
    renderUpload(backend, texture);
    backend.writes.length = 0;
    writeElements(texture, 0, PER_ROW * 16, 10);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
  });

  it('a reallocated GPU resource gets a full upload', () => {
    renderUpload(backend, texture);
    backend.get(texture).created = false; // three recreates the resource (e.g. device loss)
    backend.writes.length = 0;
    writeElements(texture, 0, 3, 11);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('falls back to the full upload when the GPU resource is not a shape it knows', () => {
    renderUpload(backend, texture);
    const r = backend.get(texture);
    if (kind === 'webgpu') {
      (r.textureDescriptorGPU as { format: string }).format = 'rgba16float';
    } else {
      r.glType = 0x140b; // HALF_FLOAT
    }
    backend.writes.length = 0;
    writeElements(texture, 0, 3, 12);
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
  });

  it('falls back to the full upload when the row write throws', () => {
    renderUpload(backend, texture);
    const target = kind === 'webgpu' ? (backend.device as { queue: object }).queue : backend.gl;
    const t = target as Record<string, (...a: unknown[]) => unknown>;
    const name = kind === 'webgpu' ? 'writeTexture' : 'texSubImage2D';
    const real = t[name];
    let failOnce = true;
    t[name] = (...args: unknown[]) => {
      if (failOnce) {
        failOnce = false;
        throw new Error('boom');
      }
      return real(...args);
    };
    writeElements(texture, 0, 3, 13);
    backend.writes.length = 0;
    renderUpload(backend, texture);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: ROWS, bytes: ROWS * ROW_BYTES }]);
    expect(gpuMirror(backend, texture)).toEqual(texture.image.data);
  });

  it('leaves non-element textures to three', () => {
    const other = new THREE.DataTexture(
      new Float32Array(4 * 4 * 4),
      4,
      4,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    other.addUpdateRange(0, 4);
    renderUpload(backend, other);
    other.addUpdateRange(0, 4);
    backend.writes.length = 0;
    renderUpload(backend, other);
    expect(backend.writes).toEqual([{ firstRow: 0, rows: 4, bytes: 4 * 4 * 16 }]);
    expect(other.updateRanges).toHaveLength(2); // untouched: three never clears them
  });

  it('upload counters tally exactly the rows written', () => {
    if (kind === 'webgpu') wrapWebGPUUploads((backend.device as { queue: object }).queue);
    else wrapWebGLUploads(backend.gl as object);
    renderUpload(backend, texture);
    perfCounters.reset();
    writeElements(texture, PER_ROW * 7, PER_ROW * 9, 14);
    renderUpload(backend, texture);
    expect(perfCounters.get('gpu.uploadBytes.texture')).toBe(2 * ROW_BYTES);
    expect(perfCounters.get('gpu.uploadCalls')).toBe(1);
  });

  it('installs once', () => {
    const wrapped = backend.updateTexture;
    expect(installElementTextureRowUploads({ backend })).toBe(kind);
    expect(backend.updateTexture).toBe(wrapped);
  });
});

describe('installElementTextureRowUploads', () => {
  it('leaves a classic WebGLRenderer and unknown shapes alone', () => {
    expect(installElementTextureRowUploads({ getContext: () => ({}) })).toBeNull();
    expect(
      installElementTextureRowUploads({ backend: { updateTexture: () => undefined } })
    ).toBeNull();
    expect(installElementTextureRowUploads(null)).toBeNull();
  });
});

describe('planDirtyRows', () => {
  const image = { data: new Float32Array(8 * 4 * 4), width: 8, height: 4 };
  const rowFloats = 8 * 4;

  it('covers every range with whole rows', () => {
    expect(planDirtyRows([{ start: rowFloats + 4, count: 8 }], image)).toEqual({
      firstRow: 1,
      rowCount: 1,
    });
    expect(
      planDirtyRows(
        [
          { start: 4, count: 4 },
          { start: rowFloats * 2, count: rowFloats + 1 },
        ],
        image
      )
    ).toEqual({ firstRow: 0, rowCount: 4 });
  });

  it('refuses empty or out-of-bounds ranges', () => {
    expect(planDirtyRows([], image)).toBeNull();
    expect(planDirtyRows([{ start: -4, count: 8 }], image)).toBeNull();
    expect(planDirtyRows([{ start: 0, count: 0 }], image)).toBeNull();
    expect(planDirtyRows([{ start: rowFloats * 4 - 4, count: 8 }], image)).toBeNull();
  });
});
