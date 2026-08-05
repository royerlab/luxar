/**
 * The invariant Stage 1 exists for, over EVERY rejection reason it has.
 *
 * `whole-node-loader.test.ts` asserts "no chunk fetched" on the paths it happens to
 * exercise. This enumerates all twenty constructible reasons instead, because the
 * guarantee is a property of the STAGE, not of the paths someone thought to test — and
 * the gate has already been bypassed four times by cases nobody had written a test for.
 *
 * Each store deliberately contains REAL chunk data, so a leak shows up as a fetch that
 * succeeded rather than being masked by a missing key.
 */

import { describe, it, expect } from 'vitest';
import * as zarr from '../../../../data/zarr';
import { MeshWholeNodeLoader } from '../../../../data/mesh/mesh-whole-node-loader';
import { ArrayRefRegistry } from '../../../../data/array-decoder/decoder';
import type { MeshMetadata } from '../../../../types/mesh';

const META = new Set(['.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json']);
class S {
  requested: string[] = [];
  constructor(public e: Map<string, Uint8Array>) {}
  get(k: string) {
    this.requested.push(k);
    return Promise.resolve(this.e.get(k));
  }
  chunks() {
    return this.requested.filter((k) => !META.has(k.slice(k.lastIndexOf('/') + 1)));
  }
}
const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));
function base(attrs: MeshMetadata) {
  const e = new Map<string, Uint8Array>();
  e.set('/.zgroup', enc({ zarr_format: 2 }));
  e.set('/mesh/.zgroup', enc({ zarr_format: 2 }));
  e.set('/mesh/.zattrs', enc(attrs));
  return e;
}
function arr(
  e: Map<string, Uint8Array>,
  p: string,
  shape: number[],
  dtype = '<f4',
  chunks?: number[],
  attrs: unknown = {}
) {
  e.set(
    `${p}/.zarray`,
    enc({
      zarr_format: 2,
      shape,
      chunks: chunks ?? shape,
      dtype,
      compressor: null,
      filters: null,
      fill_value: 0,
      order: 'C',
    })
  );
  e.set(`${p}/.zattrs`, enc(attrs));
  // Provide a real chunk too, so a fetch WOULD succeed if the gate let it through.
  const n = (chunks ?? shape).reduce((a, b) => a * b, 1);
  e.set(`${p}/${(chunks ?? shape).map(() => 0).join('.')}`, new Uint8Array(n * 4));
}
const A = (o: Partial<MeshMetadata> = {}): MeshMetadata => ({
  type: 'mesh',
  n_vertices: 4,
  n_faces: 4,
  ndim: 3,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
  ...o,
});

/** Every Stage-1 rejection reason I can construct, by name. */
const CASES: Array<[string, () => { store: S; attrs: MeshMetadata }]> = [
  [
    'n_vertices not integer',
    () => {
      const a = A({ n_vertices: 1.5 });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'n_faces zero',
    () => {
      const a = A({ n_faces: 0 });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'ndim below the 2-dimension floor',
    () => {
      const a = A({ ndim: 0 });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'vertex cap',
    () => {
      const n = 2 ** 30;
      const a = A({ n_vertices: n });
      const e = base(a);
      arr(e, '/mesh/vertices', [n, 3], '<u2', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'footprint over budget',
    () => {
      const n = 40_000_000;
      const a = A({ n_vertices: n, ndim: 4 });
      const e = base(a);
      arr(e, '/mesh/vertices', [n, 4], '<f4', [64, 4]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'single chunk over budget',
    () => {
      const a = A({ n_faces: 100 });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [100, 3], '<u4', [268435456, 3]);
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'unrecognised dtype',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3], '<c16');
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'vertices width != ndim',
    () => {
      const a = A({ ndim: 4 });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'faces wrong shape',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 4], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'faces float dtype',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<f4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'has_normals no array',
    () => {
      const a = A({ has_normals: true, normal_dims: [0, 1, 2] });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'bad normal_dims',
    () => {
      const a = A({ has_normals: true, normal_dims: [0, 1, 1] });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      arr(e, '/mesh/normals', [4, 3]);
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'undersized colors',
    () => {
      const a = A({ has_colors: true });
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3]);
      arr(e, '/mesh/faces', [4, 3], '<u4');
      arr(e, '/mesh/colors', [2, 3]);
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'unknown encoding',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [4, 3], '<f4', undefined, { encoding: { name: 'future_thing' } });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'array_ref over budget',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', target: 'evil', original_shape: [4, 3] },
      });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      arr(e, '/evil', [268435456, 3], '<f4', [268435456, 3]);
      return { store: new S(e), attrs: a };
    },
  ],
  [
    // The chain-accounting case (#1253): the target's OWN broadcast expansion is what
    // must be charged, not just the stub's declared logical shape.
    'array_ref to a broadcast target over budget',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', target: 'uniform', original_shape: [4, 3] },
      });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      arr(e, '/uniform', [1, 3], '<f4', [1, 3], {
        encoding: { name: 'broadcasted', n_elements: 45_000_000 },
      });
      return { store: new S(e), attrs: a };
    },
  ],
  [
    // Boundary: the walker allows MAX_ARRAY_REF_HOPS opens and refuses the next. Three
    // hops resolve; a fourth is the rejection. Untested before this pass.
    'array_ref chain one hop too long',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', target: 'h1', original_shape: [4, 3] },
      });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      for (const [from, to] of [
        ['h1', 'h2'],
        ['h2', 'h3'],
        ['h3', 'h4'],
      ] as const) {
        arr(e, `/${from}`, [0, 3], '<f4', undefined, {
          encoding: { name: 'array_ref', target: to, original_shape: [4, 3] },
        });
      }
      arr(e, '/h4', [4, 3], '<f4', [4, 3]);
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'array_ref cyclic',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', target: 'lp', original_shape: [4, 3] },
      });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      arr(e, '/lp', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', target: 'lp', original_shape: [4, 3] },
      });
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'array_ref no target',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', original_shape: [4, 3] },
      });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
  [
    'array_ref unopenable',
    () => {
      const a = A();
      const e = base(a);
      arr(e, '/mesh/vertices', [0, 3], '<f4', undefined, {
        encoding: { name: 'array_ref', target: 'nope', original_shape: [4, 3] },
      });
      arr(e, '/mesh/faces', [4, 3], '<u4');
      return { store: new S(e), attrs: a };
    },
  ],
];

describe('INVARIANT: no Stage-1 rejection reason fetches a chunk', () => {
  it.each(CASES)('%s', async (_label, make) => {
    const { store, attrs } = make();
    const loader = new MeshWholeNodeLoader(
      '/mesh',
      attrs,
      zarr.root(store as never).resolve('mesh'),
      {
        zarrStore: store as never,
        arrayRefRegistry: new ArrayRefRegistry(),
      }
    );
    let rejected = false;
    try {
      await loader.loadMesh({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [1e10, 1e10, 1e10],
      } as never);
    } catch {
      rejected = true;
    }
    expect(rejected, 'must be refused at Stage 1').toBe(true);
    expect(store.chunks(), 'must fetch NO chunk').toEqual([]);
  });
});
