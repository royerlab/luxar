/**
 * Stage-1 metadata preflight: every rejection, and the acceptances that prove
 * the rejections are not just "reject everything".
 *
 * The preflight is pure over (attrs, array metadata), so these tests hand it
 * hand-built handles rather than a store. That the preflight fetches no chunks
 * is a property of the LOADER calling it before any read, and is pinned in
 * `mesh-loader.test.ts` against a request-recording store — the two halves of
 * the same guarantee, tested where each is actually decidable.
 */

import { describe, it, expect } from 'vitest';
import { preflightMesh, parseDtype } from '../../../../data/mesh/mesh-preflight';
import type { MeshArrayHandles } from '../../../../data/mesh/mesh-preflight';
import { LoaderError } from '../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { MAX_MESH_VERTICES, MESH_DECODE_BUDGET_BYTES } from '../../../../config/constants';
import type * as zarr from '../../../../data/zarr';
import type { MeshMetadata } from '../../../../types/mesh';

const PATH = '/surface';

/** Minimal stand-in for a zarr array's metadata surface. */
function fakeArray(
  shape: number[],
  dtype: string,
  chunks?: number[],
  attrs: Record<string, unknown> = {}
): zarr.Array<zarr.DataType, zarr.Readable> {
  return {
    shape,
    chunks: chunks ?? shape,
    dtype,
    attrs,
  } as unknown as zarr.Array<zarr.DataType, zarr.Readable>;
}

/** A 4-vertex / 4-face 3D tetrahedron, in the dtypes the writer actually emits. */
function tetHandles(overrides: Partial<MeshArrayHandles> = {}): MeshArrayHandles {
  return {
    // The writer quantizes COORDINATE to per-channel uint16, so `<u2` here is
    // the realistic on-disk dtype, not float32 (verified against a written store).
    vertices: fakeArray([4, 3], '<u2', [4, 3], {
      encoding: { name: 'linear_perchannel_u16', original_dtype: 'float32' },
    }),
    // The INDEX encoder narrows to the smallest unsigned dtype that fits, so a
    // small mesh's faces land as uint8.
    faces: fakeArray([4, 3], '|u1', [4, 3], {
      encoding: { name: 'uint8', original_dtype: 'uint32' },
    }),
    ...overrides,
  };
}

function tetAttrs(overrides: Partial<MeshMetadata> = {}): MeshMetadata {
  return {
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
    ...overrides,
  };
}

/** Assert the call rejects as a Validation-kind LoaderError matching `pattern`. */
function expectReject(fn: () => unknown, pattern: RegExp): void {
  let thrown: unknown;
  try {
    fn();
  } catch (e) {
    thrown = e;
  }
  expect(thrown).toBeInstanceOf(LoaderError);
  const err = thrown as LoaderError;
  // The kind is not decoration: it is persisted on the failure record, and
  // 'Validation' is what stops the retry policy re-fetching a malformed store
  // on every reconnect.
  expect(err.kind).toBe('Validation');
  expect(err.path).toBe(PATH);
  expect(err.message).toMatch(pattern);
}

describe('parseDtype', () => {
  it.each([
    ['uint8', 1, true],
    ['uint16', 2, true],
    ['uint32', 4, true],
    ['uint64', 8, true],
    ['int32', 4, true],
    ['float32', 4, false],
    ['float64', 8, false],
    ['float16', 2, false],
  ])('accepts the friendly name %s', (dtype, itemSize, integer) => {
    expect(parseDtype(dtype as string)).toEqual({ itemSize, integer });
  });

  it.each([
    ['|u1', 1, true],
    ['<u2', 2, true],
    ['>u4', 4, true],
    ['<i8', 8, true],
    ['<f4', 4, false],
    ['>f8', 8, false],
    ['=f2', 2, false],
  ])('accepts the numpy typestring %s', (dtype, itemSize, integer) => {
    expect(parseDtype(dtype as string)).toEqual({ itemSize, integer });
  });

  it('both spellings occur in practice, so both must parse', () => {
    // zarrita reports friendly names; a raw .zarray carries typestrings. If only
    // one form parsed, the byte budget would read 0 bytes for the other — a
    // budget that admits everything, which is worse than no budget at all
    // because it looks like protection.
    expect(parseDtype('uint16')).toEqual(parseDtype('<u2'));
    expect(parseDtype('float32')).toEqual(parseDtype('<f4'));
  });

  it('accepts a typestring with no byte-order char — numpy does', () => {
    // `np.dtype('u2')` is valid, so a store may spell it that way.
    expect(parseDtype('u2')).toEqual({ itemSize: 2, integer: true });
  });

  it('returns null for anything unrecognised rather than guessing a size', () => {
    // Note `<U2` and `<c16`: both MATCH the typestring shape but name kinds
    // (unicode string, complex) that no mesh array can be, so they must fall
    // through to null rather than yielding a plausible itemSize. Guessing a size
    // for one of these would let it into the budget as if it were numeric.
    for (const bad of ['', 'complex128', '<c16', '<U2', 'not-a-dtype', '<f0']) {
      expect(parseDtype(bad)).toBeNull();
    }
  });
});

describe('preflightMesh — acceptance', () => {
  it('admits a plain tetrahedron and reports what it established', () => {
    const result = preflightMesh(PATH, tetAttrs(), tetHandles());
    expect(result).toEqual({
      nVertices: 4,
      nFaces: 4,
      ndim: 3,
      colorComponents: undefined,
      normalDims: undefined,
    });
  });

  it('admits every optional array, and reads the colour channel count', () => {
    const result = preflightMesh(
      PATH,
      tetAttrs({
        has_normals: true,
        normal_dims: [0, 1, 2],
        has_colors: true,
        has_scalars: true,
        shading: 'smooth',
      }),
      tetHandles({
        normals: fakeArray([4, 3], '<f4'),
        colors: fakeArray([4, 4], '|u1'),
        scalars: fakeArray([4], '|u1'),
      })
    );
    expect(result.colorComponents).toBe(4);
    expect(result.normalDims).toEqual([0, 1, 2]);
  });

  it('accepts a LUT-encoded normals array whose STORED shape differs', () => {
    // A real case, not hypothetical: a fixture whose normals take few distinct
    // values gets LUT-encoded, and the stored codes array carries a different
    // shape than the logical (V, 3). Checking the stored shape would reject a
    // perfectly valid store.
    expect(() =>
      preflightMesh(
        PATH,
        tetAttrs({ has_normals: true, normal_dims: [0, 1, 2], shading: 'smooth' }),
        tetHandles({
          normals: fakeArray([12], '|u1', [12], {
            encoding: { name: 'lut_uint8', original_shape: [4, 3], original_dtype: 'float32' },
          }),
        })
      )
    ).not.toThrow();
  });

  it('accepts a BROADCAST scalars array, which stores one value for V vertices', () => {
    expect(() =>
      preflightMesh(
        PATH,
        tetAttrs({ has_scalars: true }),
        tetHandles({
          scalars: fakeArray([1], '<f4', [1], {
            encoding: { name: 'broadcasted', n_elements: 4 },
          }),
        })
      )
    ).not.toThrow();
  });

  it('accepts a BROADCAST colors array (uniform colour) stored as (1, 4)', () => {
    // A uniform colour is stored broadcast as (1, d), with `n_elements`
    // recording the logical vertex count — the same convention the scalars
    // branch accepts. The component axis (d) is still read back verbatim.
    const result = preflightMesh(
      PATH,
      tetAttrs({ has_colors: true }),
      tetHandles({
        colors: fakeArray([1, 4], '|u1', [1, 4], {
          encoding: { name: 'broadcasted', n_elements: 4, original_dtype: 'uint8' },
        }),
      })
    );
    expect(result.colorComponents).toBe(4);
  });

  it('accepts a BROADCAST colors array (uniform colour) stored as (1, 3)', () => {
    const result = preflightMesh(
      PATH,
      tetAttrs({ has_colors: true }),
      tetHandles({
        colors: fakeArray([1, 3], '|u1', [1, 3], {
          encoding: { name: 'broadcasted', n_elements: 4, original_dtype: 'uint8' },
        }),
      })
    );
    expect(result.colorComponents).toBe(3);
  });
});

describe('preflightMesh — (a) counts and the vote-key cap', () => {
  it('rejects n_vertices above MAX_MESH_VERTICES', () => {
    const n = MAX_MESH_VERTICES + 1;
    expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({ n_vertices: n }),
          tetHandles({ vertices: fakeArray([n, 3], '<u2', [1024, 3]) })
        ),
      /exceeds the maximum of .*2\^27/
    );
  });

  it('does NOT trip the cap at exactly MAX_MESH_VERTICES — the bound is inclusive', () => {
    // The largest vertex ordinal is n_vertices - 1, so 2^27 vertices means a
    // maximum ordinal of 2^27 - 1, still strictly under the vote-key stride.
    //
    // This asserts the absence of the CAP error rather than the absence of any
    // error, because a 2^27-vertex mesh cannot pass the byte budget at the
    // default 512 MiB: even uint16-quantized 3D vertices are 768 MiB. The two
    // gates are checked in this order deliberately — see the budget-is-binding
    // test below — so an off-by-one in the cap would show up here as the wrong
    // error, which is exactly what this pins.
    const n = MAX_MESH_VERTICES;
    let message = '';
    try {
      preflightMesh(
        PATH,
        tetAttrs({ n_vertices: n }),
        tetHandles({ vertices: fakeArray([n, 3], '<u2', [1024, 3]) })
      );
    } catch (e) {
      message = (e as Error).message;
    }
    expect(message).not.toMatch(/exceeds the maximum/);
    expect(message).toMatch(/per-node budget/);
  });

  it('checks the cap BEFORE the budget, so an absurd count gets the precise reason', () => {
    // Both gates would reject a 2^30-vertex mesh. The cap runs first so the
    // message names the actual problem (pick-key aliasing) instead of blaming
    // bytes, which would send an author off decimating a mesh that is
    // fundamentally too large to address.
    const n = 2 ** 30;
    expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({ n_vertices: n }),
          tetHandles({ vertices: fakeArray([n, 3], '<u2', [1024, 3]) })
        ),
      /exceeds the maximum of .*2\^27/
    );
  });

  it.each([
    ['n_vertices', { n_vertices: 0 }],
    ['n_vertices', { n_vertices: 1.5 }],
    ['n_faces', { n_faces: 0 }],
    ['ndim', { n_faces: 4, ndim: 0 }],
  ])('rejects a non-positive-integer %s', (_field, override) => {
    expectReject(
      () => preflightMesh(PATH, tetAttrs(override as Partial<MeshMetadata>), tetHandles()),
      /must be a positive integer/
    );
  });
});

describe('preflightMesh — (b) the byte budget', () => {
  it('rejects a summed declared footprint over the budget', () => {
    // 40M vertices x 4 dims x 4 bytes = 640 MiB, over the 512 MiB ceiling, and
    // under the 2^27 vertex cap — so this exercises the budget, not the cap.
    const n = 40_000_000;
    expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_vertices: n, ndim: 4 }), {
          vertices: fakeArray([n, 4], '<f4', [65536, 4]),
          faces: fakeArray([4, 3], '<u4', [4, 3]),
        }),
      /declared arrays total .* over the .* per-node budget/
    );
    expect(n).toBeLessThan(MAX_MESH_VERTICES);
  });

  it('rejects an oversized single CHUNK even when the total is tiny', () => {
    // The case a shape-only budget misses entirely: zarr v2 does not require
    // chunks <= shape, so a 100-triangle array can declare a 268M-triangle
    // chunk. Zarr allocates chunk-shaped buffers, so that one chunk IS a ~3 GB
    // allocation.
    expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_faces: 100 }), {
          vertices: fakeArray([4, 3], '<f4', [4, 3]),
          faces: fakeArray([100, 3], '<u4', [268_435_456, 3]),
        }),
      /declares a single chunk of .* over the .* per-node budget/
    );
  });

  it('budgets the DECLARED dtype, not the canonical one', () => {
    // faces is logically uint32, but an external int64 store costs 8 bytes per
    // index. Budgeting a canonical 4 would let it fetch twice the audited bytes.
    // 24M faces x 3 x 8 = 576 MiB (rejected); the same shape at 4 bytes would be
    // 288 MiB (admitted) — so this asserts the itemsize is genuinely read.
    const f = 24_000_000;
    const int64Handles: MeshArrayHandles = {
      vertices: fakeArray([4, 3], '<f4', [4, 3]),
      faces: fakeArray([f, 3], '<i8', [65536, 3]),
    };
    expectReject(
      () => preflightMesh(PATH, tetAttrs({ n_faces: f }), int64Handles),
      /declared arrays total/
    );
    // Same shape, 4-byte dtype: admitted. If the budget ignored dtype these two
    // would agree, and the test above would be passing for the wrong reason.
    expect(() =>
      preflightMesh(PATH, tetAttrs({ n_faces: f }), {
        vertices: fakeArray([4, 3], '<f4', [4, 3]),
        faces: fakeArray([f, 3], '<u4', [65536, 3]),
      })
    ).not.toThrow();
  });

  it('counts the label CSR arrays toward the footprint', () => {
    // v1 never fetches these, but they are part of the declared node. Budgeting
    // them from the start means the ceiling does not silently loosen when the
    // label loader lands.
    const bytes = MESH_DECODE_BUDGET_BYTES;
    expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ has_labels: true }), {
          ...tetHandles(),
          labelOffsets: fakeArray([5], '<u4', [5]),
          labelBytes: fakeArray([bytes], '|u1', [65536]),
        }),
      /declared arrays total/
    );
  });

  it('rejects an unrecognised dtype rather than budgeting it as free', () => {
    expectReject(
      () => preflightMesh(PATH, tetAttrs(), tetHandles({ colors: fakeArray([4, 3], '<c16') })),
      /unrecognised dtype/
    );
  });
});

describe('preflightMesh — (c) shape and dtype cross-checks', () => {
  it('rejects a vertices width that disagrees with ndim', () => {
    // The check that stops a `panic = "abort"` trap: the slab kernel strides by
    // ndim, so a narrower row reads past the end of the array.
    expectReject(
      () => preflightMesh(PATH, tetAttrs({ ndim: 4 }), tetHandles()),
      /vertices declares shape \[4, 3\].*\(n_vertices, ndim\) = \(4, 4\)/s
    );
  });

  it('rejects a vertices row count that disagrees with n_vertices', () => {
    expectReject(
      () => preflightMesh(PATH, tetAttrs(), tetHandles({ vertices: fakeArray([3, 3], '<f4') })),
      /vertices declares shape \[3, 3\]/
    );
  });

  it.each([
    ['a non-3 face width', [4, 4]],
    ['a face count mismatch', [5, 3]],
    ['a 1-D faces array', [12]],
  ])('rejects %s', (_label, shape) => {
    expectReject(
      () =>
        preflightMesh(PATH, tetAttrs(), tetHandles({ faces: fakeArray(shape as number[], '<u4') })),
      /faces declares shape/
    );
  });

  it('rejects a FLOAT faces dtype', () => {
    // A float index truncates in the uint32 coercion, silently rewriting
    // topology. Mirrors the write-side validate_faces_for_writing rule.
    expectReject(
      () => preflightMesh(PATH, tetAttrs(), tetHandles({ faces: fakeArray([4, 3], '<f4') })),
      /an integer dtype is required/
    );
  });

  it.each([
    ['int8', '|i1'],
    ['uint8', '|u1'],
    ['int16', '<i2'],
    ['int32', '<i4'],
    ['uint32', '<u4'],
    ['int64', '<i8'],
    ['uint64', '<u8'],
  ])('accepts an external %s faces dtype', (_label, dtype) => {
    // Signed and 64-bit stores are legal here; their wrap-around hazards are
    // Stage 2's job, on the source-typed VALUES. Rejecting them at Stage 1
    // would refuse valid external meshes.
    expect(() =>
      preflightMesh(PATH, tetAttrs(), tetHandles({ faces: fakeArray([4, 3], dtype as string) }))
    ).not.toThrow();
  });

  it.each([
    ['normals', 'has_normals', [3, 3], /normals declares shape/],
    ['colors', 'has_colors', [4, 2], /colors declares shape/],
    ['scalars', 'has_scalars', [3], /scalars declares shape/],
  ])('rejects an undersized %s array', (slot, flag, shape, pattern) => {
    // These bind as ENABLED vertex attributes on an indexed draw. An undersized
    // one does not trap — drawElements reads past the buffer and mis-shades
    // every vertex it covers, backend-dependently. Catchable from the shape.
    expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({
            [flag as string]: true,
            ...(flag === 'has_normals' ? { normal_dims: [0, 1, 2] } : {}),
          } as Partial<MeshMetadata>),
          tetHandles({ [slot as string]: fakeArray(shape as number[], '<f4') })
        ),
      pattern as RegExp
    );
  });

  it('accepts colors at 3 OR 4 channels', () => {
    for (const channels of [3, 4] as const) {
      const result = preflightMesh(
        PATH,
        tetAttrs({ has_colors: true }),
        tetHandles({ colors: fakeArray([4, channels], '|u1') })
      );
      expect(result.colorComponents).toBe(channels);
    }
  });

  it.each([
    // n_elements ≠ n_vertices: the broadcast would not cover all V vertices.
    ['a mismatched n_elements', { name: 'broadcasted', n_elements: 2 }],
    // Not actually a broadcast encoding: a (1, d) store here decodes via the
    // DIRECT path and silently zero-fills vertices 1..V-1 (black) — so the
    // shape must be rejected unless it is a GENUINE broadcast.
    ['a non-broadcast encoding', { name: 'none', n_elements: 4 }],
  ])('rejects a (1, 3) colors store with %s', (_label, encoding) => {
    expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({ has_colors: true }),
          tetHandles({ colors: fakeArray([1, 3], '|u1', [1, 3], { encoding }) })
        ),
      /colors declares shape/
    );
  });
});

describe('preflightMesh — presence flags must match the store', () => {
  it.each([
    ['has_normals', { has_normals: true, normal_dims: [0, 1, 2] }],
    ['has_colors', { has_colors: true }],
    ['has_scalars', { has_scalars: true }],
  ])('rejects %s set with no array behind it', (flag, override) => {
    // The load-bearing direction: the geometry builder reads the flag to decide
    // whether to bind `normal` / `aScalar` and to enable the colormap path, so a
    // lying flag produces a draw referencing a buffer never uploaded.
    expectReject(
      () => preflightMesh(PATH, tetAttrs(override as Partial<MeshMetadata>), tetHandles()),
      new RegExp(`${flag} is set but the array is missing`)
    );
  });

  it.each([
    ['normals', 'has_normals'],
    ['colors', 'has_colors'],
    ['scalars', 'has_scalars'],
  ])('rejects a %s array the flags disown', (slot, flag) => {
    expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs(),
          tetHandles({
            [slot as string]: fakeArray(slot === 'scalars' ? [4] : [4, 3], '<f4'),
          })
        ),
      new RegExp(`${flag} says is absent`)
    );
  });
});

describe('preflightMesh — (d) normal_dims well-formedness', () => {
  it.each([
    ['missing entirely', undefined, /exactly 3 dimension indices are required/],
    ['too short', [0, 1], /exactly 3 dimension indices are required/],
    ['too long', [0, 1, 2, 3], /exactly 3 dimension indices are required/],
    ['out of range high', [0, 1, 3], /outside \[0, ndim=3\)/],
    ['negative', [-1, 1, 2], /outside \[0, ndim=3\)/],
    ['non-integer', [0, 1.5, 2], /outside \[0, ndim=3\)/],
    ['duplicated', [0, 1, 1], /three DISTINCT dimensions/],
  ])('rejects normal_dims %s', (_label, dims, pattern) => {
    expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({
            has_normals: true,
            shading: 'smooth',
            normal_dims: dims as number[] | undefined,
          }),
          tetHandles({ normals: fakeArray([4, 3], '<f4') })
        ),
      pattern as RegExp
    );
  });

  it('is not checked when the node has no normals', () => {
    // normal_dims describes a normals array; with no normals there is nothing to
    // orient, so a stray value must not fail the node.
    expect(() =>
      preflightMesh(PATH, tetAttrs({ normal_dims: [9, 9, 9] }), tetHandles())
    ).not.toThrow();
  });

  it('accepts a non-ascending frame — order carries the x/y/z roles', () => {
    const result = preflightMesh(
      PATH,
      tetAttrs({ has_normals: true, shading: 'smooth', normal_dims: [2, 0, 1] }),
      tetHandles({ normals: fakeArray([4, 3], '<f4') })
    );
    expect(result.normalDims).toEqual([2, 0, 1]);
  });
});
