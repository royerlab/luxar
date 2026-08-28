/**
 * Stage-1 metadata preflight: every rejection, and the acceptances that prove
 * the rejections are not just "reject everything".
 *
 * The preflight is pure over (attrs, array metadata), so these tests hand it
 * hand-built handles rather than a store. That the preflight fetches no chunks
 * is a property of the LOADER calling it before any read, and is pinned in
 * `whole-node-loader.test.ts` against a request-recording store — the two halves of
 * the same guarantee, tested where each is actually decidable.
 */

import { describe, it, expect } from 'vitest';
import { preflightMesh, parseDtype } from '../../../../data/mesh/preflight';
import type { MeshArrayHandles } from '../../../../data/mesh/preflight';
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
    has_uvs: false,
    has_texture: false,
    shading: 'flat',
    double_sided: true,
    ordering: 'none',
    ...overrides,
  };
}

/** Assert the call rejects as a Validation-kind LoaderError matching `pattern`. */
async function expectReject(fn: () => unknown, pattern: RegExp): Promise<void> {
  let thrown: unknown;
  try {
    await fn();
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
  ])('accepts the friendly name %s', async (dtype, itemSize, integer) => {
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
  ])('accepts the numpy typestring %s', async (dtype, itemSize, integer) => {
    expect(parseDtype(dtype as string)).toEqual({ itemSize, integer });
  });

  it('both spellings occur in practice, so both must parse', async () => {
    // zarrita reports friendly names; a raw .zarray carries typestrings. If only
    // one form parsed, the byte budget would read 0 bytes for the other — a
    // budget that admits everything, which is worse than no budget at all
    // because it looks like protection.
    expect(parseDtype('uint16')).toEqual(parseDtype('<u2'));
    expect(parseDtype('float32')).toEqual(parseDtype('<f4'));
  });

  it('accepts a typestring with no byte-order char — numpy does', async () => {
    // `np.dtype('u2')` is valid, so a store may spell it that way.
    expect(parseDtype('u2')).toEqual({ itemSize: 2, integer: true });
  });

  it('returns null for anything unrecognised rather than guessing a size', async () => {
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
  it('admits a plain tetrahedron and reports what it established', async () => {
    const result = await preflightMesh(PATH, tetAttrs(), tetHandles());
    // accountedBytes is checked in the dedicated `accountedBytes` describe block
    // below against the independently-computed arithmetic — it is a production
    // value now, not a byproduct, so it earns its own precise assertions rather
    // than a placeholder here.
    //
    // `toEqual` (not `toMatchObject`): a relaxed `toMatchObject` here lets an
    // unconsumed extra field on `MeshPreflightResult` pass unnoticed, AND under
    // `toMatchObject` a MISSING key would satisfy an expected `undefined` just
    // as well as a present one — the exact-keys check below closes that second
    // gap explicitly.
    expect(result).toEqual({
      nVertices: 4,
      nFaces: 4,
      ndim: 3,
      colorComponents: undefined,
      normalDims: undefined,
      texture: undefined,
      accountedBytes: result.accountedBytes,
    });
    expect(Object.keys(result).sort()).toEqual(
      [
        'accountedBytes',
        'colorComponents',
        'nFaces',
        'ndim',
        'normalDims',
        'nVertices',
        'texture',
      ].sort()
    );
    expect(Number.isFinite(result.accountedBytes)).toBe(true);
  });

  it('admits every optional array, and reads the colour channel count', async () => {
    const result = await preflightMesh(
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

  it('accepts a LUT-encoded normals array whose STORED shape differs', async () => {
    // A real case, not hypothetical: a fixture whose normals take few distinct
    // values gets LUT-encoded, and the stored codes array carries a different
    // shape than the logical (V, 3). Checking the stored shape would reject a
    // perfectly valid store.
    await expect(
      preflightMesh(
        PATH,
        tetAttrs({ has_normals: true, normal_dims: [0, 1, 2], shading: 'smooth' }),
        tetHandles({
          normals: fakeArray([12], '|u1', [12], {
            encoding: { name: 'lut_uint8', original_shape: [4, 3], original_dtype: 'float32' },
          }),
        })
      )
    ).resolves.toBeDefined();
  });

  it('accepts a BROADCAST colors array — the shape a uniform colour really takes', async () => {
    // The regression this pins. `add_mesh(..., colors=(1, 0, 0))` is a first-class
    // API, and an INCIDENTALLY uniform (V, 3) array is broadcast-encoded too. Both
    // land on disk as `shape: [1, 3]` with `n_elements: V` and — crucially — NO
    // `original_shape`, verified against a written store. A logical-shape rule that
    // consults only `original_shape` therefore sees a 1-row array and refuses a
    // perfectly ordinary mesh.
    await expect(
      preflightMesh(
        PATH,
        tetAttrs({ has_colors: true }),
        tetHandles({
          colors: fakeArray([1, 3], '<f4', [1, 3], {
            encoding: { name: 'broadcasted', n_elements: 4, original_dtype: 'float32' },
          }),
        })
      )
    ).resolves.toBeDefined();
  });

  it('reads the channel count of a broadcast RGBA colour from its stored row', async () => {
    const result = await preflightMesh(
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

  it('does not treat a NON-broadcast array as broadcast just because n_elements is set', async () => {
    // Defense in depth (credit: the alternative fix in #1239 gated on the encoding
    // name, which is stricter than gating on the count alone). Only the two broadcast
    // encoders stamp `n_elements` today, but the priority order should not depend on
    // that staying true — and a hostile store must not cover V vertices by claiming a
    // (1, d) shape with a bare `n_elements`. Stage 2 would catch it, but one stage
    // later and naming the wrong thing.
    await expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({ has_colors: true }),
          tetHandles({
            colors: fakeArray([1, 3], '<f4', [1, 3], {
              encoding: { name: 'uint8', n_elements: 4 },
            }),
          })
        ),
      /colors describes 1 x 3 values but must be 4 x 3/
    );
  });

  it('still rejects a broadcast colour whose n_elements disagrees with n_vertices', async () => {
    // Accepting `n_elements` must not become "accept any broadcast array": the
    // decoder expands to `n_elements` rows, so a mismatch is exactly the
    // undersized-attribute over-read the shape checks exist to stop.
    await expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({ has_colors: true }),
          tetHandles({
            colors: fakeArray([1, 3], '<f4', [1, 3], {
              encoding: { name: 'broadcasted', n_elements: 3 },
            }),
          })
        ),
      /colors describes 3 x 3 values but must be 4 x 3/
    );
  });

  it('accepts a BROADCAST scalars array, which stores one value for V vertices', async () => {
    await expect(
      preflightMesh(
        PATH,
        tetAttrs({ has_scalars: true }),
        tetHandles({
          scalars: fakeArray([1], '<f4', [1], {
            encoding: { name: 'broadcasted', n_elements: 4 },
          }),
        })
      )
    ).resolves.toBeDefined();
  });
});

describe('preflightMesh — (a) counts and the vote-key cap', () => {
  it('rejects n_vertices above MAX_MESH_VERTICES', async () => {
    const n = MAX_MESH_VERTICES + 1;
    await expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs({ n_vertices: n }),
          tetHandles({ vertices: fakeArray([n, 3], '<u2', [1024, 3]) })
        ),
      /exceeds the maximum of .*2\^27/
    );
  });

  it('does NOT trip the cap at exactly MAX_MESH_VERTICES — the bound is inclusive', async () => {
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
      await preflightMesh(
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

  it('checks the cap BEFORE the budget, so an absurd count gets the precise reason', async () => {
    // Both gates would reject a 2^30-vertex mesh. The cap runs first so the
    // message names the actual problem (pick-key aliasing) instead of blaming
    // bytes, which would send an author off decimating a mesh that is
    // fundamentally too large to address.
    const n = 2 ** 30;
    await expectReject(
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
  ])('rejects a non-positive-integer %s', async (_field, override) => {
    await expectReject(
      () => preflightMesh(PATH, tetAttrs(override as Partial<MeshMetadata>), tetHandles()),
      /must be a positive integer/
    );
  });

  it.each([[0], [1], [1.5]])('rejects ndim %s — a triangle needs two dimensions', async (ndim) => {
    // Stricter than the sibling counts on purpose, and stricter than Points/Lines: their
    // primitives are meaningful in 1D, a triangle is not. In 1D every face is collinear,
    // so without this the node loads cleanly and renders nothing with no diagnostic.
    // Mirrors `add_mesh`'s own vertex-width check.
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ ndim } as Partial<MeshMetadata>), {
          vertices: fakeArray([4, ndim], '<f4'),
          faces: fakeArray([4, 3], '<u4'),
        }),
      /at least 2/
    );
  });

  it('accepts ndim 2 — a planar mesh is legitimate', async () => {
    // The acceptance half: the floor is 2, not 3. Flat triangles in a plane have area.
    await expect(
      preflightMesh(PATH, tetAttrs({ ndim: 2 }), {
        vertices: fakeArray([4, 2], '<f4'),
        faces: fakeArray([4, 3], '<u4'),
      })
    ).resolves.toBeDefined();
  });
});

describe('preflightMesh — (b) the byte budget', () => {
  it('rejects a summed declared footprint over the budget', async () => {
    // 40M vertices x 4 dims x 4 bytes = 640 MiB, over the 512 MiB ceiling, and
    // under the 2^27 vertex cap — so this exercises the budget, not the cap.
    const n = 40_000_000;
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_vertices: n, ndim: 4 }), {
          vertices: fakeArray([n, 4], '<f4', [65536, 4]),
          faces: fakeArray([4, 3], '<u4', [4, 3]),
        }),
      /account for .* over the .* per-node budget/
    );
    expect(n).toBeLessThan(MAX_MESH_VERTICES);
  });

  it('rejects an oversized single CHUNK even when the total is tiny', async () => {
    // The case a shape-only budget misses entirely: zarr v2 does not require
    // chunks <= shape, so a 100-triangle array can declare a 268M-triangle
    // chunk. Zarr allocates chunk-shaped buffers, so that one chunk IS a ~3 GB
    // allocation.
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_faces: 100 }), {
          vertices: fakeArray([4, 3], '<f4', [4, 3]),
          faces: fakeArray([100, 3], '<u4', [268_435_456, 3]),
        }),
      /declares a single chunk of .* over the .* per-node budget/
    );
  });

  it('budgets the DECLARED dtype, not the canonical one', async () => {
    // faces is logically uint32, but an external int64 store costs 8 bytes per
    // index. Budgeting a canonical 4 would let it fetch twice the audited bytes.
    //
    // At 20M faces the int64 store accounts for 720 MB (stored 480 + decoded 240)
    // and is rejected, while the same shape as uint32 accounts for 480 MB (stored
    // 240 + decoded 240) and is admitted. The pair is what proves the stored
    // itemsize is genuinely read rather than assumed.
    const f = 20_000_000;
    const int64Handles: MeshArrayHandles = {
      vertices: fakeArray([4, 3], '<f4', [4, 3]),
      faces: fakeArray([f, 3], '<i8', [65536, 3]),
    };
    await expectReject(
      () => preflightMesh(PATH, tetAttrs({ n_faces: f }), int64Handles),
      /account for .* over the/
    );
    // Same shape, 4-byte dtype: admitted. If the budget ignored dtype these two
    // would agree, and the test above would be passing for the wrong reason.
    await expect(
      preflightMesh(PATH, tetAttrs({ n_faces: f }), {
        vertices: fakeArray([4, 3], '<f4', [4, 3]),
        faces: fakeArray([f, 3], '<u4', [65536, 3]),
      })
    ).resolves.toBeDefined();
  });

  it('counts the label CSR arrays toward the footprint', async () => {
    // v1 never fetches these, but they are part of the declared node. Budgeting
    // them from the start means the ceiling does not silently loosen when the
    // label loader lands.
    const bytes = MESH_DECODE_BUDGET_BYTES;
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ has_labels: true }), {
          ...tetHandles(),
          labelOffsets: fakeArray([5], '<u4', [5]),
          labelBytes: fakeArray([bytes], '|u1', [65536]),
        }),
      /account for .* over the/
    );
  });

  it('charges what an array DECODES to, not only what it stores', async () => {
    // The budget-bypass this pins. A broadcast array stores one row and the decoder
    // expands it to `n_elements` rows, so a ~12-byte declaration can materialize
    // gigabytes. Budgeting the stored footprint alone admits it.
    const rows = 60_000_000;
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_vertices: rows, has_colors: true }), {
          vertices: fakeArray([rows, 3], '<u2', [65536, 3]),
          faces: fakeArray([4, 3], '<u4', [4, 3]),
          colors: fakeArray([1, 4], '|u1', [1, 4], {
            encoding: { name: 'broadcasted', n_elements: rows },
          }),
        }),
      /account for .* over the/
    );
  });

  it('bounds ndim, which has no cap of its own, via the decoded footprint', async () => {
    // `n_vertices: 4, ndim: 2^26` passes every count check and its STORED footprint
    // is trivial, but `vertices` decodes to 4 x 2^26 float32 values — over a
    // gigabyte. The logical term is the only thing standing between that
    // declaration and the allocation.
    const wide = 2 ** 26;
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ ndim: wide }), {
          vertices: fakeArray([4, wide], '|u1', [4, 4096]),
          faces: fakeArray([4, 3], '<u4', [4, 3]),
        }),
      /account for .* over the/
    );
  });

  it('charges a narrow dtype at its DECODED width, not its stored width', async () => {
    // Every decoder-routed array yields a Float32Array, so a uint8 store decodes at
    // 4x its stored bytes — and `faces` is widened to u32 regardless of the narrow
    // dtype the INDEX encoder chose. A stored-only budget under-counts by 4x here,
    // in the dangerous direction.
    const n = 100_000_000;
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_vertices: n }), {
          // 300 MB stored as uint8, but 1.2 GB once decoded to float32.
          vertices: fakeArray([n, 3], '|u1', [65536, 3]),
          faces: fakeArray([4, 3], '<u4', [4, 3]),
        }),
      /account for .* over the/
    );
    expect(n * 3 * 1).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
  });

  it('folds the largest chunk buffer INTO the total, not just checking it alone', async () => {
    // Constructible because zarr v2 allows `chunks > shape`: pair a near-budget
    // accounted sum with a tiny array declaring a near-budget oversized chunk. Each
    // term passes its own check, so an independent per-chunk test admits a store that
    // peaks near 2x the ceiling at fetch time.
    //
    // 20M vertices x 3 x f32 = 240 MB stored + 240 MB decoded = 480 MB accounted;
    // a 4-triangle faces array declaring a 6M-triangle chunk adds ~72 MB. Neither
    // term alone exceeds 512 MiB; together they do.
    const n = 20_000_000;
    const chunkTriangles = 6_000_000;
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs({ n_vertices: n }), {
          vertices: fakeArray([n, 3], '<f4', [65536, 3]),
          faces: fakeArray([4, 3], '<u4', [chunkTriangles, 3]),
        }),
      /largest single chunk buffer/
    );
    // Each term individually fits, which is what makes the fold load-bearing.
    expect(n * 3 * 4 * 2).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
    expect(chunkTriangles * 3 * 4).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
  });

  it('REFUSES an encoding the budget cannot account for, rather than admitting it', async () => {
    // Fail-closed. The budget was bypassed four times by one category — "the bytes the
    // loader fetches are not the bytes this handle declares" — because an unrecognised
    // encoding fell through to "use the stored shape". The set is now closed against
    // the contract's `ENCODING_NAMES`, so a future encoder that changes the
    // stored-to-decoded relationship fails the node loudly instead of slipping past
    // the ceiling.
    await expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs(),
          tetHandles({
            faces: fakeArray([4, 3], '<u4', [4, 3], {
              encoding: { name: 'some_future_scheme' },
            }),
          })
        ),
      /cannot account for/
    );
  });

  it('accepts an absent encoding attr — a plain unencoded array', async () => {
    // Anti-vacuity for the gate above: it must refuse an unknown NAME, not any array
    // lacking encoding metadata.
    await expect(
      preflightMesh(PATH, tetAttrs(), {
        vertices: fakeArray([4, 3], '<f4'),
        faces: fakeArray([4, 3], '<u4'),
      })
    ).resolves.toBeDefined();
  });

  it.each([
    ['broadcasted', { name: 'broadcasted', n_elements: 4 }],
    ['linear_perchannel_u16', { name: 'linear_perchannel_u16', original_dtype: 'float32' }],
    ['lut_uint8', { name: 'lut_uint8', original_shape: [4, 3], original_dtype: 'float32' }],
    ['uint8', { name: 'uint8', original_dtype: 'uint32' }],
  ])('accepts the real encoding %s that the writer emits', async (_label, encoding) => {
    // The four this writer actually produces for mesh arrays, so the closed set cannot
    // be closed too tightly.
    await expect(
      preflightMesh(
        PATH,
        tetAttrs(),
        tetHandles({
          vertices: fakeArray([4, 3], '<u2', [4, 3], { encoding }),
        })
      )
    ).resolves.toBeDefined();
  });

  it('rejects an unrecognised dtype rather than budgeting it as free', async () => {
    await expectReject(
      () => preflightMesh(PATH, tetAttrs(), tetHandles({ colors: fakeArray([4, 3], '<c16') })),
      /unrecognised dtype/
    );
  });
});

describe('preflightMesh — (c) shape and dtype cross-checks', () => {
  it('rejects a vertices width that disagrees with ndim', async () => {
    // The check that stops a `panic = "abort"` trap: the slab kernel strides by
    // ndim, so a narrower row reads past the end of the array.
    await expectReject(
      () => preflightMesh(PATH, tetAttrs({ ndim: 4 }), tetHandles()),
      /vertices describes 4 x 3 values but must be 4 x 4/
    );
  });

  it('rejects a vertices row count that disagrees with n_vertices', async () => {
    await expectReject(
      () => preflightMesh(PATH, tetAttrs(), tetHandles({ vertices: fakeArray([3, 3], '<f4') })),
      /vertices describes 3 x 3 values/
    );
  });

  it.each([
    ['a non-3 face width', [4, 4]],
    ['a face count mismatch', [5, 3]],
    ['a 1-D faces array', [12]],
  ])('rejects %s', async (_label, shape) => {
    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs(), tetHandles({ faces: fakeArray(shape as number[], '<u4') })),
      /faces describes/
    );
  });

  it('rejects a FLOAT faces dtype', async () => {
    // A float index truncates in the uint32 coercion, silently rewriting
    // topology. Mirrors the write-side validate_faces_for_writing rule.
    await expectReject(
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
  ])('accepts an external %s faces dtype', async (_label, dtype) => {
    // Signed and 64-bit stores are legal here; their wrap-around hazards are
    // Stage 2's job, on the source-typed VALUES. Rejecting them at Stage 1
    // would refuse valid external meshes.
    await expect(
      preflightMesh(PATH, tetAttrs(), tetHandles({ faces: fakeArray([4, 3], dtype as string) }))
    ).resolves.toBeDefined();
  });

  it.each([
    ['normals', 'has_normals', [3, 3], /normals describes/],
    ['colors', 'has_colors', [4, 2], /colors describes/],
    ['scalars', 'has_scalars', [3], /scalars describes/],
  ])('rejects an undersized %s array', async (slot, flag, shape, pattern) => {
    // These bind as ENABLED vertex attributes on an indexed draw. An undersized
    // one does not trap — drawElements reads past the buffer and mis-shades
    // every vertex it covers, backend-dependently. Catchable from the shape.
    await expectReject(
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

  it('accepts colors at 3 OR 4 channels', async () => {
    for (const channels of [3, 4] as const) {
      const result = await preflightMesh(
        PATH,
        tetAttrs({ has_colors: true }),
        tetHandles({ colors: fakeArray([4, channels], '|u1') })
      );
      expect(result.colorComponents).toBe(channels);
    }
  });
});

describe('preflightMesh — presence flags must match the store', () => {
  it.each([
    ['has_normals', { has_normals: true, normal_dims: [0, 1, 2] }],
    ['has_colors', { has_colors: true }],
    ['has_scalars', { has_scalars: true }],
  ])('rejects %s set with no array behind it', async (flag, override) => {
    // The load-bearing direction: the geometry builder reads the flag to decide
    // whether to bind `normal` / `aScalar` and to enable the colormap path, so a
    // lying flag produces a draw referencing a buffer never uploaded.
    await expectReject(
      () => preflightMesh(PATH, tetAttrs(override as Partial<MeshMetadata>), tetHandles()),
      new RegExp(`${flag} is set but its array\\(s\\) are missing`)
    );
  });

  it('does NOT reject an array the flags disown — the loader cannot produce that', async () => {
    // The converse direction is unreachable through the real loader: `initialize`
    // opens an optional array only when its flag is set. Asserting a rejection here
    // would be testing a state production cannot construct, and the docs would be
    // claiming an enforcement that never fires. So this pins the absence.
    await expect(
      preflightMesh(PATH, tetAttrs(), tetHandles({ normals: fakeArray([4, 3], '<f4') }))
    ).resolves.toBeDefined();
  });

  it.each([
    ['has_labels', { has_labels: true }, 'labelOffsets'],
    ['has_image_labels', { has_image_labels: true }, 'imageLabelOffsets'],
    ['has_keys', { has_keys: true }, 'keyOffsets'],
  ])('rejects %s with only half of its CSR pair', async (flag, override, presentSlot) => {
    // v1 never fetches the label arrays, but a half-present pair is a store that
    // fails confusingly the moment picking lands — so it is refused now, while the
    // error can still name the actual problem.
    await expectReject(
      () =>
        preflightMesh(
          PATH,
          tetAttrs(override as Partial<MeshMetadata>),
          tetHandles({ [presentSlot as string]: fakeArray([5], '<u4') })
        ),
      new RegExp(`${flag} is set but its array\\(s\\) are missing`)
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
  ])('rejects normal_dims %s', async (_label, dims, pattern) => {
    await expectReject(
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

  it('is not checked when the node has no normals', async () => {
    // normal_dims describes a normals array; with no normals there is nothing to
    // orient, so a stray value must not fail the node.
    await expect(
      preflightMesh(PATH, tetAttrs({ normal_dims: [9, 9, 9] }), tetHandles())
    ).resolves.toBeDefined();
  });

  it('accepts a non-ascending frame — order carries the x/y/z roles', async () => {
    const result = await preflightMesh(
      PATH,
      tetAttrs({ has_normals: true, shading: 'smooth', normal_dims: [2, 0, 1] }),
      tetHandles({ normals: fakeArray([4, 3], '<f4') })
    );
    expect(result.normalDims).toEqual([2, 0, 1]);
  });
});

/**
 * `accountedBytes` is now a production value — the mesh reveal ladder
 * (`mesh-progressive-loader.ts`'s `MeshProgressiveLoader.assertWithinByteBudget`,
 * on the ladder's first load; deliberately NOT the `createProgressiveMeshLoader`
 * factory, which has no failure containment around it) sums it across every
 * level to charge the whole ladder against ONE budget. So it needs its own
 * pinning, independent of the acceptance/rejection tests above: not just that
 * SOME number comes back, but that the number IS the arithmetic the ceiling
 * comparison uses (stored bytes + decoded bytes, summed over every array,
 * plus the single largest chunk buffer) — not a lookalike that happens to be
 * in the right ballpark.
 */
describe('preflightMesh — accountedBytes', () => {
  it('equals the independently-computed stored+decoded+max-chunk arithmetic', async () => {
    // Case 1: a plain tetrahedron, unencoded float32/uint32, chunks == shape.
    // vertices: stored 12*4=48, decoded 12*4=48 -> 96; chunk 12*4=48.
    // faces:    stored 12*4=48, decoded 12*4=48 -> 96; chunk 12*4=48.
    // accountedBytes-sum = 192, max chunk = 48 -> total 240.
    const plain = await preflightMesh(PATH, tetAttrs(), {
      vertices: fakeArray([4, 3], '<f4', [4, 3]),
      faces: fakeArray([4, 3], '<u4', [4, 3]),
    });
    expect(plain.accountedBytes).toBe(240);

    // Case 2: same tetrahedron, plus every optional array.
    // normals (float32, [4,3]): stored 48, decoded 48 -> 96; chunk 48.
    // colors (uint8, RGBA [4,4]): stored 16, decoded 64 -> 80; chunk 16.
    // scalars (uint8, [4]): stored 4, decoded 16 -> 20; chunk 4.
    // accountedBytes-sum = 192 (vertices+faces) + 96 + 80 + 20 = 388.
    // max chunk over all five arrays = 48 -> total 436.
    const withOptional = await preflightMesh(
      PATH,
      tetAttrs({
        has_normals: true,
        normal_dims: [0, 1, 2],
        has_colors: true,
        has_scalars: true,
        shading: 'smooth',
      }),
      {
        vertices: fakeArray([4, 3], '<f4', [4, 3]),
        faces: fakeArray([4, 3], '<u4', [4, 3]),
        normals: fakeArray([4, 3], '<f4', [4, 3]),
        colors: fakeArray([4, 4], '|u1', [4, 4]),
        scalars: fakeArray([4], '|u1', [4]),
      }
    );
    expect(withOptional.accountedBytes).toBe(436);

    // Case 3: a chunk declared LARGER than the array's own shape (the zarr v2
    // exploit the budget's chunk term exists to catch, per preflight.ts's own
    // comment) dominates the max-chunk term rather than being folded away.
    // faces chunk 1000*3 elements * 4 bytes = 12000, which becomes the max.
    // accountedBytes-sum is unchanged at 192 -> total 192 + 12000 = 12192.
    const bigChunk = await preflightMesh(PATH, tetAttrs(), {
      vertices: fakeArray([4, 3], '<f4', [4, 3]),
      faces: fakeArray([4, 3], '<u4', [1000, 3]),
    });
    expect(bigChunk.accountedBytes).toBe(12192);

    // Case 4: the oversized chunk on an OPTIONAL array (colors), not
    // vertices/faces. A guard that folded the chunk term into the loop only
    // for the two required arrays — rather than every array the loop visits —
    // would miss this: it still passes cases 1-3 (neither exercises an
    // optional array's chunk at all).
    // vertices: stored 48, decoded 48 -> 96; chunk 48.
    // faces:    stored 48, decoded 48 -> 96; chunk 48.
    // colors (uint8 RGB [4,3]): stored 12, decoded 48 -> 60; chunk 1000*3*1=3000.
    // accountedBytes-sum = 96 + 96 + 60 = 252; max chunk = 3000 -> total 3252.
    const bigOptionalChunk = await preflightMesh(PATH, tetAttrs({ has_colors: true }), {
      vertices: fakeArray([4, 3], '<f4', [4, 3]),
      faces: fakeArray([4, 3], '<u4', [4, 3]),
      colors: fakeArray([4, 3], '|u1', [1000, 3]),
    });
    expect(bigOptionalChunk.accountedBytes).toBe(3252);
  });

  it('is the exact quantity the ceiling gates: admitted AT budget, refused one byte over', async () => {
    // Uses the same chunks-independent-of-shape affordance as the arithmetic
    // above, but pushed to hit the ceiling on the nose. All dtypes are uint8
    // (itemSize 1) so every term is a whole number of bytes, letting a single
    // extra chunk element move the total by exactly one byte.
    //
    // Fixed part (vertices + faces, both [4,3] uint8, chunks == shape):
    //   each array: stored 12*1=12, decoded 12*4=48 -> 60; chunk 12*1=12.
    //   accountedBytes-sum = 120; baseline max chunk = 12.
    // The vertices array's CHUNK is then set independently of its shape (zarr
    // v2 permits chunks > shape IN MAGNITUDE — it does not permit a different
    // RANK; `len(chunks) == len(shape)` is a zarr v2 structural requirement, so
    // the oversized chunk keeps rank 2 (`[n, 1]`, same element count as `[n]`)
    // rather than dropping to rank 1) to land the total exactly on, then one
    // byte past, MESH_DECODE_BUDGET_BYTES.
    const fixedSum = 120;
    const atBudgetChunk = MESH_DECODE_BUDGET_BYTES - fixedSum;
    const overBudgetChunk = atBudgetChunk + 1;

    const atBudget = await preflightMesh(PATH, tetAttrs(), {
      vertices: fakeArray([4, 3], '|u1', [atBudgetChunk, 1]),
      faces: fakeArray([4, 3], '|u1', [4, 3]),
    });
    expect(atBudget.accountedBytes).toBe(MESH_DECODE_BUDGET_BYTES);

    await expectReject(
      () =>
        preflightMesh(PATH, tetAttrs(), {
          vertices: fakeArray([4, 3], '|u1', [overBudgetChunk, 1]),
          faces: fakeArray([4, 3], '|u1', [4, 3]),
        }),
      /account for.*over the.*budget/
    );
  });
});

/**
 * Texture and UV admission.
 *
 * The declared texture dimensions are the only bound on an encoded texture's
 * decode, which makes them the one place in this file where a check that merely
 * *runs* is not enough — it has to run BEFORE the arithmetic. Several of these
 * cases exist specifically to pin that ordering: a `NaN` or negative dimension
 * must be reported as a bad declaration, never silently multiplied into a
 * comparison that passes.
 */
describe('preflightMesh — textures and UVs', () => {
  /** Attrs for a textured mesh, defaulting to a small valid encoded declaration. */
  const texAttrs = (o: Partial<MeshMetadata> = {}): MeshMetadata =>
    tetAttrs({
      has_uvs: true,
      has_texture: true,
      texture_encoding: 'jpeg',
      texture_width: 64,
      texture_height: 32,
      texture_channels: 3,
      texture_color_space: 'srgb',
      ...o,
    });

  /** Handles for a textured mesh: per-vertex UVs plus an encoded byte blob. */
  const texHandles = (o: Partial<MeshArrayHandles> = {}): MeshArrayHandles =>
    tetHandles({
      uvs: fakeArray([4, 2], '<f4'),
      texture: fakeArray([2048], '|u1'),
      ...o,
    });

  const reject = async (attrs: MeshMetadata, handles: MeshArrayHandles) =>
    await expect(preflightMesh(PATH, attrs, handles)).rejects.toThrow(LoaderError);

  const message = async (attrs: MeshMetadata, handles: MeshArrayHandles): Promise<string> => {
    try {
      await preflightMesh(PATH, attrs, handles);
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
    throw new Error('expected a rejection');
  };

  it('admits a textured mesh and returns the validated declaration', async () => {
    const result = await preflightMesh(PATH, texAttrs(), texHandles());
    // Returned, not just checked: the loader allocates from THIS, so that a
    // second unvalidated copy of the numbers never reaches the decode.
    expect(result.texture).toEqual({
      encoding: 'jpeg',
      width: 64,
      height: 32,
      channels: 3,
      decode: 'codec',
    });
  });

  it('admits a raw texture whose array matches its declared surface', async () => {
    const result = await preflightMesh(
      PATH,
      texAttrs({ texture_encoding: 'raw', texture_channels: 4 }),
      texHandles({ texture: fakeArray([32, 64, 4], '|u1') })
    );
    expect(result.texture?.decode).toBe('raw');
  });

  it('admits an HDR raw texture stored as quantized codes', async () => {
    // The realistic HDR-on-disk case: AUTO encoding sends a float texture
    // through `geolog_perchannel_u16`, so the stored dtype is u16 and the
    // logical shape lives in `original_shape`. Checking the STORED shape would
    // reject this, which is writer output.
    const result = await preflightMesh(
      PATH,
      texAttrs({ texture_encoding: 'raw', texture_channels: 3 }),
      texHandles({
        texture: fakeArray([6144], '<u2', [6144], {
          encoding: {
            name: 'geolog_perchannel_u16',
            original_dtype: 'float32',
            original_shape: [32, 64, 3],
          },
        }),
      })
    );
    expect(result.texture?.decode).toBe('raw');
  });

  it.each([
    ['a missing encoding', { texture_encoding: undefined }, /texture_encoding/],
    ['an unknown encoding', { texture_encoding: 'avif' as never }, /texture_encoding/],
    ['a zero width', { texture_width: 0 }, /texture_width/],
    ['a negative height', { texture_height: -32 }, /texture_height/],
    ['a NaN width', { texture_width: Number.NaN }, /texture_width/],
    ['a fractional width', { texture_width: 64.5 }, /texture_width/],
    ['a non-numeric width', { texture_width: '64' as never }, /texture_width/],
    ['a zero channel count', { texture_channels: 0 }, /texture_channels/],
    ['a 2-channel texture', { texture_channels: 2 }, /texture_channels/],
  ])('refuses %s', async (_label, override, pattern) => {
    await expect(preflightMesh(PATH, texAttrs(override), texHandles())).rejects.toThrow(pattern);
  });

  it('refuses a dimension over the per-axis GPU limit', async () => {
    // 100000 x 2 is the case the BYTE budget cannot catch: 800 KB accounted,
    // comfortably admitted, then silently clamped at upload.
    const msg = await message(
      texAttrs({ texture_width: 100_000, texture_height: 2 }),
      texHandles()
    );
    expect(msg).toMatch(/per-axis limit/);
  });

  it('refuses a codec texture whose declared surface blows the byte budget', async () => {
    // The decompression bomb: a small stored payload declaring a huge surface.
    // 12000 x 12000 x 4 = 576 MiB, from 2 KB of stored bytes. Deliberately kept
    // UNDER the per-axis cap so this exercises the byte budget rather than the
    // dimension check — a 30000-wide texture is refused one gate earlier and
    // would make this test pass without the budget ever running.
    const msg = await message(
      texAttrs({ texture_width: 12_000, texture_height: 12_000 }),
      texHandles()
    );
    expect(msg).toMatch(/per-node budget/);
  });

  it('charges the codec surface at 4 bytes per pixel regardless of channels', async () => {
    // A 1-channel declaration must be charged the same as a 4-channel one: an
    // ImageBitmap is always RGBA8. Reading `texture_channels` here would
    // under-charge by 4x and admit a bomb four times the ceiling.
    const bomb = { texture_width: 16_000, texture_height: 16_000 };
    for (const channels of [1, 3, 4]) {
      await expect(
        preflightMesh(PATH, texAttrs({ ...bomb, texture_channels: channels }), texHandles())
      ).rejects.toThrow(/per-node budget/);
    }
  });

  it('admits a large codec texture just inside the budget', async () => {
    // The other side of the boundary above: without this, "rejects everything
    // big" would pass the bomb tests just as well.
    const result = await preflightMesh(
      PATH,
      texAttrs({ texture_width: 4096, texture_height: 4096 }),
      texHandles()
    );
    expect(result.texture?.width).toBe(4096);
    // 4096^2 * 4 = 64 MiB of surface, plus the small vertex arrays.
    expect(result.accountedBytes).toBeGreaterThan(64 * 1024 * 1024);
    expect(result.accountedBytes).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
  });

  it('charges KTX2 at one byte per pixel plus the complete mip tail', async () => {
    const result = await preflightMesh(
      PATH,
      texAttrs({ texture_encoding: 'ktx2', texture_width: 4096, texture_height: 4096 }),
      texHandles()
    );
    expect(result.texture?.decode).toBe('ktx2');
    const surfaceWithMips = Math.ceil((4096 * 4096 * 4) / 3);
    expect(result.accountedBytes).toBeGreaterThan(surfaceWithMips);
    expect(result.accountedBytes).toBeLessThan(surfaceWithMips + 1024 * 1024);
  });

  it('refuses single-channel KTX2 rather than silently expanding it to RGB', async () => {
    await expect(
      preflightMesh(PATH, texAttrs({ texture_encoding: 'ktx2', texture_channels: 1 }), texHandles())
    ).rejects.toThrow(/only RGB or RGBA.*use 'raw'/);
  });

  it('does NOT double-charge a raw texture', async () => {
    // The raw surface is already charged by the generic per-array loop, so the
    // dedicated term must skip it. A raw 4096x4096x4 uint8 texture accounts for
    // 64 MiB stored + 256 MiB decoded = 320 MiB; adding the codec term as well
    // would push it past the 512 MiB ceiling and falsely reject writer output.
    const result = await preflightMesh(
      PATH,
      texAttrs({
        texture_encoding: 'raw',
        texture_width: 4096,
        texture_height: 4096,
        texture_channels: 4,
      }),
      texHandles({ texture: fakeArray([4096, 4096, 4], '|u1', [512, 4096, 4]) })
    );
    expect(result.accountedBytes).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
    expect(result.accountedBytes).toBeGreaterThan(320 * 1024 * 1024);
  });

  it('refuses a raw texture whose array disagrees with its declaration', async () => {
    const msg = await message(
      texAttrs({ texture_encoding: 'raw', texture_channels: 3 }),
      texHandles({ texture: fakeArray([16, 16, 3], '|u1') })
    );
    expect(msg).toMatch(/but its array describes/);
  });

  it.each([
    ['has_uvs with no uvs array', { uvs: undefined }, /has_uvs is set/],
    ['has_texture with no texture array', { texture: undefined }, /has_texture is set/],
  ])('refuses %s', async (_label, override, pattern) => {
    await expect(
      preflightMesh(PATH, texAttrs(), texHandles(override as Partial<MeshArrayHandles>))
    ).rejects.toThrow(pattern);
  });

  it.each([
    ['uvs without a texture', { has_texture: false }],
    ['a texture without uvs', { has_uvs: false }],
  ])('refuses %s', async (_label, override) => {
    // Each half alone renders SOMETHING wrong rather than failing, which is why
    // it is refused at load rather than left to the material.
    const attrs = texAttrs(override);
    const handles = texHandles(
      'has_texture' in override ? { texture: undefined } : { uvs: undefined }
    );
    const msg = await message(attrs, handles);
    expect(msg).toMatch(/only\s+meaningful together|is set but/);
  });

  it.each([
    ['3 components', [4, 3]],
    ['1 component', [4, 1]],
    ['the wrong row count', [8, 2]],
  ])('refuses uvs with %s', async (_label, shape) => {
    await reject(texAttrs(), texHandles({ uvs: fakeArray(shape, '<f4') }));
  });
});
