/**
 * Stage 1 of the mesh loader's admission gate: the **metadata preflight**.
 *
 * Runs purely on the node attrs and each array's zarr `.zarray` metadata
 * (declared shape, `chunks`, dtype). It touches **no chunk data** — every check
 * here is decidable before a single byte of geometry is fetched, and that is the
 * whole point of the stage existing separately.
 *
 * ## Why a second stage isn't enough
 *
 * The writer's `validate_*_for_writing` family protects only stores Luxar
 * produced. The viewer loads arbitrary `?src=` URLs, and the mesh loader is
 * whole-node (`docs/specs/MESH_NODE_SPEC.md` §7): it fetches and decodes every
 * array in full, up front. A check that runs only *after* decode therefore
 * arrives too late for the quantities that gate admission — a hostile or
 * corrupt store can declare enormous arrays and exhaust tab memory before the
 * `LoaderError` containment ("one node lost, not the scene") is ever reachable.
 *
 * So everything decidable from metadata is checked here, first, and the
 * value-level checks that genuinely need materialized arrays run afterwards in
 * Stage 2 (`mesh-validate.ts`).
 *
 * ## What is checked, and what each check prevents
 *
 * - **`n_vertices <= MAX_MESH_VERTICES`** — above `2^27` the pick vote key
 *   aliases across nodes (§6.5). Refused before allocation, not after a
 *   multi-gigabyte fetch.
 * - **Byte budget** — the summed declared footprint AND each array's largest
 *   single per-chunk decode allocation, both against `MESH_DECODE_BUDGET_BYTES`.
 * - **Shape and dtype cross-checks** — against `n_vertices`/`n_faces`/`ndim`
 *   and the §3.2 array table.
 * - **`normal_dims` well-formedness** — 3 distinct integers in `[0, ndim)`.
 *
 * ### Which gate actually binds
 *
 * At the default 512 MiB budget the **byte budget is far tighter than the vertex
 * cap**, and it is worth knowing which error to expect. A 3D float32 mesh runs
 * out of budget at ~44.7M vertices and a uint16-quantized one at ~89.5M, both
 * well under `2^27` (134.2M) — so in practice no mesh reaches the cap by growing
 * legitimately; `2^27` vertices of quantized 3D coordinates alone declare 768
 * MiB.
 *
 * The cap is still checked, and checked **first**, for two reasons. It gives a
 * hostile or nonsensical declaration (`n_vertices: 2^30`) the message that names
 * the real problem — pick keys aliasing, which no amount of decimation fixes —
 * rather than blaming bytes. And it keeps the bound enforced rather than assumed
 * if the budget is ever raised.
 *
 * The shape checks are not cosmetic tidiness. Two of them stop a
 * `panic = "abort"` WASM trap, which escapes as an opaque uncatchable
 * `RuntimeError: unreachable` rather than a node-scoped error:
 *
 * - a `vertices` width that disagrees with `ndim` makes the §5.4 slab kernel
 *   index out of slice bounds;
 * - a `faces` array of the wrong shape or a float dtype either mis-strides the
 *   topology or truncates in the u32 coercion.
 *
 * And the optional-array shape checks stop a quieter failure: `normals` /
 * `colors` / `scalars` bind as *enabled vertex attributes* on an indexed draw
 * (§6.1). An undersized one doesn't trap — it makes `drawElements` read past
 * the buffer (an invalid-operation draw or silent zeros, backend-dependent) and
 * mis-shades every vertex it covers. A declared-undersized array is catchable
 * here from its shape alone.
 *
 * @module data/mesh/mesh-preflight
 */

import type * as zarr from '../zarr';
import { MAX_MESH_VERTICES, MESH_DECODE_BUDGET_BYTES } from '../../config/constants';
import type { ArrayMetadata } from '../array-decoder/decoder';
import { LoaderError } from '../scene-loader/nodes/load-leaf-error-dispatch';
import type { MeshMetadata } from '../../types/mesh';

/**
 * Metadata-only handles for the arrays a mesh node declares.
 *
 * Every entry is a zarr `Array` opened with `kind: 'array'`, which fetches
 * `.zarray` + `.zattrs` and nothing else. Optional entries are `undefined` when
 * the node's presence flags say the array is absent.
 *
 * The label/image-label CSR arrays are included even though v1 does not fetch
 * them (picking lands in a later phase). They are part of the node's declared
 * footprint, so budgeting them from the start means the ceiling does not
 * silently loosen when the label loader arrives.
 */
export interface MeshArrayHandles {
  vertices: zarr.Array<zarr.DataType, zarr.Readable>;
  faces: zarr.Array<zarr.DataType, zarr.Readable>;
  normals?: zarr.Array<zarr.DataType, zarr.Readable>;
  colors?: zarr.Array<zarr.DataType, zarr.Readable>;
  scalars?: zarr.Array<zarr.DataType, zarr.Readable>;
  labelOffsets?: zarr.Array<zarr.DataType, zarr.Readable>;
  labelBytes?: zarr.Array<zarr.DataType, zarr.Readable>;
  imageLabelOffsets?: zarr.Array<zarr.DataType, zarr.Readable>;
  imageLabelBytes?: zarr.Array<zarr.DataType, zarr.Readable>;
}

/**
 * The zarr array name behind each handle slot.
 *
 * Not always the slot name: the CSR label arrays are `label_offsets` /
 * `label_bytes` on disk. Lives here rather than in the loader because it
 * describes {@link MeshArrayHandles}, and because the preflight's error messages
 * must name the array the USER can go look at, not our camelCase slot.
 */
export const MESH_ARRAY_NAMES: Record<keyof MeshArrayHandles, string> = {
  vertices: 'vertices',
  faces: 'faces',
  normals: 'normals',
  colors: 'colors',
  scalars: 'scalars',
  labelOffsets: 'label_offsets',
  labelBytes: 'label_bytes',
  imageLabelOffsets: 'image_label_offsets',
  imageLabelBytes: 'image_label_bytes',
};

/** What Stage 1 established, for Stage 2 and the geometry builder to rely on. */
export interface MeshPreflightResult {
  /** Vertex count, `<= MAX_MESH_VERTICES` */
  nVertices: number;
  /** Triangle count, `>= 1` */
  nFaces: number;
  /** Coordinate dimensionality; equals the `vertices` declared width */
  ndim: number;
  /** Channels per color entry, when `colors` is present */
  colorComponents?: 3 | 4;
  /** The validated `normal_dims`, when `normals` is present */
  normalDims?: number[];
}

/**
 * Bytes per element for a zarr dtype string, and whether it is an integer type.
 *
 * Both spellings occur in practice: zarrita reports friendly names
 * (`'uint16'`, `'float32'`) while a raw `.zarray` carries numpy typestrings
 * (`'<u2'`, `'|u1'`, `'>f4'`). Accepting only one form would make the budget
 * silently read `0` bytes for the other — the failure mode being a budget that
 * admits everything.
 *
 * Returns `null` for anything unrecognised, which callers treat as a rejection
 * rather than a free pass.
 */
export function parseDtype(dtype: string): { itemSize: number; integer: boolean } | null {
  const friendly: Record<string, { itemSize: number; integer: boolean }> = {
    int8: { itemSize: 1, integer: true },
    uint8: { itemSize: 1, integer: true },
    int16: { itemSize: 2, integer: true },
    uint16: { itemSize: 2, integer: true },
    int32: { itemSize: 4, integer: true },
    uint32: { itemSize: 4, integer: true },
    int64: { itemSize: 8, integer: true },
    uint64: { itemSize: 8, integer: true },
    float16: { itemSize: 2, integer: false },
    float32: { itemSize: 4, integer: false },
    float64: { itemSize: 8, integer: false },
    bool: { itemSize: 1, integer: true },
  };
  const hit = friendly[dtype];
  if (hit) return hit;

  // numpy typestring: optional byte-order char, then a kind char, then width.
  const m = /^[<>|=]?([iuUfbc])(\d+)$/.exec(dtype);
  if (!m) return null;
  const kind = m[1];
  const width = Number(m[2]);
  if (!Number.isFinite(width) || width <= 0) return null;
  // 'b' is numpy's boolean kind and carries no width digits in `|b1`'s usual
  // spelling, but `|b1` does match here with width 1 — treat it as integer.
  if (kind === 'i' || kind === 'u' || kind === 'b') return { itemSize: width, integer: true };
  if (kind === 'f') return { itemSize: width, integer: false };
  return null;
}

/** Product of a shape's extents, or `null` when the shape is unusable. */
function elementCount(shape: readonly number[] | undefined): number | null {
  if (!Array.isArray(shape) || shape.length === 0) return null;
  let n = 1;
  for (const extent of shape) {
    if (!Number.isInteger(extent) || extent < 0) return null;
    n *= extent;
  }
  return Number.isFinite(n) ? n : null;
}

/**
 * The array's LOGICAL layout — what the values MEAN and how many the decoder will
 * materialize — as opposed to how they are stored.
 *
 * Three sources, in the same priority order `ArrayDecoder` itself applies, and all
 * three occur in stores this writer produces:
 *
 * 1. **`encoding.n_elements`** (broadcast). The stored array holds ONE row and the
 *    decoder expands it to `n_elements` rows. Critically, the broadcast encoder
 *    stamps `n_elements` and **not** `original_shape` — so an `original_shape`-only
 *    reading of "logical" mistakes a uniform colour for a 1-row array and rejects
 *    it. `add_mesh(..., colors=(1, 0, 0))` is a first-class API, and an
 *    *incidentally* uniform `(V, 3)` array is broadcast-encoded too, so this is the
 *    common case rather than an exotic one.
 * 2. **`encoding.original_shape`** (LUT, per-channel quantization). The stored array
 *    holds codes whose shape differs from the values'.
 * 3. **`array.shape`** for a plain unencoded array.
 *
 * Returning a layout rather than a shape is deliberate: the byte budget needs the
 * total logical `count` (what gets allocated), while the cross-checks need `rows`
 * and `components` separately. Collapsing them to one array invites reading the
 * wrong one, which is exactly how the broadcast-colour rejection got in.
 */
interface LogicalLayout {
  /** Logical rows — one per vertex for every mesh array. */
  rows: number;
  /** Components per row: 3 for normals, 3 or 4 for colours, 1 for scalars. */
  components: number;
  /** Total values the decoder materializes (`rows * components`). */
  count: number;
}

function logicalLayout(array: zarr.Array<zarr.DataType, zarr.Readable>): LogicalLayout | null {
  const attrs = (array.attrs ?? {}) as unknown as ArrayMetadata;
  const encoding = attrs.encoding;

  const trailing = (shape: readonly number[]): number =>
    shape.length >= 2 ? shape[shape.length - 1] : 1;

  // 1. Broadcast: rows come from n_elements, components from the stored row.
  const n = encoding?.n_elements;
  if (typeof n === 'number' && Number.isInteger(n) && n >= 0) {
    const components = trailing(array.shape);
    if (!Number.isInteger(components) || components < 1) return null;
    return { rows: n, components, count: n * components };
  }

  // 2/3. Declared logical shape, else the stored one.
  const original = encoding?.original_shape;
  const shape = Array.isArray(original) && original.length > 0 ? original : array.shape;
  const total = elementCount(shape);
  if (total === null) return null;
  const components = trailing(shape);
  if (!Number.isInteger(components) || components < 1) return null;
  return { rows: total / components, components, count: total };
}

/** Human-readable byte count for error messages. */
function mib(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

/**
 * Fail the node with a Stage-1 rejection.
 *
 * A `function` declaration with an explicit `never` return type on purpose:
 * that is what lets TypeScript narrow control flow through a call to it, so
 * every check below reads as a guard clause without a trailing `return`. The
 * same body written as an arrow assigned to an un-annotated `const` would
 * type-check but narrow nothing, and each caller would need dead-code padding.
 *
 * `kind: 'Validation'` matters beyond the message: it is persisted on the
 * failure record, so the retry policy treats a malformed store as deterministic
 * instead of re-fetching it on every reconnect.
 */
function rejectMesh(path: string, message: string): never {
  throw new LoaderError('Validation', path, new Error(message));
}

/**
 * Run the Stage-1 metadata preflight.
 *
 * @throws LoaderError `kind: 'Validation'` on any rejection, having fetched no
 *   chunk data.
 */
export function preflightMesh(
  path: string,
  attrs: MeshMetadata,
  arrays: MeshArrayHandles
): MeshPreflightResult {
  // --- (a) counts, and the vote-key cap -------------------------------------
  const nVertices = attrs.n_vertices;
  const nFaces = attrs.n_faces;
  const ndim = attrs.ndim;

  if (!Number.isInteger(nVertices) || nVertices < 1) {
    rejectMesh(path, `n_vertices must be a positive integer, got ${String(nVertices)}`);
  }
  if (!Number.isInteger(nFaces) || nFaces < 1) {
    rejectMesh(path, `n_faces must be a positive integer, got ${String(nFaces)}`);
  }
  if (!Number.isInteger(ndim) || ndim < 1) {
    rejectMesh(path, `ndim must be a positive integer, got ${String(ndim)}`);
  }
  if (nVertices > MAX_MESH_VERTICES) {
    rejectMesh(
      path,
      `n_vertices ${nVertices.toLocaleString()} exceeds the maximum of ` +
        `${MAX_MESH_VERTICES.toLocaleString()} (2^27). Above this the pick vote key ` +
        'aliases across nodes and picks resolve to the wrong node. Split the mesh ' +
        'into several nodes, or decimate it.'
    );
  }

  // --- (b) byte budget -----------------------------------------------------
  //
  // TWO terms per array, because the loader allocates twice over: it fetches the
  // stored bytes, and then the decoder materializes the LOGICAL values, and on the
  // admission path those coexist.
  //
  // Budgeting only the stored footprint is not conservative, it is wrong in the
  // dangerous direction — the stored side can be arbitrarily smaller than what gets
  // allocated:
  //
  //   * a BROADCAST array stores one row and decodes to `n_elements` rows, so a
  //     ~12-byte declaration can materialize gigabytes;
  //   * every decoder-routed array yields a Float32Array, so a `uint8` store decodes
  //     at 4x its stored bytes and a `uint16` at 2x;
  //   * `faces` is widened to u32 regardless of the narrow dtype the INDEX encoder
  //     chose, so a `uint8` faces array also costs 4x on decode.
  //
  // So the logical term is charged at 4 bytes per value — the width of the widest
  // thing any of these paths materializes (`Float32Array`, or `Uint32Array` for
  // faces). Colours kept in their native dtype cost less than that, which makes this
  // an over-estimate for them and never an under-estimate.
  //
  // Charging the logical term is also what bounds `ndim`, which has no cap of its
  // own: `n_vertices: 4, ndim: 2^25` passes every count check, and its stored
  // footprint can be tiny, but `vertices` decodes to 4 x 2^25 floats. The budget is
  // the only thing standing between that declaration and a 512 MB allocation.
  //
  // The DECLARED dtype drives the stored term (never a canonical one): `faces` is
  // logically uint32 but the INDEX encoder narrows it to the smallest unsigned dtype
  // that fits, while an external int64 store costs 8 bytes per index — so a
  // canonical 4 would be wrong in both directions.
  const DECODED_BYTES_PER_VALUE = 4;
  let accountedBytes = 0;
  // The largest single chunk buffer, folded into the total below rather than only
  // checked on its own. A chunk buffer exists DURING decode, alongside the arrays,
  // so checking it independently lets a store sit just under budget on both terms
  // and peak near 2x — directly constructible, since zarr v2's `chunks > shape`
  // allowance means a tiny array can declare a near-budget chunk. Folding is what
  // makes this constant's docstring true.
  let maxChunkBytes = 0;
  for (const [slot, array] of Object.entries(arrays)) {
    if (!array) continue;
    const name = MESH_ARRAY_NAMES[slot as keyof MeshArrayHandles] ?? slot;
    const parsed = parseDtype(String(array.dtype));
    if (!parsed) {
      rejectMesh(path, `${name} has an unrecognised dtype '${String(array.dtype)}'`);
    }

    const stored = elementCount(array.shape);
    if (stored === null) {
      rejectMesh(path, `${name} declares an unusable shape [${String(array.shape)}]`);
    }
    const layout = logicalLayout(array);
    if (layout === null) {
      rejectMesh(path, `${name} declares an unusable logical shape`);
    }
    accountedBytes += stored * parsed.itemSize + layout.count * DECODED_BYTES_PER_VALUE;

    // Per-chunk term. Not redundant with the sum: zarr v2 does not require
    // `chunks <= shape`, so a `"shape": [100, 3]` array declaring
    // `"chunks": [268435456, 3]` slips a ~3 GB first-chunk allocation past a
    // shape-only budget. Zarr allocates chunk-shaped buffers, and edge chunks
    // are padded to the full chunk shape, so the declared chunk shape IS the
    // allocation.
    const perChunk = elementCount(array.chunks);
    if (perChunk === null) {
      rejectMesh(path, `${name} declares an unusable chunk shape [${String(array.chunks)}]`);
    }
    const chunkBytes = perChunk * parsed.itemSize;
    if (chunkBytes > maxChunkBytes) maxChunkBytes = chunkBytes;
    if (chunkBytes > MESH_DECODE_BUDGET_BYTES) {
      rejectMesh(
        path,
        `${name} declares a single chunk of ${mib(chunkBytes)} (chunks ` +
          `[${String(array.chunks)}], dtype ${String(array.dtype)}), over the ` +
          `${mib(MESH_DECODE_BUDGET_BYTES)} per-node budget. One chunk is one ` +
          "allocation, so this is refused regardless of the array's total size."
      );
    }
  }
  const peakBytes = accountedBytes + maxChunkBytes;
  if (peakBytes > MESH_DECODE_BUDGET_BYTES) {
    rejectMesh(
      path,
      `this node's arrays account for ${mib(peakBytes)} (stored bytes, what they ` +
        'decode to, and the largest single chunk buffer), over the ' +
        `${mib(MESH_DECODE_BUDGET_BYTES)} per-node budget. A mesh is loaded whole, so ` +
        'this is what the load would allocate. Decimate the mesh or split it across ' +
        'nodes.'
    );
  }

  // --- (c) shape and dtype cross-checks ------------------------------------
  //
  // Against the LOGICAL layout, not the stored shape: a broadcast uniform colour
  // stores one row, and a LUT-encoded normals array stores codes. Checking the
  // stored shape rejects both, and both are stores this writer really produces.
  const checkLayout = (
    slot: keyof MeshArrayHandles,
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    rows: number,
    components: readonly number[],
    extra = ''
  ): LogicalLayout => {
    const layout = logicalLayout(array);
    if (layout === null) {
      rejectMesh(path, `${MESH_ARRAY_NAMES[slot]} declares an unusable logical shape`);
    }
    if (layout.rows !== rows || !components.includes(layout.components)) {
      rejectMesh(
        path,
        `${MESH_ARRAY_NAMES[slot]} describes ${layout.rows} x ${layout.components} ` +
          `values but must be ${rows} x ${components.join(' or ')}.` +
          (extra ? ` ${extra}` : '')
      );
    }
    return layout;
  };

  checkLayout(
    'vertices',
    arrays.vertices,
    nVertices,
    [ndim],
    'A width that disagrees with ndim would index out of slice bounds in the nD slab kernel.'
  );
  checkLayout('faces', arrays.faces, nFaces, [3]);

  const fDtype = parseDtype(String(arrays.faces.dtype));
  if (!fDtype?.integer) {
    rejectMesh(
      path,
      `faces has dtype '${String(arrays.faces.dtype)}'; an integer dtype is ` +
        'required. A float index truncates in the uint32 coercion, silently ' +
        'rewriting the topology.'
    );
  }

  // Presence flags must be backed by an array that is actually there. The
  // geometry builder reads these flags to decide whether to bind `normal` /
  // `aScalar` and to enable the colormap path, so a flag with nothing behind it
  // produces a draw referencing a buffer that was never uploaded.
  //
  // Only this direction is checked. The converse — an array the flags disown — is
  // NOT reachable through the loader, which opens an optional array only when its
  // flag is set, so asserting it here would be testing a state production cannot
  // construct. The label CSR arrays are checked as a PAIR for the same reason the
  // others are checked at all: v1 never fetches them, but a `has_labels` with one
  // array missing is a store that will fail confusingly the moment picking lands.
  for (const [flagName, flag, required] of [
    ['has_normals', attrs.has_normals, [arrays.normals] as const],
    ['has_colors', attrs.has_colors, [arrays.colors] as const],
    ['has_scalars', attrs.has_scalars, [arrays.scalars] as const],
    ['has_labels', attrs.has_labels, [arrays.labelOffsets, arrays.labelBytes] as const],
    [
      'has_image_labels',
      attrs.has_image_labels,
      [arrays.imageLabelOffsets, arrays.imageLabelBytes] as const,
    ],
  ] as const) {
    if (flag && required.some((a) => !a)) {
      rejectMesh(path, `${flagName} is set but its array(s) are missing from the store.`);
    }
  }

  // Optional per-vertex arrays. Each binds as an enabled vertex attribute on an
  // indexed draw, so an undersized one over-reads rather than failing loudly.
  if (arrays.normals) checkLayout('normals', arrays.normals, nVertices, [3]);
  let colorComponents: 3 | 4 | undefined;
  if (arrays.colors) {
    colorComponents = checkLayout('colors', arrays.colors, nVertices, [3, 4]).components as 3 | 4;
  }
  if (arrays.scalars) checkLayout('scalars', arrays.scalars, nVertices, [1]);

  // --- (d) normal_dims well-formedness ------------------------------------
  let normalDims: number[] | undefined;
  if (attrs.has_normals) {
    const dims = attrs.normal_dims;
    if (!Array.isArray(dims) || dims.length !== 3) {
      rejectMesh(
        path,
        `has_normals is set but normal_dims is ${JSON.stringify(dims)}; exactly 3 ` +
          'dimension indices are required. Stored normals with no frame cannot be ' +
          'oriented — for a (t, x, y, z) mesh the "first three dimensions" are ' +
          '(t, x, y).'
      );
    }
    for (const d of dims) {
      if (!Number.isInteger(d) || d < 0 || d >= ndim) {
        rejectMesh(
          path,
          `normal_dims ${JSON.stringify(dims)} has an entry outside [0, ndim=${ndim}).`
        );
      }
    }
    if (new Set(dims).size !== 3) {
      rejectMesh(path, `normal_dims ${JSON.stringify(dims)} must name three DISTINCT dimensions.`);
    }
    normalDims = [...dims];
  }

  return { nVertices, nFaces, ndim, colorComponents, normalDims };
}
