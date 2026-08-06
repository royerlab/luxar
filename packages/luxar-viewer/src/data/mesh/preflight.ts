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
 * Stage 2 (`validate.ts`).
 *
 * ## What is checked, and what each check prevents
 *
 * - **`n_vertices <= MAX_MESH_VERTICES`** — above `2^27` the pick vote key
 *   aliases across nodes (§6.5). Refused before allocation, not after a
 *   multi-gigabyte fetch.
 * - **Byte budget** — stored bytes, decoded bytes and the largest single chunk
 *   allocation, SUMMED against `MESH_DECODE_BUDGET_BYTES`. Charged against the array
 *   whose bytes are actually fetched, which for an `array_ref` is the TARGET, not the
 *   `(0, k)` stub that points at it.
 * - **Shape and dtype cross-checks** — against `n_vertices`/`n_faces`/`ndim`
 *   and the §3.2 array table.
 * - **`normal_dims` well-formedness** — 3 distinct integers in `[0, ndim)`.
 *
 * ### Which gate actually binds
 *
 * At the default 512 MiB budget the **byte budget is far tighter than the vertex
 * cap**, and it is worth knowing which error to expect. A 3D float32 mesh runs
 * out of budget at ~22.4M vertices and a uint16-quantized one at ~29.8M, both
 * well under `2^27` (134.2M) — so in practice no mesh reaches the cap by growing
 * legitimately. (Those figures halve the stored-only arithmetic because the decoded
 * term is charged too: a float32 vertex costs 4 bytes stored AND 4 decoded. The
 * float32 number is measured, not derived — see the boundary test.)
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
 * @module data/mesh/preflight
 */

import * as zarr from '../zarr';
import { MAX_MESH_VERTICES, MESH_DECODE_BUDGET_BYTES } from '../../config/constants';
import { ArrayDecoder, type ArrayMetadata } from '../array-decoder/decoder';
import { LoaderError, classifyLoaderError } from '../scene-loader/nodes/load-leaf-error-dispatch';
import type { MeshMetadata } from '../../types/mesh';
import type { EncodingName } from '../../types/format-contract';

/**
 * Metadata-only handles for the arrays a mesh node declares.
 *
 * Every entry is a zarr `Array` opened with `kind: 'array'`, which fetches
 * `.zarray` + `.zattrs` and nothing else. Optional entries are `undefined` when
 * the node's presence flags say the array is absent.
 *
 * The label/image-label CSR arrays are included even though *this* loader never
 * fetches them. Picking has landed (`rendering/picking/mesh/`) and returns the vertex
 * ordinal that indexes this CSR, and the read itself lives outside this package — the
 * shared lazy `data/loaders/picking/label-loader.ts` the Points/Lines/GSplats hover path
 * already uses, which resolves a label only once something is hovered. The arrays are
 * part of the node's declared footprint regardless, so budgeting them here keeps the
 * ceiling over the whole node rather than over the subset this loader happens to pull.
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

/**
 * What Stage 1 established.
 *
 * The first three fields are the ones Stage 2 reads: the loader destructures
 * `{ nVertices, nFaces, ndim }` and every `validateMaterializedLength` call
 * derives its expected length from them.
 *
 * The last two are deliberately NOT a source of truth, and it matters that this
 * is stated, because the codebase's own rule is that a field no consumer reads
 * is indistinguishable from a field that is wrong (see
 * `scene-loader/geometry-descriptors.ts`). Both are byproducts of Stage 1
 * VALIDATION — the checks are the point, the values are how a test can observe
 * that the checks ran. Their real consumers read elsewhere on purpose:
 *
 * - `colorComponents` — the loader calls the shared `colorComponentsOf`, the same
 *   helper the three sibling loaders use, rather than threading this through. The
 *   two agree because Stage 1 has already rejected anything but 3 or 4, which is
 *   the strictness `colorComponentsOf` (non-4 ⇒ 3) does not have on its own.
 * - `normalDims` — the winding frame reaches `projectMeshTo3D` from the node's
 *   `attrs.normal_dims` via the handler, since the projection runs per slice move
 *   and does not hold a preflight.
 *
 * So: read these two in tests, not in production code.
 */
export interface MeshPreflightResult {
  /** Vertex count, `<= MAX_MESH_VERTICES` */
  nVertices: number;
  /** Triangle count, `>= 1` */
  nFaces: number;
  /** Coordinate dimensionality; equals the `vertices` declared width */
  ndim: number;
  /** Channels per color entry, when `colors` is present. Observation only — see above. */
  colorComponents?: 3 | 4;
  /** The validated `normal_dims`, when `normals` is present. Observation only — see above. */
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

/**
 * What "shape check" actually means here, stated precisely because it is more lenient
 * than the spec's `(V, D)` wording suggests.
 *
 * A shape is reduced to (rows, components) = (product / trailing extent, trailing
 * extent), so `[2, 2, 3]` and `[4, 1, 1, 1, 3]` are both accepted for a 4-vertex 3D
 * mesh. That is deliberate rather than sloppy: the decoder returns a FLAT typed array,
 * and C-order flattening makes those declarations byte-identical to `[4, 3]` — verified
 * end to end, same buffer and same `vertexCount`. Insisting on exactly 2-D would refuse
 * stores that are equivalent in every way the loader can observe.
 *
 * What it still refuses is anything whose flattened size or trailing extent disagrees
 * with `n_vertices`/`ndim`, which is the part that would mis-stride the slab kernel.
 */

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
  //
  // Gated on the encoding NAME as well as the count, which matters in two ways. It
  // keeps a hostile store from claiming a `(1, d)` shape covers V vertices by
  // stamping `n_elements` alone (Stage 2's length check would catch that anyway, but
  // one stage later and with a message that names the wrong thing). And it keeps this
  // branch from hijacking an array that legitimately carries BOTH `n_elements` and
  // `original_shape` — only the two broadcast encoders stamp `n_elements` today, but
  // priority order should not depend on that staying true.
  const n = encoding?.n_elements;
  if (encoding?.name === 'broadcasted' && typeof n === 'number' && Number.isInteger(n) && n >= 0) {
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

/**
 * Encoding names the budget knows how to charge, keyed so a NEW contract encoding is a
 * compile error here rather than a silent bypass.
 *
 * This exists because the budget was bypassed four times by one category of mistake:
 * *the bytes the loader fetches are not the bytes this handle declares.* `broadcasted`
 * expands a `(1, k)` stub to `n_elements` rows; a narrow dtype widens 4x on decode;
 * an unbounded `ndim` inflates the logical count; and `array_ref` reads a different
 * array entirely. Each was fixed as an instance, and a fourth still appeared.
 *
 * The root cause was that an unrecognised encoding fell through to "use the stored
 * shape" — fail-OPEN. So the set is now closed and explicit:
 *
 * - `handledByLayout` — the logical size is recoverable from the encoding's own attrs
 *   (`n_elements` / `original_shape`) or from the stored shape, and the fetched bytes
 *   are this array's own.
 * - `followsRef` — the fetched bytes belong to another array; the chain is walked.
 *
 * Anything absent is REFUSED. A future encoder that changes the stored-vs-logical
 * relationship therefore fails the node loudly instead of slipping past the ceiling,
 * which is the whole point of having a ceiling.
 */
const ENCODING_BUDGET_KIND: Record<EncodingName, 'handledByLayout' | 'followsRef'> = {
  none: 'handledByLayout',
  broadcasted: 'handledByLayout',
  array_ref: 'followsRef',
  lut_uint8: 'handledByLayout',
  lut_uint16: 'handledByLayout',
  rgb_uint8: 'handledByLayout',
  rgb_uint16: 'handledByLayout',
  bounded_scalar_uint8: 'handledByLayout',
  bounded_scalar_uint16: 'handledByLayout',
  geolog_scalar_uint8: 'handledByLayout',
  geolog_scalar_uint16: 'handledByLayout',
  log_scalar_uint8: 'handledByLayout',
  log_scalar_uint16: 'handledByLayout',
  linear_perchannel_u8: 'handledByLayout',
  linear_perchannel_u16: 'handledByLayout',
  log_perchannel_u8: 'handledByLayout',
  log_perchannel_u16: 'handledByLayout',
  signed_log_perchannel_u8: 'handledByLayout',
  signed_log_perchannel_u16: 'handledByLayout',
  geolog_perchannel_u8: 'handledByLayout',
  geolog_perchannel_u16: 'handledByLayout',
  float16: 'handledByLayout',
  float32: 'handledByLayout',
  uint8: 'handledByLayout',
  uint16: 'handledByLayout',
  uint32: 'handledByLayout',
  // The INDEX encoder emits this for faces when an index needs 64 bits.
  uint64: 'handledByLayout',
};

/**
 * Refuse an encoding the budget cannot account for.
 *
 * Fail-closed on purpose — see {@link ENCODING_BUDGET_KIND}. An absent `encoding` attr
 * is fine (an unencoded array), but a NAMED encoding outside the contract's vocabulary
 * means this store was written by something whose stored-vs-logical relationship we
 * have not reasoned about.
 */
function assertBudgetableEncoding(
  path: string,
  name: string,
  array: zarr.Array<zarr.DataType, zarr.Readable>
): void {
  const encodingName = ((array.attrs ?? {}) as unknown as ArrayMetadata).encoding?.name;
  if (encodingName === undefined) return;
  if (!Object.hasOwn(ENCODING_BUDGET_KIND, encodingName)) {
    rejectMesh(
      path,
      `${name} declares encoding '${encodingName}', which the byte budget cannot ` +
        'account for. Refused rather than admitted, because an encoding whose ' +
        'stored-to-decoded relationship is unknown can bypass the per-node ceiling.'
    );
  }
}

/**
 * Refuse an `array_ref` whose resolved target is `broadcasted`.
 *
 * The budget charges the target's STORED and per-chunk terms plus the referring
 * stub's LOGICAL count. That accounting is only sound when the target decodes to
 * roughly its stored size. A `broadcasted` target breaks it: it physically stores
 * one `(1, K)` row — a few bytes of stored/chunk term — but `ArrayDecoder` expands
 * it to one row PER logical element at fetch time, so a wide row behind a small
 * stub sails under the ceiling and then allocates gigabytes. That is exactly the
 * "too late after decode" exhaustion the preflight exists to prevent.
 *
 * Refusing it costs nothing legitimate: the encoder broadcast-encodes a uniform
 * array at priority 1, BEFORE dedup is consulted, so the writer never emits an
 * `array_ref` that points at a `broadcasted` target.
 */
function assertRefTargetDecodesInPlace(
  path: string,
  name: string,
  fetched: zarr.Array<zarr.DataType, zarr.Readable>
): void {
  const encodingName = ((fetched.attrs ?? {}) as unknown as ArrayMetadata).encoding?.name;
  if (encodingName === 'broadcasted') {
    rejectMesh(
      path,
      `${name} is an array_ref to a '${encodingName}' target. A broadcast target ` +
        'stores one row but decodes to one row per logical element, so its stored ' +
        'and chunk sizes say nothing about what the fetch would allocate — the ' +
        'per-node budget cannot bound it. Refused rather than admitted blind.'
    );
  }
}

/**
 * The number of float32 values the decoder will materialize from a RESOLVED
 * `array_ref` target, upper-bounded from metadata alone.
 *
 * Every decoder path allocates one output value per STORED value —
 * `Float32Array(stored_count)` — with a single exception: a row-mode LUT expands
 * each stored index into `k` values, `k` taken from the target's `original_shape`
 * (see `ArrayDecoder.decodeLUT`, which allocates `n x k`). So the bound is
 * `stored_count` for every encoding but a row-mode LUT, and `stored_count x k`
 * for that one. `broadcasted` is refused before this is reached — its size is
 * driven by the caller's `expectedElements`, not the target's own metadata, so it
 * cannot be bounded here at all.
 *
 * This deliberately multiplies the target's STORED count by `k` rather than reading
 * `original_shape[0]`: `n` is the target's real stored index count, and a hostile
 * store can declare an `original_shape` whose leading extent disagrees with it —
 * charging the declared rows would then under-count the `n x k` the decoder
 * actually allocates.
 */
function targetDecodedFloatCount(fetched: zarr.Array<zarr.DataType, zarr.Readable>): number | null {
  const storedCount = elementCount(fetched.shape);
  if (storedCount === null) return null;
  const encoding = ((fetched.attrs ?? {}) as unknown as ArrayMetadata).encoding;
  const name = encoding?.name;
  const isLut = name === 'lut_uint8' || name === 'lut_uint16';
  // Row mode is the default when `lut_mode` is absent (the Python encoder omits it
  // for 1-D arrays); only 'scalar' is 1:1. Mirrors `ArrayDecoder`'s k/mode logic.
  if (isLut && encoding?.lut_mode !== 'scalar') {
    const original = encoding?.original_shape;
    const k = Array.isArray(original) && original.length > 1 ? original[1] : 1;
    if (!Number.isInteger(k) || k < 1) return null;
    const total = storedCount * k;
    return Number.isFinite(total) ? total : null;
  }
  return storedCount;
}

/**
 * Whether this array's bytes live in ANOTHER array, so the budget must follow a
 * reference to find them.
 *
 * Reads {@link ENCODING_BUDGET_KIND} rather than testing for `'array_ref'` directly,
 * and that indirection is the point: with a literal test the table's *values* were
 * decorative — only its key set was consulted — so a future ref-following encoding
 * added as `followsRef` would have been silently treated as `handledByLayout` and
 * walked right past the ceiling. That is the same "two places encode one fact" shape
 * as the four bypasses the table exists to prevent, so the table is now the single
 * source of truth for both questions it answers.
 */
function followsReference(array: zarr.Array<zarr.DataType, zarr.Readable>): boolean {
  const encodingName = ((array.attrs ?? {}) as unknown as ArrayMetadata).encoding?.name;
  if (encodingName === undefined) return false;
  return (
    Object.hasOwn(ENCODING_BUDGET_KIND, encodingName) &&
    ENCODING_BUDGET_KIND[encodingName as EncodingName] === 'followsRef'
  );
}

/**
 * Maximum `array_ref` hops the preflight will follow before refusing.
 *
 * A target may itself be encoded — including as another `array_ref` — and
 * `ArrayDecoder.decodeArrayRef` recurses without a depth limit of its own. Three
 * hops is far more than the writer ever produces (it emits at most one level of
 * indirection), so this exists to bound a hostile or cyclic store rather than to
 * accommodate a real one.
 */
const MAX_ARRAY_REF_HOPS = 3;

/**
 * Follow an `array_ref` chain, metadata-only, and return the array whose bytes are
 * actually fetched.
 *
 * The referring array is a STUB — the Python encoder writes it at `(0, k)` and puts
 * the real shape in `encoding.original_shape` — so budgeting the handle the loader
 * opened charges ~48 bytes and says nothing about what gets read. `ArrayDecoder`
 * resolves `encoding.target` against the store root and reads that array in full, so
 * the budget has to follow the same path. It only *warns* on a length mismatch; the
 * throw comes from Stage 2, i.e. after the allocation the budget exists to prevent.
 *
 * Metadata-only throughout: `zarr.open(..., { kind: 'array' })` reads `.zarray` and
 * `.zattrs` and no chunk. Cycles are bounded by both the hop limit and a seen-set, so
 * a store whose target points back at itself is refused rather than hanging.
 */
async function resolveRefTarget(
  path: string,
  name: string,
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  storeRoot: zarr.Location<zarr.Readable> | undefined
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const seen = new Set<string>();
  let current = array;
  for (let hop = 0; hop <= MAX_ARRAY_REF_HOPS; hop++) {
    if (!followsReference(current)) return current;
    const encoding = ((current.attrs ?? {}) as unknown as ArrayMetadata).encoding;

    const target = encoding?.target;
    if (typeof target !== 'string' || target.length === 0) {
      rejectMesh(path, `${name} declares an array_ref with no target path.`);
    }
    if (!storeRoot) {
      rejectMesh(
        path,
        `${name} is an array_ref to '${target}', but no store root is available to ` +
          'resolve it. The budget cannot bound what would be fetched, so the node is ' +
          'refused rather than admitted blind.'
      );
    }
    if (seen.has(target)) {
      rejectMesh(path, `${name} array_ref chain revisits '${target}' — the store is cyclic.`);
    }
    seen.add(target);
    if (hop === MAX_ARRAY_REF_HOPS) {
      rejectMesh(
        path,
        `${name} array_ref chain exceeds ${MAX_ARRAY_REF_HOPS} hops (via '${target}').`
      );
    }
    try {
      current = await zarr.open(storeRoot.resolve(target), { kind: 'array' });
    } catch (error) {
      // A genuinely absent/misdeclared target is a deterministic `Validation`
      // rejection. A transient failure opening its metadata (network, abort) is
      // NOT — persisting it as `Validation` would have the retry policy treat a
      // flaky network as permanent and never re-fetch on reconnect. Propagate the
      // real kind, matching the required-array open in the loader.
      if (!zarr.isNotFoundError(error)) {
        throw new LoaderError(classifyLoaderError(error), path, error);
      }
      rejectMesh(path, `${name} is an array_ref to '${target}', which was not found in the store.`);
    }
  }
  return current;
}

/**
 * Dtypes an UNENCODED `colors` array may use — §3.2's `uint8/uint16/float32`,
 * in both spellings (zarrita reports friendly names, a raw `.zarray` carries
 * numpy typestrings).
 *
 * Colours are the one mesh array whose dtype is MEANING-BEARING at the GPU: the
 * shared colour path preserves the native type on its direct branch, and the
 * renderer's normalization is defined per dtype (uint8 as 0–255, uint16 as
 * 0–65535, float32 read as-is in [0, 1]). An unencoded `int32` or `int64`
 * colours array has no defined normalization — it widens to a float buffer
 * whose 0–255-ish values all clamp ≥ 1.0 and shade the mesh flat white, with
 * nothing to say why. Every other array is decoder-routed to value-preserving
 * `Float32Array`, where any recognised dtype is fine — which is why this check
 * exists for colours alone.
 *
 * ENCODED colours are exempt: their stored dtype holds codes (LUT indices,
 * quantized levels), and the decode path materializes a defined buffer
 * regardless — the closed {@link ENCODING_BUDGET_KIND} vocabulary already
 * bounds what "encoded" can mean.
 */
const UNENCODED_COLOR_DTYPES = new Set([
  'uint8',
  '|u1',
  '<u1',
  '>u1',
  '=u1',
  'uint16',
  '|u2',
  '<u2',
  '>u2',
  '=u2',
  'float32',
  '<f4',
  '>f4',
  '=f4',
]);

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
export async function preflightMesh(
  path: string,
  attrs: MeshMetadata,
  arrays: MeshArrayHandles,
  storeRoot?: zarr.Location<zarr.Readable>
): Promise<MeshPreflightResult> {
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
  // A floor of 2, mirroring `add_mesh`'s vertex check rather than the generic
  // "positive integer" the other counts get. A triangle needs two dimensions to enclose
  // area; in 1D every face is collinear, so the node would load cleanly and draw nothing,
  // with no diagnostic. This is deliberately stricter than Points/Lines, whose primitives
  // ARE meaningful in 1D.
  if (!Number.isInteger(ndim) || ndim < 2) {
    rejectMesh(
      path,
      `ndim must be an integer of at least 2, got ${String(ndim)}. A triangle needs two ` +
        'dimensions to have any area — in 1D every face is collinear and the surface ' +
        'renders nothing.'
    );
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

    // The array whose BYTES are fetched, which is not this handle when the encoding
    // is an `array_ref`: the referring array is a `(0, k)` stub, so charging it
    // budgets ~48 bytes for a read that can pull gigabytes. Metadata-only.
    assertBudgetableEncoding(path, name, array);
    const fetched = await resolveRefTarget(path, name, array, storeRoot);
    // The ENDPOINT is the only hop these two gates can fire on, and that is worth
    // stating rather than defending with a loop over every step:
    //
    //  - `assertRefTargetDecodesInPlace` refuses a `broadcasted` target. `broadcasted`
    //    is not a `followsRef` encoding, so `resolveRefTarget` STOPS the moment it
    //    reaches one — a broadcast array is therefore always `fetched`, never an
    //    interior hop.
    //  - `assertBudgetableEncoding` refuses an unaccountable encoding. An interior hop
    //    is by construction one the resolve loop chose to follow, i.e. `array_ref`,
    //    which the table lists as `followsRef` and is accountable by definition.
    //
    // A chain-wide loop here therefore cannot reject anything this does not; verified by
    // mutation in both directions. The walk itself still earns its keep through the cycle
    // and hop-cap refusals inside `resolveRefTarget`.
    //
    // `fetched !== array` is what keeps a legitimately broadcast NODE-LOCAL colour array
    // out of this refusal: `colors=(1, 3)` makes the colours array itself `broadcasted`
    // with no ref involved (#1238), and refusing that would reject real writer output.
    if (fetched !== array) {
      assertBudgetableEncoding(path, name, fetched);
      assertRefTargetDecodesInPlace(path, name, fetched);
    }

    const parsed = parseDtype(String(fetched.dtype));
    if (!parsed) {
      rejectMesh(path, `${name} has an unrecognised dtype '${String(fetched.dtype)}'`);
    }

    const stored = elementCount(fetched.shape);
    if (stored === null) {
      rejectMesh(path, `${name} declares an unusable shape [${String(fetched.shape)}]`);
    }
    // The referring array's LOGICAL layout is what this NODE means by the values:
    // `original_shape` records that, and the cross-checks below compare against it.
    const layout = logicalLayout(array);
    if (layout === null) {
      rejectMesh(path, `${name} declares an unusable logical shape`);
    }
    // But it is NOT necessarily what gets allocated, and the ENDPOINT is where that
    // difference lives. `ArrayDecoder.decodeArrayRef` recurses with the TARGET's own
    // attrs, so the target's encoding expands to the target's numbers however modest the
    // stub's `original_shape` is — a row-mode LUT allocates `stored_indices x k`, with
    // `k` from the TARGET's `original_shape`. Charge the larger of the stub's own
    // requirement and the target's true decode size.
    //
    // Interior hops are deliberately NOT charged, and this is the correction worth
    // recording: `decodeArrayRef` passes `expectedElements` through UNCHANGED and reads
    // nothing from an interior stub but `encoding.target`. So an interior stub's
    // `original_shape` drives no allocation whatsoever. Summing over the chain would
    // charge bytes that are never allocated, which does not merely waste budget — it can
    // FALSELY REJECT a legitimate node whose interior stub describes a large logical
    // view. The chain is still walked, for the cycle and hop-cap refusals below.
    let decodedValues = layout.count;
    if (fetched !== array) {
      const targetDecoded = targetDecodedFloatCount(fetched);
      if (targetDecoded === null) {
        rejectMesh(path, `${name}'s array_ref target declares an unusable shape`);
      }
      decodedValues = Math.max(decodedValues, targetDecoded);
    }
    accountedBytes += stored * parsed.itemSize + decodedValues * DECODED_BYTES_PER_VALUE;

    // Per-chunk term. Not redundant with the sum: zarr v2 does not require
    // `chunks <= shape`, so a `"shape": [100, 3]` array declaring
    // `"chunks": [268435456, 3]` slips a ~3 GB first-chunk allocation past a
    // shape-only budget. Zarr allocates chunk-shaped buffers, and edge chunks
    // are padded to the full chunk shape, so the declared chunk shape IS the
    // allocation.
    const perChunk = elementCount(fetched.chunks);
    if (perChunk === null) {
      rejectMesh(path, `${name} declares an unusable chunk shape [${String(fetched.chunks)}]`);
    }
    const chunkBytes = perChunk * parsed.itemSize;
    if (chunkBytes > maxChunkBytes) maxChunkBytes = chunkBytes;
    if (chunkBytes > MESH_DECODE_BUDGET_BYTES) {
      rejectMesh(
        path,
        `${name} declares a single chunk of ${mib(chunkBytes)} (chunks ` +
          `[${String(fetched.chunks)}], dtype ${String(fetched.dtype)}), over the ` +
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
  // others are checked at all: this loader never fetches them, but the hover path's
  // shared label loader does, and a `has_labels` with one array missing is a store
  // that fails confusingly there rather than here.
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
    // Same predicate the colour loader's direct branch uses, so this gate covers
    // exactly the loads whose native dtype reaches the GPU (see
    // UNENCODED_COLOR_DTYPES for why colours alone need it).
    const colorAttrs = (arrays.colors.attrs ?? {}) as unknown as ArrayMetadata;
    const colorDtype = String(arrays.colors.dtype);
    if (!ArrayDecoder.isEncoded(colorAttrs) && !UNENCODED_COLOR_DTYPES.has(colorDtype)) {
      rejectMesh(
        path,
        `colors has dtype '${colorDtype}'; an unencoded colors array must be uint8, ` +
          'uint16 or float32 (§3.2) — the dtypes with a defined colour ' +
          'normalization. Anything else mis-shades every vertex with nothing to ' +
          'say why.'
      );
    }
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
