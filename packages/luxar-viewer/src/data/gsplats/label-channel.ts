/**
 * Exact GSplat categorical-label decoding and projection helpers.
 *
 * Exact unsigned ids remain decimal strings so `uint64` values survive JavaScript
 * number precision. GPU-facing indices are compact and 1-based; 0 means unlabelled.
 *
 * @module data/gsplats/label-channel
 */

/** One exact categorical label entry. */
export interface GSplatLabelEntry {
  id: string;
  name: string;
}

/** Compact per-splat indices paired with their exact vocabulary. */
export interface GSplatLabelChannel {
  indices: Uint32Array;
  vocabulary: readonly GSplatLabelEntry[];
}

function exactIdAt(
  values: Uint8Array | Uint16Array | Uint32Array | BigUint64Array,
  index: number
): string {
  const value = values[index];
  return typeof value === 'bigint' ? value.toString() : String(value);
}

/** Compact exact unsigned ids into stable 1-based GPU indices. */
export function compactGSplatLabelIds(
  values: Uint8Array | Uint16Array | Uint32Array | BigUint64Array,
  vocabulary: Record<string, string>,
  target?: Uint32Array
): GSplatLabelChannel {
  const entries = Object.entries(vocabulary).map(([id, name]) => ({ id, name }));
  const compactById = new Map(entries.map((entry, index) => [entry.id, index + 1]));
  if (target && target.length !== values.length) {
    throw new Error(
      `GSplats label target length ${target.length} does not match value length ${values.length}`
    );
  }
  const indices = target ?? new Uint32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    const id = exactIdAt(values, index);
    const compact = compactById.get(id);
    if (compact === undefined) {
      throw new Error(`GSplats label id ${id} is missing from label_vocabulary`);
    }
    indices[index] = compact;
  }
  return { indices, vocabulary: entries };
}

/** Project compact label indices through the same visible-splat mapping as geometry. */
export function projectGSplatLabelIndices(
  indices: Uint32Array,
  sourceIndices: Uint32Array | undefined,
  visibleCount: number
): Uint32Array {
  if (!sourceIndices) return indices.slice(0, visibleCount);
  const projected = new Uint32Array(visibleCount);
  for (let index = 0; index < visibleCount; index++) {
    const sourceIndex = sourceIndices[index];
    if (sourceIndex === undefined || sourceIndex >= indices.length) {
      throw new Error(
        `GSplats label source index ${String(sourceIndex)} is outside ${indices.length} label indices`
      );
    }
    projected[index] = indices[sourceIndex];
  }
  return projected;
}

/** Resolve one storage slot back to its exact id and display name. */
export function gsplatLabelAt(
  channel: Pick<GSplatLabelChannel, 'indices' | 'vocabulary'>,
  storageIndex: number
): GSplatLabelEntry | null {
  const compact = channel.indices[storageIndex] ?? 0;
  return compact === 0 ? null : (channel.vocabulary[compact - 1] ?? null);
}
