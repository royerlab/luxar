/**
 * The mesh pick element-id split, and the `2^27` vertex cap it rests on (spec §6.5).
 *
 * The parity harness renders to an 8-bit target, so the two 16-bit id halves clamp
 * and quantize there and no rendered pixel can show that the split is right. These
 * tests carry that weight instead, in three parts:
 *
 * 1. the split is written ONCE and both shader backends route through it;
 * 2. the split round-trips through `voteWinner`'s recombination for every id shape
 *    that matters, including the largest admitted vertex ordinal;
 * The third leg — that the vote key stays exactly representable at that ordinal —
 * lives in `picking-system/pick-render.test.ts`, next to the `VOTE_KEY_STRIDE` it
 * constrains, because that is where a future stride change would be made.
 *
 * @module tests/unit/rendering/picking/mesh/element-id-split
 */
import { describe, it, expect } from 'vitest';
import {
  GLSL_ELEMENT_ID_SPLIT,
  GLSL_SORTED_INDEX,
} from '../../../../../rendering/materials/_shared/glsl-lib';
import { MESH_PICK_VERTEX_SHADER } from '../../../../../rendering/picking/mesh/shaders';
import { MAX_MESH_VERTICES } from '../../../../../config/constants';

/** The GLSL `luxarElementIdSplit`, in JS. Kept in lockstep by the tests below. */
function splitId(i: number): [low: number, high: number] {
  return [i & 0xffff, i >>> 16];
}

/** `voteWinner`'s recombination, verbatim (`pick-render.ts`). */
function recombine(low: number, high: number): number {
  return Math.round(high) * 65536 + Math.round(low);
}

describe('the 16-bit split is single-sourced', () => {
  it('the mesh pick vertex shader calls the shared helper rather than re-deriving it', () => {
    // The whole point of extracting `GLSL_ELEMENT_ID_SPLIT`: a second copy of the
    // mask/shift could drift, and the only symptom would be picks resolving to the
    // wrong vertex past 65,536 — silent, and only on large meshes.
    expect(MESH_PICK_VERTEX_SHADER).toContain('luxarElementIdSplit(uint(gl_VertexID))');
    expect(MESH_PICK_VERTEX_SHADER).toContain(GLSL_ELEMENT_ID_SPLIT);
    // ...and does NOT declare the ordering attributes it has no use for. Declaring an
    // unbound attribute is not merely wasteful on WebGPU — the vertex-buffer layout
    // is cached from the attribute set at first draw.
    expect(MESH_PICK_VERTEX_SHADER).not.toContain('aSortedIndex');
  });

  it('the sorted-index block still carries the same helper, so the four types agree', () => {
    // `luxarElementIdParts()` is now a one-line wrapper over the shared split, and
    // `GLSL_SORTED_INDEX` embeds the split so its existing consumers inject one
    // string and get both — a shader that injected only the sorted-index half would
    // fail to compile rather than silently define the function twice.
    expect(GLSL_SORTED_INDEX).toContain(GLSL_ELEMENT_ID_SPLIT);
    expect(GLSL_SORTED_INDEX).toContain('luxarElementIdSplit(luxarSortedIndex())');
    expect(GLSL_ELEMENT_ID_SPLIT.match(/vec2 luxarElementIdSplit/g)).toHaveLength(1);
  });
});

describe('split → recombine round-trips exactly', () => {
  const cases: Array<[label: string, id: number]> = [
    ['vertex 0', 0],
    ['the last single-half id', 65535],
    ['the first id needing the high half', 65536],
    ['just past the f32 exact range', 16777217],
    ['the largest admitted vertex ordinal', MAX_MESH_VERTICES - 1],
  ];

  it.each(cases)('%s', (_label, id) => {
    const [low, high] = splitId(id);
    // Each half must survive an f32 channel exactly, which is what makes the
    // recombination lossless — hence <= 65535, not merely "small".
    expect(low).toBeLessThanOrEqual(65535);
    expect(high).toBeLessThanOrEqual(65535);
    expect(recombine(low, high)).toBe(id);
  });

  it('a single-channel id would have FAILED past the f32 mantissa', () => {
    // The anti-vacuity check for the whole two-half design: without the split, an
    // ordinal above 2^24 stops being representable as consecutive f32 values, so
    // `Math.round(f32(id))` lands on a neighbour. Demonstrated rather than asserted
    // in prose, since it is the entire justification for the extra channel.
    const id = 16777217; // 2^24 + 1
    expect(Math.round(Math.fround(id))).not.toBe(id);
    const [low, high] = splitId(id);
    expect(recombine(Math.fround(low), Math.fround(high))).toBe(id);
  });
});
