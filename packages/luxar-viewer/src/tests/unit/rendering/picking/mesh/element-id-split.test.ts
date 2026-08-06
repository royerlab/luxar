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
import {
  MESH_PICK_VERTEX_SHADER,
  MESH_PICK_FRAGMENT_SHADER,
} from '../../../../../rendering/picking/mesh/shaders';
import { MAX_MESH_VERTICES } from '../../../../../config/constants';

/**
 * The GLSL `luxarElementIdSplit`, in JS — but DERIVED from the shader source rather
 * than hand-copied.
 *
 * A hand-written copy is what this test originally had, and it was vacuous: swapping
 * the two halves in the real GLSL left all eight assertions green, because the test was
 * only ever round-tripping its own reimplementation against `voteWinner`. Since GLSL
 * cannot be executed here, the source text is the only witness available — so parse the
 * component ORDER out of it and let the round-trip below run on that.
 *
 * Caught by mutation (swap the halves → this now fails); see the
 * "puts the LOW half in .x" test below for the assertion that pins it.
 */
function splitId(i: number): [low: number, high: number] {
  const [firstIsLow] = glslSplitOrder();
  const low = i & 0xffff;
  const high = i >>> 16;
  // `.x` feeds the G channel and `.y` the A channel (`vec4(nodeId, .x, brightness, .y)`),
  // and `voteWinner` recombines as `A * 65536 + G` — so `.x` MUST be the low half.
  return firstIsLow ? [low, high] : [high, low];
}

/**
 * Read the `vec2(...)` component order out of the real GLSL helper.
 *
 * Returns `[firstIsLow]`: whether the FIRST component is the masked (low 16 bits)
 * expression rather than the shifted (high) one.
 */
function glslSplitOrder(): [firstIsLow: boolean] {
  const body = /vec2 luxarElementIdSplit\(uint i\)\s*\{([^}]*)\}/.exec(GLSL_ELEMENT_ID_SPLIT);
  if (!body) throw new Error('luxarElementIdSplit not found in GLSL_ELEMENT_ID_SPLIT');
  const call = /vec2\(([^,]+),([^)]+)\)/.exec(body[1]);
  if (!call) throw new Error('no vec2(...) in luxarElementIdSplit');
  const [, first, second] = call;
  const isMask = (e: string) => /&\s*0xFFFFu/.test(e);
  const isShift = (e: string) => />>\s*16u/.test(e);
  if (isMask(first) && isShift(second)) return [true];
  if (isShift(first) && isMask(second)) return [false];
  throw new Error(`luxarElementIdSplit components are neither mask+shift: ${first} | ${second}`);
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

  it('puts the LOW half in .x and the HIGH half in .y, which is what the readback assumes', () => {
    // The assertion whose absence made this file vacuous. `voteWinner` decodes
    // `Math.round(a) * 65536 + Math.round(g)` from `vec4(nodeId, .x, brightness, .y)` —
    // so `.x` must be the masked low half and `.y` the shifted high half. Swap them and
    // every pick id above 65,535 decodes to a different vertex, silently, on large
    // meshes only. GLSL cannot be executed here, so the source text is the witness.
    const [firstIsLow] = glslSplitOrder();
    expect(firstIsLow, 'vec2 first component must be `i & 0xFFFFu` (the LOW half)').toBe(true);
  });

  it('the fragment stage routes .x to G and .y to A — the write site the readback decodes', () => {
    // Pinning the helper's internal order (above) is only half the contract: swapping
    // the two components at the WRITE site instead — `vec4(vNodeId, vElementId.y,
    // brightness, vElementId.x)` — would leave the helper assertion green while
    // `voteWinner`'s `A * 65536 + G` decodes every id above 65,535 to the wrong
    // vertex, exactly the mutation the helper test cannot see. The TSL twin's write
    // order is pinned by the `mesh-pick.fragment.glsl.txt` codegen snapshot.
    const write =
      /fragColor\s*=\s*vec4\(\s*vNodeId\s*,\s*vElementId\.(\w)\s*,\s*brightness\s*,\s*vElementId\.(\w)\s*\)/.exec(
        MESH_PICK_FRAGMENT_SHADER
      );
    expect(write, 'pick-encoding write not found in the fragment shader').not.toBeNull();
    expect(write![1], 'the G channel must carry the LOW half (.x)').toBe('x');
    expect(write![2], 'the A channel must carry the HIGH half (.y)').toBe('y');
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
