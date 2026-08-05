/**
 * The two mesh ceilings in `config/constants.ts`.
 *
 * `MAX_MESH_VERTICES` is a MIRROR of the Python constant, so it gets a literal
 * assertion here in the same shape the truncation-radius mirror uses: each side
 * asserts the value and names the other. A drift would let `add_mesh` emit a
 * store the viewer refuses — the exact fail-fast contract the pair exists to
 * uphold.
 *
 * `MESH_DECODE_BUDGET_BYTES` has no Python twin (the write side cannot know what
 * a tab survives), so what is pinned instead is the reasoning: it must stay well
 * under what a tab can hold, because the tab has to survive the ~3-4x transient
 * multiple, not the ceiling.
 */

import { describe, it, expect } from 'vitest';
import { MAX_MESH_VERTICES, MESH_DECODE_BUDGET_BYTES } from '../../../config/constants';

describe('MAX_MESH_VERTICES', () => {
  it('is 2^27, the pick vote-key stride', () => {
    // MIRROR: MAX_MESH_VERTICES in
    // packages/luxar/src/luxar/typing_utils/constants.py must hold this value.
    // If you change one, change the other — a Python test pins that side
    // (validation/tests/test_mesh_validation.py::test_vertex_cap_is_the_alias_free_bound).
    expect(MAX_MESH_VERTICES).toBe(2 ** 27);
  });

  it('is the exact alias-free bound', () => {
    // The largest ordinal a node contributes is n_vertices - 1, so admitting
    // exactly 2^27 vertices keeps every ordinal strictly under the stride. Off by
    // one and picks alias across nodes for exactly one vertex, with no diagnostic.
    expect(MAX_MESH_VERTICES - 1).toBeLessThan(2 ** 27);
  });
});

describe('MESH_DECODE_BUDGET_BYTES', () => {
  it('is 512 MiB', () => {
    expect(MESH_DECODE_BUDGET_BYTES).toBe(512 * 1024 * 1024);
  });

  it('leaves room for the ~3-4x transient multiple inside a 64-bit tab', () => {
    // On the admission path the decoded sources coexist with derived copies (the
    // u32-coerced faces, the extracted display-space position, the GPU upload).
    // Raising this toward "what a tab survives" would be the mistake: the tab has
    // to survive the multiple.
    expect(MESH_DECODE_BUDGET_BYTES * 4).toBeLessThanOrEqual(2 * 1024 * 1024 * 1024);
  });

  it('is the binding constraint in practice, not the vertex cap', () => {
    // Worth pinning because it is counter-intuitive: the cap is 134.2M vertices,
    // but a 3D float32 mesh runs out of BUDGET at ~44.7M. So a legitimately
    // growing mesh always hits the budget first, and the cap exists for hostile
    // or nonsensical declarations (where it gives the accurate diagnosis) rather
    // than for real ones.
    const maxVerticesFrom3DFloat32Budget = Math.floor(MESH_DECODE_BUDGET_BYTES / (3 * 4));
    expect(maxVerticesFrom3DFloat32Budget).toBeLessThan(MAX_MESH_VERTICES);
  });
});
