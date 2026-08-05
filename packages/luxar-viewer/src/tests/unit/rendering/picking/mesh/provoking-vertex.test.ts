/**
 * The WebGL `flat`-varying provoking-vertex alignment (spec §6.5).
 *
 * Every arm matters because the function runs on whatever the `PickingSystem` was
 * handed: a real WebGL2 context, a WebGPU device, or — in every picking unit test —
 * a partial renderer double with no `getContext` at all. It must never throw on any
 * of them, and it must actually flip the convention when it can.
 *
 * @module tests/unit/rendering/picking/mesh/provoking-vertex
 */
import { describe, it, expect, vi } from 'vitest';
import { alignProvokingVertexWithWebGPU } from '../../../../../rendering/picking/mesh/provoking-vertex';

const FIRST = 0x9123;

/** A renderer whose context exposes the extension. */
function rendererWithExtension() {
  const provokingVertexWEBGL = vi.fn();
  return {
    provokingVertexWEBGL,
    renderer: {
      getContext: () => ({
        getExtension: (name: string) =>
          name === 'WEBGL_provoking_vertex'
            ? { FIRST_VERTEX_CONVENTION_WEBGL: FIRST, provokingVertexWEBGL }
            : null,
      }),
    },
  };
}

describe('alignProvokingVertexWithWebGPU', () => {
  it('switches the convention to FIRST when the extension is present', () => {
    const { renderer, provokingVertexWEBGL } = rendererWithExtension();
    expect(alignProvokingVertexWithWebGPU(renderer)).toBe(true);
    // The exact enum matters: any other mode would leave WebGL and WebGPU reporting
    // different corners, which is the whole thing this exists to prevent.
    expect(provokingVertexWEBGL).toHaveBeenCalledWith(FIRST);
    expect(provokingVertexWEBGL).toHaveBeenCalledTimes(1);
  });

  it('reports false and does nothing when the extension is absent', () => {
    const renderer = { getContext: () => ({ getExtension: () => null }) };
    expect(alignProvokingVertexWithWebGPU(renderer)).toBe(false);
  });

  it('reports false when the extension exists but lacks the entry point', () => {
    // A partially-implemented extension object is not hypothetical — the browser
    // surface has changed shape before, and calling a non-function would throw
    // inside the PickingSystem constructor, taking the whole viewer down.
    const renderer = {
      getContext: () => ({
        getExtension: () => ({ FIRST_VERTEX_CONVENTION_WEBGL: FIRST }),
      }),
    };
    expect(alignProvokingVertexWithWebGPU(renderer)).toBe(false);
  });

  it('survives a renderer with no getContext (the unit-test double)', () => {
    // The arm that actually fired in practice: dereferencing `getContext` on a
    // partial double threw and broke 57 picking-system tests.
    expect(alignProvokingVertexWithWebGPU({})).toBe(false);
    expect(alignProvokingVertexWithWebGPU(null)).toBe(false);
    expect(alignProvokingVertexWithWebGPU(undefined)).toBe(false);
    expect(alignProvokingVertexWithWebGPU({ getContext: 'nope' })).toBe(false);
  });

  it('survives a context that returns null or throws', () => {
    expect(alignProvokingVertexWithWebGPU({ getContext: () => null })).toBe(false);
    expect(
      alignProvokingVertexWithWebGPU({
        getContext: () => {
          throw new Error('context lost');
        },
      })
    ).toBe(false);
    expect(
      alignProvokingVertexWithWebGPU({
        getContext: () => ({
          getExtension: () => {
            throw new Error('context lost');
          },
        }),
      })
    ).toBe(false);
  });

  it('treats a WebGPU-shaped context as a no-op rather than an error', () => {
    // The WebGPU backend's getContext returns a device-ish object with no
    // getExtension. Nothing to align there — WGSL already samples the first vertex.
    expect(alignProvokingVertexWithWebGPU({ getContext: () => ({ queue: {}, limits: {} }) })).toBe(
      false
    );
  });
});
