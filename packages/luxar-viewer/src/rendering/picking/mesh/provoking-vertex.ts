/**
 * Align the WebGL backend's `flat`-varying provoking vertex with WebGPU's.
 *
 * A `flat` varying is sourced from ONE corner of the triangle, and the two
 * backends do not default to the same one:
 *
 * | Backend | Convention |
 * |---|---|
 * | OpenGL ES 3.0 (`glslVersion: GLSL3`) | **last** vertex of the primitive, fixed by spec |
 * | WGSL `@interpolate(flat)` as three emits it | **first**-vertex sampling |
 *
 * That only becomes observable when a `flat` varying differs BETWEEN a triangle's
 * corners — which, in the shipped materials, is exactly one case: the mesh pick
 * shader's `vElementId`, whose source is `gl_VertexID` / `vertexIndex`
 * (spec §6.5). Left alone, the same click on the same triangle `(i0, i1, i2)`
 * reports `i2` on WebGL and `i0` on WebGPU.
 *
 * `WEBGL_provoking_vertex` lets the WebGL side adopt WebGPU's first-vertex rule, so
 * the two agree wherever the extension exists. Where it does not, the divergence
 * stands as a documented §6.4 exception and the pick contract is unchanged: at
 * vertex granularity the answer is "*a* corner vertex of the front-most triangle
 * under the cursor" — the cursor is over the face, not a vertex, so every corner is
 * equally valid and no consumer may assume a specific one.
 *
 * ## Why context-wide state is acceptable here
 *
 * `provokingVertexWEBGL` is not scoped to a program or a draw call; it changes the
 * convention for the whole context. That is safe because every OTHER `flat` varying
 * in the shipped materials is a per-**instance** constant — point and gsplat quads
 * and line segments all carry the same id, sharpness and cap flags at every corner
 * of their expanded quad, so which corner supplies the value is unobservable. Mesh
 * is the only geometry whose `flat` inputs vary within a primitive, hence the only
 * one that can see the flip at all.
 *
 * @module rendering/picking/mesh/provoking-vertex
 */

import { log, Modules } from '../../../utils/log';

/** The subset of the extension object this module uses. */
interface ProvokingVertexExtension {
  readonly FIRST_VERTEX_CONVENTION_WEBGL: number;
  provokingVertexWEBGL(provokeMode: number): void;
}

/**
 * Switch a WebGL2 context to first-vertex `flat` sampling, if it can.
 *
 * Idempotent and failure-tolerant by design: called once per `PickingSystem`, on a
 * context that may be WebGPU (no-op), may lack the extension (no-op), or may be a
 * unit-test double (no-op). A missing extension is normal, not an error — hence
 * `log.info` rather than a warning. Both arms log exactly once per session, which
 * is what makes a pick-id divergence diagnosable from a console dump alone.
 *
 * Takes the RENDERER rather than its context so that every duck-type lives here:
 * `getContext` itself is absent on the partial renderer doubles the picking-system
 * unit tests construct, so a call site that dereferenced it would throw before
 * reaching any of the guards below.
 *
 * @param renderer The renderer, or anything shaped unlike one.
 * @returns `true` when the convention was switched, `false` when left at the
 *   platform default. Returned rather than inferred so a test can assert both arms.
 */
export function alignProvokingVertexWithWebGPU(renderer: unknown): boolean {
  // Duck-typed rather than `instanceof WebGL2RenderingContext`: the WebGPU backend
  // returns a GPUDevice-ish object here, and jsdom has no WebGL2RenderingContext
  // constructor at all, so `instanceof` would throw in unit tests.
  const host = renderer as { getContext?: () => unknown } | null;
  if (!host || typeof host.getContext !== 'function') return false;

  let ext: ProvokingVertexExtension | null = null;
  try {
    const ctx = host.getContext() as { getExtension?: (name: string) => unknown } | null;
    if (!ctx || typeof ctx.getExtension !== 'function') return false;
    ext = ctx.getExtension('WEBGL_provoking_vertex') as ProvokingVertexExtension | null;
  } catch {
    // A context that throws (lost context, or a stub) is not a reason to fail
    // picking — the documented last-vertex fallback still holds.
    return false;
  }
  if (!ext || typeof ext.provokingVertexWEBGL !== 'function') {
    log.info(
      Modules.RENDERER,
      'WEBGL_provoking_vertex unavailable; mesh pick flat varyings keep the ' +
        'GL last-vertex convention (WebGPU reports the first corner instead — ' +
        'both are valid corners of the picked triangle)'
    );
    return false;
  }

  ext.provokingVertexWEBGL(ext.FIRST_VERTEX_CONVENTION_WEBGL);
  log.info(
    Modules.RENDERER,
    'Provoking vertex set to FIRST_VERTEX_CONVENTION_WEBGL — mesh pick ids now ' +
      'match the WebGPU backend'
  );
  return true;
}
