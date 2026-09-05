/**
 * Which geometry kinds take the partial-extend tolerance, in one place.
 *
 * `deriveNodeViewState` widens a node's slice tolerance across dimensions the
 * node only PARTIALLY extends through. Lines opt out: their segment bounds
 * already encode the non-displayed extent, so applying it again double-counts
 * during clipping. Every other kind opts in — a mesh has no per-element bounds
 * at all, so there is nothing to double-count.
 *
 * This is the *data* half of `GEOMETRY_DESCRIPTORS`, split out into a
 * module with no runtime imports so leaf handlers and refinement loops can read
 * one boolean without pulling in every node loader and loader factory. The
 * descriptor still exposes the flag
 * (`GEOMETRY_DESCRIPTORS[kind].applyPartialExtendTolerance`) and now sources it
 * from here, so there remains exactly one definition;
 * `tests/unit/data/scene-loader/partial-extend-tolerance.test.ts` pins the two
 * together, and fails on any NEW hardcoded site.
 *
 * Twelve call sites used to hardcode the boolean, against a table documented as
 * the single source of truth that only two of them read. Hardcoding is not a
 * compile error and the wrong value is not a crash — it is a clipping result
 * that is quietly too generous or too tight, which no test would name.
 *
 * @module data/scene-loader/partial-extend-tolerance
 */

import type { GeometryKind } from '../data-loader-types';

/**
 * Whether `deriveNodeViewState` widens the slice tolerance for each kind.
 *
 * `true` for every kind whose elements carry no non-displayed extent of their
 * own; `false` for lines, whose segment bounds already encode it. Read this —
 * do not restate the boolean at a call site. `GEOMETRY_DESCRIPTORS` exposes the
 * same value per kind and sources it from here.
 */
export const PARTIAL_EXTEND_TOLERANCE: Record<GeometryKind, boolean> = {
  points: true,
  lines: false,
  gsplats: true,
  mesh: true,
};
