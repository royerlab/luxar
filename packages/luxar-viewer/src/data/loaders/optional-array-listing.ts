/**
 * Store-listing gate for OPTIONAL per-element arrays.
 *
 * The points and lines leaf loaders open `colors` / `radii` / `widths` /
 * `sharpnesses` speculatively: a missing array is legal, so the historical
 * pattern was "open, catch the 404, use the default". On a hosted scene that
 * is three extra round trips per node before any geometry moves — the
 * 100-node benchmark example spent ~300 requests on them (2026-09 audit,
 * finding 10). `buildSceneGraph` now records the child arrays the consolidated
 * listing reports for each node (`SceneNode.arrays`), so a loader can skip the
 * probe when the listing says the array is not there.
 *
 * Fail-open by design: with no listing (`arrays` undefined — fallback
 * enumeration, or a listing that showed no arrays at all) the loader probes as
 * before. The gate only ever REMOVES a request the listing proves would 404; it
 * never suppresses an array the listing did not rule out.
 *
 * @module data/loaders/optional-array-listing
 */

import type { SceneNode } from '../data-loader-types';

/** True unless the node carries a listing that lacks `name`. */
export function isArrayListed(node: Pick<SceneNode, 'arrays'>, name: string): boolean {
  const arrays = node.arrays;
  return arrays ? arrays.has(name) : true;
}
