/**
 * Zarr's root metadata documents, and how to read a node's attributes out of
 * whichever one a store happens to use.
 *
 * Two Luxar probes fetch root metadata over raw HTTP, deliberately bypassing
 * zarrita: the cache-validation token probe
 * (`cache/multi-level-caching-store/validation-queue.ts`) and the scene-identity
 * watchdog (`data/scene-identity-watchdog.ts`). Both need to know what the
 * document is called and how it is shaped, and they sit in different layers —
 * `.dependency-cruiser.cjs` forbids `cache/` from importing `data/` — so this
 * lives in `types/`, the lowest layer, which both may reach.
 *
 * They previously each carried their own copy. Two copies of a format contract
 * is precisely the drift that produced the bugs this module exists to prevent:
 * a fix applied to one probe and not the other looks correct in review and
 * fails silently in exactly one code path.
 */

/**
 * Root metadata documents that carry a node's attributes, NEWEST FORMAT FIRST.
 *
 * Format 3 nests attributes inside `zarr.json`; format 2 uses a separate
 * `.zattrs`. Luxar writes 3 while existing stores stay 2, so a probe has to
 * accept either — and tries `zarr.json` first because that is what new datasets
 * are, making the second request the exception rather than the rule.
 */
export const ROOT_ATTR_DOCS = ['zarr.json', '.zattrs'] as const;

/**
 * A node's user attributes, from either root document shape.
 *
 * A format-2 `.zattrs` *is* the attributes object. A format-3 `zarr.json` is the
 * whole node record — `zarr_format`, `node_type`, `consolidated_metadata`, and
 * the attributes nested under `attributes` — so reading `content_hash` off its
 * top level always yields `undefined`. For the watchdog that means reporting
 * every poll as a change; for cache validation it means no token at all.
 *
 * Returns `{}` rather than throwing for a non-object, or for a format-3 record
 * whose `attributes` is absent or not an object — both callers treat "no
 * attributes" as a normal, answerable state. Falling through to the record
 * itself in that case would expose `zarr_format` and `node_type` AS the node's
 * attributes, which is worse than an empty answer and would compare unequal to
 * the attrs the scene was loaded with on every poll.
 *
 * Mirrored on the Python side by `_node_attrs` in `test_cli_integration.py`,
 * which probes the same documents over HTTP; keep the two in step.
 */
export function rootAttributes(parsed: unknown): Record<string, unknown> {
  if (parsed === null || typeof parsed !== 'object') return {};
  const record = parsed as Record<string, unknown>;
  if (record.zarr_format === 3) {
    const attributes = record.attributes;
    return attributes !== null && typeof attributes === 'object'
      ? (attributes as Record<string, unknown>)
      : {};
  }
  return record;
}
