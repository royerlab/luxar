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

/** The format-3 member of {@link ROOT_ATTR_DOCS}, named once. */
const V3_ROOT_DOC: (typeof ROOT_ATTR_DOCS)[number] = 'zarr.json';

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
 * The content signal is a `zarr_format: 3` member, and alone it is a guess: a
 * format-2 document whose USER attributes happen to carry a `zarr_format` key is
 * indistinguishable from a format-3 one and would be answered as `{}` instead of
 * verbatim. Unreachable for a Luxar store, but a foreign store is not ours to
 * constrain.
 *
 * `docName`, the document the bytes came from, settles it — both probes know it.
 * It may only DEMOTE: a name that is not `zarr.json` vetoes the unwrap, but a
 * name that IS `zarr.json` never forces one on a body that does not look like a
 * v3 record. Promoting on the name alone would answer `{}` for any non-v3 body
 * served from a `zarr.json` address — a shape no real server produces, but one
 * that fakes and misconfigured proxies do.
 *
 * Mirrored on the Python side by `_zarr_compat.attrs_from_node_doc`, which takes
 * the same optional `doc_name` with the same demote-only rule; keep them in step.
 */
export function rootAttributes(parsed: unknown, docName?: string): Record<string, unknown> {
  if (parsed === null || typeof parsed !== 'object') return {};
  const record = parsed as Record<string, unknown>;
  const looksV3 = record.zarr_format === 3;
  const namedV2 = docName !== undefined && docName !== V3_ROOT_DOC;
  if (looksV3 && !namedV2) {
    const attributes = record.attributes;
    return attributes !== null && typeof attributes === 'object'
      ? (attributes as Record<string, unknown>)
      : {};
  }
  return record;
}

/**
 * Which of {@link ROOT_ATTR_DOCS} a URL addresses, or `undefined` if neither.
 *
 * The probes hold the URL they fetched rather than a bare file name, and a URL
 * may carry a query string, so the match is on the path's last segment.
 */
export function rootAttrDocOf(url: string): (typeof ROOT_ATTR_DOCS)[number] | undefined {
  const path = url.split(/[?#]/, 1)[0];
  return ROOT_ATTR_DOCS.find((doc) => path.endsWith(`/${doc}`) || path === doc);
}
