#### Four more silent readers, found auditing the format-3 move

A post-merge audit of the format-3 change turned up four defects of the same
class it was written to remove — the class where nothing raises and the wrong
answer is an ordinary-looking value.

**The document readers disagreed with zarr itself.** They consulted format 2
first, while zarr resolves a node carrying BOTH documents as format 3 — it warns
and says so. Measured on such a store, `read_node_attrs` answered the stale v2
attributes while `zc.open_group` and a plain `zarr.open_group` both answered the
v3 ones. `read_array_meta` is the one with teeth: a leftover `.zarray` hands back
the wrong `shape`, and that is what `batch-fit validate` checks tiles against —
measured, a corrupt `zarr.json` beside a stale `.zarray` returned shape 555,
which would have been accepted as a good tile.

The rule is now the same in all four readers: a present `zarr.json` is the
answer, and being unusable is an answer too. `is_consolidated` needed that
strengthening most — falling back to a leftover `.zmetadata` reports an
INTERRUPTED v3 save as finished, and that boolean is the sentinel `batch-fit`
uses to tell a written tile from a half-written one. Luxar's own writers never
produce a mixed store; an interrupted in-place migration or an rsync over an
older copy does.

**A test that had quietly stopped testing.** `error-recovery.spec.ts` lets
metadata requests through and aborts chunk requests, but its allowlist named only
format 2's documents. On a format-3 store `zarr.json` fell through to the chunk
branch and was aborted after the first two requests, so the viewer failed at
metadata load rather than mid-chunk — a different path entirely. Its assertion is
a disjunction, so it kept passing. The corrupted-metadata test directly above it
already carries a comment about hitting exactly this, having been fixed one
matcher at a time; this was the one left behind.

**A format-3 branch covered only by accident.** `isGeneratedFixtureComplete`
decides whether a generated fixture is usable, and reporting a v3 fixture as
incomplete is what stops the E2E suite from starting at all — both global setups
then demand a regeneration that can never satisfy them. Its tests pinned the v2
branch explicitly and the v3 branch only via a sweep over whatever the ambient
generator happened to write, which `LUXAR_ZARR_FORMAT=2` silently reverts to v2
alone. Four explicit cases now: a complete v3 root, one written before
consolidation finished, one whose document is not parseable, and a stale
`.zmetadata` left beside an unfinished v3 root.

**Where a document's format comes from.** `attrs_from_node_doc` and the viewer's
`rootAttributes` inferred it from a `zarr_format: 3` member, which misreads a
format-2 document whose USER attributes carry that key — answering `{}` and
dropping every authored value. Both now take the document NAME, which every
caller already knows. The name may only DEMOTE, never PROMOTE: it vetoes an
unwrap but never forces one on a body that does not look like a v3 record.
Promoting on the name alone would answer `{}` for any non-v3 body served from a
`zarr.json` address — a shape no real writer produces, but one that test fakes
and misconfigured proxies do.
