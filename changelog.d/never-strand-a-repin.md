#### A re-pin can no longer brick a dataset

Re-pinning `gsplats_3d_drosophila_gastrulation` to the bytes its Zenodo record
holds made the demo unbuildable by anyone. Every cache held the previous
generation, there is no in-repo copy, and the record is an unpublished draft — so
the cache was quarantined as a mismatch and nothing was left to fall back to.
Ten of thirty datasets are hosted-only, so this was a class rather than an
incident.

The defect is an ordering one: the cache leg quarantined a mismatch **before**
discovering whether any source existed, so it destroyed the only copy in
existence and then reported that nothing was available. Publication happens to
unblock it, which is why it presented as a publication problem; it is not.

Same shape as #854, where `cached_download` quarantined a COMPLETE file against a
stale `expected_size` and re-downloaded it on every launch. There the fix was to
stop quarantining. Here that alone would be unsafe, because a digest mismatch is
ambiguous: corrupt bytes and a superseded-but-valid generation are
indistinguishable from the digest. So the ambiguity is removed first.

File entries gain an optional `superseded_sha256` — digests this project pinned
in an earlier generation — which `gen_data_manifest` now maintains itself,
appending the outgoing digest whenever a pin changes. That information was
previously discarded at exactly the moment it stopped being recoverable.

`ensure_dataset` gains a third, deliberately weaker verdict. Bytes matching a
superseded digest are kept **only** when nothing can replace them (no in-repo
copy, no fetch route), and are announced as out of date rather than corrupt.
Where a route exists the superseded copy loses, so an update propagates the
instant it becomes obtainable. Bytes matching nothing still quarantine.

Acceptance reaches back exactly **one** generation even though the manifest keeps
the whole history. The list's effect is how far back "acceptable" reaches, and the
oldest entry is the likeliest to be genuinely wrong; anything older degrades to a
build failure, which is loud and recoverable, rather than to a silently stale
artifact.

Measured against a real cache of 41 pinned files: 30 match the current pin, 7 a
recorded superseded digest, and 4 match nothing ever recorded. So the list is not
a complete answer — the ordering fix is what helps regardless of whether a digest
was recorded — and the four unrecorded cases still need a fetch route.
