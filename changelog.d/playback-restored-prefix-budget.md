#### Let restored playback prefixes spend their remaining frame budget

Progressive playback now treats a SliceCache-restored LOD prefix like the same
slice loaded cold: while frame budget remains, it streams additional
cache-resident levels and stops at the first cold or slow one. Previously the
restored path committed its cached prefix unchanged, so a shallow cache entry
could render less detail than having no entry at all.

Only an empty ladder keeps the unconditional first-level floor. A restored
prefix is already showable, so a zero-budget tick performs no extra decode. If
budget remains, the worst case is one cold load before the residency result is
known — the same bound as the empty path already had.
