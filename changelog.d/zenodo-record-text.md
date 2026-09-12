#### Persist the Zenodo record descriptions in the repo

The four published record descriptions are hand-maintained on Zenodo and, until
now, existed nowhere else. An accidental overwrite — or a well-meaning
regeneration — would have destroyed prose that took real work, with no diff to
recover it from. They are captured verbatim under `scripts/zenodo_record_text/`
with a per-record `description_sha256`, so drift between the repo and the live
records is detectable instead of invisible.

The record renderer now also follows the published descriptions' rule of
carrying the quality columns only where at least one row has a real figure, so
the generated report does not misleadingly show two columns of dashes where the
authoritative record omits them.
