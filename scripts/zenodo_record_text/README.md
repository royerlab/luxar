# Zenodo record descriptions, as published

The four Zenodo record descriptions, captured verbatim from the live records.
They are stored byte-for-byte as the API returns them, including markup and line
wrapping; never reformat these files, only re-capture them.

**These files are the authoritative text, not `gen_zenodo_records.py`'s output.**
The published descriptions are hand-maintained: they carry framing, provenance
and caveats the generator does not produce, and their column sets differ from
the generator's. The generator renders a *report* over the same measurements; it
has never been the thing uploaded.

They live here because they lived nowhere else. Until 2026-09-02 the only copy
was on Zenodo, so an accidental overwrite — or a well-meaning regeneration —
would have destroyed prose that took real work, with no diff to recover it from.
`records.json` records the field values alongside a `description_sha256`, so
drift between this directory and the live records is detectable rather than
invisible.

## Refreshing after an edit on Zenodo

Descriptions are edited on Zenodo by the maintainer, so this directory follows
rather than leads. Re-capture with a token in `ZENODO_TOKEN`:

    python3 scripts/zenodo_record_text/capture.py

Check for drift without a token or modifying the snapshots:

    python3 scripts/zenodo_record_text/capture.py --check

Commit the result. A changed `description_sha256` with no accompanying commit
means someone edited a record and the repo has not caught up.

## The quality-column rule

A record carries the PSNR / foreground columns **iff at least one row on that
record has a real figure**. The live descriptions omit them from `h2afva` and
`droso-timelapse`, where nothing is scored, and keep them on `cc-by` and
`cc-by-sa`, where something is. Two columns of dashes cannot be told apart from
"unmeasurable" or from evasion.

`render_record` implements this and
`test_quality_columns_are_a_record_level_choice` holds it, so the generated
report follows the authoritative records rather than contradicting them.
