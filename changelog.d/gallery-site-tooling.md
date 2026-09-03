#### Bring the demo site's route generation and count audit into the repo

Two pieces of the deploy path lived only in an operator's unversioned directory.
One of them is load-bearing: `gen_redirects.py` generates the stable
`/d/<demo-key>` routes, and the root README now links 29 tile titles at them, so
losing it means nobody can regenerate a public contract.

`scripts/gallery/gen_redirects.py` emits a static Cloudflare Pages `_redirects`
mapping each demo key onto the current dated data prefix. `/d/<key>` is an
identity; the prefix is a location, so anything durable — README, papers, issues
— links to the former and survives the next deploy. It matters that this fails
loudly: the origin answers a missing store with `200 text/html`, so a broken link
renders a blank viewer rather than a 404.

Two behaviours are deliberate and easy to lose in a rewrite:

- **A demo key is not always its store name** — 8 of 87 manifest entries differ
  (`cosmicflows_laniakea` → `cosmicflows_laniakea_full`, `nd_transforms` →
  `nd_transforms_bench`, …). The store comes from each entry's `dataset`, never
  its `id`, and routes are emitted for both spellings.
- **`--check-contract`** fails the build naming any README-linked key without a
  route. Its source is the union of `media-manifest.json` tile keys and the
  README's own `/d/` links, rather than the `docs/images/readme/gallery/`
  basenames, which no longer exist. The check fails closed when neither source
  is present or their union is empty.

There is no `/d/*` catch-all on purpose: an unknown key should not quietly land
on the gallery, because that hides a typo behind a page that looks fine.

`scripts/gallery/audit_readme_demo_count.py` is report-only and runs at deploy
time, comparing the README's stated live-demo count against the gallery being
deployed. Nothing watched that number before (`sync_demo_counts.py` owns the
*bundled* count) and it had drifted twice.

Documentation: `VIEWER_GUIDE.md` gains "Opening your own data in the hosted
viewer" — that guide points people at `luxarviewer.dev/?src=` and mentioned CORS
zero times, which is the failure they actually hit. `DEMO_SITE_RUNBOOK.md` gains
a section covering the current live state, these routes, the resolver defects
behind #2454, and the environment traps that cost hours.
