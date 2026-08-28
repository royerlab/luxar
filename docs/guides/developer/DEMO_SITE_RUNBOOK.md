# Demo Site Runbook

How the public Luxar demo gallery and viewer are hosted, and how to publish a
new wave of demos to them without breaking anything.

This is an operational document. It records the architecture, the publish
sequence, and — most importantly — the failure modes that produce a *plausible
wrong answer* rather than an error. No credentials appear here; they live in
`~/.config/luxar-r2/credentials.env` on the operator's machine.
The publishing harness named below (`gen_landing.py`, `snapshot_hashes.py`,
`compare_snapshot.py`, and the upload script) also lives outside this
repository on that machine.

---

## 1. Architecture

Three hostnames on the `luxarviewer.dev` zone, each serving a different thing:

| Hostname | Serves | Backed by |
|---|---|---|
| `luxarviewer.dev` | The viewer alone, at the root | Cloudflare Pages project `luxar-viewer` |
| `demos.luxarviewer.dev` | The gallery page, its media, and its viewer | Cloudflare Pages project `luxar-demos` |
| `data.luxarviewer.dev` | The `.luxar.zarr` stores | Cloudflare R2 bucket `luxar-demos`, **direct** |

**Neither Pages project has a `functions/` directory or an R2 binding.** Both
are pure static assets. This is the single most important property of the
setup, and it is deliberate: Pages Functions share the Workers free limit of
100,000 requests/day, a limit that *fails* rather than bills. Serving the
corpus through a Function put one invocation on every chunk fetch, which does
not survive real traffic.

Instead:

- **Scene data** is fetched **cross-origin** from `data.luxarviewer.dev`, an R2
  custom domain that bypasses Workers entirely. The gallery emits absolute
  `?src=https://data.luxarviewer.dev/...` URLs for this reason. (Requires CORS —
  §4.3.)
- **Gallery media** (`/media/*`, ~180 files, ~485 MiB) are **deployed files**,
  served by Pages' CDN for free and unmetered.

The historical marker header `x-luxar-fn: r2` is how to confirm no Worker is in
the path. It should now appear on *nothing*:

```bash
curl -sI "https://demos.luxarviewer.dev/media/earthquakes.webp" | grep -i x-luxar-fn
curl -sI "https://data.luxarviewer.dev/data/<prefix>/<store>.luxar.zarr/<existing-object>" | grep -i x-luxar-fn
# no output from either = correct
```

Beware when checking this: responses cached *before* the Function was removed
still carry the header. Add a cache-busting query string, or you will conclude
the Worker is still deployed when it is not (§3.2 again, applied to your own
verification).

A verify-everything sweep of the whole chain lives in the operator's harness;
its essentials are reproduced in §5.

### 1.1 The viewer takes an absolute `src`

`https://luxarviewer.dev/?src=<absolute-url>` opens any store the browser can
reach, which is the point of hosting the viewer at the apex — it is a general
tool, not a demo appendage. That requires CORS on whatever origin holds the
data (§4.3). The gallery uses the same absolute form because its data lives on
the separate R2 hostname; a relative `/data/...` URL has no server on the Pages
origin.

---

## 2. Publishing a wave

A "wave" is: some demos changed on `dev`, rebuild them, and update the site.

```
build the changed demos
  -> capture gallery media (stills + orbit videos)
  -> luxar optimise --profile archive        # 1 MB chunk target
  -> hash-compare against the LAST LOCAL BUILD
  -> upload only what changed, to a NEW dated prefix
  -> rebuild the page against that prefix, with an ABSOLUTE data host
  -> deploy (page + media as static assets)
  -> audit the live site
  -> purge the superseded prefix
```

The published corpus deliberately uses `archive` rather than the general
object-storage `hosting` profile. On a representative live store it reduced
the chunk count from 5,004 to 224 (22×), accepting larger partial reads in
exchange for far fewer stored objects. Re-measure browser traffic and request
cost before changing that tradeoff.

The page generator takes the data prefix as an argument, so pointing a wave at
a new prefix is a parameter change, not an edit:

```bash
python gen_landing.py gallery.json deploy/index.html \
    https://data.luxarviewer.dev/data/<new-prefix>
```

Passing a site-relative `/data/<prefix>` there is the mistake to avoid: it
still renders a working page, but every chunk fetch becomes same-origin and
needs something on the gallery origin to serve it. The audit asserts the URLs
are absolute for exactly this reason.

### 2.1 Always publish to a new dated prefix

Prefixes are dated (`data/2026-08-26a/`). Never overwrite a live prefix in
place. `luxar optimise` assigns a **fresh `content_hash` by design**, and a
warm viewer cache validating on an unchanged hash would serve stale chunks. A
new prefix sidesteps the whole class of problem: new URL, no stale cache, and
the old prefix stays intact as a rollback until the audit passes.

Unchanged stores are **server-side copied** within R2 rather than re-uploaded —
same bytes, no egress, and it keeps the wave cheap.

### 2.2 Hash-compare correctly, or not at all

To decide which stores actually changed, compare **this local build against the
previous local build**. Do *not* compare a local build against the published
store: `optimise` gives every output a new `content_hash`, so that pairing
reports "changed" for everything and is meaningless. This has already cost one
near-miss 1.7 GB needless republish.

Snapshot hashes *before* rebuilding, then diff:

```bash
python snapshot_hashes.py  > before.json   # walk datasets/demos, record content_hash
# ... rebuild ...
python compare_snapshot.py before.json     # report only genuine changes
```

### 2.3 "Unchanged" and "never built" are different answers

A store that failed to rebuild (missing optional dependency, no GPU, no
credentials) still has its old hash on disk and will report IDENTICAL. That is
indistinguishable from a successful no-op rebuild unless you also check
freshness. This caught `cell_tracking_challenge` reporting IDENTICAL at 64 hours
old.

**Check freshness on FILES, not on the store directory.** A directory's mtime
updates only when its direct children change, so `stat` on
`<store>.luxar.zarr` can report a fresh timestamp over a tree nothing rewrote.
Not hypothetical: `desi_galaxies` passed a directory-mtime check while all
9,064 of its files were four days old.

Walk the tree and count how many files predate the run:

```python
mtimes = [p.stat().st_mtime for p in store.rglob("*") if p.is_file()]
stale = sum(1 for m in mtimes if m < run_started)
# a real rebuild leaves ~0 stale; a skipped one leaves ~all
```

### 2.4 A demo may reuse a cached scene and still exit 0

Some demos short-circuit when their output already exists. `desi_galaxies`
prints `Using cached scene: …`, emits its own warning that the cached scene is
stale, then finishes with `Dataset generated at …` and exit status 0. A wave
driver reading exit codes learns nothing.

The staleness it warned about was real and had shipped: both LOD ladders held a
finest-level node of 9,751,955 points against its own 4,000,000-point demo
ceiling, which "can silently lose their tail on a 4096-class GPU". The published
tile carried that for six days.

That 4,000,000 is **not** a universal cap — it is a safety margin local to that
demo (`SCENE_MAX_POINTS_PER_NODE`, and its comment says so). The real per-node
caps are per geometry type, in `typing_utils/constants.py`:

| geometry | cap |
|---|---:|
| Lines | 2,793,472 segments |
| Points | 5,591,040 points |
| GSplats | 4,194,304 splats |

Comparing a points node against 4,000,000 over-flags it.

Two remedies, both printed by the demo itself:

```bash
luxar demo run <key> -- --recompute       # recompute from source
rm -rf datasets/demos/<key>.luxar.zarr    # or drop it and let the demo unpack
                                          # the current shipped Git-LFS asset
```

Grep build logs for `Using cached scene` after any sweep. One gallery tile in
the 86-entry manifest took that path — few enough to miss, and the one that had
a real defect behind it.

### 2.5 Rebuild against the commit you think you are on

A sweep is only valid for the code it ran against, and `dev` moves under a long
one. A full pass over the 86 gallery tiles takes roughly ninety minutes here,
during which several PRs can land. Record the HEAD the sweep started from, and
re-check it at the end; if a store-affecting commit landed mid-sweep, the
results are stale for every demo it touches.

Discarding a partial sweep and restarting at current `dev` is usually cheaper
than finishing one you know is stale and then reasoning about which subset to
redo — that subset calculation is where scoping errors happen.

---

## 3. Hazards that fail silently

Every item here produced a plausible wrong answer in production rather than an
error. They are grouped by what lies to you.

### 3.1 Stale media (historical — and how the fix works now)

Scene URLs are dated; **media URLs are not**. When media was served from R2
through a Pages Function, re-uploading a still to the same key left every edge
serving the old bytes, and the only remedy was a `CACHE_EPOCH` constant in
`pages/functions/_r2.js` that had to be bumped by hand on every wave that
re-captured media. Forgetting it produced a correct page with last week's
thumbnails — and it did, at least once.

Moving media to Pages static assets removed that failure mode rather than
mitigating it: a deployment replaces the asset and Pages invalidates its own
CDN, so there is no epoch to remember. **Re-capturing media now just means
re-running the deploy.**

Do not reintroduce a hand-maintained cache-buster. If media ever moves back
behind a long-TTL rule on another host, it needs hashed filenames instead —
see §4.2.

### 3.2 A cache policy change does not reach cached objects

Changing R2 CORS, or any response-header policy, affects **only objects fetched
after the change**. Anything already in the edge cache keeps serving the old
response until its TTL expires. With a 30-day TTL that is a month of breakage
that looks fine on every fresh test you run.

Test the *cached* path, not a cache-busted one:

```bash
curl -s -o /dev/null "$URL"                                   # warm it
curl -sD - -o /dev/null -H "Origin: https://luxarviewer.dev" "$URL" \
  | grep -iE 'cf-cache-status|access-control-allow-origin'
# want: cf-cache-status: HIT *and* the CORS header present
```

If cached objects lack the header, purge (Cloudflare → Caching → Configuration
→ Purge Everything). There is no narrower fix.

### 3.3 `curl -I` misreports cache status

`HEAD` requests report `cf-cache-status: DYNAMIC` even where a `GET` cleanly
shows `MISS → HIT → HIT`. Diagnosing cache behaviour with `-I` will tell you
your cache rule failed when it is working. Use `-o /dev/null -D -` with a real
GET.

### 3.4 Zarr v2 vs v3 — never name a metadata document

The corpus holds **both** on-disk formats, permanently. Any tool that hardcodes
`.zattrs`, `.zarray`, `.zgroup`, `.zmetadata`, or `zarr.json` will silently
mis-handle half the stores. In zarr v3 there is one `zarr.json` per node with
attributes nested under `attributes`, and the consolidated index lives at
`zarr.json` → `consolidated_metadata` → `metadata`.

Two rules, learned the hard way:

- Read through a **bi-format helper**, never a literal document name. A link
  audit that hardcoded `zarr.json` scored a v2 store's 9-byte "Not found" body
  as "no links" and passed.
- **Absence must raise, not return empty.** Two separate probes both reported
  "no `part_N` found → correct" while walking zero nodes. A checker that cannot
  distinguish "nothing there" from "could not look" is worse than no checker.

### 3.5 Scale estimates: measure the affected, not the eligible

Counting things that *could* be affected instead of measuring what *is* has
produced order-of-magnitude errors twice — a tile estimate wrong by >10×, and a
model predicting 37,152 wasted requests on a scene that emits 5. Static models
over the corpus are upper bounds. Ground-truth the head of the distribution in
a real browser before reporting a number.

### 3.6 R2 billing: 404s count

R2 bills per operation. Cloudflare's pricing page exempts exactly one error
class — HTTP 401 — and says nothing about 404, which is billed as a Class B
read. In practice the edge caches 404s too, so with a cache rule in place R2
sees one per URL per PoP per TTL and the cost collapses. The real cost of a
404 storm is **first-paint latency**, not the bill.

### 3.7 ffmpeg: every encoder option must precede the output filename

With ffmpeg 6.1.1, put `-passlogfile` after the pass-2 output filename:

```bash
ffmpeg -y -i in.webm -c:v libvpx-vp9 -b:v 480k -pass 2 -an -row-mt 1 out.webm -passlogfile P
```

warns that the option is trailing:

```
Trailing option(s) found in the command: may be ignored.
```

Options placed after an output filename apply to the *next* output, so ffmpeg
ignores the custom prefix and looks for the default `ffmpeg2pass-0.log` instead
of the `P-0.log` that a correctly ordered pass 1 wrote. Pass 2 exits 251:

```
Error opening file ffmpeg2pass-0.log.
[vost#0:0/libvpx-vp9] Error reading log file 'ffmpeg2pass-0.log' for pass-2 encoding
Error opening output file out.webm.
```

The reverse mismatch is equally broken: a trailing pass-1 option writes its
statistics to `ffmpeg2pass-0.log`, then a correctly ordered pass 2 looks for
`P-0.log`. If both passes trail the option, they both use the default filename
and can appear to work despite the broken ordering.

What makes this worth a numbered hazard rather than a footnote is the *false
explanation waiting next to it*. WebM/Matroska output can carry no stream
timestamps — including the gallery masters that ffmpeg assembles from explicit
per-angle screenshots — so `ffprobe` reports `duration_ts=N/A` and
`nb_frames=N/A`. "The input is undecodable" is therefore plausible and wrong.
Confirm decodability before blaming the input:

```bash
ffmpeg -v error -stats -i in.webm -f null -      # reports frame=120 -- it decodes fine
```

Correct ordering:

```bash
ffmpeg -y -i in.webm -c:v libvpx-vp9 -b:v 480k -pass 1 -passlogfile P -an -f null /dev/null
ffmpeg -y -i in.webm -c:v libvpx-vp9 -b:v 480k -pass 2 -passlogfile P -an -row-mt 1 out.webm
```

The gallery harness normally encodes VP9 by quality (`-crf 24 -b:v 0`); a
target-bitrate two-pass re-encode is a last resort after the recapture controls
in §4.2.

### 3.8 Judge a re-encode on frames, never on byte count

Hitting a size target says nothing about whether the tile still depicts its
subject. `hilbert_curve_3d` re-encoded from 10 MB to 295 KB hit a 300 KB target
exactly and lost the fine wire detail that *is* the subject — the cube's outline
survived, so every automated check passed. Extract matched frames from source
and output and look at them.

Measured knee for that scene: 295 KB visibly degraded, 587 KB resolved
throughout, 1174 KB indistinguishable from source. Dense point clouds and
high-motion synthetic scenes need roughly double what a smooth microscopy
volume does.

For the normal size-control workflow, use the sanctioned `WEBM_CRF` or
`WEBP_QUALITY` knobs in §4.2, recapture, and compare representative frames.

---

### 3.9 Every quality figure on a hosted archive predates its own fix

`#1914` (2026-08-23) fixed five call sites that scored a fit against the **raw**
volume, when a fit reconstructs `V - image_min`. Every currently hosted archive
was stamped *before* that date, so **every hosted PSNR/SSIM figure is invalid**,
in a direction that depends on each volume's pedestal and so cannot be corrected
by arithmetic. Only a rebuilt archive carries a correct figure.

Consequences for this site:

- **Never read a quality figure off a hosted archive** to build a page, a note,
  or a comparison. Re-measure with current code, against one materialised
  reference volume.
- **A figure is only comparable to another measured on the same side of
  2026-08-23.** Comparing a pinned figure to a fresh one measures the scoring
  change, not the data.
- The public page is unaffected: no rendered field carries a quality figure
  (`gen_landing.py` never reads `note`). The exposure is in the manifest `note`
  fields, which are internal.

**And the noise floor is larger than it looks.** Three identical rebuilds of one
archive spread 0.08–0.35 dB with counts within ±0.3%. Treat anything under
~0.4 dB as noise; a difference only means something above that.

### 3.10 One path, two generations: hash before asserting

`demo_gsplats_2d_cmu1_pathology.py` says its hosted archives "are still flat
leaves". Two sessions measured `cmu1_ch0.gsplats.zarr.zip` and got answers that
could not both be true — flat laddered leaf with **no `kind` attr anywhere**, and
`{'partition': 1, 'lod': 64}` with 4,867 metadata docs.

**Both were right.** The manifest pins two generations of the same filename:

    cmu1_ch0   sha256         cd22645f...   bytes          38,201,205   <- in-repo copy
               hosted_sha256  29faffc1...   hosted_bytes   84,492,218   <- what the record serves

This Mac's cache holds the in-repo generation (all three channels hash to
`sha256` exactly). obsidian's cache holds the hosted generation (all three hash
to `hosted_sha256` exactly). Same path string, same demo, 2.2-2.75x apart in
bytes and structurally unrelated.

`_support/datasets/data_fetch.py` documents why: the two contracts were one field
until a refit replaced the hosted artifact without touching the in-repo copy, and
splitting them is what kept Zenodo publication off the critical path of every
demo-data PR.

**The discriminator, and the only reliable one:** hash the file and match it
against `sha256` vs `hosted_sha256`. That names the generation in one command.
Byte size alone is suggestive; a digest is decisive. Neither session did this
before asserting, and each had numbers to show.

#### The consequence worth knowing

`data_fetch.py` states the resolution order plainly: *"while an in-repo payload
is present it wins over a newer hosted artifact, so a checkout with a stale LFS
object keeps serving the older generation (loudly)."*

So **the same demo builds a structurally different scene depending on which
generation the building machine has in cache.** cmu1 grafts its archives
verbatim, so on this Mac it produces a flat scene (12 element nodes, 20,591,415
elements — exactly the sum of the three in-repo archives) and on a machine with
the hosted copy it would produce a partitioned, laddered one.

Two things follow for this site:

- **A tile's structure is a property of the build host, not just the recipe.**
  Record which host built a tile when its demo grafts pinned archives.
- **A tile can be accidentally correct.** cmu1's live tile is flat because the
  publishing machine held the stale in-repo generation — not because anything
  chose that. Refreshing the LFS payload to match hosted would change the live
  scene's structure with no code change and no manifest change visible in a diff.

#### And the docstring

It is describing the *hosted* archives, which do carry a partition and 64 lod
groups per channel. So it is **wrong**, as the first instinct had it — but not for
the reason v1 of this section gave, and the correction cannot be made from an
in-repo measurement alone.

Derive topology from each group's declared `kind` (children of `kind=lod` are
substitutive levels, children of `kind=partition` are parts) rather than from
node-name patterns. And note that **no `kind` attr anywhere means flat, not
unreadable** — the in-repo cmu1 generation is well-formed with zero kinds.

### 3.11 Flattening an archive only cuts requests if the demo grafts it

Two authoring paths, opposite outcomes from the same archive change:

| authoring call | effect of flattening the archive |
|---|---|
| `add_gsplats_from_file` with **no** recipe — grafts archive shape | scene node count drops with the archive |
| a declared `recipe=` — rebuilds structure locally | **download bytes only**; the scene re-creates its own structure |

Measured on this corpus:

- **`h2afva_timelapse`** and **`h2afva_stack`** graft, so flattening is a real
  request win — 704 → 176 element nodes for the timelapse, 41 parts → 1 for the
  stack.
- **`codex_pancreas`** fits locally through `save_with_lod(recipe="adaptive")`
  and grafts its own output, so its 2,764 groups are authored, not inherited.
  Changing the *archive* would not move them; changing the recipe's
  `max_elements` would.
- **`cmu1_pathology`** grafts verbatim, so it inherits whichever generation is
  in cache (3.10): flat from the in-repo copy, partitioned + laddered from the
  hosted one. Flattening the hosted archives is a real win *and* collapses that
  divergence — but measure the generation before claiming either.

So **split the claim per demo** before promising a load win. "Fewer nodes" and
"fewer bytes" are different wins, only the grafting demos get the first, and
some demos are already correct.

## 4. Cloudflare configuration

### 4.1 Cache rule on the data subdomain

Rules → Cache Rules, filter `hostname equals data.luxarviewer.dev`, action
"Eligible for cache" + "Ignore cache-control header and use this TTL".

TTL is a tradeoff, not a tuning knob: prefixes are dated and therefore
immutable, so a long TTL is safe for *data*. But the same rule governs how long
a mistake persists — a CORS or header change is invisible to already-cached
objects for the full TTL (§3.2). One day is a reasonable compromise; anything
longer wants a purge in the change procedure.

### 4.2 Media has no such protection

`/media/*` URLs are **not** dated. They are currently Pages static assets, which
is safe because a deploy invalidates them (§3.1). If media ever moves onto
`data.luxarviewer.dev` or any other host behind the long-TTL cache rule, it
needs a short TTL or hashed filenames *first*, otherwise a re-captured still is
unfixable without a full purge.

Two Pages limits bound this set: **25 MiB per file** and **20,000 files**. The
count is comfortable (180), but the largest clip sits at **24.87 MiB** —
0.13 MiB under the cap. A single oversized file fails the whole deployment, so
the gallery harness checks every PNG, WebP and WebM immediately after it is
written. It warns at 20 MiB, fails at the 25 MiB boundary, and prints the total
plus the five largest files at the end of the run. Treat a warning as a prompt
to choose a deliberate encoding adjustment with `WEBM_CRF` or `WEBP_QUALITY`
in `generate-gallery.spec.ts`, then recapture and inspect the affected demos;
do not silently trade quality for size with an automatic re-encode loop. Judge
the result on matched frames rather than bytes alone (§3.8).

### 4.3 CORS on the R2 bucket

Required for **both** viewers now, since the gallery also fetches data
cross-origin. `AllowedOrigins` must list every hostname that hosts a viewer:

```
https://luxarviewer.dev          apex viewer
https://demos.luxarviewer.dev    gallery viewer
https://luxar-demos.pages.dev    Pages fallback URL
```

Per-deployment preview URLs (`<hash>.luxar-demos.pages.dev`) cannot be
enumerated, so **data will not load in a Pages preview deploy**. Verify against
the canonical hostname.

Directory `.luxar.zarr` stores, including the entire published corpus, fetch
metadata and chunks with simple GETs. They require
`Access-Control-Allow-Origin`, but no `Range` request header or preflight.

Zipped `.zarr.zip` stores use byte-range requests. Their host must honour
`Range`, include `range` in `AllowedHeaders`, and expose `Content-Range`,
`Content-Length`, `Accept-Ranges`, and `ETag`. The range headers let the viewer
validate partial responses; `ETag` preserves archive identity across
cross-origin cache validation.

Verify both paths. Curl does not enforce CORS, so inspect the response headers
explicitly:

```bash
curl -sI -H "Origin: https://luxarviewer.dev" "$DIRECTORY_OBJECT_URL"
# want: access-control-allow-origin

curl -sI -X OPTIONS -H "Origin: https://luxarviewer.dev" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: range" "$ZIP_URL"
# want: 204, access-control-allow-headers including "range"

curl -sD - -o /dev/null -H "Origin: https://luxarviewer.dev" \
  -H "Range: bytes=0-0" "$ZIP_URL"
# want: 206 and access-control-expose-headers listing all four headers above
```

### 4.4 One hostname, one Pages project

A domain attached to two Pages projects is ambiguous. After splitting the
viewer out of the gallery, confirm each project claims only its own:

```bash
curl -s "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/pages/projects/<project>/domains" \
  -H "Authorization: Bearer $CF_API_TOKEN"
```

---

## 5. Verifying a wave

Run all of these; each catches a class the others cannot see.

1. **Live-site audit** — every tile, scene URL, still and video resolves; no
   placeholders; bylines match `DEMO_META`; bucket store count equals tile
   count. Target: **0 problems**.
2. **Browser probe** — load a sample of scenes in the real viewer, assert
   non-zero elements and no console errors. A store can be perfectly published
   and still render nothing.
3. **Cache and CORS** — §3.2 and §4.3, against *cached* objects.
4. **Pre-flight before purging the old prefix** — deleting from R2 is
   irreversible, so first prove nothing references it: grep the gallery page,
   the apex page, *and* the deployed JS bundles for the old prefix string, and
   confirm every store in the old prefix also exists in the new one.

### 5.1 Do not publish these

`chromatrace_choir_umap` and `chromatrace_choir_umap_sequence` are unpublished
research and must never reach the bucket. The external publish script carries
both an exclusion list and a hard abort guard that fails the run if their media
appears in the upload set. Keep both — the list alone has no teeth.

---

## 6. Framing tiles

Gallery framing lives in `scripts/gallery/manifest.json`; see
`scripts/gallery/README.md` for the field list. Two things worth knowing here:

- **`fillTarget` defaults to 0.95, which is wrong for round or dense
  subjects.** A globe or a capsid at 0.95 is cropped past its own silhouette
  and reads as a texture wall. 0.65 is the value that keeps recurring for these.
- **Judge a tile on the render, not on the metric.** The `coverage` score uses
  a percentile bounding box that ignores exactly the frame-edge outliers that
  make a tile unreadable — it rated a broken framing 93% and the correct one
  63%. `border-lit` is the better signal, but it too is inflated by legitimately
  bright limbs (a luminous cloud shell gains rim luminance seen edge-on). Look
  at the image.
- **Worst-orbit-pose border-lit does not answer whether the still is framed
  correctly.** An elongated subject projects wider as it rotates. Tribolium's
  correct poster frame still measures 48.4% because the embryo reaches the edge
  at `rock +15°`; shrinking it to satisfy that warning would underfill the tile.

---

## 7. Reading a scene's LOD and partition structure

Structure decisions get made from these numbers, so getting the accounting
right matters more than it looks. Every rule below is here because assuming the
obvious reading produced a wrong answer.

### 7.1 Substitutive levels are alternatives; additive rungs are deltas

A `kind=lod` group's `child_0/1/2` are **decimated copies of one another**, so a
scene's content is the **max** over levels, never the sum. Within one level, the
`additive_N` rungs are **deltas** and do sum — verified on `desi_galaxies`, whose
`child_0` rungs run 2000, 2000, 4000, 8000, 16000 … and total exactly 152,262,
matching the level.

Summing across levels instead reported that store as "11,123,187 elements" when
it holds 9,751,955 with 1.37M of ladder redundancy above it.

A logical node's size is likewise the **sum over its parts**, not the largest
single array. `desi_galaxies`' finest level reads 900,000 if you take the
biggest `positions` array and 9,751,955 if you total its parts.

And the rule that moves the most numbers: **only the RESIDENT slice counts.** A
node stacked on a hidden axis is measured per hidden coordinate, not by its
total — `demos/_lod_policy.py` states this. `human_multiome_peak_umap` totals
6,248,730 across six attribute views but is **1,041,455 resident**; measuring the
total over-flags it against any cap.

Two traps inside that, both found the hard way by the session doing the demo
rework:

- Measuring **one part** under-reports by the part count — a first pass read
  `nuclear_pore_complex` at 164,633 when it is 4,937,064, a 30x error, because
  the path measured was a single `part_N`.
- Summing **each part's largest slice** over-reports, because different parts
  peak on different hidden coordinates. Group globally by hidden coordinate
  first, *then* take the max.

Never infer element counts from physical array shapes. An `array_ref` encoding
stores a deduplicated array with a zero first dimension; use the node's
`n_points` / `n_splats`, or `encoding.original_shape` when inspecting that array.

### 7.2 `shape=[0]` means `array_ref`

A byte-identical duplicate of another array in the same store is encoded as an
empty `(0,)` / `(0, D)` array with `encoding.name="array_ref"`; `target` names
the source array and `original_shape` records the logical shape. Readers resolve
the target. A zero-shaped array with that encoding is normal; one without it is
wrong. See [Array Encodings](../user/LUXAR_ZARR_FORMAT.md#array-encodings).

### 7.3 The BSP tree is `bsp_tree`, on the `kind=partition` wrapper

New stores use zarr format 3: each node's attributes are nested under
`attributes` in its `zarr.json`; there is no `.zattrs`. A format-2-only scanner
therefore finds nothing and reports, wrongly, that partitions carry no BSP
metadata. Use Luxar's bi-format metadata readers when inspecting stores.

The serialized form is a nested dict with `left` / `right` and an `axis` per
internal node. The algebra on it lives in `core/group/partition.py`:
`prune_serialized_bsp_tree`, `map_serialized_bsp_tree`,
`reconstruct_serialized_bsp_tree`, `serialized_bsp_tree_separates`,
`serialized_bsp_tree_straddles_centers`,
`serialized_bsp_tree_axis_overlap_floors`, and `persist_pruned_bsp_tree`.

Measured on published prefix `2026-08-27b`:

| store | node | depth | leaves | axes |
|---|---|---:|---:|---|
| `ocean_currents_earth` | `currents` | 4 | 16 | 0,1,2 |
| `biodiversity_planetary_scale` | `Migrations by slice` | 2 | 3 | 0,1 |
| `biodiversity_planetary_scale` | `By taxon & period` | 1 | 2 | 2 |

A depth-1 single-axis entry like `By taxon & period` is a planar cut rather than
a spatial tree, and is worth checking: the serialized `axis` is a **centre-column
index** the viewer maps through `displayDims`, so a split on a non-displayed
dimension culls nothing.

### 7.4 Plain additive LOD silently rewrites `indexed` edges

`indexed` is the only line type that groups connected vertices into whole
component units, via `lod/lines.py::_indexed_connected_components`; for chain
inputs, those units are whole polylines. But the additive writer then discards
the explicit edge list and rebuilds each component as a chain in **ascending
vertex order**. Unless the original edge set already equals those consecutive
pairs, the ladder invents edges and drops real ones.

A plain `add_lines(..., line_type="indexed", additive_lod=...)` writes real
`additive_N` rungs and emits **no warning**. Do not use that path for indexed
graphs or streamlines whose topology must be preserved.

Converting to `segments` avoids invented edges, but `identify_polylines` returns
`n // 2` arrays of shape `(2,)`, so the laddering unit becomes a single segment.
A prefix is then scattered segments, i.e. fragmented polylines rather than
whole ones.

Additive LOD composed under `substitutive_lod=` behaves differently: indexed
levels are suppressed. An explicit additive request emits a `UserWarning` and
the levels load all-at-once; the default composed ladder is skipped with an
informational message.

Making this work needs a **verified** opt-in: check that every component really
is an ascending chain (its edge set equals its consecutive-vertex pairs) and
raise, not warn, when an explicit request cannot be honoured.

### 7.5 Compare like with like

Published and local copies of the same store can differ **structurally**, not
just in freshness. `desi_galaxies` has no partition when published and a
depth-2 BSP locally; `nuclear_pore_complex` is 41,288 elements published and
9,874,128 locally after a demo change. Label the source of every number, and
never put both in one table.
