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
demo (`SCENE_MAX_POINTS_PER_NODE`, and its comment says so). The conservative
4096-class per-node caps are per geometry type, in `typing_utils/constants.py`:

| geometry | cap |
|---|---:|
| Lines | 2,793,472 segments |
| Points | 5,591,040 points |
| GSplats | 4,194,304 splats |

DESI's 9,751,955 breaches the real 5,591,040 Points cap too, so that warning
was not a false positive. Comparing a points node against 4,000,000 over-flags
only the band from 4,000,000 to 5,591,040.

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

### 3.9 Quality figures from before 2026-08-23 use the wrong basis

`#1914` (2026-08-23) fixed five call sites that scored a fit against the **raw**
volume, when a fit reconstructs `V - image_min`. A figure stamped before that
date is invalid in a direction that depends on the volume's pedestal and cannot
be corrected by arithmetic. Archives rebuilt afterward can carry correct
figures.

Consequences for this site:

- **Check the archive root `timestamp` and the sidecar `quality_note` date**
  before using a figure in a page, note, or comparison. Re-measure pre-fix or
  unstamped figures with current code against one materialised reference volume.
- **A figure is only comparable to another measured on the same side of
  2026-08-23.** Comparing a pinned figure to a fresh one measures the scoring
  change, not the data.
- The public page is unaffected: no rendered field carries a quality figure
  (`gen_landing.py` never reads `note`). The exposure is in the manifest `note`
  fields, which are internal.

### 3.10 One path, two generations: hash before asserting

`demo_gsplats_2d_cmu1_pathology.py` used to say its hosted archives were "still
flat leaves" (corrected here; see **And the docstring** below). Two caches
yielded answers that could not both be true for
`cmu1_ch0.gsplats.zarr.zip` — flat laddered leaf with **no `kind` attr anywhere**,
and `{'partition': 1, 'lod': 64}` with 4,867 metadata docs.

**Both were right.** The manifest pins two generations of the same filename:

    cmu1_ch0   sha256         cd22645f...   bytes          38,201,205   <- in-repo copy
               hosted_sha256  29faffc1...   hosted_bytes   84,492,218   <- what the record serves

A cache holding the in-repo generation has all three channels matching `sha256`;
a cache holding the hosted generation has all three matching `hosted_sha256`.
Same path string, same demo, 2.2-2.75x apart in bytes and structurally unrelated.

`_support/datasets/data_fetch.py` documents why: the two contracts were one field
until a refit replaced the hosted artifact without touching the in-repo copy, and
splitting them is what kept Zenodo publication off the critical path of every
demo-data PR.

**The discriminator, and the only reliable one:** hash the file and match it
against `sha256` vs `hosted_sha256`. That names the generation in one command.
Byte size alone is suggestive; a digest is decisive. Neither measurement was
matched to a digest before its structural claim was made.

#### The consequence worth knowing

`data_fetch.py` states the resolution order plainly: *"while an in-repo payload
is present it wins over a newer hosted artifact, so a checkout with a stale LFS
object keeps serving the older generation (loudly)."*

So **the same demo builds a structurally different scene depending on which
generation the building machine has in cache.** cmu1 grafts its archives
verbatim: the in-repo generation produces a flat scene (12 element nodes,
20,591,415 elements — exactly the sum of the three archives), while the hosted
generation produces a partitioned, laddered one.

Three things follow for this site:

- **A tile's structure is a property of the artifact generation, not just the
  recipe.** Record the digest used to build a tile when its demo grafts pinned
  archives.
- **A tile can be accidentally correct.** cmu1's live tile is flat because the
  publishing machine held the stale in-repo generation — not because anything
  chose that.
- **The trigger is an LFS payload refresh** — which looks like routine
  housekeeping, touches no code, and changes no reviewable line. That is what
  makes this worth a runbook entry rather than a comment.

**But 3.11's rule narrows the exposure by authoring path.** Four of the fourteen
diverged datasets are loaded into `GSplatData` and re-added as raw arrays, so
their bytes and splat content can change but their archive topology is
discarded. Seven pass the `GSplatData` object to `add_gsplats_from_data`, which
preserves additive rungs and lowers multiple substitutive levels into a
`kind=lod` scene group. Three graft the artifact directly:

| authoring path | datasets |
|---|---|
| re-add raw arrays | `gsplats_kidney`, `gsplats_cells3d`, `gsplats_ct_totalsegmentator`, `gsplats_visible_human_head` |
| pass through `add_gsplats_from_data` | `gsplats_cryoem_virus`, `gsplats_milkyway_dust`, `gsplats_celegans`, `gsplats_dapi`, `gsplats_multichannel`, `gsplats_nexrad_supercell`, `gsplats_opencell_map4` |
| graft artifact | `gsplats_flylight_mcfo_63x`, `gsplats_cmu1_pathology`, `desi_galaxies` |

Digest-confirmed copies show structural divergence for two pass-through
datasets: `cryoem_virus` and `milkyway_dust` change from flat to four
substitutive levels with additive rungs. Of the grafted datasets, only
`cmu1_pathology` is digest-confirmed on both sides, so the confirmed armed set is
**three**. The hosted topology of `flylight_mcfo_63x`, `desi_galaxies`,
`celegans`, and `nexrad_supercell` remains unclassified; inspect a
digest-confirmed copy before counting any of them. Check the scene-build call as
well as the archive before treating a divergence as a structural risk.

#### Scope: this is not a cmu1 quirk

Measured across `data_manifest.json` — of 45 pinned file entries, 23 carry a
`hosted_sha256`, and **all 23 differ from their repo `sha256`**. Fourteen
datasets are affected:

| dataset | files | repo MB | hosted MB | ratio |
|---|---:|---:|---:|---:|
| `gsplats_cmu1_pathology` | 3 | 113.8 | 278.0 | 2.44x |
| `gsplats_cells3d` | 2 | 0.6 | 1.3 | 2.25x |
| `gsplats_cryoem_virus` | 1 | 11.1 | 16.0 | 1.44x |
| `gsplats_ct_totalsegmentator` | 2 | 7.2 | 10.2 | 1.42x |
| `gsplats_milkyway_dust` | 1 | 7.8 | 10.6 | 1.36x |
| `gsplats_visible_human_head` | 2 | 25.6 | 34.5 | 1.35x |
| `gsplats_nexrad_supercell` | 1 | 10.1 | 12.9 | 1.28x |
| `gsplats_dapi` | 1 | 0.1 | 0.1 | 1.22x |
| `gsplats_celegans` | 1 | 72.0 | 80.8 | 1.12x |
| `desi_galaxies` | 1 | 74.3 | 76.8 | 1.03x |
| `gsplats_kidney` | 3 | 2.1 | 2.1 | 0.99x |
| `gsplats_multichannel` | 2 | 0.4 | 0.4 | 0.95x |
| `gsplats_flylight_mcfo_63x` | 1 | 8.2 | 7.7 | 0.94x |
| `gsplats_opencell_map4` | 2 | 1.6 | 1.5 | 0.93x |

So the presence of `hosted_sha256` **is** the divergence signal — there is
currently no dataset carrying the field whose two generations agree.

**The ratio column predicts nothing about structure, in either direction.** It is
here to size the download, not the risk:

- `cells3d` at **2.25x** is flat -> flat. Only the splat count moved (20,323 vs
  41,975); both scenes have 2 element nodes.
- `cryoem_virus` at **1.44x** is flat -> `kind=lod` with 4 substitutive levels.
- `cmu1` at **2.44x** is flat leaf -> partition + 64 lod groups.

Ratios below 1.0 are refits that shrank, and they are not exempt either. Only a
kind-based read of a **digest-confirmed hosted copy** settles topology.

Practical consequence: for any of these fourteen, a local measurement describes
whichever generation that host cached. Hash it before attaching the result to a
generation.

#### And the docstring

`demo_gsplats_2d_cmu1_pathology.py:512-518` used to describe the *hosted*
archives as flat, even though they carry a partition and 64 lod groups per
channel. It now records that the in-repo flat generation wins while it remains
in the tree, despite the record already serving the partitioned generation.
Neither statement can be established from an in-repo measurement alone.

Derive topology from each group's declared `kind` (children of `kind=lod` are
substitutive levels, children of `kind=partition` are parts) rather than from
node-name patterns. And note that **no `kind` attr anywhere means flat, not
unreadable** — the in-repo cmu1 generation is well-formed with zero kinds.

### 3.11 Flattening only cuts requests when archive topology reaches the scene

Four authoring paths have different outcomes from the same archive change:

| authoring call | effect of flattening the archive |
|---|---|
| load into `GSplatData`, then `scene.add_gsplats(...)` | archive topology is discarded; bytes and splat content can still change |
| load into `GSplatData`, then `scene.add_gsplats_from_data(...)` | additive rungs and substitutive levels are lowered into the scene, so scene node count changes with the artifact |
| `add_gsplats_from_file` or `extract_shipped_scene` | grafted scene node count changes with the artifact |
| `save_with_lod(recipe=...)`, then graft that output | structure is authored locally by the recipe |

Applied to the current demo code:

- **`h2afva_timelapse`** and **`h2afva_stack`** graft, so flattening changes
  their request topology as well as their archive bytes.
- **`codex_pancreas`** fits locally through `save_with_lod(recipe="adaptive")`
  and grafts its own output, so its groups are authored rather than inherited.
  Its structure changes through recipe settings such as `max_elements`, not by
  flattening a separately supplied artifact.
- **`cmu1_pathology`** grafts verbatim, so it inherits whichever generation is
  in cache (3.10): flat from the in-repo copy, partitioned + laddered from the
  hosted one. Flattening the hosted archives is a real win *and* collapses that
  divergence — but measure the generation before claiming either.
- **`cryoem_virus`**, **`milkyway_dust`**, **`dapi`**, **`multichannel`**, and
  **`opencell_map4`** pass `GSplatData` through, so their digest-confirmed hosted
  ladders and levels become scene nodes rather than being flattened by the demo.

So **split the claim per demo** before promising a load win. "Fewer nodes" and
"fewer bytes" are different wins; only paths that preserve archive topology get
the first, and some demos are already correct.

### 3.12 Before flattening, compute the RESIDENT count — the total will mislead you

Section 7 gives the counting rule (only the resident slice counts, and its two
traps). This is the flatten-specific consequence, because flattening is where the
wrong number is most tempting: it collapses a tree into one node, and that node's
**total** is what the compiler prints.

Worked case. Flattening `h2afva_51tp` collapses 4,446 groups to 7 and yields one
node of 121,163,285 splats:

    node total, 51 timepoints   121,163,285   <- what ElementCapacityWarning prints
    resident slice, worst case    2,629,840   <- what the GPU commits
    cap (4096-class GPU)          4,194,304   -> 1.59x UNDER, fine

Read as a total that is 28.89x over cap and looks like a blocker. It is not: only
one of the 51 timepoints is ever resident. `_lod_policy.py` says so directly —
*"the compiler also warns on the node total rather than the resident slice, so
that warning is expected for a sliced nD node that satisfies the runtime limit."*
**An `ElementCapacityWarning` on a sliced nD node is not a finding.**

**The exception is a STATIC object**, which has no hidden axis to reduce the
committed set. The hosted `cmu1` generation (`hosted_sha256`; ch0
`29faffc1...`) flattens its three channels to 8,823,953 / 9,924,486 / 10,830,790
splats; the in-repo `sha256` generation flattens to 6,896,619 / 7,093,383 /
6,601,413. Both are 2D — nothing to slice on — and every channel exceeds the
cap, so the overflow shows as a Hilbert-contiguous clean-edged hole that reads
as missing data. There, parts stop being optional and become load-bearing.

Do not try to confirm that by opening the tile: it renders **whole** on a
developer Mac, because a Metal `maxTextureSize=16384` path caps at 16,777,216
rather than 4,194,304. A clean render on your own hardware carries no information
about the floor — see 3.16.

So the check before flattening is arithmetic, not a run: a `kind=lod` group
contributes only its finest child, a `kind=partition` sums its parts,
`additive_N` rungs are deltas that re-partition the level's own content, so never
add them on top of it — and equally, a ladder never *reduces* the leaf below its
own total, so it is not a remedy for an over-cap node (3.16) — then divide by the
hidden-axis extent if there is one. Only if the **resident** figure exceeds the
cap does the demo need `partition=dict(max_elements=…)` landing in the same
change as the flatten.

### 3.13 A ladder's `counts` are in different UNITS per geometry, and the wrong one writes zero rungs

On `add_lines`, an explicit `counts` list is in **polylines** while
`"stream:<c>"` is in **vertices**. Measured on 4,000 polylines x 27 vertices:

    counts=[39062, 78124, 108000]   -> clamped to the polyline count -> 0 RUNGS
    "stream:39062"                  -> 3 rungs (39,069 / 39,069 / 29,862 vertices)

The clamp is silent: no error, no warning, and a store with zero rungs still
loads and still renders. `check_demo_ladders.py` is the only thing that catches
it, and only above 200,000 elements — so a mid-sized demo can ship a ladder that
does not exist.

### 3.14 One viewport does not validate a substitutive ladder

A whole-object substitutive ladder anchors its finest level at **0.5 screen
occupancy**, and adding levels cannot move that anchor. So the *viewport aspect
ratio* — not the ladder — can decide whether a scene is bounded.

Measured on `cosmicflows_laniakea_full` at the authored pose: two substitutive
levels settled at ~892k segments on **16:9**, while the wide outer basins still
selected ~2.1M and ~3.2M **fine** segments at **4:3** and **1:1**.

A ladder that looks bounded on a 16:9 capture can therefore be unbounded on a
square window. Validate at several aspect ratios, or prefer an additive ladder
where the geometry allows one (not `indexed` lines — see Section 7.4), whose
prefix is bounded by construction rather than by framing.

### 3.15 A streaming ladder's first rung is first paint — size it in bytes, not chunks

desi's 2,000-element first rung (`SCENE_FIRST_CHUNK`,
`demo_desi_galaxies.py:207`) is sized so its eager coarsest **substitutive**
level lands in one zarr chunk. An additive-only leaf has no coarse level, so
**its first rung *is* first paint**, and 2,000 elements is far below a sensible
download budget.

Measured group counts before and after converting six embedding demos from
substitutive to additive:

| demo | today | at `stream:2000` | at `stream:39062` |
|---|---:|---:|---:|
| `mouse_multiome_peak_umap` | 13 | 12 | 7 |
| `esm3_protein_landscape` | 13 | 12 | 7 |
| `zebrahub_multiome_peak_umap` | 17 | 15 | 11 |
| `cellxgene_census_umap` | 18 | 14 | 9 |
| `human_multiome_peak_umap` | 17 | 17 | 13 |
| `arxiv_papers_kaggle` | 18 | 18 | 13 |

39,062 is a 200 ms budget at 25 Mbps and 16 B/element. At 2,000 every extra rung
is another node, so the ladder costs requests without buying a faster first
paint — the same accounting trap as counting nodes instead of bytes, one level
down.

### 3.16 State the window before you read the number, and check it is longer than the phenomenon

The most expensive error in this campaign was not a wrong mechanism. It was a
**phenomenon that did not exist**, and two mechanisms invented in later work to
explain it.

#### The false observation

`gsplats_2d_cmu1_pathology`'s live tile appeared to commit **10,295,708** of its
store's **20,591,415** elements — exactly 50.0%, per channel — with `isLoading`
stuck true, no console error, and half the data apparently unreachable. It was
called unpublishable, held from a wave, filed as an open problem, and carried
into later work as evidence for an unbounded-loop hazard.

#### What is actually true

    Progressive: 4/4 LODs loaded (7093383 splats) — complete   @ 63,442ms
    Progressive: 4/4 LODs loaded (6601413 splats) — complete   @ 64,183ms
    Progressive: 4/4 LODs loaded (6896619 splats) — complete   @ 67,864ms

    totalElements  20,591,415      isLoading  false
    progress lines 12 total, 0.18/s

Every channel reaches 4/4. The three sum to 20,591,415 — the whole store. It
completes in about 68 seconds. Nothing is truncated, nothing stalls, and at
0.18 lines/s nothing spins.

#### Why the number looked real

The gap between the last level-1 line (41,750 ms) and the first completion
(63,442 ms) is **21.7 seconds of silence**. The probe declared the total "settled"
after three identical samples at 5 s intervals — **15 seconds**.

**15 < 21.7.** The measurement stopped inside a quiet period mid-load and reported
a snapshot as a terminal state. `isLoading: true` was not a symptom; it was the
literal truth.

#### Why it survived so long

- **The artifact was stable and reproducible.** Two independent probe runs agreed
  to the unit, because both shared the 15 s settle rule. Reproducibility measured
  the rule, not the system.
- **It was quantitatively beautiful.** 50.0% in every channel, across three
  different leaf totals, matching a cap prediction exactly. That is what bought it
  credibility — and the exactness came from all three channels having 4 rungs, so
  any stop after level 1 yields 50% everywhere. Three "independent" confirmations
  were one constraint counted three times.
- **The search terms encoded the hypothesis.** The first grep set was
  `clamp|truncat|exceed|capacity|MAX_SPLATS`. The answer was in
  `Progressive: 4/4 … — complete`, which that set structurally could not match. A
  filter built from a theory can only ever confirm it.

#### The rules

1. **Fix the observation window before looking at the result, and justify it
   against the expected duration of the thing being measured.** Twelve rungs at
   the 10–26 s per level actually observed is minutes; a 15 s settle rule and even
   a 180 s cap are both inside that range.
2. **"Stopped changing" is not "finished."** Prefer a signal that matches the
   phase being measured: `— complete` for a full ladder, or `isLoading: false`
   only for the first committed view, rather than inferring either from a
   stationary number.
3. **A stable artifact is not a real effect.** If two runs agree, check they do not
   share a stopping rule, a cache, or a filter.
4. **Grep for what the code says, not for what you suspect.** Loaders, policies and
   compilers usually log their own reason; find that string first.

#### What genuinely survived

- **The load is slow for an ordinary reason**: the store is 20,591,415 splats,
  roughly **172 MB**, delivered in ~68 s — about **20 Mbps**. That is a large
  download, not a defect. A request-count explanation was considered and
  **measured false**: the published store has **508 chunks** with a nominal
  **~1024 KB uncompressed chunk shape** (the `archive` profile's 1 MB target),
  against **50,316 chunks at 1.0–2.7 KB** in the upstream `.gsplats.zarr`
  archives. `optimise --profile archive` does re-chunk
  grafted subtrees, so the archives' fragmentation never reaches a published tile.
  It does still hit whoever downloads those archives directly — a demo build pays
  38 MB in 16,852 pieces — which is an authoring-side fix worth making upstream.
- **The cap risk is real, and a clean render does not test it.** Each channel is
  6.6M–7.1M splats with no hidden axis. It renders whole on a 16384-class
  developer GPU — the probe launches `--use-angle=metal`, where `maxTextureSize`
  is 16384, giving a gsplats cap of `4096 x 16384 / 4 = 16,777,216`. 6.9M is
  comfortably under *that*.
  `constants.py` is explicit: *"maxTextureSize is a GPU property (16384 on modern
  desktop, 4096 on the conservative floor), so the only bound an AUTHOR can rely on
  is the 4096-class one."* So **a clean render on developer hardware is the
  expected observation and carries no information about the floor** — on a
  4096-class GPU the same node clamps and loses a Hilbert-contiguous wedge
  (#1957 erased the North Atlantic by clamping 2.3% of a Lines node). Never
  validate a cap question on one GPU class.
- **The authoring guard sees the accumulated quantity.**
  `warn_if_over_element_cap` runs once at the parent with the ladder total; rung
  checks are deliberately suppressed as redundant, and the graft path performs
  the same aggregate check. The 6.6M–7.1M CMU-1 channels therefore warn against
  the 4,194,304-splat conservative floor.
- **An additive ladder still never bounds the committed set** — a prefix converges
  to 100% of the leaf. Only a partition, or a hidden axis to slice on, reduces
  what is resident.

### 3.17 Dropping substitutive LODs without re-chunking leaves the store far slower than it needs to be

Removing substitutive levels is the right call for most single-object scenes, but
it is **half an operation**. The recipe is three steps, in this order:

    flatten  ->  lod --recipe stream  ->  optimise --profile archive

The ordering cost is independently measured on the Drosophila 500-timepoint
archive: requests per timepoint step fell from **173 to 2** after
`optimise --profile archive`
(`demo_gsplats_4d_drosophila_embryogenesis.py:163-165`).

The reason is not inherited source chunking: `flatten` and `lod` rewrite every
array at the 64 KB authoring target, discarding even an existing 1 MB layout.
Running `optimise` before either command is therefore undone, and a timepoint
slice again spans many small chunks. **Additive-only and re-chunking are a
package**, and `optimise` must run *last* so every rewritten array gets the 1 MB
layout.

The `h2afva_51tp` rebuild exposed two further traps:

- **Don't stop at `flatten`.** A bare flat leaf loses the ladder entirely; the
  target is a leaf *plus* rungs (`lod --recipe stream`), which is what
  `_lod_policy.py`'s `stream 4 nodes` row describes.
- **A partition can be worth nothing on a time-stacked node.** The writer already
  lexsorts by the time barrier, so per-timepoint chunk locality exists without any
  partition — adding one buys no request reduction there. (It still earns its place
  on a *static* node over the element cap, 3.12.)

Result on an unpublished rebuild of `h2afva_51tp` (the manifest still pins the
original partitioned generation): 1,873,559,527 → 1,115,714,088 bytes
(**−40.5%**), 176 substitutive levels → 0, 704 element nodes → 8 (one leaf plus
its rungs), 125,751 chunks → 2,316, with all 51 timepoints intact at uniform
spacing and none blended.

### 3.18 A probe must emit the evidence that its own window was valid

3.16 says to check the observation window is longer than the phenomenon. That is
advice, and advice does not run. Make it a **reported field** instead, so a
meaningless run announces itself.

Worked example from an ad-hoc browser-console probe; there is no checked-in script
to rerun. The depth-sort flashing bug lives only on *count-changed* commits;
equal-count re-commits take a path that always worked. Three runs against the same
live bundle reported:

    host A   firstCommitWaitMs  501                    commits 99   countChanged 99   distinctSteps 89   unsorted 0   -> FIXED
    host B   firstCommitWaitMs  (none — fixed 8 s warm-up)   commits 73   countChanged  0   distinctSteps  1   unsorted 0   -> INCONCLUSIVE
    host B   firstCommitWaitMs 1752                    commits 85   countChanged  0   distinctSteps  1   unsorted 0   -> INCONCLUSIVE

Host B's runs looked like passes on the headline numbers and were worth nothing:
zero denominators, with every observed commit an equal-count progressive re-commit
on a path that never had the bug.

The 1752 ms diagnostic did **not** refute a cold-cache explanation. In the viewer,
`isLoading === false` reports first-commit latency, not full-ladder completion, and
the debug interface is installed only after the initial `loadDataset` call returns.
Progressive refinement may continue afterwards; the 85 re-commits show that it did.
The number therefore bounds time to the debug interface and first committed view,
not time to a settled ladder, so a load-bound observation window remained plausible.

Open the viewer with `?debug`, keep the 180 s first-commit timeout separate from the
30 s observation window, and **publish both the wait and its exit reason**:

```js
const firstCommitTimeoutMs = 180000;
const observationWindowMs = 30000;
const tWait = Date.now();
let firstCommitObserved = false;
while (Date.now() - tWait < firstCommitTimeoutMs) {
  const st = window.__luxarDebug?.getState?.();
  if (st && st.isLoading === false) {
    firstCommitObserved = true;
    break;
  }
  await new Promise(r => setTimeout(r, 500));
}
window.__firstCommitWaitMs = Date.now() - tWait;
window.__firstCommitObserved = firstCommitObserved;
window.__observationWindowMs = observationWindowMs;
if (!firstCommitObserved) {
  throw new Error('first-commit timeout; probe inconclusive');
}
// Run the measurement for exactly observationWindowMs from here.
```

Rules:

- **Gate on a signal that matches the phase being measured, not a fixed sleep.**
  `isLoading: false` is suitable for first commit; full-ladder work needs the
  loader's `Progressive: n/n LODs loaded — complete` signal. A hardcoded warm-up
  is only an untested claim about the system's timescale.
- **Report the wait and the exit reason.** `firstCommitObserved: false` means the
  180 s cap expired and the run is inconclusive; the elapsed value alone cannot
  distinguish timeout from success. This timeout precedes the separate 30 s
  observation window, so do not compare one duration to the other.
- **Print the denominator next to every ratio.** `0 unsorted` and `0 of 0` render
  identically in a summary line and mean opposite things — one is a pass, the other
  is no measurement. A zero denominator is never a pass; emit an explicit
  `INCONCLUSIVE` verdict instead of a green one.
- **Assert that the driver ran, not just that the output looks clean.** The effect
  here lives on time-axis steps, so the probe must report how many *distinct*
  axis positions it observed. One host saw 89 distinct timepoints with counts
  ranging 10,229–30,296; another saw 85 commits at a single position, and only the
  step-count field distinguishes "the fix works" from "nothing was exercised".
  Where a probe depends on the system animating itself, have it detect that and
  drive the axis directly when it is not.

Note both failures here were the *same* mistake by different hands: a fixed 8 s
warm-up written by the session that had already documented that a stopping rule is
a claim about timescale, and a 15 s settle rule (3.16) written by the session that
had just relayed that lesson. Knowing the rule is not the control; emitting the
diagnostic is.

### 3.19 Node reduction cuts requests only when arrays fit in one chunk

Section 2's rule — hosted cost is **requests**, not bytes — is right, but "one
request per node" is an *upper bound*, not a measurement. After
`optimise --profile archive` (1 MB target) a big array spans many chunks while a
small one spans exactly one, so cutting node count only cuts fetches in one of two
regimes. The following counts come from the consolidated metadata at published
prefix `2026-08-27b`; `arrays` excludes zero-shaped `array_ref` placeholders,
`chunks` is the root `chunk_layout.chunks_after` value, and the eager columns apply
the viewer's `default_level` deferral rule in
`packages/luxar-viewer/src/data/scene-loader/nodes/load-lod-group-node.ts`:

| store | groups | arrays | chunks | eager arrays | eager chunks | whole chunks:arrays |
|---|---:|---:|---:|---:|---:|---:|
| `gsplats_2d_codex_pancreas` | 2767 | 10320 | 10320 | 3440 | 3440 | **1.00** |
| `desi_galaxies` | 87 | 320 | 389 | 80 | 80 | 1.22 |
| `biodiversity_planetary_scale` | 29 | 108 | 187 | 108 | 187 | 1.73 |
| `cosmicflows_laniakea_full` | 15 | 79 | 224 | 79 | 224 | 2.84 |
| `gsplats_2d_cmu1_pathology` | 18 | 60 | 508 | 60 | 508 | **8.47** |

Use the eager columns for a first-paint claim and the whole-store ratio for the
cost to reach full detail. They differ only where substitutive levels defer
non-default children.

Two regimes, and the ratio tells you which one you are in:

- **Node-bound (ratio ≈ 1).** Every eager array is a single chunk, so removing an
  eagerly loaded array removes approximately one first-paint fetch.
  `codex_pancreas` is exactly 1.00 both store-wide and for its eager subset; the
  viewer initially fetches 3,440 arrays/chunks, not all 10,320. The 63-request
  stacked-leaf versus 689-request partition measurement in
  `packages/luxar/src/luxar/demos/_lod_policy.py` is not a reusable sublinear
  node-to-request law: it compares a byte-bound leaf with a node-bound partition.
- **Byte-bound (ratio >> 1).** Arrays span many chunks, so request count tracks
  total bytes and is nearly indifferent to node count. `cmu1` fetches 508 chunks
  from 60 arrays; halving its node count would barely move that.

**The ratio is a property of a pipeline STAGE, not of a store.** `optimise` is
what moves these stores toward the node-bound regime. The same published roots
record both pipeline stages in `chunk_layout`, so this comparison is reproducible
from their `zarr.json` files without a separate local build:

| store | arrays | chunks before | before ratio | chunks after | after ratio |
|---|---:|---:|---:|---:|---:|
| `gsplats_2d_codex_pancreas` | 10320 | 19888 | 1.93 | 10320 | **1.00** |
| `desi_galaxies` | 320 | 8600 | 26.88 | 389 | 1.22 |
| `biodiversity_planetary_scale` | 108 | 2081 | 19.27 | 187 | 1.73 |
| `cosmicflows_laniakea_full` | 79 | 5004 | 63.34 | 224 | 2.84 |
| `gsplats_2d_cmu1_pathology` | 60 | 7580 | 126.33 | 508 | 8.47 |

`biodiversity_planetary_scale` therefore reads **19.27 before optimise against
1.73 after it** — same generation and structure, but a factor of 11 fewer chunks
per physical array because optimise re-chunks to a 1 MB target. This is consistent
with 3.17's measurement in the other direction (173 requests as-built, 2 after
optimise).

So a node reduction's payoff is **contingent on the publish step**: optimisation
can move a small-array store into the node-bound regime, but it does not guarantee
that outcome. Halving eager arrays is a direct request win on a node-bound
published artefact and close to meaningless on a byte-bound as-built one. The win
belongs to the combination, not to the authoring change alone.

Practical consequence: **before claiming a node reduction buys a faster load, check
the chunks:arrays ratio of the artefact you will actually serve** — post-optimise,
and of the generation you are publishing, not whichever one happens to be live. A
store with many small nodes gains directly; a store with few large ones gains
almost nothing and its lever is total bytes instead (3.17).

Corollary for the other direction: adding nodes is only expensive in the
node-bound regime. A per-part additive ladder that multiplies groups is cheap on a
byte-bound store and costly on a node-bound one — so the same structural change
has opposite cost depending on chunk layout.

#### Worked example: is a per-part ladder worth its nodes?

This was an ad hoc local full-data experiment on 2026-08-28, based on
`demo_nuclear_pore_complex.py`; neither the probe nor its output was checked in.
It is not the published `2026-08-27b` preview, which has 41,288 points in one
unpartitioned node. The local experiment used 9,874,128 elements and a 32-part
BSP, with both variants taken through `optimise --profile archive`:

    version        groups  arrays  chunks   first commit
    un-laddered        36     160     224    9,874,128 elements
    4 rungs/part      161     640     672    ~1,250,000 elements

Post-optimise chunks:arrays is **1.40** un-laddered and **1.05** laddered, so both
variants are node-bound and the 128 rung groups cost real fetches: **+448
requests** to reach full detail, 3x the un-laddered total.

But that is the wrong total to compare. First paint needs only the **first rung**
of each part — roughly a quarter of the arrays, ~128-160 chunks — against **all
224** for the un-laddered store, which must also commit 9.87M elements in one go.
So the ladder is cheaper at first paint on *both* axes and more expensive only in
the total to reach full detail, which arrives progressively and off the critical
path.

For a gallery tile that is the right trade. State it that way round: a ladder does
not reduce total requests, it moves them after first paint.

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

Two traps inside that:

- Measuring **one part** under-reports by the part count — a first pass read
  `nuclear_pore_complex` at 164,633 when it is 4,937,064 resident per state, a
  30x error, because the path measured was a single `part_N`.
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
raise, not warn, when an explicit request cannot be honoured. Track that writer
fix in #2320.

### 7.5 Compare like with like

Published and local copies of the same store can differ **structurally**, not
just in freshness. `desi_galaxies` has no partition when published and a
depth-2 BSP locally; `nuclear_pore_complex` is 41,288 elements published and
9,874,128 across both local states after a demo change. Label the source of every
number, and never put both in one table.
