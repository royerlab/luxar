# Demo Site Runbook

How the public Luxar demo gallery and viewer are hosted, and how to publish a
new wave of demos to them without breaking anything.

This is an operational document. It records the architecture, the publish
sequence, and — most importantly — the failure modes that produce a *plausible
wrong answer* rather than an error. No credentials appear here; they live in
`~/.config/luxar-r2/credentials.env` on the operator's machine.

---

## 1. Architecture

Three hostnames on the `luxarviewer.dev` zone, each serving a different thing:

| Hostname | Serves | Backed by |
|---|---|---|
| `luxarviewer.dev` | The viewer alone, at the root | Cloudflare Pages project `luxar-viewer` |
| `demos.luxarviewer.dev` | The gallery page + its embedded viewer | Cloudflare Pages project `luxar-demos` |
| `data.luxarviewer.dev` | The `.luxar.zarr` stores | Cloudflare R2 bucket `luxar-demos`, **direct** |

Two properties of that split matter operationally:

- **`luxar-viewer` has no `functions/` directory and no R2 binding.** It is
  static assets only. This is deliberate: Pages Functions are billed per
  invocation, and the viewer does not need one.
- **`data.luxarviewer.dev` is an R2 custom domain, so it bypasses Pages
  Functions entirely.** A request there never runs a Worker. The way to confirm
  this is the *absence* of the `x-luxar-fn: r2` marker header that the gallery's
  Function sets:

  ```bash
  curl -sI https://data.luxarviewer.dev/data/<prefix>/<store>.luxar.zarr/zarr.json | grep -i x-luxar-fn
  # no output = R2 direct, Function bypassed (this is what you want)
  ```

A verify-everything sweep of the whole chain lives in the operator's harness;
its essentials are reproduced in §5.

### 1.1 The viewer takes an absolute `src`

`https://luxarviewer.dev/?src=<absolute-url>` opens any store the browser can
reach, which is the point of hosting the viewer at the apex — it is a general
tool, not a demo appendage. That requires CORS on whatever origin holds the
data (§4.3). The gallery's own embedded viewer keeps *relative* `src=/data/...`
paths, because there the data and page are same-origin by construction.

---

## 2. Publishing a wave

A "wave" is: some demos changed on `dev`, rebuild them, and update the site.

```
build the changed demos
  -> capture gallery media (stills + orbit videos)
  -> luxar optimise --profile archive        # 1 MB chunk target
  -> hash-compare against the LAST LOCAL BUILD
  -> upload only what changed, to a NEW dated prefix
  -> bump CACHE_EPOCH, rebuild the page, deploy
  -> audit the live site
  -> purge the superseded prefix
```

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
indistinguishable from a successful no-op rebuild unless you also check mtime.
This caught `cell_tracking_challenge` reporting IDENTICAL at 64 hours old.

Always assert freshness alongside the hash.

---

## 3. Hazards that fail silently

Every item here produced a plausible wrong answer in production rather than an
error. They are grouped by what lies to you.

### 3.1 `CACHE_EPOCH` — stale media

Scene URLs are dated; **media URLs are not**. Re-uploading a still or video to
the same key leaves every edge and browser serving the old bytes. The gallery
Function (`pages/functions/_r2.js`) carries a `CACHE_EPOCH` constant appended
as a cache-buster; **bump it on any wave that re-captures media**. Symptom if
you forget: the page is correct, the data is correct, and the thumbnails are
last week's.

### 3.2 A cache policy change does not reach cached objects

Changing R2 CORS, or any response-header policy, affects **only objects fetched
after the change**. Anything already in the edge cache keeps serving the old
response until its TTL expires. With a 30-day TTL that is a month of breakage
that looks fine on every fresh test you run.

Test the *cached* path, not a cache-busted one:

```bash
curl -s -o /dev/null "$URL"                                   # warm it
curl -sI -H "Origin: https://luxarviewer.dev" "$URL" \
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

### 3.7 `gh issue comment --edit-last` is account-scoped

On a shared account it edits the last comment made by the *account*, not by
your session — one agent overwrote another's comment this way. Recover with
`gh api -X PATCH /repos/:owner/:repo/issues/comments/<id>`. For the same reason
`gh pr list --author @me` returns the whole fleet's PRs, not yours.

---

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

`/media/*` URLs are **not** dated. If media ever moves behind the same
long-TTL rule it needs either a short TTL or hashed filenames first, otherwise
§3.1 becomes unfixable without a full purge.

### 4.3 CORS on the R2 bucket

Required for the apex viewer to read cross-origin. `AllowedHeaders` **must**
include `range` — the viewer issues partial reads, and without it every chunk
fetch fails preflight while `zarr.json` appears to work fine.

Verify with a real preflight, not a GET carrying an `Origin` header (curl does
not enforce CORS; browsers do):

```bash
curl -sI -X OPTIONS -H "Origin: https://luxarviewer.dev" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: range" "$URL"
# want: 204, access-control-allow-headers including "range"
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
research and must never reach the bucket. The publish script carries both an
exclusion list and a hard abort guard that fails the run if their media appears
in the upload set. Keep both — the list alone has no teeth.

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
