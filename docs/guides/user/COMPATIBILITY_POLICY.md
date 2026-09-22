# Compatibility & Deprecation Policy

What Luxar promises to keep working, for how long, and how it tells you when
something is about to change. This page is the contract; the format-level detail
lives in [Formats & Migration](./FORMAT_AND_MIGRATION.md), and the day-to-day
mechanics for contributors are in `CONTRIBUTING.md` ("Deprecations and renames").

Everything below applies **from the first PyPI / npm release onwards**. Before it
there are no users to protect, so a rename is a hard cut with no alias — the
"no backwards compatibility burden" doctrine. Two pre-release renames are the
precedent for that style: the LOD recipe names (`additive` → `stream`, …) error
with a did-you-mean pointer (`LEGACY_RECIPE_NAMES` in `luxar.gsplats.lod.recipes`),
and the renamed `--method` flags are rejected the same way
(`_reject_renamed_method_flags` in `luxar.cli.lod`). Neither will ever gain a
soft alias: they never shipped. Post-release renames follow the window and the
mechanisms described here instead.

## What is public

The policy covers exactly these surfaces. Anything not listed — a private module
(`_`-prefixed), an undocumented attribute, the shape of a log line, a test
fixture — may change in any release without notice.

| Surface | What counts as public |
|---|---|
| Python package `luxar` | Every name in the top-level `luxar.__all__`, and every module documented in the API reference (`docs/api/*.rst`): its `__all__` where it declares one, its documented functions and classes otherwise. Keyword argument **names** are part of a function's public signature. |
| CLI `luxar` | Every command and sub-command, every non-hidden flag and its accepted values, exit statuses, and the on-disk artefacts a command writes. Human-readable output (tables, narration) is **not** stable. |
| npm package `@luxar/viewer` | The runtime and type exports of the package barrel (`packages/luxar-viewer/src/index.ts`); the `StorageKeys` registry (the `localStorage` keys a host page may read or clear); the URL query parameters the standalone viewer reads (`readUrlParams`); and the events `LuxarApp` emits to its embedder (`app.on(...)`). |
| On-disk formats | `.luxar.zarr` scenes and `.gsplats.zarr` stores, identified by their `format_version` (see [the version-check rule](#the-version-check-rule)). The exhaustive field-level specs are [Luxar zarr format](./LUXAR_ZARR_FORMAT.md) and [GSplats zarr format](../../specs/GSPLATS_ZARR_FORMAT.md). |

## Versioning: CalVer, and its npm spelling

Luxar releases are dated. The Python package version is **CalVer**,
`YYYY.MM.DD` with zero-padded month and day (`2026.06.05`). npm requires semver,
which forbids leading zeros, so the viewer package carries the **same date with
the padding stripped** (`2026.6.5`). They are one release under two spellings;
`scripts/check_version_consistency.py` (`hatch run check-versions`) fails the build
if the two ever name different dates.

Because the version is a date, it carries **no compatibility signal by itself**.
A dated version does not tell you whether an upgrade is breaking — this page,
the changelog title prefixes below, and the deprecation notices do.

## The version-check rule

Both on-disk formats carry a version, and every reader — the Python
`LuxarScene` / `load_gsplats` and the viewer — applies **one rule** to it. The
current and supported versions are single-sourced in
[`format-contract/contract.yaml`](../../../format-contract/contract.yaml) and
projected into Python and TypeScript, so the writer and every reader agree on
the vocabulary.

| The store's version is … | The reader … |
|---|---|
| in the **supported** set | loads it silently. |
| **same major, newer minor** than anything supported (a `0.3` scene read by a `0.2` reader; a `3.5` gsplats store read by a `3.4` reader) | **warns and loads**. A minor bump is additive by definition, so an older reader is expected to render the parts it understands. |
| anything else — an **older** version that has dropped out of the supported set, a **newer major**, or a value that does not parse as `major.minor` | **refuses**, naming the version it found and the fix (for gsplats: `luxar gsplat migrate-format <in> <out>`). |

The same table holds for scenes and for gsplats, in Python and in the viewer.
A missing version is refused when the store declares a `format_type` (it is a
Luxar store that has lost its header) and tolerated when it does not (a plain
zarr group that was never a Luxar store).

Consequences for authors of a format change:

- **Minor bump** (`0.2` → `0.3`): only additive changes — new optional attrs, new
  node kinds an old reader may skip. An older reader must still load the store.
- **Major bump**: anything an older reader would misread. Ship a migration
  (`luxar gsplat migrate-format` is the model) in the same release.
- Widening the supported set is free; **narrowing it** (dropping an old version
  from `supported`) is a removal and follows the deprecation window below.

## The deprecation window

A public name that changes keeps working, with a notice, for **two releases or
six months, whichever is longer**, counted from the release that first shipped
the notice. Only after both have elapsed may the old spelling be removed. The
notice always states the release the deprecation started in and the earliest
point it may be removed, so nobody has to look the window up:

```text
optimise is deprecated since Luxar 2026.10.01 and will be removed after 2027.04.01; use optimize instead.
```

Removing a feature outright (no replacement) follows the same window, with the
"use … instead" clause dropped.

### Mechanisms

Each surface has one way to say "deprecated", so every notice reads the same:

| Surface | Mechanism |
|---|---|
| Python function, class, module, or attribute | The old name stays importable as a thin alias that calls `luxar.utils.warn_deprecated(old, new, since=…, remove_after=…)` and forwards. The category is `DeprecationWarning`, attributed to the *caller's* line, so `-W error` and `pytest.warns` both see it. |
| Python keyword argument | `luxar.utils.deprecated_kwarg_alias(kwargs, old, new, …)` at the top of the function body moves the old key onto the new one and warns; passing both spellings is a `TypeError`. |
| CLI flag | The old flag stays declared as a `hidden=True` typer option defaulting to `None` (so it leaves `--help`), and `luxar.cli.utils.deprecated_option(value, old_flag, new_flag, …)` prints the notice to **stderr** — a command-line process hides `DeprecationWarning` by default, so the CLI does not rely on it. |
| CLI command or sub-command | The old command stays registered, hidden, and prints the same stderr notice before delegating. |
| Viewer URL parameter | The old key is added to `URL_PARAM_ALIASES` in `packages/luxar-viewer/src/config/url-params.ts`, mapping to its current spelling; the parser resolves it and logs one deprecation warning per key per session. |
| Viewer barrel export | The old export stays, re-exporting the new symbol, with a `@deprecated` JSDoc tag naming the replacement and the window. |
| Viewer `StorageKeys` entry | Not automated. A renamed key is read under its old name once, written back under the new one, and the old entry removed — a one-shot migration in the module that owns the key, documented beside the `StorageKeys` registry. |

### What is frozen

Some things are not deprecated and replaced — they are never changed at all,
because a change would silently invalidate data or caches already in the wild:

- **`content_hash` inputs.** The digest the compiler stamps on every group (and
  `luxar optimize` re-stamps) is a post-order xxhash64 over the zarr tree: the
  chunk **bytes**, the codec **identifiers** and their configuration, the array
  metadata, the node attributes except the hash itself and the software version
  that wrote it, and the plain payload files a node declares. Before hashing,
  every finite float nested in a hashed group attribute document is persisted
  at 12 significant decimal digits. That write-time canonicalization absorbs
  platform drift below roughly 1e-12 relative without changing the frozen input
  set. Exact array encoding attributes remain the producer's responsibility:
  companded rails are stored at float32 precision, covariance certificates at
  four significant digits, and fitted coordinate-grid steps at twelve. Larger
  differences remain producer defects rather than something the hasher hides.
  Published stores and their warm caches are untouched until a scene is
  recompiled and republished. The inputs and canonicalization rule are frozen
  from that point onward. The viewer validates its persistent cache against this
  hash, so a later change to either would either serve stale chunks or discard
  every warm cache on earth.
  (Re-chunking a store *does* change its hash — different chunk keys cover
  different rows — which is why `luxar optimize` tells you to publish under a
  new URL prefix.)
- **The legacy scene header key.** Published, immutable scene records carry
  `luxar_version: "0.1"`. Readers keep accepting that spelling for the 0.1
  format forever; it is the one deprecation with no removal date.

## Viewer persistence: storage-format bumps reset, never migrate

The viewer persists three kinds of state in the browser, each stamped with its
own integer version. When a reader meets a different version it **logs one line
and falls back to defaults**; it does not attempt to migrate the old shape:

| State | Version constant | On mismatch |
|---|---|---|
| User settings (`luxar.settings`) | `SETTINGS_VERSION` in `src/config/user-settings.ts` | defaults |
| Per-scene rendering settings (`luxar.rendering.<scene-id>`) | `RENDERING_SETTINGS_VERSION` in `src/ui/rendering-controls/settings-persistence.ts` | defaults (the scene's own `viewer_config` applies again) |
| OPFS chunk cache | `OPFS_ENCODING_VERSION` in `src/cache/types.ts` | the cache is dropped and refilled |

Bumping any of these is therefore cheap for the code and mildly annoying for the
user (a re-download, a re-tweaked slider), which is the intended trade: settings
and caches are reproducible from the scene, so a migration would be effort spent
protecting state that costs nothing to rebuild. Bump the constant whenever the
stored shape changes incompatibly and say so in the changelog.

## The `three` peer dependency

`@luxar/viewer` declares `three` as a **peer** dependency pinned with a tilde
range (`~0.185.1`): patch updates flow automatically, **minor** updates never do,
because Three.js ships behaviour changes on minors. A minor bump is a deliberate
release event — the triggers, the by-hand edit to both `package.json` ranges, and
the verification sequence are in `packages/luxar-viewer/THREE_VERSION_NOTES.md`
under "When to bump" (a package file, kept beside the pin it describes rather
than in this docs tree). Host pages must install a `three` inside that range; the
supported range is stated in the viewer's `package.json` and in the release notes
of any release that moves it.

## How changes are announced

Every changelog fragment (`changelog.d/`) that touches this policy starts its
title with a fixed prefix, so the release notes can be scanned for compatibility
events without reading every entry:

- `Deprecated:` — a name now emits the notice above; the old spelling still
  works until the stated removal point.
- `Removed:` — a deprecation window has closed and the old spelling is gone.
- `Breaking:` — an incompatible change that could not be given a window
  (a format major bump, a frozen-input correction). These are rare and each one
  says what to do.

See `changelog.d/README.md` for the fragment format.
