# Directory navigation

Server-agnostic directory browsing for the dataset picker. Wraps a
remote HTTP root and probes it with a small chain of detection
strategies, so the same UI works against WebDAV servers, Apache/nginx
HTML listings, the JSON listings emitted by `luxar serve`, and static
hosts that ship a hand-written `.luxar-index.json` manifest.

The single source file `directory-navigator.ts` exports the
`DirectoryNavigator` class and the `DirectoryEntry` /
`NavigationResult` types — no submodules.

## Public surface

```typescript
export interface DirectoryEntry {
  name: string;
  path: string;
  type: 'file' | 'directory' | 'zarr';
  size?: number;
  modified?: Date;
}

export interface NavigationResult {
  entries: DirectoryEntry[];
  currentPath: string;
  parentPath?: string;
  strategy: 'webdav' | 'html' | 'index' | 'manual';
  isZarr: boolean;
}

export class DirectoryNavigator {
  constructor(baseUrl: string);
  navigate(path?: string): Promise<NavigationResult>;
  getFullUrl(path: string): string;
  canListDirectories(): Promise<boolean>;
}
```

Re-exported from `data/index.ts` as `DirectoryNavigator`,
`DirectoryEntry`, and `NavigationResult`.

## Strategy chain

`navigate(path)` walks four probes in order and returns on the first
match; if none succeed it falls back to `'manual'` so the caller can
prompt for a path:

1. **`'webdav'` (Zarr short-circuit)** — `HEAD` for `<url>/.zgroup`.
   When the path is itself a Zarr dataset, returns empty `entries`
   with `isZarr: true`.
2. **`'webdav'` (listing)** — `PROPFIND` with `Depth: 1`. Parses
   `DAV:response` elements; `.zarr` + `collection` → `type: 'zarr'`.
3. **`'html'`** — first negotiates JSON (`Accept: application/json`)
   to detect `luxar serve`'s `{ entries: [...] }` payload, then falls
   back to HTML scraping of three patterns: nginx `<pre><a>`,
   Apache/IIS `<tr>` rows, generic `<li><a>`. Deduped by name.
4. **`'index'`** — `GET` for `<url>/.luxar-index.json`. Uses each
   entry's explicit `type`, else infers from `.zarr` or `isDirectory`.
5. **`'manual'`** — fallback. Empty entries.

Each probe is wrapped in a 10-second `AbortController`
(`FETCH_TIMEOUT`); failures silently fall through to the next.

## Invariants

- **Trailing slash on `baseUrl`.** The constructor normalises it;
  callers can pass either form.
- **`currentPath` is internal state.** `navigate` updates it before
  probing so the listing parsers can compose child paths as
  `${currentPath}/${name}`. Two concurrent `navigate` calls on the
  same instance race — callers must serialise per instance.
- **`.zarr` suffix is the Zarr signal in listings.** Folders ending in
  `.zarr` are surfaced as `type: 'zarr'`; the deeper `.zgroup` HEAD
  probe runs only for the top-level Zarr short-circuit.
- **`'..'` is never an entry.** All listing parsers skip the parent
  link; navigation up the tree goes through `NavigationResult.parentPath`.
- **`canListDirectories()` is a feature probe.** Returns true unless
  every strategy fell through to `'manual'` with empty entries.

## Consumers

- `data/index.ts` — re-exports the class and types.
- `ui/dataset-browser.ts` — constructs one navigator per browser
  session, driven by the URL bar and breadcrumb controls.
- `tests/unit/data/nav/directory-navigator.test.ts` and
  `tests/unit/ui/dataset-browser.test.ts` — unit coverage, mock
  `fetch` per strategy.

## See also

- `../README.md` — data-package overview.
- `../../ui/dataset-browser.ts` — the only production caller.
