# dataset-browser

Private helpers for `../dataset-browser.ts`. The browser itself is a
DOM panel that wires a [`DirectoryNavigator`](../../data/nav/README.md)
to a breadcrumb + entry list; this folder holds the pure pieces that
don't need a DOM or a navigator to test.

## Contents

| File           | Purpose                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| `url-utils.ts` | `extractBaseUrl(url, origin)` / `extractPath(url)` — pure URL parsing for the browser's initial state |

## Public surface

```typescript
export function extractBaseUrl(url: string, origin: string): string;
export function extractPath(url: string): string;
```

`extractBaseUrl` computes the directory to list when the browser opens:
empty input falls back to `${origin}/`; a URL whose path ends in
`.zarr` (with or without a trailing slash) is trimmed to its parent
directory so the listing shows the dataset's siblings; anything else
returns `origin + pathname` unchanged. Parse failures return the input
verbatim — callers don't need a try/catch.

`extractPath` returns the relative dataset path (everything up to and
including the first `.zarr` segment) so the browser can pre-populate
its breadcrumb when re-opened on an already-loaded dataset. Returns
`''` for empty input, parse failures, or URLs with no `.zarr` segment.

## Consumers

- `../dataset-browser.ts` — imports both functions and wraps them in
  private methods of the same name that bind `this.origin`.
- `src/tests/unit/ui/dataset-browser/url-utils.test.ts` — unit coverage
  for both helpers across the documented cases (empty, trailing-slash,
  no-`.zarr`, parse failure).

## See also

- [`../README.md`](../README.md) — UI package overview, dataset
  browser section.
- [`../../data/nav/README.md`](../../data/nav/README.md) — the
  `DirectoryNavigator` that the browser drives once a base URL is
  resolved.
