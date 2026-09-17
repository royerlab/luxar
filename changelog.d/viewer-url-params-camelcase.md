#### Viewer URL parameters are camelCase, and the viewer knows its own version

The query grammar mixed kebab-case flags (`?no-cache`, `?lod-finest`,
`?webgpu-force-webgl`) with camelCase values (`?cacheBudgetMB`, `?depthSort`).
Links outlive documentation, so the convention is settled before the first
release: every parameter is camelCase, and the twenty kebab spellings retire
as a hard cut with no aliases (`noCache`, `noSliceCache`, `noOpfs`,
`cacheDebug`, `clearCache`, `noLodFade`, `noLinks`, `noLodEnergy`,
`lodFinest`, `noBlendWarmup`, `noPrefetch`, `prefetchDebug`, `cacheStats`,
`webgpuForceWebgl`, `perfTimestamp`, `lodBias`, `noDensityGuard`,
`densityCap`, `bakeEnv`, `envResolution` — the last two are also what the
Python environment-bake driver emits). The wire spellings live in one exported
`URL_PARAM_KEYS` table; `URL_PARAM_ALIASES` with alias-aware `hasParam` /
`getParam` (one console warning per deprecated alias per page load) is the
mechanism a future rename will use, empty today. A parity test requires every
key to be documented in the viewer README and the viewer guide, and a
convention test rejects any kebab value outright.

The viewer also reports a real version. `__LUXAR_VIEWER_VERSION__` is defined
from `package.json` in the app, library and vitest builds and exported as
`VIEWER_VERSION` from the barrel; `window.__luxarDebug.version` reads it
instead of the literal `'1.0.0'` it used to carry. The public export list is
now one file, `scripts/public-api-exports.json`, that both the barrel test and
the post-build `check-lib-exports` assert against with exact equality, so an
export can neither disappear from the npm bundle nor appear unannounced (the
build check used to guard five of fourteen).
