# Embed Demo

Minimal example showing how to mount the Luxar viewer inside a third-party
host page. The page intentionally has its own non-trivial styling (serif
font, beige background, custom margins, bulleted list) so that any
regression on the embed-isolation contract is visible at a glance.

## What this proves

1. **CSS isolation** — `luxar-viewer/styles.css` adds component styles for
   the viewer's UI overlays but does not touch `body`, `html`, `*`, lists,
   buttons, or scrollbars. The host page's styling is intact.
2. **No console hijack** — `import { LuxarApp } from 'luxar-viewer'` is
   side-effect-free: it does not monkey-patch `console.*`. Only an
   explicit call to `consoleInterceptor.patch()` (which embedders almost
   never want) does that.
3. **Clean dispose** — `app.dispose()` removes all viewer DOM, listeners,
   GPU resources, and CSS variables. The host page can mount and unmount
   the viewer freely.

## Run

From `packages/luxar-viewer`:

```bash
pnpm build:lib                       # builds dist/lib/{luxar-viewer.js,.css,types}
python -m http.server 8765           # serve from packages/luxar-viewer, NOT examples/embed
open http://localhost:8765/examples/embed/
```

Serve from `packages/luxar-viewer` (not from this directory): `index.html`
loads the built library via `../../dist/lib/luxar-viewer.{js,css}`, so the
HTTP root must be the package root for those paths to resolve.

(The page uses an `<script type="importmap">` to point `three` at jsdelivr
so it works without npm-resolution. In a real app you install `three`
from npm and let your bundler resolve it.)

The page also wires two buttons — **Dispose viewer** and **Re-initialize**
— so you can mount, tear down, and remount the viewer by hand and watch
the host page survive each cycle.

## The minimal embedding code

```html
<canvas id="luxar-canvas"></canvas>
<script type="importmap">
  { "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js" } }
</script>
<link rel="stylesheet" href="/path/to/luxar-viewer.css" />
<script type="module">
  import { LuxarApp } from '/path/to/luxar-viewer.js';

  const app = new LuxarApp();
  await app.init({
    canvas: document.getElementById('luxar-canvas'),
    src: 'https://example.com/data.zarr',
    updateBrowserUrl: false,   // don't rewrite the host URL on dataset change
  });

  // Later:
  // app.dispose();
</script>
```

### Via a bundler (npm)

In a real app you install from npm and let your bundler resolve the
specifiers — no importmap or `/path/to/` URLs:

```ts
import { LuxarApp } from 'luxar-viewer';
import 'luxar-viewer/styles.css';   // component styles, scoped under .luxar-*

const app = new LuxarApp();
await app.init({ canvas, src: 'https://example.com/data.zarr', updateBrowserUrl: false });
```

## Tested embed surface

| Option           | Type             | Default | Why an embedder cares                                                 |
| ---------------- | ---------------- | ------- | --------------------------------------------------------------------- |
| `canvas`         | HTMLCanvasElement| —       | The canvas the viewer renders into. Required.                         |
| `container`      | HTMLElement      | `document.body` | Host element all overlays/panels/toasts mount into. A non-`body` container is made a containing block so fixed overlays scope to it; restored on `dispose()`. This demo passes the framed box. |
| `src`            | string           | config  | Initial Zarr URL.                                                     |
| `debug`          | boolean          | false   | Exposes `window.__luxarDebug` for Playwright / dev console.           |
| `loaderConfig`   | LoaderConfig     | —       | Cache and prefetch flags.                                             |
| `updateBrowserUrl` | boolean        | false   | Already off by default for embed safety; the standalone bootstrap opts in to `true`. |
| `wasmPath`       | string           | —       | Override for bundlers that don't resolve `import.meta.url` for WASM.  |
| `workerPath`     | string           | —       | Same, but for the data worker.                                        |

## Out of scope

Multi-instance (multiple viewers on the same host page), `:root`-scoped
themes, Shadow DOM wrappers, and React/web-component bindings are not
part of the v1 embed.
