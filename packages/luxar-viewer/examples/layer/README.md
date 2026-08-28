# LuxarLayer host example

This page is a minimal host application: it creates and owns the Three.js renderer, scene,
camera, controls, canvas, resize handling, context recovery, and render loop. `LuxarLayer` only
loads Luxar geometry into that scene and performs its required per-frame bookkeeping.

Run it from `packages/luxar-viewer`:

```bash
pnpm test:generate-fixtures
pnpm dev
```

Open `http://127.0.0.1:5173/examples/layer/`. The default dataset is served by the repository's
Playwright data server on port 9000; when running manually, start it from the repository root:

```bash
python3 packages/luxar-viewer/tools/range-http-server.py 9000 --bind 127.0.0.1
```

Pass another scene with `?src=<absolute-url>`. In an installed application, replace the relative
source-barrel import in `layer.js` with:

```js
import { LuxarLayer } from '@royerlab/luxar-viewer';
```

The important ordering is visible in `animate()`: `layer.update()` runs before
`renderer.render(scene, camera)` on every frame.
