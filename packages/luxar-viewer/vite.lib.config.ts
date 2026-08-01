/**
 * Vite library build configuration for the publishable Luxar Viewer package.
 *
 * Builds `src/index.ts` (the side-effect-free public barrel) as an ES module
 * with `three` externalized as a peer dependency. Run via `pnpm build:lib`.
 *
 * - `vite.config.ts` (default) builds the standalone HTML app from
 *   `index.html` and is what `pnpm build` / `pnpm dev` use.
 * - `vite.lib.config.ts` (this file) builds the npm-publishable package and
 *   is what `pnpm build:lib` uses.
 *
 * Type declarations are emitted by a separate `tsc --emitDeclarationOnly`
 * step (see `tsconfig.lib.json`) — chained from `pnpm build:lib`.
 */

import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Library consumers control their own deployment path; emit relative URLs
  // so workers/assets resolve via `import.meta.url` rather than from the
  // document root.
  base: './',
  build: {
    outDir: 'dist/lib',
    emptyOutDir: true,
    target: 'esnext',

    lib: {
      // Two entries:
      //  - `luxar-viewer.js`        — the public barrel (no CSS imports
      //    — importing it must stay side-effect-free, see
      //    tests/unit/api/barrel-side-effects.test.ts).
      //  - `luxar-viewer-styles.js` — a tiny shim whose only purpose is
      //    to pull in `styles/index.css` so Vite emits a single
      //    `luxar-viewer.css` sidecar for consumers to import directly.
      entry: {
        'luxar-viewer': resolve(__dirname, 'src/index.ts'),
        'luxar-viewer-styles': resolve(__dirname, 'src/lib-styles-entry.ts'),
      },
      formats: ['es'],
      fileName: (_format, name) => `${name}.js`,
      cssFileName: 'luxar-viewer',
    },

    rollupOptions: {
      // `three` is a peer dependency — embedders bring their own copy so the
      // viewer and the host page share a single THREE.* runtime (essential
      // for `instanceof` checks and texture interop). Externalize the WHOLE
      // `three` package, not just the bare id: the TSL materials statically
      // import the `three/webgpu` and `three/tsl` subpaths, and array
      // externals match ids EXACTLY, so a plain `['three']` would inline those
      // subpaths (and the `three.core.js` they pull in) — shipping a second
      // THREE core. The `/^three(\/.*)?$/` regex externalizes `three` and every
      // `three/*` subpath, all of which the embedder's peer `three` supplies.
      external: [/^three(\/.*)?$/],
      output: {
        // Don't fold `three` into the bundle even if it's referenced.
        globals: {
          three: 'THREE',
        },
      },
    },

    // Workers loaded via `new Worker(new URL(...))` are emitted as separate
    // chunks under dist/lib/assets/. Same for WASM.
    minify: false, // Library consumers run their own minification.
    // No sourcemaps in the published package: the lib build exists only to be
    // packed for npm, and the `.js.map` files were ~10 MB / two-thirds of the
    // tarball. Consumers debug against their own bundler's output.
    sourcemap: false,
  },

  resolve: {
    alias: { '@': resolve(__dirname, 'src') },
  },

  worker: {
    format: 'es',
  },

  optimizeDeps: {
    exclude: ['comlink'],
  },
});
