import { defineConfig } from 'vite';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
// Explicit `.ts` extension: Vite's future native config loader (Node's own TS
// support) does not do extensionless resolution, and 8.2 warns about it.
import { checkoutIdentityPlugin, ensureCheckoutIdentity } from './tools/e2e-server-identity.ts';

const viewerRoot = fileURLToPath(new URL('.', import.meta.url));
const projectRoot = resolve(viewerRoot, '../..');

export default defineConfig(({ command }) => ({
  plugins:
    command === 'serve'
      ? [checkoutIdentityPlugin(ensureCheckoutIdentity(projectRoot, viewerRoot))]
      : [],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'esnext', // Required for Workers and WASM support
    // Four chunks exceed rollup's 500 kB default and all four are deliberate:
    // `blosc` and `zstd` (codec WASM glue) and `three-webgpu` are lazily
    // imported and never touch the initial payload, and `index` is the app
    // itself. At the default limit the warning fired on every single build and
    // could only ever be a false positive, so it taught people to ignore it.
    // Raised to just above the largest INTENTIONAL chunk: a new offender still
    // trips it, and `pnpm build:check` separately proves the lazy ones stay
    // lazy (that is the assertion with teeth — see scripts/check-eager-chunks.mjs).
    chunkSizeWarningLimit: 1700,
    rolldownOptions: {
      output: {
        // Split the `three` package into core / tsl / webgpu so each
        // subsystem gets its own cacheable chunk. Without this, the
        // tree-shaken three modules collapse into a single ~1.2 MB
        // chunk that re-downloads on any three-internal change.
        codeSplitting: {
          groups: [
            {
              // `three.core.js` is imported by BOTH `three.module.js` (the
              // WebGL entry) and `three.webgpu.js`. Without a group of its own
              // the shared core lands in one of theirs, and the observed
              // outcome was the worst one: the `three` chunk ended up
              // importing `three-webgpu`, which pins the whole WebGPU/node
              // system into the eager graph no matter how carefully the
              // application code defers it (issue #1679). Splitting the core
              // out first — highest priority, so it wins over the two entry
              // rules below — lets `three` and `three-webgpu` each depend on
              // it instead of on each other.
              name: 'three-core',
              test: /[\\/]three[\\/]build[\\/]three\.core/,
              priority: 30,
            },
            {
              name: 'three-webgpu',
              test: /[\\/]three[\\/]build[\\/]three\.webgpu/,
              priority: 20,
            },
            {
              name: 'three-tsl',
              test: /[\\/]three[\\/]build[\\/]three\.tsl/,
              priority: 20,
            },
            {
              name: 'three-ktx2',
              test: /[\\/]three[\\/]examples[\\/]jsm[\\/](?:loaders[\\/]KTX2Loader|libs[\\/](?:ktx-parse|zstddec))/,
              priority: 20,
              includeDependenciesRecursively: false,
            },
            {
              name: 'three',
              test: /[\\/]three[\\/]/,
              priority: 10,
            },
          ],
        },
      },
    },
  },
  resolve: { alias: { '@': resolve(viewerRoot, 'src') } },
  worker: {
    format: 'es', // Use ES modules for workers
  },
  optimizeDeps: {
    exclude: ['comlink'], // Comlink works better without bundling
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    open: true,
    // Proxy configuration - forward requests to Python backend
    proxy: {
      '/data': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/examples': {
        // E2E tests need access to example datasets
        // Playwright auto-starts: hatch run luxar serve examples/
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass(req) {
          if (req.url?.startsWith('/examples/layer/')) return req.url;
        },
      },
    },
  },
}));
