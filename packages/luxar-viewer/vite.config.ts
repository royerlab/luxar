import { defineConfig } from 'vite';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { checkoutIdentityPlugin, ensureCheckoutIdentity } from './tools/e2e-server-identity';

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
    rolldownOptions: {
      output: {
        // Split the `three` package into core / tsl / webgpu so each
        // subsystem gets its own cacheable chunk. Without this, the
        // tree-shaken three modules collapse into a single ~1.2 MB
        // chunk that re-downloads on any three-internal change.
        codeSplitting: {
          groups: [
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
      },
    },
  },
}));
