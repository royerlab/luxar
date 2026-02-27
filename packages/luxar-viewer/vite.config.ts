import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'esnext', // Required for Workers and WASM support
  },
  resolve: { alias: { '@': resolve(__dirname, 'src') } },
  worker: {
    format: 'es', // Use ES modules for workers
  },
  optimizeDeps: {
    exclude: ['comlink'], // Comlink works better without bundling
  },
  server: {
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
});
