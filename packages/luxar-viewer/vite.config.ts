import { defineConfig } from "vite";
import { resolve } from "path";
import fs from "fs";

export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
  resolve: { alias: { "@": resolve(__dirname, "src") } },
  server: {
    port: 5173,
    open: true,
    // Serve examples directory from parent project for E2E tests
    fs: {
      allow: [
        resolve(__dirname, "src"),  // Allow viewer source
        resolve(__dirname, "../.."),  // Allow parent project (for /examples/)
      ]
    },
    // Proxy configuration
    proxy: {
      "/data": {
        target: "http://localhost:8000",
        changeOrigin: true
      },
      "/examples": {
        target: "http://localhost:8001",
        changeOrigin: true
      }
    }
  }
});

