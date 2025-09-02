import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
  resolve: { alias: { "@": resolve(__dirname, "src") } },
  server: {
    port: 5173,
    open: true,
    // NEW ➜ forward /data/* to FastAPI on :8000
    proxy: {
      "/data": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    }
  }
});

