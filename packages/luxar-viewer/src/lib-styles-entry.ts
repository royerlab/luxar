/**
 * CSS-only entry for the library build.
 *
 * Importing this file pulls in `styles/index.css` (the embed-safe CSS).
 * The library build emits the bundled CSS as `dist/lib/luxar-viewer.css`,
 * which is what consumers reach for via `import '@luxar/viewer/styles.css'`
 * (mapped through package.json `exports`).
 *
 * This file exists ONLY so Vite has a JS module to attach the CSS to —
 * the public API barrel (`src/index.ts`) must stay side-effect-free.
 */
import './styles/index.css';
