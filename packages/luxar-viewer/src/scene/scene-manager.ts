/**
 * Re-export stub. The SceneManager class lives in
 * `./scene-manager/index.ts` so the existing scene-setup helpers
 * (camera-framing, camera-setup, scene-disposal, webgl-context-recovery)
 * become folder-package siblings of the orchestrator rather than
 * floating beside the monolithic file. Consumers import `SceneManager`
 * from this path unchanged.
 *
 * Step 13 of the god-object refactor. Subsequent commits may pull
 * specific concerns (DPR/clipping/resize policy, camera-mode swap,
 * global appearance setters) out of index.ts into focused
 * scene-manager/<concern>.ts modules.
 */

export { SceneManager } from './scene-manager/index';
