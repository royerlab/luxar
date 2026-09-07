/**
 * Wire the scene environment's LIVE behaviour into the running app.
 *
 * `rendering/environment/scene-environment.ts` knows how to capture; this module tells
 * it WHEN. A scene-derived environment (`viewer_config.environment.source = "scene"`)
 * depends on what is resident and how it looks, so it is marked stale on:
 *
 * - a geometry commit (`eventBus 'geometry-committed'`, the same moment the pick buffer
 *   is invalidated),
 * - a slice change (`sceneDimsManager` listener),
 * - an appearance change (`window 'luxar-layers-changed'`),
 *
 * and re-captured on the next frame after a short debounce, once the loader has settled
 * (same predicate the adaptive-DPR manager reads). Camera motion is deliberately NOT a
 * trigger: a cube map from a fixed probe is view-independent, and the one camera-driven
 * change — LOD residency — arrives as a commit.
 *
 * Under `?bake-env` this module also runs the one-shot bake once the first load has
 * settled: capture, read back, expose the container on `__luxarDebug.environment.lastBake`
 * for the `luxar env bake` driver, and download it for a human at the keyboard.
 *
 * @module core/app/init/environment-wiring
 */

import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { eventBus } from '../../../utils/cross-layer/event-bus';
import { log, Modules } from '../../../utils/log';
import { getSceneLoader } from '../../../data/scene-loader-manager';
import { bakeEnvironment, containerToBase64 } from '../../../rendering/environment/bake';
import { parseProbeSpec } from '../../../rendering/environment/probe';
import { downloadBlob } from '../../../ui/recording-panel/screenshot-exporter';
import type { LuxarAppOptions } from '../options';
import { buildInfo } from '../../../config/build-info';

/** What the wiring needs from the app. */
export interface EnvironmentWiringDeps {
  sceneManager: SceneManager;
  animationController: AnimationController;
  events: EventGroup;
  options: LuxarAppOptions;
  /** The load-activity predicate (true = nothing in flight). */
  isSettled: () => boolean;
}

/**
 * Consecutive settled frames a bake waits for after the loader first reports settled:
 * the post-load refinement drain releases the lock between passes, so one settled frame
 * is not yet "the scene as published".
 */
export const BAKE_SETTLED_FRAMES = 30;

export function wireSceneEnvironment(deps: EnvironmentWiringDeps): void {
  const { sceneManager, animationController, events, options, isSettled } = deps;
  sceneManager.attachEnvironmentRuntime(isSettled);

  animationController.addPerFrameCallback('environment-capture', () => {
    sceneManager.environment?.tick();
  });
  events.add(() => animationController.removePerFrameCallback('environment-capture'));

  const markStale = (): void => sceneManager.environment?.markStale();
  events.add(eventBus.on('geometry-committed', markStale));
  sceneDimsManager.addListener(markStale);
  events.add(() => sceneDimsManager.removeListener(markStale));
  window.addEventListener('luxar-layers-changed', markStale);
  events.add(() => window.removeEventListener('luxar-layers-changed', markStale));

  if (options.bakeEnvironment) scheduleBake(deps, options.bakeEnvironment);
}

function scheduleBake(
  deps: EnvironmentWiringDeps,
  request: NonNullable<LuxarAppOptions['bakeEnvironment']>
): void {
  const { sceneManager, animationController, events, isSettled } = deps;
  let settledFrames = 0;
  let started = false;
  const id = 'environment-bake';
  animationController.addPerFrameCallback(
    id,
    () => {
      if (started) return;
      // Not before a dataset load has begun and produced a scene root.
      const hasScene = sceneManager.scene.children.some((c) => c.name === 'LuxarScene');
      if (!getSceneLoader('default') || !hasScene) return;
      settledFrames = isSettled() ? settledFrames + 1 : 0;
      if (settledFrames < BAKE_SETTLED_FRAMES) return;
      started = true;
      animationController.removePerFrameCallback(id);
      void runBake(sceneManager, request);
    },
    { continuous: true }
  );
  events.add(() => animationController.removePerFrameCallback(id));
}

/** The `__luxarDebug.environment` slot the driver polls, created if bootstrap has not. */
function debugEnvironmentSlot(): NonNullable<
  NonNullable<typeof window.__luxarDebug>['environment']
> {
  const debug = (window.__luxarDebug ??= {} as NonNullable<typeof window.__luxarDebug>);
  debug.environment ??= { kind: () => 'none', captureCount: () => 0, hasBaked: () => false };
  return debug.environment;
}

async function runBake(
  sceneManager: SceneManager,
  request: NonNullable<LuxarAppOptions['bakeEnvironment']>
): Promise<void> {
  const slot = debugEnvironmentSlot();
  try {
    const result = await bakeEnvironment(resolveBakeRequest(sceneManager, request));
    slot.lastBake = {
      header: result.header,
      base64: containerToBase64(result.bytes),
      byteLength: result.bytes.byteLength,
    };
    log.info(
      Modules.RENDERER,
      `Environment bake ready: ${result.header.resolution}px cube, ${result.bytes.byteLength} bytes ` +
        `(probe ${result.header.probe.spec})`
    );
    downloadBlob(
      new Blob([result.bytes.buffer as ArrayBuffer], { type: 'application/octet-stream' }),
      'scene.env.bin'
    );
  } catch (error) {
    slot.bakeError = String(error);
    log.error(Modules.RENDERER, `Environment bake failed: ${String(error)}`);
  }
}

/** Turn the URL request into a full bake request, or throw a message naming what is missing. */
function resolveBakeRequest(
  sceneManager: SceneManager,
  request: NonNullable<LuxarAppOptions['bakeEnvironment']>
): Parameters<typeof bakeEnvironment>[0] {
  const environment = sceneManager.environment;
  if (!environment) throw new Error('no scene environment');
  const root = sceneManager.scene.children.find((c) => c.name === 'LuxarScene');
  const contentHash = root?.userData?.sceneContentHash as string | undefined;
  if (!contentHash) throw new Error('the scene carries no content_hash; finalize it first');
  const probe = request.probe ? parseProbeSpec(request.probe) : null;
  if (request.probe && !probe) throw new Error(`malformed probe '${request.probe}'`);
  // A bake must not be lit by a previously baked map: it captures the SCENE.
  environment.setBaked(null);
  return {
    environment,
    renderer: sceneManager.renderer,
    capabilities: sceneManager.capabilities,
    probe: probe ?? undefined,
    resolution: request.resolution,
    sceneContentHash: contentHash,
    appearance: { viewer_config: sceneManager.getSceneViewerConfig() ?? {} },
    viewerVersion: buildInfo().version,
  };
}
