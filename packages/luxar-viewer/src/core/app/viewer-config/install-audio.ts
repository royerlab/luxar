/**
 * Bind the sound layer to a loaded scene.
 *
 * Applies the authored `viewer_config.audio` defaults (the listener's persisted
 * mute / master gain win), then hands the engine the scene root so it finds the
 * sound-node placeholders `loadSoundNode` attached and starts decoding their
 * clips.
 *
 * @module core/app/viewer-config/install-audio
 */

import type * as THREE from 'three';

import type { AudioEngine } from '../../../audio/audio-engine';
import { extractAudioConfig } from '../../../config/zarr-bridge/audio-config';
import type { SoundWaypointCondition } from '../../../types/audio';

/** What {@link installSceneAudio} needs from the app. */
export interface SceneAudioPorts {
  /** The app's sound engine (only the scene-binding surface). */
  audioEngine: Pick<
    AudioEngine,
    'detachScene' | 'applySceneConfig' | 'attachScene' | 'notifyWaypoint'
  >;
  /** The loaded scene's root (`LuxarScene`), if it is in the scene. */
  sceneRoot: THREE.Object3D | undefined;
  /** Re-push the Layers panel's per-layer sound mutes onto the new nodes. */
  pushAudioMutes(): void;
  /** The `when` of the waypoint the opening story snapped to, if any. */
  openingWaypointWhen: SoundWaypointCondition | undefined;
}

/**
 * Apply `audio` (the scene's raw `viewer_config.audio`) and attach the scene's
 * sound nodes to the engine, replaying the opening waypoint's arrival.
 */
export function installSceneAudio(audio: unknown, ports: SceneAudioPorts): void {
  const { audioEngine } = ports;
  audioEngine.detachScene();
  audioEngine.applySceneConfig(extractAudioConfig(audio));
  if (ports.sceneRoot) {
    audioEngine.attachScene(ports.sceneRoot);
    ports.pushAudioMutes();
  }
  // The waypoints install first and snap to the opening waypoint before any
  // sound node exists, so the load-time arrival is replayed here — otherwise
  // the opening story's `on_arrive` narration would never fire.
  if (ports.openingWaypointWhen) audioEngine.notifyWaypoint('arrive', ports.openingWaypointWhen);
}
