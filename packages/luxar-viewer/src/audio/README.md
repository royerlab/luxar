# audio

The viewer's sound layer: `sound` scene-graph nodes played through Web Audio via
three's `AudioListener` / `PositionalAudio` / `Audio` (no new dependency). Design:
`docs/guides/specs/SOUND_SPEC.md`. Phase 1 ships non-spatial and spatial playback,
slab-rule audibility on hidden dimensions, the `continuous` / `once` triggers,
three buses with voice-over-ambient ducking, the autoplay gate, the rail mute,
`viewer_config.audio` and the remote `setAudio` / `playSound` / `stopSound`.

## Quick Start

Nothing here is constructed until a scene with a sound node attaches; a plain
points scene never creates an `AudioContext`. `LuxarApp` owns one engine:

```typescript
import { AudioEngine } from '../audio/audio-engine';

const engine = new AudioEngine({
  getCamera: () => sceneManager.camera,
  onCameraReplaced: (cb) => subscribe(sceneManager, 'camera-changed', cb),
  getDims: () => sceneDimsManager.getDims(),
  onDimsChanged: (cb) => subscribe(sceneDimsManager, cb),
  getSceneGraph: () => getSceneLoader()?.sceneGraph ?? null,
  getSceneScale: () => sceneManager.getSceneScale(),
  container: getViewerContainer,
  emit: (event, payload) => embedderEvents.emit(event, payload),
  notifyUiChanged: () => window.dispatchEvent(new CustomEvent('luxar-audio-changed')),
});

// After the scene's nodes are attached and the opening slice applied:
engine.applySceneConfig(extractAudioConfig(viewerConfig?.audio));
engine.attachScene(luxarSceneRoot);

engine.setAudio({ muted: true }); // the rail mute
engine.play('narration_hemoglobin'); // remote control, regardless of the slab
engine.getState(); // { state, muted, masterGain, panningModel, buses, playing, hasSoundNodes }
```

## Modules

| File               | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio-engine.ts`  | `AudioEngine` — owns the `AudioContext` through three's listener (its gain is the master), three bus `GainNode`s (`ambient` routed through a duck gain, `voice`, `effects`), the ducker (voice ducks ambient by `duck_db`), the autoplay gate, one `SoundNode` per node. `attachScene(root)` traverses for placeholders carrying `userData.sound`, evaluates the slab once, subscribes to dimension changes, decodes clips sequentially. `detachScene()` on dataset switch. Persists mute + master gain. |
| `sound-node.ts`    | `SoundNode` — K `PositionalAudio` voices or one `Audio`; bus routing (`gain.disconnect(); gain.connect(bus)` — three wires the gain to the listener input in its constructor and `play()` only reconnects the source side); fades ≥ 30 ms on every edge; `continuous` loops and restarts on a rising edge, `once` fires per rising edge; gate-closed edges are deferred and replayed on open (so an opening `once` narration survives the tap).                                                          |
| `audibility.ts`    | Pure slab rule: `buildSoundBaseViewState(dims)` (mesh tolerance semantics: half a step on a discrete hidden dim, one cell on a continuous one), `computeRowAudibility` (through `deriveNodeViewState` for `extend_to_all` + inverse `nd_transform`, then `mesh_vertex_visibility_mask`), `displayedXYZ`.                                                                                                                                                                                                 |
| `sound-attrs.ts`   | `parseSoundNodeAttrs(path, raw, sceneScale)` — defaults; `ref_distance` / `max_distance` left absent on disk default to `scale / 20` / `scale`; `on_depart` / `on_arrive` warn and play as `once` (Phase 2).                                                                                                                                                                                                                                                                                             |
| `autoplay-gate.ts` | `AutoplayGate` — the "Tap to enable sound" overlay (`styles/components/audio-gate.css`), dismissed by the first pointer or key event anywhere. A kiosk launched with `--autoplay-policy=no-user-gesture-required` never shows it; a muted scene never shows it either.                                                                                                                                                                                                                                   |
| `fades.ts`         | `rampGain` (cancel + hold + linear ramp, floor `MIN_FADE_MS`), `dbToGain`.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `audio-prefs.ts`   | `loadAudioPrefs` / `saveAudioPrefs` on `StorageKeys.audio` (global, not per scene: a muted kiosk stays muted across datasets).                                                                                                                                                                                                                                                                                                                                                                           |

Vocabulary shared with the data layer and the embedder API lives in
`types/audio.ts` (`SoundSourceDescriptor`, `SoundNodeAttrs`, `AudioState`,
`AudioPatch`, `AudioBusName`, `PanningModel`). The data layer's half is
`data/scene-loader/nodes/load-sound-node.ts`; the config bridge is
`config/zarr-bridge/audio-config.ts`.

## Layering

`audio` sits between `data` and `scene` (`.dependency-cruiser.cjs`): it may import
`data/*` pure helpers, `wasm/typescript`, `utils/`, `three`; the camera, the dims
manager, the scene graph and the embedder emitter arrive as PORTS
(`AudioEngineDeps`). That is what lets `audio-engine.test.ts` run in node with a
fake `AudioContext` injected via `THREE.AudioContext.setContext`.

## Kiosk

Chrome refuses to start an `AudioContext` without a gesture. For an unattended
display launch it with:

```
google-chrome --kiosk --autoplay-policy=no-user-gesture-required "http://host:5173/?src=…&kiosk"
```

The context then starts on load and the gate never appears. See
`docs/guides/specs/REMOTE_CONTROL_SPEC.md` §4.2 for the rest of the kiosk block.

## Phase 2 (not here)

`on_depart` / `on_arrive` via the waypoint driver's events, `attach_to`, cluster
sounds in the stories demo, Layers-panel rows with per-node gain. Sound nodes are
absent from the Layers panel today (the panel gate is `isGeometryType || group`),
and a parent group's eye does not silence its sounds.
