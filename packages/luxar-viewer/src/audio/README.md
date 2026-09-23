# audio

The viewer's sound layer: `sound` scene-graph nodes played through Web Audio via
three's `AudioListener` / `PositionalAudio` / `Audio` (no new dependency). Design:
`docs/guides/specs/SOUND_SPEC.md`. All four phases of that spec are here:
non-spatial, spatial (`positions` rows or `attach_to` a node's centre) and
first-order **ambisonic** playback; slab-rule audibility on hidden dimensions;
the `continuous` / `once` triggers on the slab's edges and `on_depart` /
`on_arrive` on the waypoint driver's events; three buses with voice-over-ambient
ducking; the autoplay gate; the rail mute; Layers-panel rows (eye = mute,
slider = gain); a capture tap for the Recording panel; `viewer_config.audio`;
the remote `setAudio` / `playSound` / `stopSound`.

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

// The waypoint driver's events (the app forwards them with the waypoint's `when`):
engine.notifyWaypoint('arrive', { story: 3 }); // fires the on_arrive nodes whose rows belong

// Layers-panel rows and the Recording panel:
engine.setNodeMuted('/sounds/hum', true); // one node, or every node under a group path
engine.setNodeGain('/sounds/hum', 0.5);
const tap = engine.acquireCaptureStream(); // the mix the listener hears, for MediaRecorder
engine.releaseCaptureStream(tap!);
```

`resolveNodeCenter(name)` is an optional port for `attach_to`; the ambisonic
fields need no port — the engine wraps the listener's `updateMatrixWorld`, which
three calls from the camera every rendered frame.

## Modules

| File               | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio-engine.ts`  | `AudioEngine` — owns the `AudioContext` through three's listener (its gain is the master), three bus `GainNode`s (`ambient` routed through a duck gain, `voice`, `effects`), the ducker (voice ducks ambient by `duck_db`), the autoplay gate, one `SoundNode` per node. The gate is decided from `context.state`, not from the outcome of `resume()`, and reopens from `onstatechange`, a one-shot gesture listener, or `enableSound()`; `isBlocked()` reports "wanted but not started" separately from `isMuted()`. `attachScene(root)` traverses for placeholders carrying `userData.sound`, evaluates the slab once, subscribes to dimension changes, decodes clips sequentially. `detachScene()` on dataset switch. Persists mute + master gain.                                                                                                                                                                              |
| `sound-node.ts`    | `SoundNode` — K `PositionalAudio` voices, one `Audio`, or one `FoaAudio`; bus routing (`gain.disconnect(); gain.connect(bus)` — three wires the gain to the listener input in its constructor and `play()` only reconnects the source side); fades ≥ 30 ms on every edge; `continuous` loops and restarts on a rising edge, `once` fires per rising edge; `on_depart` / `on_arrive` ignore the slab's rising edge and start from `triggerFromWaypoint` (an `on_arrive` clip still fades out when its story is left); gate-closed edges and triggers are deferred and replayed on open, and a deferred waypoint trigger is superseded by the next waypoint event so at most one is ever waiting (`clearDeferredTrigger`). `attach_to` places the voices at the named node's centre (re-resolved on every evaluation). Three mute flags — the rail, the node's own Layers eye, an ancestor group's eye — and a live per-node `gain`. |
| `foa-decoder.ts`   | `FoaDecoder` — first-order ambisonic (AmbiX: ACN `W Y Z X`, SN3D) rotate-and-decode graph from plain Web Audio nodes: 4-ch splitter → nine rotation gains over the dipoles → virtual-cardioid pair at ±60° → stereo merger. `setCameraQuaternion` folds the world→AmbiX axis mapping (front = −Z, left = −X, up = +Y) so the field stays fixed to the world as the camera turns; the engine calls it from the listener's per-frame update. `FoaAudio` is a three `Audio` whose source runs through the decoder. No Omnitone: it decodes to binaural only (wrong on room speakers), fetches HRIRs from a CDN at runtime, and is unmaintained.                                                                                                                                                                                                                                                                                       |
| `audibility.ts`    | Pure slab rule: `buildSoundBaseViewState(dims)` (mesh tolerance semantics: half a step on a discrete hidden dim, one cell on a continuous one), `computeRowAudibility` (through `deriveNodeViewState` for `extend_to_all` + inverse `nd_transform`, then `mesh_vertex_visibility_mask`), `displayedXYZ`; `buildWaypointViewState` / `computeWaypointMembership` rewrite a waypoint's `when` clause as a slab (exact value = ±0.5, range = midpoint ± half-width, unnamed hidden dims = anything) so "belongs to a waypoint" runs the SAME kernel.                                                                                                                                                                                                                                                                                                                                                                                  |
| `sound-attrs.ts`   | `parseSoundNodeAttrs(path, raw, sceneScale)` — defaults; `ref_distance` / `max_distance` left absent on disk default to `scale / 20` / `scale`, with resolved `max_distance` clamped to at least `ref_distance`; `attach_to` implies spatial; `ambisonic: "foa"` forces non-spatial; unknown enum values warn and fall back.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `autoplay-gate.ts` | `AutoplayGate` — the "Tap to enable sound" overlay (`styles/components/audio-gate.css`), dismissed by the first pointer or key event anywhere. A kiosk launched with `--autoplay-policy=no-user-gesture-required` never shows it; a muted scene never shows it either. It is a hint, not the only route: the rail's Sound button carries a persistent `blocked` state that resumes the context when clicked.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `fades.ts`         | `rampGain` (cancel + hold + linear ramp, floor `MIN_FADE_MS`), `dbToGain`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `audio-prefs.ts`   | `loadAudioPrefs` / `saveAudioPrefs` on `StorageKeys.audio` (global, not per scene: a muted kiosk stays muted across datasets).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

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
`docs/guides/specs/REMOTE_CONTROL_SPEC.md` §4.3 for the rest of the kiosk block.

## Waypoint triggers

`WaypointDriver` (core) emits `waypoint-departed { index }` when the matched
waypoint changes away from one and `waypoint-arrived { index, completed }` when
the new waypoint's flight resolves (immediately after a snap, or when the
waypoint has no camera block). A flight the visitor cancels still arrives
(`completed: false`); a flight a NEWER waypoint superseded never does, so two
stories' narrations cannot overlap. `LuxarApp` forwards both to the embedder bus
and to `AudioEngine.notifyWaypoint(kind, when)`, and replays the load-time
arrival after the scene's sound nodes attach (the waypoints install first).

## Ambisonic fields

A node with `ambisonic: "foa"` is a 4-channel AmbiX AAC clip decoded through
`FoaDecoder` (above) and rotated against the camera. Chrome 152 was checked to
hand `decodeAudioData` a 4-channel `AudioBuffer` for AAC encoded by both
`afconvert` and `ffmpeg`; a clip with fewer channels plays with the missing
dipoles silent (warned). The stories demo re-encodes its stereo bed as a field
at build time (`luxar.demos._audio_synth.synthesise_foa_from_clip`).

## Recording

The Recording panel's real-time WebM path adds `acquireCaptureStream()`'s tracks
to its canvas capture when "Include Audio" is on, asking `MediaRecorder` for an
Opus-capable mime first. The tap sits after the master gain, so a muted viewer
records silence. The frame-by-frame offline path has no clock to record audio
against and stays silent.
