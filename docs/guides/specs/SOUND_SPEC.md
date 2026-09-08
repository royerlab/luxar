# Sound Spec — ambient and spatial audio as scene-graph nodes

**Status:** Design agreed with the project owner on 2026-09-06; all four phases
are implemented in draft PR #2566 (`feat/sound-layer`, stacked on PR #2536, which
it needs for the story waypoints its triggers use; it stays a draft until #2536
merges). Implementation decisions and deviations are marked inline below.

## 1. Motivation

A scene can already be *seen* from a camera pose and *read* through overlays.
The kiosk (see `REMOTE_CONTROL_SPEC.md`) wants it to be *heard* too: an ambient
bed that changes with the story, a voice that narrates a cluster on arrival, and
sounds that come from places in the data and get louder as the camera approaches.
Sound is a layer on top of rendering and independent of it — like overlays — but
spatial sources have a position in the scene's reference frame, which makes them
nodes, not overlays.

Decisions taken (owner, 2026-09-06):

| Question | Decision |
| --- | --- |
| Playback at the event | Room speakers, stereo. Equal-power panning by default; HRTF authorable. |
| First uses | Narration per story on arrival; a continuous ambient bed; spatial cluster sounds. |
| Assets | Stored in the zarr as opaque files, like overlay images. |
| Autoplay rule | Both: Chrome kiosk flag when available, a one-time "tap to enable sound" gate as fallback. |
| Data model | A first-class `sound` node type (`scene.add_sound`), nD position, slab visibility, transforms. |
| Narration | Text-to-speech at scene build time: OpenAI TTS when a key is present, else macOS `say`, else warn. |
| Ambient / cluster clips | Recorded CC0 clips (Freesound and the like), licence recorded per node. |
| Default | Authored per scene; viewer default ON with a mute control shown only when the scene has sound nodes. |
| Extras | Recording panel captures audio; ambisonic (first-order) beds rotating with the camera — decoded by a dependency-free native graph, NOT Omnitone (Phase 4 decision: Omnitone decodes to binaural only, wrong on room speakers; fetches HRIRs from a CDN at runtime, impossible on an offline kiosk; unmaintained). |
| Branch | New branch and PR stacked on #2536. |

## 2. Framework

**Web Audio API through three's own wrappers.** No new dependency.
`THREE.AudioListener` attaches to the camera and follows it every frame;
`THREE.PositionalAudio` is an `Object3D` wrapping a `PannerNode` (position,
orientation, distance model, cone, HRTF or equal-power panning);
`THREE.Audio` is the non-spatial sibling. Three's wrappers already handle the
`AudioContext` singleton, buffer sources, gain, loop and play/stop.

Not chosen: Resonance Audio (archived); Howler / Tone.js (nothing for playback we
lack; Tone matters only for synthesis, which is out of scope); Omnitone (kept for
Phase 4 ambisonics only). Doppler no longer exists in Web Audio; nothing lost.

## 3. Data model

### 3.1 Python authoring

```python
# Ambient bed for story 0, plays while story == 0 (slab rule), loops, fades.
scene.add_sound(
    "bed_overview", "assets/overview_bed.mp3",
    positions=None,                       # non-spatial
    hidden={"story": 0},                  # sugar: one row at story=0, displayed dims ignored
    trigger="continuous", gain=0.4, fade_in_ms=1500, fade_out_ms=1500,
    license="CC0", attribution="Freesound user X", source_url="https://…",
)
# Narration for story 3: plays once, 800 ms after the flight lands.
scene.add_sound(
    "narration_hsp70", narration_clip,     # bytes or path
    positions=None, hidden={"story": 3},
    trigger="on_arrive", delay_ms=800, bus="voice",
)
# A spatial hum at the Hsp70 cluster, audible at story 3, louder as you approach.
scene.add_sound(
    "hum_hsp70", "assets/hum.mp3",
    positions=[[3, 7.28, -7.41, -0.27]],   # nD row: (story, x, y, z)
    trigger="continuous", spatial=True,
    ref_distance=2.0, max_distance=30.0, rolloff=1.0, distance_model="inverse",
)
```

- **`sound` is a node type** alongside points / lines / gsplats / mesh — in the
  contract's `node_types` only, **not** in `geometry_types` / `loader_types`: it
  is heard, not drawn, so it has no blending mode, element cap, LOD, picking or
  bounds, none of the geometry tables get a row, and the viewer dispatches on
  `type === 'sound'` alone (Phase 1 decision). It has an
  optional `positions` array `(K, ndim)` exactly like a points node: one row per
  place the source exists. The **slab rule** on hidden dimensions decides
  audibility — a source whose story coordinate matches the slice is live, others
  are silent — so "sounds react to hidden dimensions" is the same mechanism
  points use, and `extend_to_all` makes a source live everywhere. `hidden={...}`
  is authoring sugar that builds the single row from dimension names.
- **Non-spatial** (`positions=None` or `spatial=False`): routed past the panner
  to its bus. Still a node: it still has the slab rule (via `hidden=`) and layer
  toggling.
- **`attach_to="<node name>"`** (Phase 2): the source follows the bounding-box
  centre of another node — "the cluster hums" without authoring coordinates.
- **Clips** are opaque files inside the node group (`audio.mp3` / `audio.m4a`,
  named by `attrs.audio_file`, written through the store-agnostic
  `_zarr_compat.write_raw_bytes` rather than a filesystem path) and read with the
  loader's opaque-file reader like overlay images. The clip bytes are part of
  `content_hash` (`PAYLOAD_FILE_ATTRS` gains `audio_file`). Formats: MP3 or AAC (`.m4a`);
  Ogg/Opus is refused because Safari does not decode it. `duration_ms`, `format`
  and `sample_rate` are stamped by the writer (via `mutagen`/`soundfile` when
  available, else left absent).
- **Licence fields** (`license`, `attribution`, `source_url`) are required for
  clip sources — the same discipline as demo citations — and surface in the
  Layers panel tooltip.
- **Triggers** (`trigger`, plus `delay_ms` ≥ 0):

  | trigger | plays when |
  | --- | --- |
  | `continuous` | looped while the node is audible (slab); fades in/out on the edge |
  | `once` | once each time the node becomes audible |
  | `on_depart` | when a story flight leaves a waypoint the node belongs to |
  | `on_arrive` | when a story flight lands on a waypoint the node belongs to (or a snap) |

  "Belongs to" means the node's row matches the waypoint's `when` under the same
  overlay rule. `on_depart`/`on_arrive` need the waypoint driver's events (§4.3).
- **Buses**: `ambient` (default), `voice`, `effects`. Each has a gain in
  `viewer_config.audio`; `voice` ducks `ambient` by `duck_db` while anything on
  it plays.
- **Spatial parameters** map one to one onto `PannerNode`: `distance_model`
  (`inverse` default, `linear`, `exponential`), `ref_distance`, `max_distance`,
  `rolloff`, optional `cone_inner_deg` / `cone_outer_deg` / `cone_outer_gain`
  and an `orientation` vector. Defaults derive from the scene scale like the fly
  speed does (`ref_distance = scale/20`, `max_distance = scale`); the viewer
  clamps its resolved `max_distance` to at least `ref_distance`.

### 3.2 On disk

```
/sounds/hum_hsp70/            # a group under any group, like other nodes
  zarr.json                   # type: "sound", attrs below
  positions                   # (K, ndim) PLAIN float32 (not the quantizing
                              # encoder: a one-row array collapses to code 0
                              # under per-channel uint16), absent for non-spatial
  audio.mp3                   # opaque
attrs: type, spatial, trigger, delay_ms, gain, bus, loop (derived), fade_in_ms,
       fade_out_ms, distance_model, ref_distance, max_distance, rolloff, cone_*,
       orientation, attach_to, format, duration_ms, license, attribution,
       source_url, layer (bool), extend_to_all, transform, nd_transform,
       ambisonic ("foa" for a first-order AmbiX ACN/SN3D field; absent otherwise),
       channels (stamped when a tag reader is present)
```

`format` is the CODEC (`mp3` / `aac`), never the spatial layout: an ambisonic
field is `format: "aac"` + `ambisonic: "foa"`, and the writer refuses a field
that is not four-channel. World front is −Z; the viewer decodes with nine gains
over the dipoles plus a virtual-cardioid pair at ±60° for the room's stereo.

`content_hash` covers the audio bytes (the same gap #1720 closed for overlay
PNGs must not reopen).

### 3.3 `viewer_config.audio`

```python
ViewerConfig(audio=AudioConfig(
    enabled=True,                 # viewer default when the scene has sound nodes
    master_gain=0.8,
    panning_model="equalpower",   # or "HRTF" for headphones
    buses={"ambient": 0.6, "voice": 1.0, "effects": 0.8},
    duck_db=-9.0,                 # ambient attenuation while voice plays
))
```

## 4. Viewer architecture

### 4.1 `src/audio/`

- **`AudioEngine`** — owns the `AudioContext` (via three's listener), the
  master gain, the three buses and the ducker; handles the autoplay gate (§4.4);
  exposes `setMasterGain`, `setMuted`, `setPanningModel`, `play(name)`,
  `stop(name)`, and emits `sound-started` / `sound-ended`.
- **`SoundNode`** — one per scene node: a `THREE.PositionalAudio` (spatial) or
  `THREE.Audio` (non-spatial) parented under the node's transform group, its
  buffer decoded once from the opaque file, fades implemented on the node gain.
  Reacts to the slab result the same way a points node does: audible ⇄ silent
  with fades, `continuous` restarts on the rising edge, `once` fires on it.
- **Listener** — a `THREE.AudioListener` child of the camera. Three updates its
  position and orientation from the camera's world matrix each frame; nothing
  per-frame is added on the Luxar side.
- **Loader** — a `SoundWholeNodeLoader` (whole node, no chunking): reads
  `positions`, runs the existing per-vertex slab kernel for audibility, fetches
  the clip through the opaque-file reader.

### 4.2 Layers panel and rail

Sound nodes are layers: eye = mute that node, plus a per-node gain slider and
the licence tooltip. A **speaker rail item** (mute all, master gain) appears
only when the loaded scene has at least one sound node; its muted state
persists like other rendering settings.

### 4.3 Triggers and the waypoint driver

`WaypointDriver` gains two events on the embedder bus,
`waypoint-departed {index}` and `waypoint-arrived {index, completed}` (arrival
fires when the `flyTo` promise resolves, or immediately after a snap). Sound
nodes with `on_depart` / `on_arrive` subscribe and match the waypoint's `when`
against their own row. A flight cancelled by the visitor still resolves
(`completed: false`), so narration still starts — but from wherever the camera
stopped, which is the right behaviour. A flight SUPERSEDED by a newer waypoint
never fires `waypoint-arrived`, so two narrations cannot overlap when a visitor
steps quickly. An `on_arrive` clip fades out on the slab's falling edge (moving
on cuts the previous story's narration); an `on_depart` clip plays out.

### 4.4 Autoplay

Browsers refuse to start an `AudioContext` without a user gesture on the page.

- **Kiosk**: launch Chrome with `--autoplay-policy=no-user-gesture-required`
  (documented next to the kiosk block in `REMOTE_CONTROL_SPEC.md` §4.2). The
  context starts on load.
- **Fallback**: if the context is `suspended` after load, the engine shows a
  minimal "Tap to enable sound" gate (an overlay, dismissed by the first
  pointer or key event anywhere), resumes the context, and only then starts
  `continuous` nodes — and re-runs the rising edges from a silent baseline, so
  an opening `once` narration is not lost to the tap (Phase 1 clarification). The remote API reports `audio.state` so a controller can
  tell the display needs its tap.

### 4.5 Remote API (extends `REMOTE_CONTROL_SPEC.md`)

`setAudio({ masterGain?, muted?, buses?, panningModel? })`, `playSound(name)`,
`stopSound(name)`, `getViewerState().audio = { state, muted, masterGain, buses,
playing: [names] }`, events `sound-started`, `sound-ended`, and the two
waypoint events above.

## 5. The stories demo

- **Narration**: `luxar.demos._narration.synthesise(text, voice, cache_dir)`:
  OpenAI TTS when `OPENAI_API_KEY` is set (model and voice pinned in the demo),
  else macOS `say` → `afconvert` to `.m4a`, else a warning and no narration
  node. Clips are cached by hash of (text, voice, engine) under the demo cache,
  so a rebuild with unchanged text costs nothing. Each story's narration is its
  panel text (title, facts, open question) read in order, `trigger="on_arrive"`,
  `delay_ms=600`, bus `voice`.
- **Ambient bed**: one CC0 clip, `trigger="continuous"`, `fade 1500 ms`, bus
  `ambient` (per-story beds or filters later). Phase 1 ships "Calm Ambient 1
  (Synthwave 4k)" by The Cynic Project (cynicmusic.com), CC0, from OpenGameArt
  (<https://opengameart.org/content/calm-ambient-1-synthwave-4k>), fetched
  checksum-pinned through `cached_download` into the `esm3_protein_stories`
  cache: soft evolving pads, no percussion, ~2.6 min loop. It replaced a
  Freesound clip the owner found "too industrial and harsh" — the bed must be
  warm and unobtrusive under the narration, and a clip is judged by listening
  to the WHOLE loop, since many start gently and turn harsh.
- **Cluster sounds** (Phase 2): one spatial source per story at the cluster
  centre, distance falloff tuned from the story's waypoint camera, live only at
  that story.

## 6. Phases

| Phase | Scope |
| --- | --- |
| 1 | `sound` node type (Python writer + validation, viewer loader, `AudioEngine`, `SoundNode`, listener), non-spatial and spatial playback, slab audibility, `continuous`/`once`, buses and ducking, autoplay gate + kiosk flag docs, rail mute, `viewer_config.audio`, remote `setAudio`/`playSound`; demo narration + ambient bed |
| 2 | `on_depart`/`on_arrive` via waypoint events, `attach_to`, cluster sounds in the demo (a pentatonic hum per story attached to the highlight node), Layers-panel rows (eye = mute, group eye mutes the subtree, inline gain slider, provenance tooltip; `LayerSummary`/`LayerPatch` gain) |
| 3 | Recording panel "Include Audio": the master-gain tap into real-time WebM (Opus mime first; the offline path stays silent) |
| 4 | Ambisonic beds (`ambisonic: "foa"`, AmbiX ACN/SN3D, four-channel AAC — Chrome decodes it to four channels) rotating with the camera through the native decoder, no Omnitone |

All four phases shipped together in draft PR #2566 (2026-09-06).

Out of scope: sonification (procedural data-driven sound), live TTS over the
remote API (Phase D of the remote-control spec owns it), multichannel outputs.

## 7. Risks and open questions

- **Format**: MP3 everywhere; AAC on Safari/Chrome/Firefox is fine too; Opus is
  not decodable on Safari. The writer refuses Ogg.
- **Store size**: a 30 s narration clip is ~0.5 MB as MP3; ten stories plus beds
  ≈ 10–15 MB, acceptable. Hosted demos may prefer URL sources later.
- **Clicks**: every start/stop goes through a short gain ramp (≥ 30 ms);
  `continuous` nodes fade on the slab edge.
- **Clock**: `on_arrive` timing uses the flight promise, not the audio clock; a
  cancelled flight still resolves. Documented as intended.
- **Two audibility rules?** No: a sound node's audibility *is* the slab rule the
  points use; `hidden=` is sugar for one row. Overlays keep `visible_range`
  (screen space, no position). The waypoint `when` clause is the third use of the
  same vocabulary.
- Should the rail mute also silence the autoplay gate's need? Yes: a muted
  scene never shows the gate.
