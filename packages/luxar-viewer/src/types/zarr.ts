/**
 * Type definitions for Zarr store attributes and metadata.
 *
 * These types provide proper typing for Zarr group and array attributes,
 * eliminating the need for 'as any' type assertions throughout the codebase.
 */

import type { ZarrKioskConfig } from '../config/kiosk';

/**
 * Per-dimension affine transform for continuous/discrete dimensions.
 * Applied as: effective_value = scale * original_value + offset
 */
export interface NdTransformAffine {
  /** Multiplicative factor (default 1.0) */
  scale?: number;
  /** Additive shift (default 0.0) */
  offset?: number;
}

/**
 * Per-dimension permutation for categorical dimensions.
 * Maps original category index to new index.
 */
export interface NdTransformPermutation {
  /** Permutation array: permutation[old_index] = new_index */
  permutation: number[];
}

/** Single dimension transform entry (affine or permutation) */
export type NdTransformEntry = NdTransformAffine | NdTransformPermutation;

/** nD transform map: dimension name → per-dim transform */
export type NdTransformMap = Record<string, NdTransformEntry>;

/**
 * Type guard to check if an nd_transform entry is a permutation.
 *
 * Defensive: accepts `null` / `undefined` / primitives because zarr metadata
 * is parsed dynamically from JSON. The naive `'permutation' in entry` form
 * throws `TypeError` for those inputs, masking malformed `nd_transform`
 * payloads as opaque loader crashes. Returns `false` instead.
 */
export function isPermutation(
  entry: NdTransformEntry | null | undefined
): entry is NdTransformPermutation {
  return entry !== null && typeof entry === 'object' && 'permutation' in entry;
}

/**
 * nD bounding box (min/max per dimension)
 */
export interface PositionBounds {
  /** Minimum value per dimension */
  min: number[];
  /** Maximum value per dimension */
  max: number[];
}

/**
 * Scene-level dimension information stored in Zarr attributes
 * Mirrors Python's luxar.core.Dimension class for full compatibility
 */
export interface SceneDimensionAttrs {
  dimensions: Array<{
    name: string;
    unit: string;
    scale?: number;
    range?: [number, number];
    display: boolean;
    discrete?: boolean;
    step?: number;
    cyclic?: boolean;
    spatial?: boolean;
    categories?: string[]; // Category labels for categorical dimensions
    description?: string;
  }>;
}

/** The authored camera block — the scene-level `camera` and each waypoint's `camera`. */
export interface ZarrCameraConfig {
  position?: [number, number, number];
  target?: [number, number, number];
  up?: [number, number, number];
  fov?: number;
  fov_preset?: string;
  near?: number;
  far?: number;
  /** Orthographic zoom factor (> 0). Only meaningful under an ortho projection. */
  zoom?: number;
  /** Named scene graph node whose bounding box center becomes the camera target */
  target_node?: string;
}

/**
 * A `when` clause: dimension NAME → exact value (matches within ±0.5 of the
 * current step) or inclusive `[min, max]`. Same syntax and semantics as an
 * overlay's `visible_range`.
 */
export type ZarrWaypointCondition = Record<string, number | [number, number]>;

/**
 * A camera pose bound to a hidden-dimension position (Python `Waypoint`).
 * First match in list order wins; see `core/app/camera/waypoint-driver.ts`.
 */
export interface ZarrWaypoint {
  when: ZarrWaypointCondition;
  camera: ZarrCameraConfig;
  /** Flight duration in ms; absent = viewer default, 0 = snap. */
  duration_ms?: number;
  easing?: 'linear' | 'ease-in-out';
  /** Rendering overrides (snake_case ViewerConfig keys) applied on arrival. */
  rendering?: Record<string, unknown>;
  /**
   * When this waypoint's dimension-bound overlays appear. `immediate` (the
   * default) shows them as the dimension changes, while the camera is still
   * flying; `on_arrival` holds overlays that would newly appear until the
   * flight resolves — a snap or a camera-less waypoint arrives at once, a
   * flight the visitor cancels counts as arrival, a flight a newer waypoint
   * supersedes never reveals. Departing overlays hide immediately either way.
   */
  reveal?: 'immediate' | 'on_arrival';
}

/**
 * Scene-wide audio defaults (Python `AudioConfig`, `viewer_config.audio`).
 * Validated and clamped by `config/zarr-bridge/audio-config.ts`.
 */
export interface ZarrAudioConfig {
  enabled?: boolean;
  master_gain?: number;
  panning_model?: 'equalpower' | 'HRTF' | string;
  buses?: Partial<Record<'ambient' | 'voice' | 'effects', number>>;
  /** Ambient attenuation while anything on the voice bus plays (dB, <= 0). */
  duck_db?: number;
}

/**
 * Viewer configuration from Python API (stored in zarr root .zattrs).
 * All fields are optional — only set fields are present.
 * Keys use snake_case to match the Python/zarr convention.
 *
 * This is also the format exported by Ctrl+Shift+S in the viewer,
 * enabling full Python → zarr → viewer → export → Python round-trips.
 */
export interface ZarrViewerConfig {
  // Camera
  camera?: ZarrCameraConfig;

  /**
   * Where the scene environment that lights `material="physical"` meshes comes
   * from (`MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3). Ignored by every other
   * material. Validated and narrowed by `extractEnvironmentConfig`.
   */
  environment?: {
    /** `room` (default), `scene` (exact cube capture from `probe`), `hdri` (`url`). */
    source?: string;
    /** `'auto'`, `'node:<path>'`, or an `[x, y, z]` world position. */
    probe?: string | [number, number, number];
    /** Cube face size in pixels for a `scene` capture, 16–1024. */
    resolution?: number;
    /** `scene.environmentIntensity`, `>= 0`. */
    intensity?: number;
    /** Equirectangular image for `hdri`, relative to the store or absolute. */
    url?: string;
  };

  // Scene identity — becomes the browser tab title (document.title); wins
  // over the `?title=` URL parameter `luxar serve --open` derives from the
  // dataset file name.
  title?: string;

  // Scene
  background_color?: string;

  // Rendering pipeline
  tone_mapping?: string;
  exposure?: number;
  global_offset?: number;
  global_gamma?: number;

  // Bloom
  bloom_enabled?: boolean;
  bloom_strength?: number;
  bloom_radius?: number;
  bloom_threshold?: number;
  bloom_levels?: number;

  // Navigation
  control_type?: string;
  auto_rotate?: boolean;
  auto_rotate_speed?: number;
  /** Turntable axis: a camera-frame axis or a fixed scene axis; see `AutoRotateAxis`. */
  auto_rotate_axis?: string;
  auto_dolly?: boolean;
  /** Peak dolly swing as a percent of the viewing distance (15 → ±15%). */
  auto_dolly_amplitude_percent?: number;
  /** Seconds per full dolly oscillation. */
  auto_dolly_period?: number;
  natural_drag?: boolean;

  // Cinematic
  cinematic_mode?: boolean;

  // Vignette
  vignette_enabled?: boolean;
  vignette_darkness?: number;
  vignette_offset?: number;

  // Detector noise
  detector_noise_enabled?: boolean;
  detector_noise_readout_sigma?: number;
  detector_noise_photon_gain?: number;
  detector_noise_fpn_sigma?: number;

  // Anti-aliasing
  fxaa_enabled?: boolean;
  msaa_enabled?: boolean;
  msaa_samples?: number;
  ssaa_enabled?: boolean;
  ssaa_multiplier?: number;

  // Chromatic lens distortion
  chromatic_lens_distortion_enabled?: boolean;
  chromatic_lens_distortion_x?: number;
  chromatic_lens_distortion_y?: number;
  chromatic_lens_dispersion?: number;
  chromatic_lens_principal_point_x?: number;
  chromatic_lens_principal_point_y?: number;
  chromatic_lens_focal_length_x?: number;
  chromatic_lens_focal_length_y?: number;
  chromatic_lens_skew?: number;

  // Fly controls
  fly_movement_speed?: number;
  fly_rotation_speed?: number;
  fly_inertial_mode?: boolean;
  fly_damping?: number;
  fly_rotation_damping?: number;

  // Dynamic clipping
  dynamic_clipping_enabled?: boolean;

  // Adaptive resolution
  adaptive_dpr_enabled?: boolean;
  allow_high_dpr?: boolean;

  // UI panel visibility
  ui?: {
    show_help?: boolean;
    show_rendering_controls?: boolean;
    show_performance_monitor?: boolean;
    show_dimensions?: boolean;
    show_scale_bar?: boolean;
    show_layers?: boolean;
    show_overlays?: boolean;
    // Unattended-display lockdown. Resolved (with `?kiosk`) by
    // `config/kiosk.ts::resolveKioskMode`, which validates the shape — so
    // this mirrors the writer's field names and nothing more.
    kiosk?: ZarrKioskConfig;
  };

  // Theme
  theme?: string;

  // Dimension navigation state
  dimensions?: {
    current_step?: number[];
    selected_dimension?: number;
  };

  // Animation state (per-dimension)
  animation?: Array<{
    playing?: boolean;
    target_fps?: number;
    loop?: string;
    direction?: string;
    /** Per-dimension step override (absent = Auto). */
    step_size?: number;
  }>;

  // Story waypoints: camera poses bound to hidden-dimension positions.
  waypoints?: ZarrWaypoint[];

  // Sound layer defaults (master gain, buses, ducking, panning).
  audio?: ZarrAudioConfig;

  // Touch-panel authoring: heading, which dimension the tiles walk, per-chapter
  // overrides, author CSS. `unknown` on purpose — the shape is validated by
  // `config/zarr-bridge/control-panel.ts`, and declaring it structurally here
  // would let a caller read a field the validator has not checked.
  control_panel?: unknown;
}

/**
 * Zarr group attributes for the root scene
 */
export interface ZarrSceneAttrs {
  /** Scene format version */
  luxar_version?: string;

  /** Scene-level dimensions */
  scene_dimensions?: SceneDimensionAttrs;

  /** Scene type identifier */
  type?: 'scene' | string;

  /** Physical units */
  units?: string;

  /** Scene-level position bounds (union of all node bounds) */
  position_bounds?: PositionBounds;

  /** Viewer configuration hints from Python API */
  viewer_config?: ZarrViewerConfig;

  /** Any additional metadata */
  [key: string]: unknown;
}

/**
 * 4x4 transformation matrix laid out as a flat 16-element array,
 * column-major (THREE.js convention). Producers must transpose from
 * NumPy's row-major before serialising; the loader's `validateTransformFormat`
 * rejects row-major payloads with translation at indices [3,7,11].
 */
export type Matrix4x4 = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

/**
 * Zarr group attributes for nodes in the scene graph
 */
export interface ZarrNodeAttrs {
  /** Node type (points, group, etc.) */
  type?: 'points' | 'group' | string;

  /**
   * Transformation matrix (16 elements for a column-major 4x4 matrix).
   * Typed as a wider readonly number array on the input side because
   * zarr metadata is parsed dynamically; the {@link hasTransform} guard
   * narrows it to {@link Matrix4x4} once the length-16 invariant has
   * been verified.
   */
  transform?: readonly number[];

  /** Per-dimension transforms for non-displayed dimensions */
  nd_transform?: NdTransformMap;

  /** Rendering attributes */
  opacity?: number;
  absorption?: number;
  gamma?: number;
  intensity?: number;
  offset?: number;
  blending_mode?: string;
  point_size?: number;

  /** Points metadata */
  n_points?: number;
  max_radius?: number;

  /** Position bounds (nD bounding box) */
  position_bounds?: PositionBounds;

  /** Dimensions to extend visibility across (points visible at all values of these dimensions) */
  extend_to_all?: string[];

  /**
   * Sound node attrs (`type: "sound"`, heard not drawn) — the plain store key
   * holding the clip and whether a `positions` array exists. The playback and
   * panner knobs are read through `audio/sound-attrs.ts::parseSoundNodeAttrs`.
   */
  audio_file?: string;
  has_positions?: boolean;
  n_positions?: number;

  /** Arrays in this group */
  arrays?: string[];

  /**
   * Insertion order among siblings, stamped by the Python `Node` on add.
   * The scene graph is rebuilt from zarr consolidated metadata, whose
   * enumeration is alphabetical; the loader sorts siblings by this index
   * to restore napari-style addition order (e.g. the layers panel).
   */
  child_index?: number;

  /** Physical units */
  units?: string;

  /**
   * Colormap name. Built-in (e.g. 'viridis', 'magma') OR 'custom' — the
   * latter pairs with a `colormap_lut` zarr array sibling.
   */
  colormap?: string;

  /** Whether per-element scalar values are present for colormap lookup. */
  has_scalars?: boolean;

  /** `[min, max]` range used to normalise scalars before LUT lookup. */
  scalar_data_range?: [number, number];

  /**
   * bytes loaded from the node's `colormap_lut` zarr array when
   * `colormap === 'custom'`. Populated by `SceneLoader.buildSceneGraph`
   * and consumed by `NodeFactory` via
   * `getColormapTexture('custom', customLutBytes)`. Not authored at the
   * zarr level — purely a runtime hand-off.
   */
  customLutBytes?: Uint8Array;

  /** Any additional attributes */
  [key: string]: unknown;
}

/**
 * Zarr store interface with contents method
 */
export interface ZarrStoreWithContents {
  /** List contents of the store */
  contents(): Promise<Array<{ path: string; kind: 'group' | 'array' }>>;

  /** Other store properties */
  [key: string]: unknown;
}

/**
 * Type guard to check if a store has contents method
 */
export function hasContentsMethod(store: unknown): store is ZarrStoreWithContents {
  return (
    typeof store === 'object' &&
    store !== null &&
    typeof (store as Record<string, unknown>).contents === 'function'
  );
}

/**
 * Type guard: narrow `attrs.transform` from `readonly number[] | undefined`
 * to a {@link Matrix4x4} 16-tuple once the length-16 invariant has been
 * verified at runtime. The geometry-format check (column-major,
 * translation at indices [12,13,14]) is performed downstream by
 * `validateTransformFormat` in `node-factory.ts`.
 */
export function hasTransform(
  attrs: ZarrNodeAttrs
): attrs is ZarrNodeAttrs & { transform: Matrix4x4 } {
  return (
    attrs.transform !== undefined && Array.isArray(attrs.transform) && attrs.transform.length === 16
  );
}

/**
 * Type guard to check if attributes contain an nD transform
 */
export function hasNdTransform(
  attrs: ZarrNodeAttrs
): attrs is ZarrNodeAttrs & { nd_transform: NdTransformMap } {
  return (
    attrs.nd_transform !== undefined &&
    attrs.nd_transform !== null &&
    typeof attrs.nd_transform === 'object' &&
    !Array.isArray(attrs.nd_transform)
  );
}

/**
 * Type guard to check if attributes are for a points node
 */
export function isPointsNode(attrs: ZarrNodeAttrs): boolean {
  return attrs.type === 'points';
}

/**
 * Type guard to check if attributes contain scene dimensions
 */
export function hasSceneDimensions(
  attrs: ZarrSceneAttrs
): attrs is ZarrSceneAttrs & { scene_dimensions: SceneDimensionAttrs } {
  return attrs.scene_dimensions !== undefined && attrs.scene_dimensions.dimensions !== undefined;
}
