/**
 * The scene environment's authored configuration and baked-map contract
 * (`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3).
 *
 * Types only — this is the vocabulary the zarr bridge (`config/zarr-bridge`), the
 * loader (`data/loaders/environment`) and the renderer (`rendering/environment`) share,
 * and `types/` is the one layer all three may import.
 *
 * @module types/environment
 */

/**
 * Where the environment comes from. `room` is three's procedural RoomEnvironment (the
 * default); `scene` is an exact cube capture of the scene from a probe — the data IS the
 * light; `hdri` is an equirectangular image at `url`.
 */
export type EnvironmentSource = 'room' | 'scene' | 'hdri';

export const ENVIRONMENT_SOURCES: readonly EnvironmentSource[] = ['room', 'scene', 'hdri'];

/**
 * Where a `scene` capture looks out from: the scene bounds centre, a named node's
 * bounding-box centre (what a marker shell around a cluster wants), or a world position.
 */
export type EnvironmentProbe = 'auto' | { node: string } | { position: [number, number, number] };

/** Cube face size bounds for a `scene` capture (the map is prefiltered; more buys little). */
export const ENVIRONMENT_RESOLUTION_MIN = 16;
export const ENVIRONMENT_RESOLUTION_MAX = 1024;
export const ENVIRONMENT_RESOLUTION_DEFAULT = 128;

/** The validated, narrowed form of `viewer_config.environment`. */
export interface EnvironmentConfig {
  source: EnvironmentSource;
  probe: EnvironmentProbe;
  resolution: number;
  /** `scene.environmentIntensity`, `>= 0`. */
  intensity: number;
  /** Equirectangular image for `hdri`, relative to the store or absolute. */
  url?: string;
}

/** Default config — what an absent `viewer_config.environment` means. */
export const DEFAULT_ENVIRONMENT_CONFIG: EnvironmentConfig = {
  source: 'room',
  probe: 'auto',
  resolution: ENVIRONMENT_RESOLUTION_DEFAULT,
  intensity: 1,
};

/** Name of the root-level sidecar group a baked map lives in (Python: `ENVIRONMENT_GROUP`). */
export const ENVIRONMENT_GROUP = 'environment';

/** Face order of a baked map — three's `CubeTexture` order. */
export const ENVIRONMENT_FACE_ORDER = ['px', 'nx', 'py', 'ny', 'pz', 'nz'] as const;

/** The container / array `format` value (Python: `luxar.environment.container.ENVIRONMENT_FORMAT`). */
export const ENVIRONMENT_FORMAT = 'cube-faces-half';

/** The sample encoding stamped on the faces array (Python: `SAMPLE_FORMAT`). */
export const ENVIRONMENT_SAMPLE_FORMAT = 'half-float-bits';

/** Magic prefix of the bake container (Python: `luxar.environment.container.MAGIC`). */
export const ENVIRONMENT_CONTAINER_MAGIC = 'LXENV001';

/**
 * The header a bake produces and the `environment/` group's attrs carry (minus the
 * attach-time bookkeeping). Mirrors `luxar.environment.container.REQUIRED_HEADER_KEYS`.
 */
export interface BakedEnvironmentHeader {
  format: typeof ENVIRONMENT_FORMAT;
  face_order: readonly string[];
  /** The backend the capture ran on; recorded, not gated on (see `rendering/environment/baked.ts`). */
  coordinate_system: 'webgl' | 'webgpu';
  probe: { spec: string; position: [number, number, number] };
  resolution: number;
  scene_content_hash: string;
  /** The appearance state frozen into the map (informational). */
  appearance?: Record<string, unknown>;
  baked_at?: string;
  viewer_version?: string;
}

/** A baked map as loaded from the store: six faces of IEEE half-float bits, RGBA. */
export interface BakedEnvironment {
  header: BakedEnvironmentHeader;
  resolution: number;
  /** Six `resolution × resolution × 4` arrays in {@link ENVIRONMENT_FACE_ORDER}. */
  faces: Uint16Array[];
}
