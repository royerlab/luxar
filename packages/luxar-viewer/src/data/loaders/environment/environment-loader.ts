/**
 * Load a baked environment map from the store's root-level `environment/` sidecar
 * group (`luxar env attach`; spec `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3).
 *
 * The group carries neither `type` nor `kind`, so `build-scene-graph.ts` skips it as a
 * metadata sidecar — nothing reads it implicitly, which is why this loader exists. Its
 * `faces` attr names the live array (`faces-<digest>`; a re-bake is a NEW path, so a
 * caching store can never serve stale faces), and `scene_content_hash` is the guard:
 * the environment group is EXCLUDED from the scene digest on the Python side, so this
 * comparison against the root's `content_hash` is exact. A mismatch means the scene
 * changed since the bake; the map is ignored with a console line and the viewer falls
 * back to its configured source or the room.
 *
 * Absence is the normal case and costs one metadata probe. Every failure path degrades
 * to "no baked map" rather than failing the scene load — the environment is lighting,
 * not data.
 *
 * @module data/loaders/environment/environment-loader
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import {
  ENVIRONMENT_FACE_ORDER,
  ENVIRONMENT_FORMAT,
  ENVIRONMENT_GROUP,
  ENVIRONMENT_SAMPLE_FORMAT,
  type BakedEnvironment,
  type BakedEnvironmentHeader,
} from '../../../types/environment';

const TAG = '[🌐] [Environment]';

/**
 * Read the baked map, or `null` when the store has none / it is stale / it is
 * malformed. `rootContentHash` is the root `content_hash` from the freshly fetched
 * root document (the same value the cache validation reads).
 */
export async function loadBakedEnvironment(
  rootLoc: zarr.Location<zarr.Readable>,
  rootContentHash: string | undefined
): Promise<BakedEnvironment | null> {
  let attrs: Record<string, unknown>;
  try {
    const group = await zarr.open(rootLoc.resolve(ENVIRONMENT_GROUP), { kind: 'group' });
    attrs = (group.attrs ?? {}) as Record<string, unknown>;
  } catch {
    return null; // no environment group — the normal case
  }

  const verdict = judgeEnvironmentAttrs(attrs, rootContentHash);
  if ('reject' in verdict) return reject(verdict.reject);
  if ('stale' in verdict) {
    log.warning(
      Modules.SCENE_LOADER,
      `${TAG} baked map is stale (scene changed since bake: ${verdict.stale.slice(0, 12)}… vs ` +
        `${(rootContentHash ?? '').slice(0, 12)}…) — ignoring it; re-run \`luxar env bake\``
    );
    return null;
  }
  const { arrayName, resolution } = verdict;

  let data: Uint16Array;
  try {
    data = await readFacesArray(rootLoc, arrayName);
  } catch (error) {
    return reject(`could not read ${ENVIRONMENT_GROUP}/${arrayName}: ${String(error)}`);
  }

  const perFace = resolution * resolution * 4;
  if (data.length !== perFace * ENVIRONMENT_FACE_ORDER.length) {
    return reject(
      `${ENVIRONMENT_GROUP}/${arrayName} has ${data.length} samples; a ${resolution}px cube needs ${perFace * 6}`
    );
  }
  const faces = ENVIRONMENT_FACE_ORDER.map((_, i) => data.subarray(i * perFace, (i + 1) * perFace));
  const header = attrs as unknown as BakedEnvironmentHeader;
  log.info(
    Modules.SCENE_LOADER,
    `${TAG} baked ${resolution}px map found (${ENVIRONMENT_GROUP}/${arrayName}, probe ${header.probe?.spec ?? '?'})`
  );
  return { header, resolution, faces };
}

function reject(why: string): null {
  log.warning(Modules.SCENE_LOADER, `${TAG} ignoring the baked map: ${why}`);
  return null;
}

type AttrsVerdict =
  { reject: string } | { stale: string } | { arrayName: string; resolution: number };

/** Validate the group attrs against the store contract; pure, so the rules are testable one by one. */
function judgeEnvironmentAttrs(
  attrs: Record<string, unknown>,
  rootContentHash: string | undefined
): AttrsVerdict {
  const shape = judgeShapeAttrs(attrs);
  if (typeof shape === 'string') return { reject: shape };
  const stamped = attrs.scene_content_hash;
  if (typeof stamped !== 'string' || !stamped) {
    return { reject: 'it records no scene_content_hash' };
  }
  if (rootContentHash === undefined) {
    return { reject: 'the scene carries no content_hash to check it against' };
  }
  if (stamped !== rootContentHash) return { stale: stamped };
  const arrayName = attrs.faces;
  if (typeof arrayName !== 'string' || !arrayName) return { reject: 'it names no faces array' };
  return { arrayName, resolution: shape };
}

/** The format / face-order / resolution rules: the resolution when they hold, else the reason. */
function judgeShapeAttrs(attrs: Record<string, unknown>): number | string {
  if (attrs.format !== ENVIRONMENT_FORMAT) {
    return `format ${JSON.stringify(attrs.format)} is not '${ENVIRONMENT_FORMAT}'`;
  }
  if (attrs.sample_format !== undefined && attrs.sample_format !== ENVIRONMENT_SAMPLE_FORMAT) {
    return `sample_format ${JSON.stringify(attrs.sample_format)} is not supported`;
  }
  if (!isThreeFaceOrder(attrs.face_order)) {
    return `face_order ${JSON.stringify(attrs.face_order)} is not ${JSON.stringify(ENVIRONMENT_FACE_ORDER)}`;
  }
  const resolution = attrs.resolution;
  if (!Number.isInteger(resolution) || (resolution as number) < 1) {
    return `resolution ${JSON.stringify(resolution)} is not a positive integer`;
  }
  return resolution as number;
}

function isThreeFaceOrder(order: unknown): boolean {
  return (
    Array.isArray(order) &&
    order.length === ENVIRONMENT_FACE_ORDER.length &&
    order.every((f, i) => f === ENVIRONMENT_FACE_ORDER[i])
  );
}

async function readFacesArray(
  rootLoc: zarr.Location<zarr.Readable>,
  arrayName: string
): Promise<Uint16Array> {
  const handle = await zarr.open(rootLoc.resolve(`${ENVIRONMENT_GROUP}/${arrayName}`), {
    kind: 'array',
  });
  const chunk = await zarr.readArray(handle);
  const raw = chunk.data as unknown;
  return raw instanceof Uint16Array ? raw : Uint16Array.from(raw as ArrayLike<number>);
}
