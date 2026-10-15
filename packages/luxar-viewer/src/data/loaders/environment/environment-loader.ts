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
 * Absence is the normal case and costs NO request when a root index fetched from the
 * network this load is available: `luxar env attach` re-consolidates after writing the
 * group, so a FRESH index is authoritative and one that lists no `environment` node
 * means there is none. A CACHED index is not: the L2 tier revalidates the root only by
 * `content_hash`, which `env attach` deliberately leaves unchanged, so a warm cache can
 * keep serving the pre-attach index. On a warm load of a format-3 store the scene loader
 * still has this load's network copy of the index (the load-time root-document fetch;
 * `data/scene-loader/cache/root-listing.ts`). Only without one (format 2, a presigned
 * or zipped source) is the group probed — one document of the store's known format —
 * and a genuine absence is remembered per dataset for the session, so a revisit does
 * not pay it again. Every failure path degrades to "no baked
 * map" rather than failing the scene load — the environment is lighting, not data.
 *
 * @module data/loaders/environment/environment-loader
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { hasContentsMethod } from '../../../types/zarr';
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
 * A current answer about root-level groups (structurally
 * `data/scene-loader/cache/root-listing.ts::RootListing`): `lists` is
 * `true` / `false` from an index fetched from the network this load, `undefined`
 * when there is none; `format` is the store's format when known.
 */
export interface EnvironmentRootListing {
  lists(group: string): boolean | undefined;
  readonly format: zarr.StoreFormat | null;
  /** Whether the answer came from the opened store's own (network-fresh) index. */
  readonly fromOpenedIndex?: boolean;
}

/**
 * Where to read the environment group's attrs from. The opened store answers
 * from its consolidated index, so when that index is not this load's network
 * copy a warm cache can hand back the PRE-re-bake attrs (`env attach` keeps
 * `content_hash`) — read them from the source store instead, as chunk packs do.
 * The faces array is digest-named, so it is read through the opened store
 * either way.
 */
function attrsLocation(
  rootLoc: zarr.Location<zarr.Readable>,
  index: boolean | EnvironmentRootListing,
  sourceRootLoc: (() => zarr.Location<zarr.Readable>) | undefined
): zarr.Location<zarr.Readable> {
  if (!sourceRootLoc) return rootLoc;
  const openedIndexFresh = typeof index === 'boolean' ? index : index.fromOpenedIndex === true;
  return openedIndexFresh ? rootLoc : sourceRootLoc();
}

/**
 * Datasets (by the caller's store key — the normalized dataset URL) whose probe
 * found no environment group this session. Only a NOT-FOUND is recorded: a network
 * fault says nothing about the sidecar and must be retried on the next load.
 */
const knownAbsent = new Set<string>();

/** Forget every remembered absence (tests only). */
export function resetEnvironmentProbeCacheForTests(): void {
  knownAbsent.clear();
}

/**
 * Whether the store's consolidated index lists the environment group: `true` /
 * `false` when there is an index, `undefined` when there is none (the answer
 * then needs a probe).
 */
async function indexListsEnvironment(store: unknown): Promise<boolean | undefined> {
  if (!hasContentsMethod(store)) return undefined;
  try {
    const contents = await store.contents();
    return contents.some(
      (entry) => entry.path.replace(/^\/+/, '') === ENVIRONMENT_GROUP && entry.kind === 'group'
    );
  } catch {
    return undefined;
  }
}

/**
 * What a root index says about the environment group: `listed` (`undefined` when
 * there is no index to ask) and whether that answer is `current` enough that a
 * `false` proves absence (see the module doc).
 */
async function environmentListing(
  store: unknown,
  index: boolean | EnvironmentRootListing
): Promise<{ listed: boolean | undefined; current: boolean; format: zarr.StoreFormat | null }> {
  if (typeof index !== 'boolean') {
    const listed = index.lists(ENVIRONMENT_GROUP);
    return { listed, current: listed !== undefined, format: index.format };
  }
  return {
    listed: await indexListsEnvironment(store),
    current: index,
    format: zarr.storeFormatOf(store),
  };
}

/** Open the environment group (one format's document when the format is known). */
async function openEnvironmentAttrs(
  location: zarr.Location<zarr.Readable>,
  format: zarr.StoreFormat | null
): Promise<Record<string, unknown>> {
  const group =
    format === null
      ? await zarr.open(location, { kind: 'group' })
      : await zarr.openGroupOfFormat(location, format);
  return (group.attrs ?? {}) as Record<string, unknown>;
}

/**
 * The environment group's attrs, or `null` when the store has none. An index that
 * does not list the group is trusted only when current (see the module doc).
 */
async function readEnvironmentAttrs(
  rootLoc: zarr.Location<zarr.Readable>,
  storeKey: string | undefined,
  index: boolean | EnvironmentRootListing,
  sourceRootLoc?: () => zarr.Location<zarr.Readable>
): Promise<Record<string, unknown> | null> {
  const { listed, current, format } = await environmentListing(rootLoc.store, index);
  if (listed === false && current) return null;
  // Not listed by a trustworthy index: this read is a probe, whose 404 is remembered.
  const probeKey = listed === true ? undefined : storeKey;
  if (probeKey !== undefined && knownAbsent.has(probeKey)) return null;
  try {
    // One format's document when the format is known — a guess at both cost up
    // to three 404s for the (normal) absent case.
    const location = attrsLocation(rootLoc, index, sourceRootLoc);
    return await openEnvironmentAttrs(location.resolve(ENVIRONMENT_GROUP), format);
  } catch (error) {
    if (probeKey !== undefined && zarr.isNotFoundError(error)) knownAbsent.add(probeKey);
    return null; // no environment group — the normal case
  }
}

/**
 * Read the baked map, or `null` when the store has none / it is stale / it is
 * malformed. `rootContentHash` is the root `content_hash` from the freshly fetched
 * root document (the same value the cache validation reads). `storeKey` (the
 * normalized dataset URL) scopes the session's negative cache for probed stores; omit
 * it to always probe. `index` says which root index is current: `true` when the
 * store's consolidated index came from the network this load (not a cache tier), so
 * its silence about `environment` is proof of absence; a {@link EnvironmentRootListing}
 * (which can also answer from the load-time root-document fetch on a warm load); or
 * the default `false`, which probes. `sourceRootLoc`, when given, is where the group's
 * attrs are read whenever the opened store's own index is not network-fresh (see
 * `attrsLocation`).
 */
export async function loadBakedEnvironment(
  rootLoc: zarr.Location<zarr.Readable>,
  rootContentHash: string | undefined,
  storeKey?: string,
  index: boolean | EnvironmentRootListing = false,
  sourceRootLoc?: () => zarr.Location<zarr.Readable>
): Promise<BakedEnvironment | null> {
  const attrs = await readEnvironmentAttrs(rootLoc, storeKey, index, sourceRootLoc);
  if (attrs === null) return null;

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
