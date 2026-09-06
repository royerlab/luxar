/**
 * Where a scene-derived environment capture looks out from.
 *
 * A cube map is exact only at its probe point (spec §3.3, "parallax"). `auto` is the
 * scene bounds centre — the showcase case; `node:<path>` is a node's world bounding-box
 * centre — what a marker shell around a cluster wants; a literal `[x, y, z]` is a world
 * position for everything else. Probes are resolved against the LIVE scene graph at
 * capture time, so a node that has not committed yet resolves to the scene centre with a
 * warning rather than to the origin.
 *
 * @module rendering/environment/probe
 */

import * as THREE from 'three';
import type { EnvironmentProbe } from '../../types/environment';
import { log, Modules } from '../../utils/log';

const _box = new THREE.Box3();

/**
 * Parse a probe spelled as text — the `?probe=` URL parameter and the CLI flag:
 * `auto`, `node:<path>`, or `x,y,z` (a bracketed `[x,y,z]` is tolerated). Returns
 * `null` for anything else.
 */
export function parseProbeSpec(spec: string | null | undefined): EnvironmentProbe | null {
  if (spec == null) return null;
  const text = spec.trim();
  if (text === 'auto') return 'auto';
  if (text.startsWith('node:')) {
    const node = text.slice('node:'.length).trim();
    return node ? { node } : null;
  }
  const parts = text.replace(/^\[|\]$/g, '').split(',');
  if (parts.length !== 3) return null;
  const xyz = parts.map((p) => Number(p.trim()));
  if (!xyz.every(Number.isFinite)) return null;
  return { position: [xyz[0], xyz[1], xyz[2]] };
}

/** The canonical text form of a probe (what the bake header records). */
export function formatProbeSpec(probe: EnvironmentProbe): string {
  if (probe === 'auto') return 'auto';
  if ('node' in probe) return `node:${probe.node}`;
  return probe.position.join(',');
}

/**
 * World-space bounding sphere of everything committed under `root`, or `null` when
 * nothing has bounds yet (a scene before its first commit).
 */
export function committedBoundingSphere(root: THREE.Object3D | null): THREE.Sphere | null {
  if (!root) return null;
  _box.makeEmpty();
  root.updateWorldMatrix(true, true);
  root.traverse((obj) => {
    const geometry = (obj as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
    if (!geometry || !obj.visible) return;
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    if (!bb || bb.isEmpty()) return;
    _box.union(bb.clone().applyMatrix4(obj.matrixWorld));
  });
  if (_box.isEmpty()) return null;
  const sphere = new THREE.Sphere();
  _box.getBoundingSphere(sphere);
  return Number.isFinite(sphere.radius) && sphere.radius > 0 ? sphere : null;
}

/**
 * Resolve a probe to a world position against the live scene graph.
 *
 * `node:<path>` matches by object name (the node factories name every node by its
 * store path, with the leading slash), falling back to the scene centre with a warning
 * when the node is absent or has no bounds. `auto` on an empty scene is the origin.
 */
export function resolveProbe(
  probe: EnvironmentProbe,
  root: THREE.Object3D | null,
  out: THREE.Vector3
): THREE.Vector3 {
  if (probe !== 'auto' && 'position' in probe) {
    return out.set(probe.position[0], probe.position[1], probe.position[2]);
  }
  if (probe !== 'auto') {
    const centre = nodeProbeCentre(probe.node, root);
    if (centre) return out.copy(centre);
  }
  const sphere = committedBoundingSphere(root);
  return sphere ? out.copy(sphere.center) : out.set(0, 0, 0);
}

/** The bounding-box centre of the named node, or `null` (with a warning) when it cannot be used. */
function nodeProbeCentre(name: string, root: THREE.Object3D | null): THREE.Vector3 | null {
  const path = name.startsWith('/') ? name : `/${name}`;
  const node = root?.getObjectByName(path) ?? root?.getObjectByName(name) ?? null;
  const sphere = committedBoundingSphere(node);
  if (sphere) return sphere.center;
  log.warning(
    Modules.RENDERER,
    `Environment probe node '${name}' ${node ? 'has no committed bounds yet' : 'was not found'}; ` +
      'capturing from the scene centre instead'
  );
  return null;
}
