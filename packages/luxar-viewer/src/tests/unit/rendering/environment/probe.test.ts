/**
 * Probe parsing and resolution for the scene-derived environment capture.
 *
 * A cube map is exact only at its probe, so where `auto` / `node:<path>` / `x,y,z`
 * land is the whole fidelity story for a marker shell; pinned against a small live
 * scene graph rather than described.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  committedBoundingSphere,
  formatProbeSpec,
  parseProbeSpec,
  resolveProbe,
} from '../../../../rendering/environment/probe';

function boxMesh(name: string, center: [number, number, number], half = 1): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(half * 2, half * 2, half * 2);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  mesh.name = name;
  mesh.position.set(...center);
  return mesh;
}

describe('parseProbeSpec / formatProbeSpec', () => {
  it.each([
    ['auto', 'auto'],
    [' auto ', 'auto'],
    ['node:clusters/shell_3', { node: 'clusters/shell_3' }],
    ['1,2,3', { position: [1, 2, 3] }],
    ['[0.5, -1, 2e1]', { position: [0.5, -1, 20] }],
  ] as const)('parses %j', (spec, expected) => {
    expect(parseProbeSpec(spec)).toEqual(expected);
  });

  it.each(['', 'centre', 'node:', '1,2', '1,x,3', null, undefined])('rejects %j', (spec) => {
    expect(parseProbeSpec(spec as string | null | undefined)).toBeNull();
  });

  it('formats back to the canonical text the bake header records', () => {
    expect(formatProbeSpec('auto')).toBe('auto');
    expect(formatProbeSpec({ node: 'shell' })).toBe('node:shell');
    expect(formatProbeSpec({ position: [1, 2.5, -3] })).toBe('1,2.5,-3');
    for (const spec of ['auto', 'node:a/b', '1,2,3']) {
      expect(formatProbeSpec(parseProbeSpec(spec)!)).toBe(spec);
    }
  });
});

describe('resolveProbe', () => {
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  root.add(boxMesh('/left', [-4, 0, 0]), boxMesh('/right', [4, 2, 0]));
  root.updateMatrixWorld(true);

  it('auto is the centre of the committed bounds; an empty scene resolves to the origin', () => {
    const out = new THREE.Vector3();
    expect(resolveProbe('auto', root, out).toArray()).toEqual([0, 1, 0]);
    expect(resolveProbe('auto', new THREE.Group(), out).toArray()).toEqual([0, 0, 0]);
    expect(resolveProbe('auto', null, out).toArray()).toEqual([0, 0, 0]);
  });

  it('a literal position is used as given', () => {
    const out = new THREE.Vector3();
    expect(resolveProbe({ position: [1, 2, 3] }, root, out).toArray()).toEqual([1, 2, 3]);
  });

  it("node:<path> is that node's world bounding-box centre, with or without the leading slash", () => {
    const out = new THREE.Vector3();
    expect(resolveProbe({ node: '/right' }, root, out).toArray()).toEqual([4, 2, 0]);
    expect(resolveProbe({ node: 'right' }, root, out).toArray()).toEqual([4, 2, 0]);
  });

  it('a missing node falls back to the scene centre rather than the origin', () => {
    const out = new THREE.Vector3();
    expect(resolveProbe({ node: '/nowhere' }, root, out).toArray()).toEqual([0, 1, 0]);
  });

  it('committedBoundingSphere ignores hidden objects and geometry without bounds', () => {
    const scene = new THREE.Group();
    const hidden = boxMesh('/hidden', [100, 0, 0]);
    hidden.visible = false;
    const empty = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    scene.add(hidden, empty, boxMesh('/a', [1, 1, 1]));
    const sphere = committedBoundingSphere(scene);
    expect(sphere).not.toBeNull();
    expect(sphere!.center.toArray()).toEqual([1, 1, 1]);
    expect(committedBoundingSphere(new THREE.Group())).toBeNull();
  });
});
