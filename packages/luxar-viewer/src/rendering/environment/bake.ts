/**
 * `?bakeEnv`: capture the scene-derived environment once and hand it back as the
 * container `luxar env attach` writes into the store.
 *
 * The capture is the very same `SceneEnvironment.captureScene` a live `scene` source
 * runs — so what gets baked is exactly what a viewer would have captured live — read
 * back as HALF floats (a PNG would clip everything above 1.0 and destroy the highlights
 * that make metals read). The container is defined once, in
 * `luxar.environment.container` (Python) and mirrored here: `LXENV001`, a little-endian u32
 * header length, a UTF-8 JSON header, then the six faces as `uint16` half bits in
 * three's `px, nx, py, ny, pz, nz` order, RGBA, GL memory (bottom-up) row order.
 *
 * @module rendering/environment/bake
 */

import type * as THREE from 'three';
import type { Renderer, RendererCapabilities } from '../renderer-capabilities';
import {
  ENVIRONMENT_CONTAINER_MAGIC,
  ENVIRONMENT_FACE_ORDER,
  ENVIRONMENT_FORMAT,
  type BakedEnvironmentHeader,
  type EnvironmentProbe,
} from '../../types/environment';
import { readPixelsCompactAsync } from '../post-processing/hdr/pixel-utils';
import { formatProbeSpec } from './probe';
import type { SceneEnvironment } from './scene-environment';

/** Everything a bake needs, injected. */
export interface BakeRequest {
  environment: SceneEnvironment;
  renderer: Renderer;
  capabilities: RendererCapabilities;
  /** Overrides for the scene's configured probe / resolution (the URL parameters). */
  probe?: EnvironmentProbe;
  resolution?: number;
  /** The scene's root `content_hash`; the attach and the viewer guard on it. */
  sceneContentHash: string;
  /** The appearance state frozen into the map (informational). */
  appearance?: Record<string, unknown>;
  viewerVersion: string;
}

/** The bake's output: the header, the faces, and the packed container. */
export interface BakeResult {
  header: BakedEnvironmentHeader;
  faces: Uint16Array[];
  bytes: Uint8Array;
}

/** Capture and read back the six faces; throws when no capture runtime is attached. */
export async function bakeEnvironment(req: BakeRequest): Promise<BakeResult> {
  const capture = req.environment.captureScene({ probe: req.probe, resolution: req.resolution });
  if (!capture) {
    throw new Error('Environment bake: the scene environment has no capture runtime attached');
  }
  const faces: Uint16Array[] = [];
  for (let face = 0; face < ENVIRONMENT_FACE_ORDER.length; face++) {
    const { pixels } = await readPixelsCompactAsync(req.renderer, req.capabilities, {
      target: capture.target as unknown as THREE.RenderTarget,
      kind: 'rgba16f',
      faceIndex: face,
      // GL memory order (bottom-up), so the baked texture uploads with `flipY=false`
      // land every texel where the render target had it (`./baked.ts`).
      flipY: true,
    });
    faces.push(pixels);
  }
  const header: BakedEnvironmentHeader = {
    format: ENVIRONMENT_FORMAT,
    face_order: [...ENVIRONMENT_FACE_ORDER],
    coordinate_system: req.capabilities.apiSurface === 'webgl2' ? 'webgl' : 'webgpu',
    probe: {
      spec: formatProbeSpec(capture.probe),
      position: [capture.probePosition.x, capture.probePosition.y, capture.probePosition.z],
    },
    resolution: capture.resolution,
    scene_content_hash: req.sceneContentHash,
    appearance: req.appearance ?? {},
    baked_at: new Date().toISOString(),
    viewer_version: req.viewerVersion,
  };
  return { header, faces, bytes: packEnvironmentContainer(header, faces) };
}

/** Serialize header + faces into the `LXENV001` container (see the module doc). */
export function packEnvironmentContainer(
  header: BakedEnvironmentHeader,
  faces: Uint16Array[]
): Uint8Array {
  const magic = new TextEncoder().encode(ENVIRONMENT_CONTAINER_MAGIC);
  const body = new TextEncoder().encode(JSON.stringify(header));
  const samples = faces.reduce((n, f) => n + f.length, 0);
  const out = new Uint8Array(magic.length + 4 + body.length + samples * 2);
  let offset = 0;
  out.set(magic, offset);
  offset += magic.length;
  new DataView(out.buffer).setUint32(offset, body.length, true);
  offset += 4;
  out.set(body, offset);
  offset += body.length;
  // Little-endian halves, whatever the host order (the Python side reads `<u2`).
  const view = new DataView(out.buffer, offset);
  let i = 0;
  for (const face of faces) {
    for (let k = 0; k < face.length; k++, i += 2) view.setUint16(i, face[k], true);
  }
  return out;
}

/** Base64 of the container, for the Playwright driver's `page.evaluate` hand-off. */
export function containerToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
