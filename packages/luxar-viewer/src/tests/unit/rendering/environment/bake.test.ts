/**
 * The bake: the container layout the Python side (`luxar.environment.container`) reads, and the
 * face readback threading (WebGL's 7th `activeCubeFaceIndex` argument) through a stub
 * renderer that fills each face with its own index.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  bakeEnvironment,
  containerToBase64,
  packEnvironmentContainer,
} from '../../../../rendering/environment/bake';
import {
  SceneEnvironment,
  type PmremGeneratorLike,
} from '../../../../rendering/environment/scene-environment';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';
import {
  ENVIRONMENT_CONTAINER_MAGIC,
  ENVIRONMENT_FACE_ORDER,
  ENVIRONMENT_FORMAT,
  type BakedEnvironmentHeader,
} from '../../../../types/environment';

const HEADER: BakedEnvironmentHeader = {
  format: ENVIRONMENT_FORMAT,
  face_order: [...ENVIRONMENT_FACE_ORDER],
  coordinate_system: 'webgl',
  probe: { spec: 'auto', position: [0, 1, 2] },
  resolution: 2,
  scene_content_hash: 'deadbeef',
};

/** Parse the container the way `luxar.environment.container.unpack` does. */
function unpack(bytes: Uint8Array): { header: unknown; samples: Uint16Array } {
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  expect(magic).toBe(ENVIRONMENT_CONTAINER_MAGIC);
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const n = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + n)));
  const rest = bytes.subarray(12 + n);
  const samples = new Uint16Array(rest.length / 2);
  for (let i = 0; i < samples.length; i++) samples[i] = view.getUint16(12 + n + i * 2, true);
  return { header, samples };
}

describe('packEnvironmentContainer', () => {
  it('lays out magic, u32 header length, JSON header, then little-endian halves', () => {
    const faces = Array.from({ length: 6 }, (_, f) => new Uint16Array(2 * 2 * 4).fill(0x3c00 + f));
    const bytes = packEnvironmentContainer(HEADER, faces);
    const { header, samples } = unpack(bytes);
    expect(header).toEqual(HEADER);
    expect(samples.length).toBe(6 * 2 * 2 * 4);
    for (let f = 0; f < 6; f++) {
      expect(samples[f * 16]).toBe(0x3c00 + f);
      expect(samples[f * 16 + 15]).toBe(0x3c00 + f);
    }
    // Byte-level: the first sample of face 0 is 0x3c00 little-endian.
    const n = new DataView(bytes.buffer).getUint32(8, true);
    expect(bytes[12 + n]).toBe(0x00);
    expect(bytes[12 + n + 1]).toBe(0x3c);
  });

  it('base64 round-trips the bytes (the Playwright hand-off)', () => {
    const bytes = packEnvironmentContainer(HEADER, [
      new Uint16Array(16),
      new Uint16Array(16),
      new Uint16Array(16),
      new Uint16Array(16),
      new Uint16Array(16),
      new Uint16Array(16),
    ]);
    const decoded = Uint8Array.from(atob(containerToBase64(bytes)), (c) => c.charCodeAt(0));
    expect(decoded).toEqual(bytes);
  });
});

describe('bakeEnvironment', () => {
  it('captures, reads the six faces through the face-index slot, and records the header', async () => {
    const scene = new THREE.Scene();
    const readCalls: Array<{ target: unknown; face: number | undefined }> = [];
    const renderer = {
      isWebGLRenderer: true,
      coordinateSystem: THREE.WebGLCoordinateSystem,
      xr: { enabled: false },
      state: { buffers: { depth: { getReversed: () => false } } },
      getRenderTarget: () => null,
      getActiveCubeFace: () => 0,
      getActiveMipmapLevel: () => 0,
      setRenderTarget: () => {},
      render: () => {},
      readRenderTargetPixelsAsync: vi.fn(
        async (
          target: unknown,
          _x: number,
          _y: number,
          _w: number,
          _h: number,
          buffer: Uint16Array,
          face?: number
        ) => {
          readCalls.push({ target, face });
          buffer.fill(0x4000 + (face ?? 0)); // half 2.0 + face id
        }
      ),
    };
    const generator: PmremGeneratorLike = {
      fromScene: () => ({ texture: new THREE.Texture(), dispose: () => {} }),
      dispose: () => {},
    };
    const target = { texture: new THREE.CubeTexture(), width: 4, height: 4, dispose: () => {} };
    const environment = new SceneEnvironment({
      scene,
      renderer,
      createGenerator: () => generator,
      createCubeTarget: () => target,
    });
    environment.attachRuntime({
      sceneRoot: () => null,
      pushCaptureCameraParams: () => {},
      restoreCameraParams: () => {},
      isSettled: () => true,
      baseUrl: () => undefined,
    });
    const capabilities = {
      apiSurface: 'webgl2',
      framebufferYDown: false,
    } as unknown as RendererCapabilities;

    const result = await bakeEnvironment({
      environment,
      renderer: renderer as unknown as THREE.WebGLRenderer,
      capabilities,
      probe: { position: [1, 2, 3] },
      resolution: 4,
      sceneContentHash: 'cafe',
      viewerVersion: '1.2.3',
    });

    expect(readCalls.map((c) => c.face)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(readCalls.every((c) => c.target === target)).toBe(true);
    expect(result.faces).toHaveLength(6);
    expect(result.faces[3][0]).toBe(0x4003);
    expect(result.faces[3].length).toBe(4 * 4 * 4);
    expect(result.header.resolution).toBe(4);
    expect(result.header.probe).toEqual({ spec: '1,2,3', position: [1, 2, 3] });
    expect(result.header.scene_content_hash).toBe('cafe');
    expect(result.header.coordinate_system).toBe('webgl');
    expect(result.header.viewer_version).toBe('1.2.3');
    expect(result.header.face_order).toEqual([...ENVIRONMENT_FACE_ORDER]);
    const { header, samples } = unpack(result.bytes);
    expect((header as BakedEnvironmentHeader).scene_content_hash).toBe('cafe');
    expect(samples.length).toBe(6 * 4 * 4 * 4);
    expect(environment.captureCount).toBe(1);
  });

  it('refuses without a capture runtime', async () => {
    const environment = new SceneEnvironment({
      scene: new THREE.Scene(),
      renderer: {},
      createGenerator: () => ({
        fromScene: () => ({ texture: new THREE.Texture(), dispose() {} }),
        dispose() {},
      }),
      createCubeTarget: () => ({
        texture: new THREE.CubeTexture(),
        width: 1,
        height: 1,
        dispose() {},
      }),
    });
    await expect(
      bakeEnvironment({
        environment,
        renderer: {} as THREE.WebGLRenderer,
        capabilities: { apiSurface: 'webgl2' } as unknown as RendererCapabilities,
        sceneContentHash: 'x',
        viewerVersion: 'v',
      })
    ).rejects.toThrow('no capture runtime');
  });
});
