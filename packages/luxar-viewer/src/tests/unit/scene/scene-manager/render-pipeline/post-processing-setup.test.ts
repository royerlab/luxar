/**
 * Unit tests for the post-processing-setup factory used by
 * SceneManager.
 *
 * The factory itself is thin (one constructor call + a log line);
 * these tests pin the canvas-size resolution path so a renamed
 * `clientWidth` / `clientHeight` field doesn't silently fall back
 * to `window.innerWidth / innerHeight`.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

vi.mock('../../../../../rendering/post-processing-manager', () => {
  const PostProcessingManager = vi.fn();
  return { PostProcessingManager };
});

import { createPostProcessing } from '../../../../../scene/scene-manager/render-pipeline/post-processing-setup';
import { PostProcessingManager } from '../../../../../rendering/post-processing-manager';
import type {
  Renderer,
  RendererCapabilities,
} from '../../../../../rendering/renderer-capabilities';

function makeRenderer(width: number, height: number): Renderer {
  const canvas = document.createElement('canvas');
  Object.defineProperty(canvas, 'clientWidth', { value: width, configurable: true });
  Object.defineProperty(canvas, 'clientHeight', { value: height, configurable: true });
  return { domElement: canvas } as unknown as Renderer;
}

describe('createPostProcessing', () => {
  it('passes canvas clientWidth/Height into PostProcessingManager constructor', () => {
    const renderer = makeRenderer(1024, 768);
    const capabilities = {} as RendererCapabilities;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const onResize = vi.fn();

    createPostProcessing({ renderer, capabilities, scene, camera, onResize });

    expect(PostProcessingManager).toHaveBeenCalledTimes(1);
    const args = vi.mocked(PostProcessingManager).mock.calls[0];
    expect(args[0]).toBe(renderer);
    expect(args[1]).toBe(capabilities);
    expect(args[2]).toBe(scene);
    expect(args[3]).toBe(camera);
    expect(args[4]).toEqual({ width: 1024, height: 768 });
    expect(args[5]).toBe(onResize);
  });

  it('falls back to window dimensions when canvas clientWidth/Height are 0', () => {
    const renderer = makeRenderer(0, 0);
    const capabilities = {} as RendererCapabilities;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();

    createPostProcessing({
      renderer,
      capabilities,
      scene,
      camera,
      onResize: vi.fn(),
    });

    const args = vi.mocked(PostProcessingManager).mock.calls.at(-1)!;
    expect(args[4]).toEqual({
      width: window.innerWidth,
      height: window.innerHeight,
    });
  });
});
