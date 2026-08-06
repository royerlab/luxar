/**
 * Render executors for the TSL ↔ GLSL parity harness: `renderGLSL`
 * drives the GLSL3 `ShaderMaterial` path through `THREE.WebGLRenderer`;
 * `renderTSL` drives the NodeMaterial path through
 * `WebGPURenderer({ forceWebGL: true })` and captures the generated
 * shader strings for the codegen-snapshot spec.
 *
 * @module tests/e2e/harnesses/tsl-harness/render
 */

import * as THREE from 'three';
import { requireWebGLSources } from '../../../../rendering/materials/_shared/shader-source';
import { buildDefaultCamera } from './shared';
// Intentional module cycle: `index.ts` assembles SHADER_REGISTRY from the
// family modules and re-exports these executors. The registry binding is
// only dereferenced inside the function bodies (long after the module
// graph has evaluated), so the cycle is benign under ESM live bindings.
import { SHADER_REGISTRY } from './index';

/** Edge length (px) of the square offscreen render target the parity harness draws into. */
export const HARNESS_SIZE = 64;

/**
 * Render the GLSL3 path of a registered shader to an offscreen target
 * and return the readback pixel buffer (RGBA8, length = w*h*4).
 */
export function renderGLSL(shaderName: string): Uint8Array {
  const entry = SHADER_REGISTRY[shaderName];
  if (!entry) throw new Error(`Unknown shader: ${shaderName}`);

  // GLSL parity render — the harness can only check parity when the
  // registry entry ships a GLSL fallback. `requireWebGLSources`
  // throws with a clear diagnostic if it doesn't (shouldn't happen
  // for any shader currently in the registry).
  const glsl = requireWebGLSources(entry.source);
  const uniforms = entry.buildUniforms();
  // Build the ShaderMaterial. We pass `defines` only when the
  // registry entry supplies it — Three.js warns "parameter 'defines'
  // has value of undefined" otherwise.
  const materialParams: THREE.ShaderMaterialParameters = {
    vertexShader: glsl.vertex,
    fragmentShader: glsl.fragment,
    uniforms,
    glslVersion: THREE.GLSL3,
    // Depth stays off for the single-primitive parity cases; entries
    // whose scenario needs fragments to COMPETE through the depth test
    // (surface-pick depth) opt in via `depthCompete`.
    depthTest: entry.depthCompete ?? false,
    depthWrite: entry.depthCompete ?? false,
    transparent: false,
  };
  if (entry.buildDefines) {
    materialParams.defines = entry.buildDefines();
  }
  if (entry.vertexColors) {
    materialParams.vertexColors = true;
  }
  const material = new THREE.ShaderMaterial(materialParams);

  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);

  const target = new THREE.WebGLRenderTarget(HARNESS_SIZE, HARNESS_SIZE, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });

  const scene = new THREE.Scene();
  // Default is an OrthographicCamera at (0,0,1) so world (0,0,0) projects to
  // NDC (0,0) — viewport centre (point shaders depend on this for centring).
  // Cases override via `buildCamera` (e.g. the perspective behind-camera guard).
  const camera = entry.buildCamera ? entry.buildCamera() : buildDefaultCamera();
  const mesh = entry.buildMesh
    ? entry.buildMesh(material)
    : new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(mesh);

  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  const pixels = new Uint8Array(HARNESS_SIZE * HARNESS_SIZE * 4);
  renderer.readRenderTargetPixels(target, 0, 0, HARNESS_SIZE, HARNESS_SIZE, pixels);

  target.dispose();
  if ('geometry' in mesh) (mesh as THREE.Mesh | THREE.Points).geometry.dispose();
  material.dispose();
  (uniforms.uInput?.value as THREE.Texture | null | undefined)?.dispose();
  renderer.dispose();

  return pixels;
}

/**
 * Render the TSL path of a registered shader via
 * `WebGPURenderer({ forceWebGL: true })` and return both the readback
 * pixels and the generated GLSL strings.
 *
 * Capturing the GLSL strings requires walking the `WebGLBackend`'s
 * pipeline cache after the render completes — there's no public
 * "give me the source" API, so we read the strings off the
 * `NodeBuilderState` that the backend stashes per-RenderObject.
 */
export async function renderTSL(
  shaderName: string
): Promise<{ pixels: Uint8Array; vertexShader: string; fragmentShader: string }> {
  const entry = SHADER_REGISTRY[shaderName];
  if (!entry) throw new Error(`Unknown shader: ${shaderName}`);
  if (!entry.source.webgpu) {
    throw new Error(`Shader ${shaderName} has no TSL factory yet`);
  }

  const uniforms = entry.buildUniforms();
  const material = entry.buildTSLMaterial
    ? entry.buildTSLMaterial(uniforms)
    : (entry.source.webgpu(uniforms) as THREE.Material);

  const { WebGPURenderer } = await import('three/webgpu');
  const renderer = new WebGPURenderer({ antialias: false, alpha: false, forceWebGL: true });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);
  await renderer.init();

  // Capture the generated GLSL / WGSL strings by patching the renderer's
  // NodeManager. `_createNodeBuilderState(nodeBuilder)` is called by
  // both the sync and async build paths and receives a builder whose
  // `.vertexShader` / `.fragmentShader` strings are fully populated.
  // We hook it once per renderer instance and restore right after.
  // (See node_modules/three/src/renderers/common/nodes/NodeManager.js:469.)

  const nodesInstance = (renderer as unknown as { _nodes: any })._nodes;

  const origCreateState = nodesInstance._createNodeBuilderState.bind(nodesInstance);
  // Ref object so TS doesn't narrow `value` to `null` through the
  // closure mutation below.
  const capturedRef: { value: { vertex: string; fragment: string } | null } = {
    value: null,
  };

  nodesInstance._createNodeBuilderState = function (nodeBuilder: any) {
    if (!capturedRef.value && nodeBuilder?.material === material) {
      capturedRef.value = {
        vertex: typeof nodeBuilder.vertexShader === 'string' ? nodeBuilder.vertexShader : '',
        fragment: typeof nodeBuilder.fragmentShader === 'string' ? nodeBuilder.fragmentShader : '',
      };
    }
    return origCreateState(nodeBuilder);
  };

  const target = new THREE.WebGLRenderTarget(HARNESS_SIZE, HARNESS_SIZE, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });

  const scene = new THREE.Scene();
  // Mirrors the GLSL path's camera setup (default ortho, or the entry's
  // `buildCamera` override). See renderGLSL for the rationale.
  const camera = entry.buildCamera ? entry.buildCamera() : buildDefaultCamera();
  const mesh = entry.buildMesh
    ? entry.buildMesh(material)
    : new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(mesh);

  renderer.setRenderTarget(target);
  // `renderAsync()` is deprecated in r184 — `init()` is already awaited
  // at renderer creation above, so plain render() is the supported form.
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  // Restore the original NodeManager method now that the capture
  // window is over.
  nodesInstance._createNodeBuilderState = origCreateState;

  const readback = await renderer.readRenderTargetPixelsAsync(
    target,
    0,
    0,
    HARNESS_SIZE,
    HARNESS_SIZE
  );
  // The renderer returns its own typed array — copy into Uint8Array so
  // the rest of the harness treats both paths uniformly.
  const pixels = new Uint8Array(readback.buffer.slice(0));

  const vertexShader = capturedRef.value?.vertex ?? '';
  const fragmentShader = capturedRef.value?.fragment ?? '';

  target.dispose();
  if ('geometry' in mesh) (mesh as THREE.Mesh | THREE.Points).geometry.dispose();
  material.dispose();
  (uniforms.uInput?.value as THREE.Texture | null | undefined)?.dispose();
  renderer.dispose();

  return { pixels, vertexShader, fragmentShader };
}
