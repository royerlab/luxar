/**
 * TSL ↔ GLSL parity harness.
 *
 * Loaded by `tsl-harness.html` (Vite serves it at `/tsl-harness.html`).
 * Exposes `window.__tslHarness` with primitives that render a fullscreen
 * pass through both backends and return the result for pixel-diffing in
 * a Playwright spec.
 *
 * Why a dedicated page rather than reusing the main viewer:
 * - Construction order is explicit and minimal — no app/state machine
 *   to wait on, no scene graph to mock around.
 * - Both backends (WebGL2 via `THREE.WebGLRenderer`, WebGPU-via-WebGL2
 *   via `WebGPURenderer({ forceWebGL: true })`) live side by side; the
 *   test toggles between them per call rather than per page load.
 * - The TSL path drives `GLSLNodeBuilder` directly so the generated
 *   GLSL strings are recoverable for snapshot-diff.
 *
 * Not in scope: real WebGPU dispatch. That requires Chrome stable +
 * `?webgpu=1` and runs in a separate spec. This harness validates
 * the WebGL2 fallback parity, which is what `forceWebGL: true` covers.
 *
 * @module tests/e2e/harnesses/tsl-harness
 */

import * as THREE from 'three';
import { vec4 } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { FXAA_SOURCE } from '../../../rendering/post-processing/fxaa/shaders';
import { BLOOM_THRESHOLD_SOURCE } from '../../../rendering/post-processing/bloom/shaders';
import { MEGA_SOURCE } from '../../../rendering/post-processing/mega/shader.glsl';
import { megaWebGPUFactory } from '../../../rendering/post-processing/mega/shader.tsl';
import { POINT_SOURCE } from '../../../rendering/materials/point/shader-glsl';
import {
  pointWebGPUFactory,
  buildPointTSLNodesFromUniforms,
} from '../../../rendering/materials/point/shader-tsl';
import { POINT_PICK_SOURCE } from '../../../rendering/picking/point/shaders';
import {
  pointPickWebGPUFactory,
  buildPointPickTSLNodesFromUniforms,
} from '../../../rendering/picking/point/pick.tsl';
import { createPointQuadGeometry } from '../../../rendering/point-geometry';
import { LINE_SOURCE } from '../../../rendering/materials/line/shader-glsl';
import {
  lineWebGPUFactory,
  buildLineTSLNodesFromUniforms,
} from '../../../rendering/materials/line/shader-tsl';
import { LINE_PICK_SOURCE } from '../../../rendering/picking/line/shaders';
import {
  linePickWebGPUFactory,
  buildLinePickTSLNodesFromUniforms,
} from '../../../rendering/picking/line/pick.tsl';
import { createInstancedLinesMesh } from '../../../rendering/line-geometry';
import { GSPLAT_SOURCE } from '../../../rendering/materials/gsplat/shader-glsl';
import {
  gsplatWebGPUFactory,
  buildGSplatTSLNodesFromUniforms,
} from '../../../rendering/materials/gsplat/shader-tsl';
import { GSPLAT_PICK_SOURCE } from '../../../rendering/picking/gsplat/shaders';
import {
  gsplatPickWebGPUFactory,
  buildGSplatPickTSLNodesFromUniforms,
} from '../../../rendering/picking/gsplat/pick.tsl';
import { createInstancedGSplatsMesh, writeSplatTexels } from '../../../rendering/gsplat-geometry';
import {
  requireWebGLSources,
  type ShaderSource,
} from '../../../rendering/materials/_shared/shader-source';

/**
 * Shape of an entry in the shader registry exposed to Playwright.
 * Adding a new shader to {@link SHADER_REGISTRY} is sufficient to make
 * it usable from the spec.
 */
interface RegistryEntry {
  readonly source: ShaderSource;
  /** Default uniforms for this shader's parity test. */
  readonly buildUniforms: () => Record<string, THREE.IUniform>;
  /**
   * GLSL3 `defines` to set on the `THREE.ShaderMaterial`. Needed for
   * shaders like `mega` that use `#define` gates for feature toggles
   * + an `LUXAR_TONE_MAPPING_MODE` numeric. Optional; defaults to
   * empty (no defines).
   */
  readonly buildDefines?: () => Record<string, string>;
  /**
   * Override for the TSL material constructor. When provided, the
   * harness calls this directly instead of `source.webgpu(uniforms)`.
   * Used to pass shader-specific factory configs (e.g.
   * `megaWebGPUFactory(uniforms, { toneMappingMode: 1 })`).
   */
  readonly buildTSLMaterial?: (uniforms: Record<string, THREE.IUniform>) => THREE.Material;
  /**
   * Override the mesh built around the material. Defaults to a
   * fullscreen `THREE.Mesh(PlaneGeometry(2, 2), material)` rendered
   * with an OrthographicCamera. Override for point-sprite tests
   * that need `THREE.Points(...)`.
   */
  readonly buildMesh?: (material: THREE.Material) => THREE.Object3D;
  /**
   * Override the camera. Defaults to an `OrthographicCamera` at (0,0,1)
   * looking at the origin (see `buildDefaultCamera`). Override for cases
   * that need a `PerspectiveCamera` — e.g. the behind-camera guard, which
   * is perspective-only (`uIsOrtho == 0`) and a no-op under the default
   * ortho camera.
   */
  readonly buildCamera?: () => THREE.Camera;
  /**
   * Set on the GLSL3 `ShaderMaterial`. Needed by shaders like
   * `point` that read the auto-injected `in vec3 color` attribute —
   * Three only emits the attribute declaration when this is true.
   * TSL reads the same attribute via `attribute<'vec3'>('color',
   * 'vec3')` and doesn't need a parallel flag.
   */
  readonly vertexColors?: boolean;
}

/**
 * Sized 8×8 test texture: gradient horizontally, ramped vertically,
 * with a single bright pixel near the centre to exercise the FXAA
 * edge-detection path. Deterministic across both backends.
 */
function buildTestTexture(): THREE.DataTexture {
  const w = 8;
  const h = 8;
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i] = Math.floor((x / (w - 1)) * 255);
      data[i + 1] = Math.floor((y / (h - 1)) * 255);
      data[i + 2] = 128;
      data[i + 3] = 255;
    }
  }
  // High-contrast pixel for FXAA to bite on.
  const cx = 4;
  const cy = 4;
  const ci = (cy * w + cx) * 4;
  data[ci] = 255;
  data[ci + 1] = 255;
  data[ci + 2] = 255;

  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Deterministic 256×1 RGBA colormap LUT for the colormap-parity cases.
 * A diagonal gradient (R ramps up, B ramps down, G a triangle) so the
 * sampled colour varies meaningfully with the lookup coordinate `t` —
 * making gamma-on-value warping observable. Matches the production
 * colormap texture layout/filtering (`colormap-textures.ts`) so both
 * backends sample it identically.
 */
function buildColormapTexture(): THREE.DataTexture {
  const n = 256;
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    data[o] = i; // R: 0 → 255
    data[o + 1] = i < 128 ? i * 2 : (255 - i) * 2; // G: triangle peak at mid
    data[o + 2] = 255 - i; // B: 255 → 0
    data[o + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/** Point mesh + a per-instance `aScalar` for the colormap-parity case. */
function buildPointColormapMesh(material: THREE.Material): THREE.Object3D {
  const mesh = buildPointInstancedMesh(material) as THREE.Mesh;
  // Mid-range scalar so gamma (pow(t, invGamma)) actually moves the
  // lookup off the t=0/1 fixed points where pow is the identity.
  mesh.geometry.setAttribute(
    'aScalar',
    new THREE.InstancedBufferAttribute(new Float32Array([0.5]), 1)
  );
  return mesh;
}

/**
 * Line mesh + per-endpoint scalars for the colormap-parity case.
 *
 * Under `USE_COLORMAP` the line shader sources colour from the LUT and
 * omits the `aStartColor`/`aEndColor` `in` declarations entirely (see
 * `line/shader-glsl.ts`), which keeps the active vertex-attribute count
 * within `GL_MAX_VERTEX_ATTRIBS` (16) even with the scalar pair added.
 * We drop the now-unused colour buffers and bind the scalars so the mesh
 * matches the shader's active attribute set.
 */
function buildLineColormapMesh(material: THREE.Material): THREE.Object3D {
  const mesh = buildLineInstancedMesh(material) as THREE.Mesh;
  mesh.geometry.deleteAttribute('aStartColor');
  mesh.geometry.deleteAttribute('aEndColor');
  mesh.geometry.setAttribute(
    'aStartScalar',
    new THREE.InstancedBufferAttribute(new Float32Array([0.2]), 1)
  );
  mesh.geometry.setAttribute(
    'aEndScalar',
    new THREE.InstancedBufferAttribute(new Float32Array([0.8]), 1)
  );
  return mesh;
}

/**
 * Trivial diagnostic shader: outputs a constant RGB. Used to verify
 * that the harness's two backends produce pixel-identical results
 * for the simplest possible fragment. If this fails, the divergence
 * is in the renderer-level setup (color space, output transform),
 * not in a per-shader port.
 */
const CONST_SHADER: ShaderSource = {
  name: 'const-rgb',
  webgl: {
    vertex: /* glsl */ `
      void main() {
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragment: /* glsl */ `
      precision highp float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(0.5, 0.25, 0.75, 1.0);
      }
    `,
  },
  webgpu: () => {
    const m = new NodeMaterial();
    m.fragmentNode = vec4(0.5, 0.25, 0.75, 1.0);
    m.toneMapped = false;
    m.depthTest = false;
    m.depthWrite = false;
    m.transparent = false;
    return m;
  },
};

/**
 * Build a real instanced-points mesh for the point parity test.
 * One point at world origin with realistic attributes; 4-vertex quad
 * base + InstancedBufferAttribute per-instance data (aCenter etc.).
 */
function buildPointInstancedMesh(
  material: THREE.Material,
  sharpness: number = 0.5,
  center: readonly [number, number, number] = [0, 0, 0]
): THREE.Object3D {
  // sharpness is the normalised [0, 1] knob -> super-Gaussian exponent
  // beta = 2^(6s - 2). Default 0.5 -> beta=2 (a true Gaussian). `center` is the
  // world-space point position (default origin); a behind-camera center is used
  // to exercise the perspective behind-camera guard.
  const geom = createPointQuadGeometry();
  geom.setAttribute(
    'aCenter',
    new THREE.InstancedBufferAttribute(new Float32Array([center[0], center[1], center[2]]), 3)
  );
  geom.setAttribute('aRadius', new THREE.InstancedBufferAttribute(new Float32Array([0.5]), 1));
  geom.setAttribute(
    'aSharpness',
    new THREE.InstancedBufferAttribute(new Float32Array([sharpness]), 1)
  );
  geom.setAttribute(
    'aColor',
    new THREE.InstancedBufferAttribute(new Float32Array([1.0, 0.5, 0.25]), 3)
  );
  // Match the production points-mesh contract (the gpu-buffer-pool points
  // adapter): the WebGLRenderer only issues an instanced draw when `instanceCount`
  // is finite, and the drawRange must cap at the 6 indices that
  // form the unit quad — otherwise r184 falls back to a single
  // non-instanced draw call and produces a degenerate parity image.
  geom.instanceCount = 1;
  geom.setDrawRange(0, 6);
  const mesh = new THREE.Mesh(geom, material);
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Build a single-segment line mesh for parity testing. Horizontal
 * segment across the viewport in NDC, generous width so it covers
 * many pixels and exposes both the perpendicular falloff and edge AA.
 */
function buildLineInstancedMesh(
  material: THREE.Material,
  start: readonly [number, number, number] = [-0.5, 0, 0],
  end: readonly [number, number, number] = [0.5, 0, 0]
): THREE.Object3D {
  // PRODUCTION assembly (createInstancedLinesMesh), not a hand-rolled
  // geometry: the previous version decorated the plain BufferGeometry
  // quad TEMPLATE with instanced attributes — never a real
  // InstancedBufferGeometry — which the WebGPU-path draw dispatch
  // (three.webgpu.js drawParams: `instanceCount = geometry.instanceCount`
  // only when isInstancedBufferGeometry) does not draw as intended.
  // Using the production creator keeps parity testing the real path and
  // makes instancing correct by construction. (Same fix the point
  // builder got earlier — see the instanceCount note there.)
  const mesh = createInstancedLinesMesh(
    {
      startPositions: new Float32Array([start[0], start[1], start[2]]),
      endPositions: new Float32Array([end[0], end[1], end[2]]),
      startColors: new Float32Array([1.0, 0.5, 0.25]),
      endColors: new Float32Array([1.0, 0.5, 0.25]),
      startWidths: new Float32Array([0.1]),
      endWidths: new Float32Array([0.1]),
      // Sharpness is the normalised [0, 1] knob -> super-Gaussian exponent
      // beta = 2^(6s - 2). 0.5 -> beta=2 (a true Gaussian, the default).
      startSharpness: new Float32Array([0.5]),
      endSharpness: new Float32Array([0.5]),
      segmentLengths: new Float32Array([1.0]),
      startClipped: new Uint8Array([0]),
      endClipped: new Uint8Array([0]),
      segmentCount: 1,
    },
    material
  );
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Build a single-splat gsplat mesh. Isotropic covariance (identity
 * Cholesky) at world origin, fixed amplitude. Test exercises 3D→2D
 * covariance projection + Mahalanobis fragment math.
 */
function buildGSplatInstancedMesh(
  material: THREE.Material,
  center: readonly [number, number, number] = [0, 0, 0],
  sigma: number = 0.1
): THREE.Object3D {
  // PRODUCTION assembly (createInstancedGSplatsMesh), not a hand-rolled
  // geometry: the previous version decorated the plain BufferGeometry
  // quad TEMPLATE (createGSplatQuadGeometry) with instanced attributes.
  // That accidentally rendered on the WebGLRenderer path but drew ZERO
  // pixels through WebGPURenderer — every gsplat parity variant's TSL
  // side was black and the tests passed vacuously under the tolerance.
  // Isotropic: L = sigma · I, packed [L00, L10, L11, L20, L21, L22].
  const mesh = createInstancedGSplatsMesh(
    {
      centers: new Float32Array([center[0], center[1], center[2]]),
      cholesky01: new Float32Array([sigma, 0]),
      cholesky23: new Float32Array([sigma, 0]),
      cholesky45: new Float32Array([0, sigma]),
      amplitudes: new Float32Array([1.0]),
      colors: new Float32Array([1.0, 0.5, 0.25]),
      splatCount: 1,
    },
    material
  );
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Pre-built splat data texture for a gsplat parity variant. Since the
 * texture-storage migration the shaders read splat data via
 * `texelFetch(uSplatTex, ...)`; the TSL texture node is FACTORY-time
 * bound, so the texture must exist in the uniforms record BEFORE the
 * material is built (the same reason the production wrapper rebuilds
 * its graph on a texture identity change). Every gsplat registry
 * entry's `buildUniforms` supplies one of these with the SAME
 * (center, sigma) its `buildMesh` passes to
 * `buildGSplatInstancedMesh`, and the production `writeSplatTexels`
 * writes the layout so the harness can never drift from the real
 * texel packing. (The mesh's own geometry-attached texture holds
 * identical data; the shaders sample the uniform's.)
 */
function buildGSplatSplatDataTexture(
  center: readonly [number, number, number] = [0, 0, 0],
  sigma: number = 0.1
): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(16), 4, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeSplatTexels(
    tex,
    {
      centers: new Float32Array([center[0], center[1], center[2]]),
      cholesky01: new Float32Array([sigma, 0]),
      cholesky23: new Float32Array([sigma, 0]),
      cholesky45: new Float32Array([0, sigma]),
      amplitudes: new Float32Array([1.0]),
      colors: new Float32Array([1.0, 0.5, 0.25]),
    },
    1
  );
  return tex;
}

/**
 * A highly anisotropic (near-flat) splat: full extent in X and Z, a near-zero
 * minor axis in Y. Under the default ortho camera (which projects the XY block)
 * this yields a near-degenerate 2D covariance — the case the 2D low-pass
 * dilation exists for. Cholesky diagonal = (sx, sy, sz); off-diagonals zero.
 */
function buildThinCovSplatTexture(sx = 0.4, sy = 0.004, sz = 0.4): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Float32Array(16), 4, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  writeSplatTexels(
    tex,
    {
      centers: new Float32Array([0, 0, 0]),
      cholesky01: new Float32Array([sx, 0]),
      cholesky23: new Float32Array([sy, 0]),
      cholesky45: new Float32Array([0, sz]),
      amplitudes: new Float32Array([1.0]),
      colors: new Float32Array([1.0, 0.5, 0.25]),
    },
    1
  );
  return tex;
}

/**
 * Default parity camera: `OrthographicCamera` at (0,0,1) looking at the
 * origin, so world (0,0,0) projects to NDC centre. Shared by every case that
 * doesn't override `buildCamera`.
 */
function buildDefaultCamera(): THREE.Camera {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  return camera;
}

/**
 * Perspective camera at (0,0,1) looking down −Z, for the behind-camera guard
 * cases. A point at world z=3 lands at view-space z=+2 (behind the camera),
 * so the perspective-only guard (`uIsOrtho == 0 && mvPosition.z >= 0`) fires.
 */
function buildBehindCamera(): THREE.Camera {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  return camera;
}

const SHADER_REGISTRY: Record<string, RegistryEntry> = {
  'const-rgb': {
    source: CONST_SHADER,
    buildUniforms: () => ({}),
  },
  fxaa: {
    source: FXAA_SOURCE,
    buildUniforms: () => ({
      uInput: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
    }),
  },
  'bloom-threshold': {
    source: BLOOM_THRESHOLD_SOURCE,
    buildUniforms: () => ({
      uInput: { value: buildTestTexture() },
      uTexelSize: { value: new THREE.Vector2(1 / 8, 1 / 8) },
      uThreshold: { value: 0.5 },
      uSmoothing: { value: 0.5 },
    }),
  },
  // Mega-shader: default configuration only (no bloom, no lens
  // distortion, no vignette, no detector noise, mode=Linear). The
  // tone-mapping mode is pinned to Linear (mode=1) because that's
  // the simplest path through THREE's toneMapping chunk and
  // matches TSL's linearToneMapping output.
  mega: {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      // THREE's tone-mapping chunk reads this; pin to 1.0 so the
      // GLSL3 ShaderMaterial doesn't double-multiply our exposure.
      toneMappingExposure: { value: 1.0 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, { toneMappingMode: 1 }) as unknown as THREE.Material,
  },
  // Mega + bloom enabled: validates the `USE_BLOOM` JS-side conditional
  // branch in the TSL factory matches the GLSL `#ifdef USE_BLOOM` path.
  'mega-bloom': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      // A second texture for bloom — uniform contents differ from the
      // HDR scene so the test fails if the shader reads the wrong one.
      uBloomTexture: { value: buildTestTexture() },
      uBloomIntensity: { value: 0.5 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_BLOOM: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useBloom: true,
      }) as unknown as THREE.Material,
  },
  // Mega + detector noise: validates the Bob Jenkins hash +
  // Anscombe Poisson + clampedLogistic Gaussian port. The noise is
  // deterministic per (uv, time) so both backends should agree
  // bit-for-bit modulo float-precision rounding.
  'mega-detector-noise': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      uTime: { value: 0.123 }, // fixed value → deterministic
      uReadoutSigma: { value: 0.02 },
      uPhotonGain: { value: 0.05 },
      uFpnSigma: { value: 0.01 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_DETECTOR_NOISE: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useDetectorNoise: true,
      }) as unknown as THREE.Material,
  },
  // Mega + vignette: validates the `USE_VIGNETTE` JS-side branch.
  'mega-vignette': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      uVignetteDarkness: { value: 0.7 },
      uVignetteOffset: { value: 0.5 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_VIGNETTE: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useVignette: true,
      }) as unknown as THREE.Material,
  },
  // Mega + ACES tone-mapping (mode 4) — the PRODUCTION DEFAULT. The other
  // mega cases pin Linear (mode 1), so without this entry the ACES port
  // between shader.glsl.ts and shader.tsl.ts (the path users actually see)
  // is never parity-checked. ACES is non-linear, so this also guards the
  // RRT/ODT matrix + curve port, not just the mode switch.
  'mega-aces': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '4' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, { toneMappingMode: 4 }) as unknown as THREE.Material,
  },
  // Point parity: full PointMaterial sprite + GOG + super-Gaussian falloff.
  // Uniforms mirror the production PointMaterial constructor; ortho mode
  // keeps `invDistance = 1` so the test is deterministic across cameras.
  // The default mesh sharpness is 0.5 -> beta=2 (a true Gaussian).
  point: {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      // Disable blending for raw-pixel parity against the harness's
      // ShaderMaterial path (which uses transparent: false). Production
      // sets AdditiveBlending; the parity test only checks fragment
      // output, not blending semantics.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
  },
  // Super-Gaussian exponent sweep: the GLSL `pow(rho, beta)` / `exp(...)`
  // falloff must match TSL across the beta range, not just at the default.
  // `point-soft` exercises a peaky cusp (s=0.1 -> beta≈0.36); `point-hard`
  // a near-flat-top hard edge (s=0.9 -> beta≈12.1).
  'point-soft': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.1),
  },
  'point-hard': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.9),
  },
  // Point with the gamma==1 fast path enabled. Same geometry as `point`
  // but invGamma=1 + `gammaOne: true`, so the fragment-stage color pow()
  // is replaced with an identity. The GLSL counterpart defines
  // `LUXAR_GAMMA_ONE`. Mirrors `line-gamma-one` (three-geometry symmetry).
  'point-gamma-one': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(buildPointTSLNodesFromUniforms(uniforms, {}), {
        gammaOne: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
  },
  // Point colormap parity: USE_COLORMAP LUT path. Gamma is applied to the
  // scalar VALUE before the LUT lookup (vertex stage); intensity/offset
  // apply POST-LUT to the mapped color (matching the gsplat shader).
  // invGamma != 1 with aScalar = 0.5 so the gamma warp is observable,
  // and non-default uIntensity/uOffset so the post-LUT gain/offset path
  // is exercised and must match across backends.
  'point-colormap': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.5 },
      uOffset: { value: 0.05 },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 1.0 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, { useColormap: true }),
        { useColormap: true }
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointColormapMesh,
  },
  // Line parity: instanced quad line with width, sharpness, GOG.
  // Ortho camera so screen-space conversion is deterministic.
  line: {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 }, // ortho frustum height
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      // Pre-baked pixel-width scales for this ortho config:
      //   uOrthoLineScale = 2 * 64 / 2 = 64 (matches old 2*resY/uFOV)
      //   uPerspectiveLineScale is unused (uIsOrtho=1) — benign 1.0.
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Line with the gamma==1 fast path enabled. Same geometry +
  // uniforms as `line`, but the TSL factory is built with
  // `gammaOne: true` so the fragment-stage pow() is replaced with an
  // identity. The codegen snapshot for this variant pins the
  // pow-free fast path; the parity test compares against a GLSL
  // shader that has `LUXAR_GAMMA_ONE` defined.
  'line-gamma-one': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        gammaOne: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Line with the no-GOG fast path. Same geometry as `line`,
  // but the TSL factory is built with `noGOG: true` so the
  // `vColor * uIntensity + uOffset` + `max(..., 0)` chain is replaced
  // with `adjusted = vColor`. The GLSL counterpart defines
  // `LUXAR_NO_GOG`.
  'line-no-gog': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 }, // gamma kept slow path; only no-GOG is exercised
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildDefines: () => ({ LUXAR_NO_GOG: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        noGOG: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineInstancedMesh,
  },
  // Line colormap parity: USE_COLORMAP LUT path with per-endpoint scalars
  // (0.2 → 0.8). Gamma applied to the value pre-LUT (gammaOne=false here);
  // intensity/offset apply POST-LUT to the mapped color (matching the
  // gsplat shader) — non-default values here so the post-LUT gain/offset
  // path is exercised and must match across backends.
  'line-colormap': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.5 },
      uOffset: { value: 0.05 },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 1.0 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, { useColormap: true }), {
        useColormap: true,
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildLineColormapMesh,
  },
  // GSplat parity: isotropic Gaussian splat at world origin with
  // identity Cholesky factor. Ortho camera for deterministic projection.
  // Tests the 3D→2D covariance Jacobian, Cholesky factorisation,
  // eigendecomposition, oriented-quad expansion, Mahalanobis fragment.
  gsplat: {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 }, // ortho frustum 2 units → 32 px/unit
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // max projection — no Σ⁻¹ ray-integral path
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) }, // exp(-T²/2) for T=3
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      // blendingMode 'max' matches uProjectionMode=1 above: the TSL
      // factory JS-specializes the graph on the mode (sum emits the
      // Σ⁻¹ ray-integral block), so an inconsistent pair compares a
      // GLSL max-projection against a TSL sum-projection — a real
      // mismatch that was hidden while the TSL side rendered nothing.
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat at an OFF-CENTER position (world y=0.5 → screen y≈48 of 64).
  // Every other sprite fixture sits at the exact viewport center and is
  // mirror-symmetric about y = H/2, which makes the parity suite BLIND
  // to top-left/bottom-left fragcoord convention bugs (a y-mirror is
  // the identity on them). This variant exists to catch exactly that
  // class — the screenCoordinate-vs-vCenterScreen mismatch made every
  // off-center TSL splat invisible while all centered parity passed.
  'gsplat-offcenter': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0.5, 0]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildGSplatInstancedMesh(material, [0, 0.5, 0]),
  },
  // TINY-SIGMA splat (sigma = 0.005, maxLateralVar = 2.5e-5) whose
  // PROJECTED extent lands in the coverage-fade band: uFx = 3200 →
  // projectedExtent = 3200·0.005·3 = 48 px, band (32, 64) → fade 0.5.
  // Guards the fade being computed UNCONDITIONALLY: the former
  // maxLateralVar > 0.01 gate skipped it for sub-0.1-sigma splats, so
  // this splat rendered at FULL amplitude (peak ~255) instead of the
  // faded ~127. Both backends shared the gate, so plain parity is
  // blind — the spec also asserts the ABSOLUTE peak brightness.
  'gsplat-tiny-sigma-fade': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 0], 0.005) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 3200.0 },
      uFy: { value: 3200.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildGSplatInstancedMesh(material, [0, 0, 0], 0.005),
  },
  // TINY-SIGMA splat pushed PAST the fade band (uFx = 12800 →
  // projectedExtent = 192 px > maxExtent = 64) — the coverage cull must
  // reject the vertex entirely. Pre-fix, the gate skipped the fade and
  // the unconditional extent clamp squashed the 192-px quad into a
  // full-intensity hard-edged rectangle (the deep-zoom artifact).
  'gsplat-tiny-sigma-reject': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 0], 0.005) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 12800.0 },
      uFy: { value: 12800.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (material) => buildGSplatInstancedMesh(material, [0, 0, 0], 0.005),
  },
  // GSplat with the gamma==1 fast path enabled. Same geometry as `gsplat`
  // but uInvGamma=1 + `gammaOne: true`, so the fragment-stage color pow()
  // is replaced with an identity. The GLSL counterpart defines
  // `LUXAR_GAMMA_ONE`. Mirrors `line-gamma-one` (three-geometry symmetry).
  'gsplat-gamma-one': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_GAMMA_ONE: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        gammaOne: true,
        blendingMode: 'max', // matches uProjectionMode=1 (see `gsplat` variant note)
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat 'normal' premultiplied coverage alpha (LUXAR_NORMAL_PREMULT ↔
  // TSL blendingMode:'normal'). The interesting channel is ALPHA: the
  // fragment writes clamp(intensity·uOpacity, 0, 1) instead of 1.0, and
  // meanAbsDiff compares full RGBA. uOpacity=0.6 keeps the coverage
  // sub-saturated so alpha varies across the splat. uProjectionMode=1
  // (PEAK) matches the TSL factory's normal-mode graph: alpha-over is the
  // surface model, so normal mode uses the 2D-projected peak, not the
  // Σ⁻¹ ray-integral (that path is now exercised only by additive variants).
  'gsplat-normal-premult': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // peak projection (normal = surface)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 0.6 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildDefines: () => ({ LUXAR_NORMAL_PREMULT: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'normal',
      }) as unknown as THREE.Material;
      // The harness compares raw fragment output — override the
      // factory-applied blend state exactly like the other variants.
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat 2D-covariance DILATION on a near-degenerate splat. The splat is
  // near-flat (tiny minor axis in Y), so its projected Σ_2D is near-singular —
  // exactly what the low-pass dilation guards. uCov2DDilation=0.3 is set on
  // BOTH backends (GLSL reads the uniform; the TSL adapter reads the same
  // value), so this asserts GLSL and TSL dilate identically. No existing
  // variant exercises a degenerate covariance, so without this a shared
  // dilation regression would be invisible. (max projection isolates the
  // dilation from the ray-integral path.)
  'gsplat-thin-cov': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildThinCovSplatTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 }, // max projection (isolate dilation)
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uCov2DDilation: { value: 0.3 }, // the term under test — same on both backends
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat colormap parity: USE_COLORMAP LUT path keyed on aAmplitude.
  // uScalarScale=0.5 maps the amplitude (1.0) to t=0.5 so gamma — applied
  // to the value pre-LUT — actually shifts the lookup (pow is identity at
  // t=1). Color GOG bypassed.
  'gsplat-colormap': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
      uColormapTex: { value: buildColormapTexture() },
      uScalarMin: { value: 0.0 },
      uScalarScale: { value: 0.5 },
    }),
    buildDefines: () => ({ USE_COLORMAP: '' }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        useColormap: true,
        blendingMode: 'max', // matches uProjectionMode=1 (see `gsplat` variant note)
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildGSplatInstancedMesh,
  },
  // GSplat-pick parity: same covariance projection as `gsplat`
  // but fragment outputs (nodeId, elementId, brightness, 1.0) and
  // depth = 1 - brightness. No GOG, no ray-integration boost.
  'gsplat-pick': {
    source: GSPLAT_PICK_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture() },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 1.5 }, // tighter for picking (vs 3.0 visual)
      uTruncateSq: { value: 2.25 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uNodeId: { value: 42 },
      uShiftC: { value: Math.exp(-0.5 * 2.25) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 2.25)) },
    }),
    buildTSLMaterial: (uniforms) =>
      gsplatPickWebGPUFactory(
        buildGSplatPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildGSplatInstancedMesh,
  },
  // Line-pick parity: same quad-expansion math as `line` but
  // fragment outputs (nodeId, elementId, brightness, 1.0) and
  // depth = 1 - brightness. No edgeAA, no GOG.
  'line-pick': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNodeId: { value: 42 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      // Pre-baked pixel-width scales (mirror `line` parity entry).
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
    }),
    buildTSLMaterial: (uniforms) =>
      linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: true,
      }) as unknown as THREE.Material,
    buildMesh: buildLineInstancedMesh,
  },
  // Point-pick parity: identical sprite layout to `point` but
  // the fragment outputs (nodeId, elementId, brightness, 1.0) and
  // depth = 1 - brightness. Pick footprint is half-radius (×0.5).
  'point-pick': {
    source: POINT_PICK_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 1 },
      uNodeId: { value: 42 },
      uResolution: { value: new THREE.Vector2(64, 64) },
    }),
    buildTSLMaterial: (uniforms) =>
      pointPickWebGPUFactory(
        buildPointPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: buildPointInstancedMesh,
  },
  // Behind-camera guard parity: perspective camera (uIsOrtho:0) with the point
  // placed behind it (world z=3 → view z=+2). The visual point shader's
  // `uIsOrtho == 0 && mvPosition.z >= 0` reject must fire IDENTICALLY in GLSL and
  // TSL, so both backends produce an empty (background) frame. Without a behind-
  // camera case the guard ships with no rendered parity coverage (every other
  // point case is ortho, where the guard is a no-op).
  'point-behind': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // Same behind-camera guard, but on the point PICK shader (the path that was
  // missing the guard before — see the visual/pick symmetry fix). Both backends
  // must cull the behind-camera point so no spurious pick sprite is emitted.
  'point-pick-behind': {
    source: POINT_PICK_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 1.0 },
      uIsOrtho: { value: 0 },
      uNodeId: { value: 42 },
      uResolution: { value: new THREE.Vector2(64, 64) },
    }),
    buildTSLMaterial: (uniforms) =>
      pointPickWebGPUFactory(
        buildPointPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // ---- B9a/B9b/B9c regression variants ----
  //
  // B9a proof pair: identical points at the SAME view depth, one centered
  // and one off-axis, under PERSPECTIVE. Post-fix (view-z sizing, matching
  // lines/gsplats) their footprints are equal; the old Euclidean-distance
  // sizing shrank the off-axis sprite by cos(theta) (~11% linear at 26.6°
  // here). The spec compares covered-pixel counts across the two renders.
  'point-persp-center': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 221.7 }, // 2*64/tan(30°) — fov 60 at 64px
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.1 }, // ~11px sprite at view depth 1
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  'point-persp-offaxis': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 221.7 }, // 2*64/tan(30°) — fov 60 at 64px
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.1 }, // ~11px sprite at view depth 1
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    // World x=0.5 at view depth 1 → 26.6° off-axis, NDC x ≈ 0.87 (on-screen
    // at fov 60, aspect 1).
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0.5, 0, 0]),
    buildCamera: buildBehindCamera,
  },
  // B9b proof: raw projected size ≈ 0.96px (radiusScale 0.06 × attr 0.5 ×
  // factor 32). The 1.5px sprite floor guarantees rasterization, and the
  // fragment's sizeScale² compensation scales the ALPHA output by
  // (0.96/1.5)² ≈ 0.41. The point is positioned so the sprite center
  // lands EXACTLY on pixel (32,32)'s center (world 1.5/96 with the
  // [-1,1] ortho frustum on 64px) — falloff there is exactly 1, so the
  // written alpha is deterministically ≈ 0.41·255 ≈ 105 (pre-fix: 255,
  // indistinguishable from the opaque clear).
  'point-subpixel': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 32.0 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.06 },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.01 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildPointInstancedMesh(m, 0.5, [0.015625, 0.015625, 0]),
  },
  // B9c: unified near fade, mid-band. Point at view depth 1 with
  // uNearCull 0.7 → smoothstep((1-0.7)/0.7) ≈ 0.39 fade — non-empty,
  // identical across backends (shared perspectiveNearFade helper).
  'point-near-fade': {
    source: POINT_SOURCE,
    buildUniforms: () => ({
      pointSizeFactor: { value: 221.7 },
      maxPointSize: { value: 32.0 },
      radiusScale: { value: 0.1 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.7 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      opacity: { value: 1.0 },
      invGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = pointWebGPUFactory(
        buildPointTSLNodesFromUniforms(uniforms, {}),
        {}
      ) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: buildPointInstancedMesh,
    buildCamera: buildBehindCamera,
  },
  // B9c: line behind-camera parity (was point-only coverage). Both
  // endpoints at world z=3 → view z=+2 → the perspective-gated
  // bothBehind cull must produce an empty frame on BOTH backends.
  'line-behind': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 64.0 },
      uOrthoLineScale: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: false,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 3], [0.5, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  'line-pick-behind': {
    source: LINE_PICK_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 0 },
      uNodeId: { value: 42 },
      uNearCull: { value: 0.01 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 64.0 },
      uOrthoLineScale: { value: 1.0 },
    }),
    buildTSLMaterial: (uniforms) =>
      linePickWebGPUFactory(buildLinePickTSLNodesFromUniforms(uniforms), {
        isOrtho: false,
      }) as unknown as THREE.Material,
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 3], [0.5, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  // B9c BUG-A regression: ortho line INSIDE the frustum but within the
  // uNearCull slab (view depth 0.15 < nearCull 0.5, camera near 0.1).
  // Pre-fix the ungated bothBehind cull hid it (while a point/gsplat at
  // the same spot drew); post-fix it renders on both backends — NDC
  // clipping is the sole ortho cull authority.
  'line-ortho-near': {
    source: LINE_SOURCE,
    buildUniforms: () => ({
      uFOV: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uIsOrtho: { value: 1 },
      uNearCull: { value: 0.5 },
      uMaxLinePixelWidth: { value: 32.0 },
      uPerspectiveLineScale: { value: 1.0 },
      uOrthoLineScale: { value: 64.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = lineWebGPUFactory(buildLineTSLNodesFromUniforms(uniforms, {}), {
        isOrtho: true,
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    // Camera at z=1 (ortho near 0.1): world z=0.85 → view depth 0.15.
    buildMesh: (m) => buildLineInstancedMesh(m, [-0.5, 0, 0.85], [0.5, 0, 0.85]),
  },
  // B9c: gsplat behind-camera parity (was point-only coverage). The
  // unified perspectiveNearFade subsumes the old standalone reject —
  // a center at view z=+2 must yield an empty frame on both backends.
  'gsplat-behind': {
    source: GSPLAT_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 3]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 3.0 },
      uTruncateSq: { value: 9.0 },
      uRayIntegralFactor: { value: 2.433 },
      uProjectionMode: { value: 1 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uOpacity: { value: 1.0 },
      uInvGamma: { value: 1.0 / 2.2 },
      uIntensity: { value: 1.0 },
      uOffset: { value: 0.0 },
      uShiftC: { value: Math.exp(-0.5 * 9) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 9)) },
    }),
    buildTSLMaterial: (uniforms) => {
      const m = gsplatWebGPUFactory(buildGSplatTSLNodesFromUniforms(uniforms), {
        blendingMode: 'max',
      }) as unknown as THREE.Material;
      m.transparent = false;
      m.blending = THREE.NoBlending;
      return m;
    },
    buildMesh: (m) => buildGSplatInstancedMesh(m, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
  'gsplat-pick-behind': {
    source: GSPLAT_PICK_SOURCE,
    buildUniforms: () => ({
      uSplatTex: { value: buildGSplatSplatDataTexture([0, 0, 3]) },
      uResolution: { value: new THREE.Vector2(64, 64) },
      uFx: { value: 32.0 },
      uFy: { value: 32.0 },
      uTruncate: { value: 1.5 },
      uTruncateSq: { value: 2.25 },
      uIsOrtho: { value: 0 },
      uNearCull: { value: 0.01 },
      uMaxExtentFactor: { value: 1.0 },
      uNodeId: { value: 42 },
      uShiftC: { value: Math.exp(-0.5 * 2.25) },
      uInvOneMinusC: { value: 1.0 / (1.0 - Math.exp(-0.5 * 2.25)) },
    }),
    buildTSLMaterial: (uniforms) =>
      gsplatPickWebGPUFactory(
        buildGSplatPickTSLNodesFromUniforms(uniforms)
      ) as unknown as THREE.Material,
    buildMesh: (m) => buildGSplatInstancedMesh(m, [0, 0, 3]),
    buildCamera: buildBehindCamera,
  },
};

const HARNESS_SIZE = 64;

/**
 * Render the GLSL3 path of a registered shader to an offscreen target
 * and return the readback pixel buffer (RGBA8, length = w*h*4).
 */
function renderGLSL(shaderName: string): Uint8Array {
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
    depthTest: false,
    depthWrite: false,
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
async function renderTSL(
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

declare global {
  interface Window {
    __tslHarness?: {
      ready: Promise<void>;
      renderGLSL: (shaderName: string) => Uint8Array;
      renderTSL: (
        shaderName: string
      ) => Promise<{ pixels: Uint8Array; vertexShader: string; fragmentShader: string }>;
      listShaders: () => string[];
    };
  }
}

const status = document.getElementById('status');
const setStatus = (msg: string) => {
  if (status) status.textContent = msg;
};

const ready = (async () => {
  setStatus('tsl-harness ready');
})();

window.__tslHarness = {
  ready,
  renderGLSL,
  renderTSL,
  listShaders: () => Object.keys(SHADER_REGISTRY),
};
