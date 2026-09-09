/**
 * Renderer capabilities — the single seam between Luxar and the raw
 * graphics API.
 *
 * Every other module in the codebase asks `RendererCapabilities` for
 * what the GPU can do, what bit depth the backbuffer has, or to read
 * pixels back. The implementation here speaks WebGL2. When the WebGPU
 * port lands, only this file changes.
 */
import * as THREE from 'three';
import type { WebGPURenderer } from 'three/webgpu';

import { detectDisplayCapabilities, type HDRCapabilities } from '../utils/hdr/hdr-detection';
import { log, Modules } from '../utils/log';

export type { WebGPURenderer } from 'three/webgpu';

/**
 * The graphics-API renderer Luxar uses.
 *
 * `THREE.WebGLRenderer` is the production default (GLSL `ShaderMaterial`).
 * `WebGPURenderer` (TSL `NodeMaterial`) is selectable via
 * `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`; it dispatches to a
 * real WebGPU adapter when available or falls back to its internal
 * WebGL2 backend. Consumers hold renderer references typed as `Renderer`.
 */
export type Renderer = THREE.WebGLRenderer | WebGPURenderer;

/**
 * Shape-narrow a `Renderer` to `THREE.WebGLRenderer`. Used in
 * `createRendererCapabilities` to gate raw-GL probes. The check
 * is structural rather than `instanceof` so we don't have to
 * runtime-import `WebGPURenderer` just to type-test against it
 * (the import would force eager loading of `three/webgpu` even
 * when WebGL is the active backend).
 *
 * Uses the positive `isWebGLRenderer` flag that
 * `THREE.WebGLRenderer`'s constructor sets on `this`. A negative
 * check (`!isWebGPURenderer`) would mis-identify any future
 * renderer type that doesn't carry the WebGPU flag. Probing for
 * `.getContext` would mis-identify a `WebGPURenderer` running on
 * its internal WebGL2 fallback because that path's `getContext()`
 * also delegates to a WebGL2 context.
 */
export function isWebGLRenderer(renderer: Renderer): renderer is THREE.WebGLRenderer {
  return (renderer as { isWebGLRenderer?: boolean }).isWebGLRenderer === true;
}

/**
 * What downstream code needs to know about the underlying graphics
 * stack. All static fields are captured once at construction.
 */
export interface RendererCapabilities {
  /**
   * The renderer API **surface** in use — discriminator for callers
   * that must branch on method signatures (readback shapes, render
   * target wiring, etc.).
   *
   * - `'webgl2'` → active renderer is `THREE.WebGLRenderer`.
   * - `'webgpu'` → active renderer is `WebGPURenderer`, **even when**
   *   WebGPURenderer's internal backend has fallen back to WebGL2.
   *   The callable surface still follows the WebGPURenderer API
   *   (e.g. `readRenderTargetPixelsAsync` returns its result instead
   *   of writing into a destination buffer, padding rules apply).
   *
   * Treat this as "which method-signature contract should I follow?",
   * not as "which physical GPU backend is running?". Probing the
   * physical backend requires inspecting `renderer.backend` and is
   * intentionally not exposed here.
   */
  readonly apiSurface: 'webgl2' | 'webgpu';
  /**
   * True when the effective framebuffer presented by the renderer has
   * row 0 at the **top** of the viewport (real WebGPU; also
   * WebGPURenderer running on its WebGL2 compat backend, which Three.js
   * normalises to match real WebGPU). False when row 0 is at the
   * **bottom** (`THREE.WebGLRenderer`).
   *
   * This is the canonical seam for every Y-orientation decision in the
   * viewer:
   *
   * - `createFullscreenTriangleGeometry` emits V-inverted UVs when this
   *   is `true` (so that screen-bottom-left at NDC (-1,-1) samples the
   *   bottom row of the source target on top-down framebuffers). The
   *   `false` branch emits the straight-V UVs used by WebGL2's
   *   bottom-up framebuffer. Either way the resulting `vUv` resolves
   *   to the canvas-relative UV at every fragment.
   * - `readPixelsCompactAsync` returns rows in canonical top-down order;
   *   when this is `false`, the primitive inverts rows on the way out.
   *
   * Disambiguates from `apiSurface`: in practice both fields move
   * together today (every WebGPURenderer reports
   * `framebufferYDown=true`), but they answer different questions.
   * `apiSurface` is the *method-signature* contract (e.g.
   * `readRenderTargetPixelsAsync`'s shape); this field is the
   * *framebuffer memory layout*. Future Three.js versions could
   * conceivably introduce a `WebGPURenderer` configuration whose
   * effective Y differs, which is why we keep this as a separate
   * capability rather than aliasing `apiSurface`.
   */
  readonly framebufferYDown: boolean;
  /** HDR / wide-gamut / float-texture detection. */
  readonly hdr: HDRCapabilities;
  /** Maximum MSAA sample count the GPU supports (0 if unsupported). */
  readonly maxMSAASamples: number;
  /**
   * Maximum 2D texture dimension (`MAX_TEXTURE_SIZE` /
   * `maxTextureDimension2D`). Sizes the gsplat splat-data texture
   * (see `rendering/element-texture-layout.ts`): width is capped at
   * `min(4096, maxTextureSize)` and height bounds per-node splat
   * capacity.
   */
  readonly maxTextureSize: number;
  /** Maximum renderbuffer dimension (`MAX_RENDERBUFFER_SIZE`). */
  readonly maxRenderbufferSize: number;
  /** `[min, max]` `gl_PointSize` range — used for debug logging. */
  readonly pointSizeRange: readonly [number, number];

  /**
   * Read the current canvas backbuffer into a freshly-allocated
   * `Uint8Array` (RGBA bytes, not vertically flipped).
   *
   * Returns a Promise so the WebGPU port (which requires async
   * `buffer.mapAsync`) replaces only this method's body, not the
   * interface contract. Under WebGL2 the inner work is synchronous
   * and the Promise resolves immediately.
   *
   * Implementations are responsible for binding the canvas
   * backbuffer before reading.
   */
  readBackbufferPixels(): Promise<{ pixels: Uint8Array; width: number; height: number }>;
}

/**
 * Detect the effective framebuffer Y orientation.
 *
 * `WebGLRenderer` always renders into a WebGL2 FBO whose memory row 0 is
 * the bottom of the viewport → `false`.
 *
 * `WebGPURenderer` always presents a **top-down** framebuffer convention
 * to user code regardless of whether the underlying backend is real
 * WebGPU or Three.js's WebGL2 compat fallback (`forceWebGL: true` /
 * natural compat). Three.js's WebGPURenderer normalises Y internally
 * so TSL/NodeMaterial output is byte-identical across its two
 * backends. The user-observed result: even under `forceWebGL`, a
 * passthrough sample at NDC (-1, -1) reads the *top-left* texel of the
 * source target, matching real WebGPU's UV (0, 0) convention.
 *
 * The discriminator is therefore the renderer class, NOT the backend
 * flag. An earlier heuristic branched on `renderer.backend.isWebGLBackend`
 * to distinguish real WebGPU from compat WebGL2, but that flag is about
 * the *backing API*, not the *effective framebuffer Y orientation*.
 * Branching on it left the image visibly flipped under
 * `?webgpu-force-webgl` because the geometry factory assumed WebGL's
 * bottom-up FBO while WebGPURenderer was producing top-down output.
 *
 * @internal — exported only so SceneManager can pass the result through
 * `createRendererCapabilities`. Outside of capability construction,
 * read `caps.framebufferYDown` instead.
 */
export function detectFramebufferYDown(renderer: Renderer): boolean {
  return !isWebGLRenderer(renderer);
}

/**
 * Whether a live WebGPU device advertises the optional `float32-filterable`
 * feature (or, under the WebGL2 compat backend, the equivalent GL extension).
 *
 * Read from the backend that is actually running rather than from the API
 * surface, for the same reason `maxTextureSize` is below: `?webgpu-force-webgl`
 * runs a WebGL2 context behind the WebGPU renderer, and assuming a real device's
 * feature set there would overestimate it.
 */
function hasWebGPUFloat32Filterable(renderer: unknown): boolean {
  const backend = (
    renderer as {
      backend?: {
        device?: { features?: { has?: (name: string) => boolean } };
        gl?: WebGL2RenderingContext;
      };
    }
  ).backend;
  const features = backend?.device?.features;
  if (features && typeof features.has === 'function') {
    return features.has('float32-filterable');
  }
  if (backend?.gl && typeof backend.gl.getExtension === 'function') {
    return !!backend.gl.getExtension('OES_texture_float_linear');
  }
  // No backend in scope yet (a pre-init or test renderer). False is the safe
  // answer: it selects a HalfFloat upload, which filters correctly everywhere.
  return false;
}

/**
 * Build a `RendererCapabilities` snapshot from a constructed
 * renderer. Call once after renderer init; pass the result to
 * consumers (PostProcessingManager, SceneManager, …).
 *
 * `framebufferYDown` is computed by {@link detectFramebufferYDown} when
 * not supplied — that's the production path. Tests pass an explicit
 * value to skip the renderer-backend probe.
 */
export function createRendererCapabilities(
  renderer: Renderer,
  framebufferYDown: boolean = detectFramebufferYDown(renderer)
): RendererCapabilities {
  const display = detectDisplayCapabilities();

  if (isWebGLRenderer(renderer)) {
    // WebGL2 path: probe raw-GL for capabilities. Post-renderer
    // `getContext()` calls are deliberately rare — this one, the
    // `readBackbufferPixels` readback below, and the provoking-vertex
    // probe in `rendering/picking/mesh` are the whole list (plus the
    // canvas-side pre-renderer call in `scene-manager`).
    const gl = renderer.getContext() as WebGL2RenderingContext;

    const maxMSAASamplesRaw = gl.getParameter(gl.MAX_SAMPLES) as number | null;
    const maxMSAASamples = typeof maxMSAASamplesRaw === 'number' ? maxMSAASamplesRaw : 0;

    // WebGL2 guarantees >= 2048; every real device in the wild reports
    // >= 4096. Fall back to the guaranteed floor if the probe misbehaves.
    const maxTextureSizeRaw = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number | null;
    const maxTextureSize =
      typeof maxTextureSizeRaw === 'number' && maxTextureSizeRaw > 0 ? maxTextureSizeRaw : 2048;
    const maxRenderbufferSizeRaw = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number | null;
    const maxRenderbufferSize =
      typeof maxRenderbufferSizeRaw === 'number' && maxRenderbufferSizeRaw > 0
        ? maxRenderbufferSizeRaw
        : 2048;

    const rawRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    const pointSizeRange: readonly [number, number] =
      rawRange &&
      typeof (rawRange as ArrayLike<number>)[0] === 'number' &&
      typeof (rawRange as ArrayLike<number>)[1] === 'number'
        ? [(rawRange as ArrayLike<number>)[0], (rawRange as ArrayLike<number>)[1]]
        : [1, 1024];

    const floatTextures = !!(
      gl.getExtension('EXT_color_buffer_float') ||
      gl.getExtension('EXT_color_buffer_half_float') ||
      gl.getExtension('WEBGL_color_buffer_float')
    );
    const colorDepth = {
      red: (gl.getParameter(gl.RED_BITS) as number) ?? 8,
      green: (gl.getParameter(gl.GREEN_BITS) as number) ?? 8,
      blue: (gl.getParameter(gl.BLUE_BITS) as number) ?? 8,
    };
    // Separate extension from the color-buffer ones above, and separately
    // required: core WebGL2 lets a FloatType texture exist and be sampled with
    // NEAREST, but LINEAR filtering on one needs `OES_texture_float_linear`. When
    // it is absent the sampler silently falls back to nearest rather than
    // erroring, so an HDR mesh texture would look blocky with nothing to say why
    // — hence a probe rather than an assumption.
    const filterableFloatTextures = !!gl.getExtension('OES_texture_float_linear');
    const hdr: HDRCapabilities = {
      ...display,
      floatTextures,
      filterableFloatTextures,
      colorDepth,
    };

    return {
      apiSurface: 'webgl2',
      framebufferYDown,
      hdr,
      maxMSAASamples,
      maxTextureSize,
      maxRenderbufferSize,
      pointSizeRange,
      readBackbufferPixels() {
        // Bind the canvas backbuffer explicitly. `runPipeline` is
        // defensive about restoring its prior render target, but we
        // can't assume the caller arrived here through that path.
        renderer.setRenderTarget(null);
        const ctx = renderer.getContext();
        const width = ctx.drawingBufferWidth;
        const height = ctx.drawingBufferHeight;
        const pixels = new Uint8Array(width * height * 4);
        ctx.readPixels(0, 0, width, height, ctx.RGBA, ctx.UNSIGNED_BYTE, pixels);
        return Promise.resolve({ pixels, width, height });
      },
    };
  }

  // WebGPU path: WebGPURenderer doesn't expose a `getContext()`
  // returning a WebGL2 context. Most capability probes that the
  // WebGL2 path uses (MAX_SAMPLES, ALIASED_POINT_SIZE_RANGE, GL
  // extensions, channel bit depths) don't have direct WebGPU
  // equivalents — WebGPU's adapter limits cover different things.
  // Use sensible defaults for probes that do not map directly to
  // WebGPU limits.
  const hdr: HDRCapabilities = {
    ...display,
    floatTextures: true, // WebGPU canvas formats include float-texture targets
    // `float32-filterable` is an OPTIONAL WebGPU feature, so this is read from
    // the live device rather than assumed from the API surface. Defaulting it to
    // true would be the worse error of the two: an HDR texture would upload as
    // FloatType on a device that cannot filter it and sample blocky, whereas
    // defaulting to false costs half the precision on a HalfFloat upload and
    // still filters correctly everywhere.
    filterableFloatTextures: hasWebGPUFloat32Filterable(renderer),
    colorDepth: { red: 8, green: 8, blue: 8 },
  };

  // WebGPU surface: read the live limit from whichever backend is
  // actually running (createRendererCapabilities runs post-init on the
  // production path). Real WebGPU exposes `device.limits`; the WebGL2
  // compat backend exposes the raw `gl` context instead — probe it so a
  // 4096-class device under `?webgpu-force-webgl` isn't overestimated.
  // Fall back to WebGPU's guaranteed default limit (8192).
  const backend = (
    renderer as unknown as {
      backend?: {
        device?: { limits?: { maxTextureDimension2D?: number } };
        gl?: WebGL2RenderingContext;
      };
    }
  ).backend;
  let maxTextureSize = 8192;
  let maxRenderbufferSize = 8192;
  const maxTextureDimension2D = backend?.device?.limits?.maxTextureDimension2D;
  if (typeof maxTextureDimension2D === 'number' && maxTextureDimension2D > 0) {
    maxTextureSize = maxTextureDimension2D;
    maxRenderbufferSize = maxTextureDimension2D;
  } else if (backend?.gl && typeof backend.gl.getParameter === 'function') {
    const glMax = backend.gl.getParameter(backend.gl.MAX_TEXTURE_SIZE) as number | null;
    if (typeof glMax === 'number' && glMax > 0) maxTextureSize = glMax;
    const renderbufferMax = backend.gl.getParameter(backend.gl.MAX_RENDERBUFFER_SIZE) as
      number | null;
    if (typeof renderbufferMax === 'number' && renderbufferMax > 0) {
      maxRenderbufferSize = renderbufferMax;
    } else {
      maxRenderbufferSize = maxTextureSize;
      log.warning(
        Modules.RENDERER,
        `MAX_RENDERBUFFER_SIZE probe failed; using MAX_TEXTURE_SIZE (${maxTextureSize})`
      );
    }
  }

  return {
    apiSurface: 'webgpu',
    framebufferYDown,
    hdr,
    maxMSAASamples: 4, // WebGPU adapters guarantee at least 4× MSAA
    maxTextureSize,
    maxRenderbufferSize,
    pointSizeRange: [1, 1024],
    readBackbufferPixels() {
      // WebGPU backbuffer readback. WebGPURenderer doesn't have a
      // `gl.readPixels(canvas, …)` equivalent — the canvas is owned
      // by the browser compositor and not directly mappable.
      //
      // The canonical capture path is
      // `PostProcessingManager.renderToImageData()`, which renders into
      // an offscreen `WebGLRenderTarget` and reads it via the
      // backend-agnostic `readRenderTargetPixelsAsync`. Direct backbuffer
      // readback (this method) is retained on the interface for tests and
      // direct readers under WebGL2.
      //
      // Under WebGPU we deliberately fail loud rather than fake a
      // success: this method has no scene/render context to capture
      // (the caller would already have rendered), and any "render
      // an empty target" stub here would silently produce a black
      // pixel buffer in place of the intended capture. Direct
      // backbuffer readback is not something WebGPU supports;
      // callers must route through renderToImageData.
      return Promise.reject(
        new Error(
          'readBackbufferPixels is not supported under the WebGPU backend. ' +
            'Use PostProcessingManager.renderToImageData() for the capture path.'
        )
      );
    },
  };
}
