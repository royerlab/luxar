// @vitest-environment jsdom
/**
 * PostProcessingManager lifecycle contracts.
 *
 * The full pipeline render is exercised by `post-processing-pipeline.spec.ts`
 * and `context-restore.spec.ts` E2E specs. These unit tests pin the
 * no-context lifecycle behaviours that can regress in ordinary
 * TypeScript edits without surfacing in a manual browser session:
 *
 *   - `dispose()` releases every transient resource it owns and is
 *     idempotent on a second call.
 *   - `setBloomEnabled(true/false)` allocates / disposes the bloom
 *     chain symmetrically and never re-allocates on a no-op call.
 *   - `setFXAAEnabled(true/false)` allocates / disposes the FXAA pass.
 *   - `resize()` disposes the previous HDR render target before
 *     allocating the new one (no orphaned GPU handles).
 *   - `rebuildAfterContextRestore()` snapshots and re-applies the
 *     effect-toggle state (bloom / vignette / lens) so a context
 *     loss while bloom was off doesn't silently re-enable it from
 *     config defaults.
 *   - The deferred-rebuild API tracks depth correctly.
 *
 * Rendering itself is not exercised: there is no GL context. The
 * constructor and lifecycle methods reach `materialManager` which
 * builds real `MegaShaderMaterial` instances — those only touch the
 * GL context when actually rendered, so construction completes off
 * the live renderer. See `output-color-space-wiring.test.ts` for
 * the established mock-renderer pattern this file reuses.
 *
 * @module tests/unit/rendering/post-processing/post-processing-manager-lifecycle
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { PostProcessingManager } from '../../../rendering/post-processing/post-processing-manager';
import { materialManager } from '../../../rendering/material-manager';
import type { Renderer, RendererCapabilities } from '../../../rendering/renderer-capabilities';
import type { DataRefractionSplit } from '../../../rendering/post-processing/post-processing-manager/refraction-split';
import { getGlassDepthTexture } from '../../../rendering/materials/_shared/glass-partition';
import { loadTslMaterials } from '../../../rendering/tsl/load';
import { log, Modules } from '../../../utils/log';

function mockCaps(
  apiSurface: 'webgl2' | 'webgpu' = 'webgl2',
  overrides: Partial<RendererCapabilities> = {}
): RendererCapabilities {
  return {
    apiSurface,
    framebufferYDown: apiSurface === 'webgpu',
    hdr: {
      p3Gamut: false,
      rec2020Gamut: false,
      hdr: false,
      deepColor: false,
      floatTextures: true,
      filterableFloatTextures: false,
      colorDepth: { red: 8, green: 8, blue: 8 },
      recommendedColorSpace: 'srgb',
    },
    maxTextureSize: 4096,
    maxRenderbufferSize: 4096,
    maxMSAASamples: 4,
    pointSizeRange: [1, 1024],
    readBackbufferPixels: () => Promise.resolve({ pixels: new Uint8Array(), width: 0, height: 0 }),
    ...overrides,
  };
}

function makeMockRenderer(opts: { pixelRatio?: number } = {}): Renderer {
  const canvas = document.createElement('canvas');
  const pixelRatio = opts.pixelRatio ?? 1;
  return {
    outputColorSpace: THREE.LinearSRGBColorSpace,
    toneMapping: THREE.NoToneMapping,
    getPixelRatio: () => pixelRatio,
    setSize: vi.fn((width: number, height: number) => {
      canvas.width = Math.floor(width * pixelRatio);
      canvas.height = Math.floor(height * pixelRatio);
    }),
    domElement: canvas,
  } as unknown as Renderer;
}

function makeManager(opts: { width?: number; height?: number } = {}): PostProcessingManager {
  const renderer = makeMockRenderer();
  materialManager.setCaps(mockCaps('webgl2'));
  return new PostProcessingManager(
    renderer,
    mockCaps('webgl2'),
    new THREE.Scene(),
    new THREE.PerspectiveCamera(),
    { width: opts.width ?? 64, height: opts.height ?? 64 }
  );
}

/**
 * Reach into PostProcessingManager's private fields for the asserts
 * that pin disposal/allocation. Tests are co-located with the module
 * and live behind the field-access cast so public API stays narrow.
 */
type ManagerInternals = {
  hdrTarget: THREE.WebGLRenderTarget;
  ldrTarget: THREE.WebGLRenderTarget;
  bloomChain: { dispose(): void; outputTexture: THREE.Texture } | null;
  fxaaPass: { dispose(): void } | null;
  megaShader: {
    dispose(): void;
    isLensDistortionEnabled(): boolean;
    isVignetteEnabled(): boolean;
    isDetectorNoiseEnabled(): boolean;
  };
  megaPass: { dispose(): void };
  disposed: boolean;
  deferRebuildDepth: number;
  renderer: { setSize: ReturnType<typeof vi.fn>; domElement: HTMLCanvasElement };
};

function peek(mgr: PostProcessingManager): ManagerInternals {
  return mgr as unknown as ManagerInternals;
}

describe('PostProcessingManager → dispose lifecycle', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('releases every transient resource owned at construction time', () => {
    const mgr = makeManager();
    const internals = peek(mgr);

    const hdrDispose = vi.spyOn(internals.hdrTarget, 'dispose');
    const ldrDispose = vi.spyOn(internals.ldrTarget, 'dispose');
    const bloomDispose = internals.bloomChain
      ? vi.spyOn(internals.bloomChain, 'dispose')
      : undefined;
    const megaShaderDispose = vi.spyOn(internals.megaShader, 'dispose');
    const megaPassDispose = vi.spyOn(internals.megaPass, 'dispose');

    mgr.dispose();

    expect(hdrDispose).toHaveBeenCalledTimes(1);
    expect(ldrDispose).toHaveBeenCalledTimes(1);
    expect(megaShaderDispose).toHaveBeenCalledTimes(1);
    expect(megaPassDispose).toHaveBeenCalledTimes(1);
    if (bloomDispose) expect(bloomDispose).toHaveBeenCalledTimes(1);
    expect(internals.disposed).toBe(true);
  });

  it('is idempotent — second dispose() is a no-op', () => {
    const mgr = makeManager();
    const internals = peek(mgr);

    mgr.dispose();
    // Capture spies AFTER first dispose so the second call's behaviour
    // is measured in isolation.
    const hdrDispose = vi.spyOn(internals.hdrTarget, 'dispose');
    const megaShaderDispose = vi.spyOn(internals.megaShader, 'dispose');

    expect(() => mgr.dispose()).not.toThrow();
    expect(hdrDispose).not.toHaveBeenCalled();
    expect(megaShaderDispose).not.toHaveBeenCalled();
  });
});

describe('PostProcessingManager → bloom toggle cycles', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('allocates a fresh BloomChain when toggled from off → on', () => {
    const mgr = makeManager();
    // The config default enables bloom; force a known-off baseline so
    // the off → on transition is observable.
    mgr.setBloomEnabled(false);
    expect(mgr.isBloomEnabled()).toBe(false);

    mgr.setBloomEnabled(true);
    expect(mgr.isBloomEnabled()).toBe(true);
    expect(peek(mgr).bloomChain).not.toBeNull();

    mgr.dispose();
  });

  it('disposes the BloomChain and drops the reference when toggled on → off', () => {
    const mgr = makeManager();
    mgr.setBloomEnabled(true); // ensure on
    const chain = peek(mgr).bloomChain;
    expect(chain).not.toBeNull();
    const disposeSpy = vi.spyOn(chain!, 'dispose');

    mgr.setBloomEnabled(false);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(peek(mgr).bloomChain).toBeNull();
    expect(mgr.isBloomEnabled()).toBe(false);

    mgr.dispose();
  });

  it('does not re-allocate on a no-op toggle (on → on)', () => {
    const mgr = makeManager();
    mgr.setBloomEnabled(true);
    const chainBefore = peek(mgr).bloomChain;

    mgr.setBloomEnabled(true);
    expect(peek(mgr).bloomChain).toBe(chainBefore);

    mgr.dispose();
  });

  it('does not throw on a no-op toggle (off → off)', () => {
    const mgr = makeManager();
    mgr.setBloomEnabled(false);
    expect(() => mgr.setBloomEnabled(false)).not.toThrow();
    expect(peek(mgr).bloomChain).toBeNull();

    mgr.dispose();
  });
});

describe('PostProcessingManager → FXAA toggle cycles', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('allocates the FxaaPass on off → on and disposes it on on → off', () => {
    const mgr = makeManager();
    // Force a known-off baseline (config default may enable FXAA).
    mgr.setFXAAEnabled(false);
    expect(peek(mgr).fxaaPass).toBeNull();

    mgr.setFXAAEnabled(true);
    const pass = peek(mgr).fxaaPass;
    expect(pass).not.toBeNull();
    const disposeSpy = vi.spyOn(pass!, 'dispose');

    mgr.setFXAAEnabled(false);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(peek(mgr).fxaaPass).toBeNull();

    mgr.dispose();
  });

  it('repeated on → on toggle keeps the same pass instance', () => {
    const mgr = makeManager();
    mgr.setFXAAEnabled(true);
    const before = peek(mgr).fxaaPass;
    mgr.setFXAAEnabled(true);
    expect(peek(mgr).fxaaPass).toBe(before);

    mgr.dispose();
  });
});

describe('PostProcessingManager → resize render-target lifecycle', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('warns once per transition into framebuffer-limited sizing', () => {
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const mgr = makeManager({ width: 5000, height: 3000 });
    const limitWarnings = () =>
      warning.mock.calls.filter(
        ([module, message]) =>
          module === Modules.POST_PROCESSING && String(message).includes('framebuffer limit')
      );

    mgr.resize(5000, 3000);
    expect(limitWarnings()).toHaveLength(1);
    mgr.resize(5001, 3000);
    mgr.resize(5002, 3000);
    expect(limitWarnings()).toHaveLength(1);

    mgr.resize(1000, 600);
    mgr.resize(5000, 3000);
    expect(limitWarnings()).toHaveLength(2);

    mgr.dispose();
    warning.mockRestore();
  });

  it('disposes the previous hdrTarget before allocating a new one', () => {
    const mgr = makeManager({ width: 64, height: 64 });
    const internals = peek(mgr);
    const previousHdr = internals.hdrTarget;
    const hdrDispose = vi.spyOn(previousHdr, 'dispose');

    mgr.resize(128, 96);

    expect(hdrDispose).toHaveBeenCalledTimes(1);
    expect(peek(mgr).hdrTarget).not.toBe(previousHdr);

    mgr.dispose();
  });

  it('reuses the same ldrTarget via setSize (single handle for the session)', () => {
    // ldrTarget reuses the target with setSize() rather than allocating
    // anew — this matches WebGLRenderTarget's contract (setSize disposes
    // the old GPU texture inside the same wrapper) and keeps the handle
    // identity stable for downstream texture-rebind logic.
    const mgr = makeManager({ width: 64, height: 64 });
    const previousLdr = peek(mgr).ldrTarget;

    mgr.resize(128, 96);

    expect(peek(mgr).ldrTarget).toBe(previousLdr);

    mgr.dispose();
  });

  it('invokes the onResize callback so SceneManager can refresh material uniforms', () => {
    const onResize = vi.fn();
    const renderer = makeMockRenderer();
    materialManager.setCaps(mockCaps('webgl2'));
    const mgr = new PostProcessingManager(
      renderer,
      mockCaps('webgl2'),
      new THREE.Scene(),
      new THREE.PerspectiveCamera(),
      { width: 64, height: 64 },
      onResize
    );

    mgr.resize(96, 96);
    expect(onResize).toHaveBeenCalled();

    mgr.dispose();
  });

  // No-op guard: every window resize reaches reallocateForSize through TWO
  // paths (the window `resize` listener AND the canvas-parent
  // ResizeObserver), and reallocateForSize historically disposed +
  // recreated the full-screen half-float HDR target unconditionally —
  // twice per resize tick, even at identical size/DPR. An identical
  // request must now be a complete no-op.
  it('skips reallocation entirely when size, DPR and MSAA are unchanged', () => {
    const onResize = vi.fn();
    const renderer = makeMockRenderer();
    materialManager.setCaps(mockCaps('webgl2'));
    const mgr = new PostProcessingManager(
      renderer,
      mockCaps('webgl2'),
      new THREE.Scene(),
      new THREE.PerspectiveCamera(),
      { width: 64, height: 64 },
      onResize
    );

    mgr.resize(128, 96); // genuine change → reallocates
    const hdrAfterFirst = peek(mgr).hdrTarget;
    const hdrDispose = vi.spyOn(hdrAfterFirst, 'dispose');
    const setSizeCalls = (renderer.setSize as ReturnType<typeof vi.fn>).mock.calls.length;
    onResize.mockClear();

    mgr.resize(128, 96); // identical → full no-op (the second path of a resize tick)

    expect(hdrDispose).not.toHaveBeenCalled();
    expect(peek(mgr).hdrTarget).toBe(hdrAfterFirst);
    expect((renderer.setSize as ReturnType<typeof vi.fn>).mock.calls.length).toBe(setSizeCalls);
    expect(onResize).not.toHaveBeenCalled();

    mgr.dispose();
  });

  it('still reallocates when only the device pixel ratio changed', () => {
    // Adaptive DPR changes renderer.getPixelRatio() without changing the
    // logical size — the guard must key on PHYSICAL size too.
    let pixelRatio = 1;
    const canvas = document.createElement('canvas');
    const renderer = {
      outputColorSpace: THREE.LinearSRGBColorSpace,
      toneMapping: THREE.NoToneMapping,
      getPixelRatio: () => pixelRatio,
      setSize: vi.fn(),
      domElement: canvas,
    } as unknown as import('../../../rendering/renderer-capabilities').Renderer;
    materialManager.setCaps(mockCaps('webgl2'));
    const mgr = new PostProcessingManager(
      renderer,
      mockCaps('webgl2'),
      new THREE.Scene(),
      new THREE.PerspectiveCamera(),
      { width: 64, height: 64 }
    );

    mgr.resize(96, 96);
    const hdrAfterFirst = peek(mgr).hdrTarget;

    pixelRatio = 2; // adaptive/manual DPR change, same logical size
    mgr.resize(96, 96);

    expect(peek(mgr).hdrTarget).not.toBe(hdrAfterFirst);

    mgr.dispose();
  });

  it('clamps oversized SSAA targets and resizes attachments before the canvas', () => {
    const renderer = makeMockRenderer({ pixelRatio: 2 });
    const capabilities = mockCaps('webgl2', {
      maxTextureSize: 16384,
      maxRenderbufferSize: 8192,
    });
    materialManager.setCaps(capabilities);
    const mgr = new PostProcessingManager(
      renderer,
      capabilities,
      new THREE.Scene(),
      new THREE.PerspectiveCamera(),
      { width: 64, height: 64 }
    );
    mgr.setSSAAEnabled(true);

    const internals = peek(mgr);
    const ldrSetSize = vi.spyOn(internals.ldrTarget, 'setSize');
    const rendererSetSize = renderer.setSize as ReturnType<typeof vi.fn>;
    rendererSetSize.mockClear();

    mgr.resize(2509, 1328);

    const [logicalWidth, logicalHeight] = rendererSetSize.mock.lastCall!;
    expect(logicalWidth).toBe(4096);
    expect(logicalHeight).toBe(2167);
    expect(peek(mgr).hdrTarget.width).toBe(8192);
    expect(peek(mgr).hdrTarget.height).toBe(4334);
    expect(peek(mgr).ldrTarget.width).toBe(8192);
    expect(peek(mgr).ldrTarget.height).toBe(4334);
    expect(renderer.domElement.width).toBe(8192);
    expect(renderer.domElement.height).toBe(4334);
    expect(ldrSetSize.mock.invocationCallOrder[0]).toBeLessThan(
      rendererSetSize.mock.invocationCallOrder[0]
    );

    mgr.dispose();
  });

  it('suspends MSAA renderbuffers while SSAA is active and restores them afterward', () => {
    const renderer = makeMockRenderer({ pixelRatio: 2 });
    const capabilities = mockCaps('webgl2', {
      maxTextureSize: 16384,
      maxRenderbufferSize: 8192,
    });
    materialManager.setCaps(capabilities);
    const mgr = new PostProcessingManager(
      renderer,
      capabilities,
      new THREE.Scene(),
      new THREE.PerspectiveCamera(),
      { width: 2509, height: 1328 }
    );

    mgr.setMSAAEnabled(true);
    expect(peek(mgr).hdrTarget.samples).toBe(4);

    mgr.setSSAAEnabled(true);
    expect(peek(mgr).hdrTarget.samples).toBe(0);
    expect(peek(mgr).hdrTarget.width).toBe(8192);
    expect(peek(mgr).hdrTarget.height).toBe(4334);

    mgr.setSSAAMultiplier(1);
    expect(peek(mgr).hdrTarget.samples).toBe(4);

    mgr.setSSAAMultiplier(1.5);
    expect(peek(mgr).hdrTarget.samples).toBe(0);

    mgr.rebuildAfterContextRestore();
    expect(peek(mgr).hdrTarget.samples).toBe(0);

    mgr.setSSAAEnabled(false);
    expect(peek(mgr).hdrTarget.samples).toBe(4);

    mgr.dispose();
  });

  it('logs the effective MSAA state while SSAA suspends configured samples', () => {
    const mgr = makeManager();
    mgr.setMSAAEnabled(true);
    const updateLog = vi.spyOn(log, 'update').mockImplementation(() => {});
    const infoLog = vi.spyOn(log, 'info').mockImplementation(() => {});

    mgr.setSSAAEnabled(true);
    expect(updateLog).toHaveBeenLastCalledWith(
      Modules.POST_PROCESSING,
      'SSAA enabled (2x); MSAA: 4x configured; suspended by SSAA'
    );

    mgr.setSSAAMultiplier(1);
    mgr.setMSAASamples(2);
    expect(infoLog).toHaveBeenLastCalledWith(Modules.POST_PROCESSING, 'MSAA samples set to 2x');

    mgr.setSSAAMultiplier(1.5);
    mgr.setMSAASamples(4);
    expect(infoLog).toHaveBeenLastCalledWith(
      Modules.POST_PROCESSING,
      'MSAA samples set to 4x configured; suspended by SSAA'
    );

    mgr.setSSAAEnabled(false);
    expect(updateLog).toHaveBeenLastCalledWith(Modules.POST_PROCESSING, 'SSAA disabled; MSAA: 4x');

    mgr.dispose();
  });

  it('reallocates after rebuildAfterContextRestore even at an identical size', () => {
    // The restore path rebuilds all transient targets; the allocation
    // memo must be invalidated so the follow-up updateRendererSize()
    // resize is not silently skipped against the fresh targets.
    const mgr = makeManager({ width: 64, height: 64 });
    mgr.resize(96, 96);
    mgr.rebuildAfterContextRestore();
    const hdrAfterRebuild = peek(mgr).hdrTarget;

    mgr.resize(96, 96); // same size — but the memo was reset by the rebuild

    expect(peek(mgr).hdrTarget).not.toBe(hdrAfterRebuild);

    mgr.dispose();
  });
});

describe('PostProcessingManager → rebuildAfterContextRestore preserves effect toggles', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('round-trips a bloom-off state across context restore', () => {
    const mgr = makeManager();
    mgr.setBloomEnabled(false);
    expect(mgr.isBloomEnabled()).toBe(false);

    mgr.rebuildAfterContextRestore();

    // Snapshot was bloom=off; defaults must NOT silently re-enable it.
    expect(mgr.isBloomEnabled()).toBe(false);

    mgr.dispose();
  });

  it('round-trips a bloom-on state across context restore', () => {
    const mgr = makeManager();
    mgr.setBloomEnabled(true);
    expect(mgr.isBloomEnabled()).toBe(true);

    mgr.rebuildAfterContextRestore();
    expect(mgr.isBloomEnabled()).toBe(true);

    mgr.dispose();
  });

  it('round-trips vignette and lens-distortion toggle state', () => {
    const mgr = makeManager();
    // Force a non-default combination so we can tell defaults from
    // the actual snapshot path. Both effects start disabled in the
    // default config; flip them on.
    mgr.setVignetteEnabled(true);
    mgr.setChromaticLensDistortionEnabled(true);
    expect(peek(mgr).megaShader.isVignetteEnabled()).toBe(true);
    expect(peek(mgr).megaShader.isLensDistortionEnabled()).toBe(true);

    mgr.rebuildAfterContextRestore();
    expect(peek(mgr).megaShader.isVignetteEnabled()).toBe(true);
    expect(peek(mgr).megaShader.isLensDistortionEnabled()).toBe(true);

    mgr.dispose();
  });

  it('replaces every transient resource so dangling pre-loss handles are dropped', () => {
    const mgr = makeManager();
    const before = peek(mgr);
    const preHdr = before.hdrTarget;
    const preLdr = before.ldrTarget;
    const preMega = before.megaShader;

    mgr.rebuildAfterContextRestore();

    const after = peek(mgr);
    expect(after.hdrTarget).not.toBe(preHdr);
    expect(after.ldrTarget).not.toBe(preLdr);
    expect(after.megaShader).not.toBe(preMega);

    mgr.dispose();
  });

  it('is a no-op after dispose()', () => {
    const mgr = makeManager();
    mgr.dispose();
    // Should not re-allocate resources after disposal.
    expect(() => mgr.rebuildAfterContextRestore()).not.toThrow();
  });
});

describe('PostProcessingManager → deferred-rebuild depth', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('matched start/end pairs return depth to zero', () => {
    const mgr = makeManager();
    expect(peek(mgr).deferRebuildDepth).toBe(0);

    mgr.startDeferRebuild();
    mgr.startDeferRebuild();
    expect(peek(mgr).deferRebuildDepth).toBe(2);

    mgr.endDeferRebuild();
    mgr.endDeferRebuild();
    expect(peek(mgr).deferRebuildDepth).toBe(0);

    mgr.dispose();
  });

  it('withDeferredRebuild always pairs start/end even when fn throws', () => {
    const mgr = makeManager();
    expect(() =>
      mgr.withDeferredRebuild(() => {
        throw new Error('inner');
      })
    ).toThrow('inner');
    expect(peek(mgr).deferRebuildDepth).toBe(0);

    mgr.dispose();
  });

  it('endDeferRebuild at depth=0 logs a warning but does not underflow', () => {
    const mgr = makeManager();
    mgr.endDeferRebuild();
    // Depth stays clamped at 0 — no negative depth, no exception.
    expect(peek(mgr).deferRebuildDepth).toBe(0);

    mgr.dispose();
  });
});

describe('PostProcessingManager → size accessors', () => {
  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  // The recording session saves a size and later hands it back to
  // resize(); the offline capture derives a "Native" frame height from
  // it. Both need the DISPLAY size, and `renderer.getSize()` is not it —
  // reallocateForSize gives the renderer the SSAA-multiplied size and
  // puts the display size on the canvas CSS instead.
  it('reports the size resize() was given, not the SSAA-multiplied one', () => {
    const mgr = makeManager({ width: 1512, height: 850 });
    mgr.setSSAAMultiplier(2);
    mgr.setSSAAEnabled(true);

    expect(mgr.getDisplaySize()).toEqual({ width: 1512, height: 850 });
    expect(mgr.getEffectiveRenderScale()).toBe(2);

    // The premise the two accessors rest on, and the reason the capture
    // path must fold the scale in itself: the RENDERER is handed
    // display × multiplier while the canvas CSS keeps the display size.
    // Without this, swapping the two would leave the accessors — and
    // every recording test — green while an offline capture silently
    // dropped the SSAA factor.
    const { renderer } = peek(mgr);
    expect(renderer.setSize).toHaveBeenLastCalledWith(3024, 1700, false);
    expect(renderer.domElement.style.width).toBe('1512px');

    mgr.resize(1920, 1080);
    expect(mgr.getDisplaySize()).toEqual({ width: 1920, height: 1080 });

    mgr.dispose();
  });

  it('hands out a copy, so a caller cannot resize the pipeline by mutation', () => {
    const mgr = makeManager({ width: 64, height: 64 });
    const size = mgr.getDisplaySize();
    size.width = 4096;
    expect(mgr.getDisplaySize().width).toBe(64);

    mgr.dispose();
  });

  it('excludes SSAA from the scale when SSAA is off', () => {
    const mgr = makeManager({ width: 64, height: 64 });
    mgr.setSSAAMultiplier(4);
    expect(mgr.getEffectiveRenderScale()).toBe(1);

    mgr.dispose();
  });
});

describe('PostProcessingManager → the refraction split (spec §3.4 Phase 3)', () => {
  type WithSplit = {
    refractionSplit: { setSize(w: number, h: number): void; dispose(): void } | null;
  };
  const split = (mgr: PostProcessingManager) => (mgr as unknown as WithSplit).refractionSplit;

  beforeEach(() => {
    materialManager.setCaps(mockCaps('webgl2'));
  });

  it('owns one on a WebGL renderer, resizes it with the targets, and disposes it with them', () => {
    const mgr = makeManager({ width: 64, height: 64 });
    const s = split(mgr);
    expect(s).not.toBeNull();
    const setSize = vi.spyOn(s!, 'setSize');
    const dispose = vi.spyOn(s!, 'dispose');

    mgr.resize(80, 40);
    expect(setSize).toHaveBeenCalledWith(80, 40);

    mgr.dispose();
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(split(mgr)).toBeNull();
  });

  it('comes back after a context restore', () => {
    const mgr = makeManager();
    const before = split(mgr);
    mgr.rebuildAfterContextRestore();
    const after = split(mgr);
    expect(after).not.toBeNull();
    expect(after).not.toBe(before);
    mgr.dispose();
  });

  it('owns one on a WebGPU renderer too (the data partition is backend-agnostic), sized with the targets', async () => {
    // The WebGPU mega-shader is a TSL material behind the lazy registry.
    await loadTslMaterials();
    materialManager.setCaps(mockCaps('webgpu'));
    const mgr = new PostProcessingManager(
      makeMockRenderer(),
      mockCaps('webgpu'),
      new THREE.Scene(),
      new THREE.PerspectiveCamera(),
      { width: 64, height: 64 }
    );
    const s = split(mgr) as DataRefractionSplit | null;
    expect(s).not.toBeNull();
    expect(s!.isWebGL).toBe(false);
    expect(s!.glassDepthTarget.width).toBe(64);
    mgr.resize(80, 40);
    expect(s!.glassDepthTarget.width).toBe(80);
    expect(s!.glassDepthTarget.height).toBe(40);
    // The depth target wraps the ONE depth texture every data material samples.
    expect(s!.glassDepthTarget.depthTexture).toBe(getGlassDepthTexture());
    mgr.dispose();
    expect(split(mgr)).toBeNull();
  });
});
