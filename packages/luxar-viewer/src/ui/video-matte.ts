/**
 * Stacked-alpha-matte video compositor.
 *
 * A transparent clip cannot be trusted to a video codec's alpha plane: Chrome
 * and Firefox render a VP9 `yuva420p` WebM transparent, but Safari and every
 * WKWebView (the exported kiosk app) decode it and DROP the alpha, so the clip
 * lands on a black square. Instead the clip is authored as one OPAQUE frame
 * twice as tall — the colour on top, the alpha channel as a grey matte of the
 * same size below (`Scene.add_video(..., alpha_matte="stacked")`) — and this
 * module recombines the halves on a `<canvas>` with a two-tap fragment shader:
 * `rgb` from the top half, `a` from the bottom half's red channel, output
 * premultiplied so the canvas composites correctly over the scene.
 *
 * One compositor per stacked video overlay. Its WebGL resources are allocated
 * lazily on the first `start()`, so clips never shown do not consume contexts.
 * It draws only while `start()`ed (the overlay manager ties that to the
 * overlay's visibility, like playback) and only when the video has a frame;
 * frames are pulled with
 * `requestVideoFrameCallback` where available (one draw per decoded frame),
 * `requestAnimationFrame` otherwise. Without WebGL (jsdom, a blocked context)
 * the first start reports failure and the manager shows the plain video —
 * colour over matte, but visible rather than broken.
 */

export const MATTE_VERTEX_SHADER = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

/**
 * Texture row 0 is the TOP of the uploaded frame, while `v_uv.y` is 0 at the
 * bottom of the canvas, hence the flip. Colour from the top half, alpha from
 * the bottom half; premultiplied output.
 */
export const MATTE_FRAGMENT_SHADER = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_frame;
void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec3 rgb = texture2D(u_frame, vec2(uv.x, uv.y * 0.5)).rgb;
  float a = texture2D(u_frame, vec2(uv.x, 0.5 + uv.y * 0.5)).r;
  gl_FragColor = vec4(rgb * a, a);
}
`;

/** The visible (colour) size of a stacked frame: full width, half the height. */
export function stackedFrameSize(videoWidth: number, videoHeight: number): [number, number] {
  return [Math.max(1, videoWidth), Math.max(1, Math.floor(videoHeight / 2))];
}

type FrameCallbackVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

export interface VideoMatteCompositor {
  /** The canvas showing the recombined, transparent frames. */
  readonly canvas: HTMLCanvasElement;
  /** Begin drawing frames (idempotent). */
  start(): void;
  /** Stop drawing; the last frame stays on the canvas. */
  stop(): void;
  /**
   * Stop and hand the WebGL context back, staying REUSABLE: the next `start()`
   * acquires a fresh one. A browser caps how many WebGL contexts may be live at
   * once (Chrome evicts the oldest when the cap is passed, and the oldest is
   * the scene's own renderer), so a tour with more stacked clips than that cap
   * would kill the renderer if every clip kept its context for the session.
   * Losing the context also clears the canvas, so `onRelease` fires to let the
   * caller put a poster back underneath.
   */
  release(): void;
  /** Stop, release the GL resources and detach for good. */
  dispose(): void;
}

function compileProgram(gl: WebGLRenderingContext): WebGLProgram | null {
  const compile = (type: number, src: string): WebGLShader | null => {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  };
  const vs = compile(gl.VERTEX_SHADER, MATTE_VERTEX_SHADER);
  const fs = compile(gl.FRAGMENT_SHADER, MATTE_FRAGMENT_SHADER);
  const program = vs && fs ? gl.createProgram() : null;
  if (!program || !vs || !fs) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

/** A GL context, program, quad and texture on `canvas`; `null` without WebGL. */
function setupGl(
  canvas: HTMLCanvasElement
): { gl: WebGLRenderingContext; program: WebGLProgram; texture: WebGLTexture } | null {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: true,
  }) as WebGLRenderingContext | null;
  if (!gl) return null;
  const program = compileProgram(gl);
  const texture = gl.createTexture();
  const quad = gl.createBuffer();
  if (!program || !texture || !quad) return null;
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'a_pos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.uniform1i(gl.getUniformLocation(program, 'u_frame'), 0);
  gl.disable(gl.BLEND);
  return { gl, program, texture };
}

export interface VideoMatteOptions {
  /**
   * Called once if a frame upload throws — a cross-origin clip served without
   * CORS taints the video and `texImage2D` raises a SecurityError — after the
   * compositor has stopped itself. The caller then shows the raw clip instead.
   */
  onFailure?: (error: unknown) => void;
  /** Called once after the first frame has been drawn to the canvas. */
  onFirstFrame?: () => void;
  /**
   * Called when `release()` has handed the context back and blanked the
   * canvas, so the caller can restore whatever sits behind it (the poster).
   * `onFirstFrame` will fire again on the next start.
   */
  onRelease?: () => void;
}

/**
 * Build a lazy compositor for `video`, a stacked-matte clip. The canvas takes the
 * colour half's size as soon as the video's metadata is known (so a CSS
 * `height: auto` keeps the clip's true aspect) and is returned unattached —
 * the caller places it where the `<video>` would have gone. The video must be
 * readable by WebGL: same-origin, a blob URL, or `crossOrigin = 'anonymous'`
 * against a CORS-enabled server (set BEFORE its `src`).
 */
export function createVideoMatteCompositor(
  video: HTMLVideoElement,
  options: VideoMatteOptions = {}
): VideoMatteCompositor {
  const canvas = document.createElement('canvas');
  canvas.className = 'luxar-overlay__matte';
  let setup: NonNullable<ReturnType<typeof setupGl>> | undefined;

  const sizeToVideo = (): void => {
    if (!video.videoWidth || !video.videoHeight) return;
    const [w, h] = stackedFrameSize(video.videoWidth, video.videoHeight);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      setup?.gl.viewport(0, 0, w, h);
    }
  };
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) sizeToVideo();
  video.addEventListener('loadedmetadata', sizeToVideo);

  let running = false;
  let failed = false;
  let handle = 0;
  let usingVfc = false;
  let hasFrame = false;
  const v = video as FrameCallbackVideo;

  const fail = (error: unknown): void => {
    failed = true;
    running = false;
    options.onFailure?.(error);
  };
  const ensureSetup = (): boolean => {
    if (setup) return true;
    setup = setupGl(canvas) ?? undefined;
    if (!setup) {
      fail(new Error('WebGL is unavailable'));
      return false;
    }
    sizeToVideo();
    return true;
  };

  const draw = (): void => {
    if (failed || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (!ensureSetup()) return;
    sizeToVideo();
    const currentSetup = setup;
    if (!currentSetup) return;
    const { gl, texture } = currentSetup;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    } catch (error) {
      // A tainted (cross-origin, no CORS) video: WebGL may never read it.
      // Stop for good and let the caller fall back to the raw clip rather
      // than throw out of a dims-manager listener every frame.
      fail(error);
      return;
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (!hasFrame) {
      hasFrame = true;
      canvas.dataset.hasFrame = '1';
      options.onFirstFrame?.();
    }
  };
  const tick = (): void => {
    if (!running) return;
    draw();
    if (running) schedule();
  };
  const schedule = (): void => {
    if (v.requestVideoFrameCallback) {
      usingVfc = true;
      handle = v.requestVideoFrameCallback(tick);
    } else {
      usingVfc = false;
      handle = requestAnimationFrame(tick);
    }
  };
  const cancel = (): void => {
    if (!handle) return;
    if (usingVfc && v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(handle);
    else cancelAnimationFrame(handle);
    handle = 0;
  };

  const release = (): void => {
    running = false;
    cancel();
    if (!setup) return;
    setup.gl.deleteTexture(setup.texture);
    setup.gl.deleteProgram(setup.program);
    setup.gl.getExtension('WEBGL_lose_context')?.loseContext();
    setup = undefined;
    // The canvas is blank again, so the next start has to redraw before it
    // shows anything — and `onFirstFrame` must fire again for the caller to
    // hide whatever it puts underneath. `failed` deliberately survives: a
    // tainted clip stays abandoned rather than retrying on every step.
    hasFrame = false;
    delete canvas.dataset.hasFrame;
    options.onRelease?.();
  };

  return {
    canvas,
    start() {
      if (running || failed) return;
      running = true;
      if (!ensureSetup()) return;
      draw();
      if (running) schedule();
    },
    stop() {
      running = false;
      cancel();
    },
    release,
    dispose() {
      release();
      video.removeEventListener('loadedmetadata', sizeToVideo);
    },
  };
}
