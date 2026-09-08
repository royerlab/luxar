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
 * One compositor per stacked video overlay. It draws only while `start()`ed
 * (the overlay manager ties that to the overlay's visibility, like playback)
 * and only when the video has a frame; frames are pulled with
 * `requestVideoFrameCallback` where available (one draw per decoded frame),
 * `requestAnimationFrame` otherwise. Without WebGL (jsdom, a blocked context)
 * `createVideoMatteCompositor` returns `null` and the manager shows the plain
 * video — colour over matte, but visible rather than broken.
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
  /** Whether at least one video frame has been drawn (the poster can go). */
  readonly hasFrame: boolean;
  /** Stop, release the GL resources and lose the context. */
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
}

/**
 * Build a compositor for `video`, a stacked-matte clip. The canvas takes the
 * colour half's size as soon as the video's metadata is known (so a CSS
 * `height: auto` keeps the clip's true aspect) and is returned unattached —
 * the caller places it where the `<video>` would have gone. The video must be
 * readable by WebGL: same-origin, a blob URL, or `crossOrigin = 'anonymous'`
 * against a CORS-enabled server (set BEFORE its `src`).
 */
export function createVideoMatteCompositor(
  video: HTMLVideoElement,
  options: VideoMatteOptions = {}
): VideoMatteCompositor | null {
  const canvas = document.createElement('canvas');
  canvas.className = 'luxar-overlay__matte';
  const setup = setupGl(canvas);
  if (!setup) return null;
  const { gl, program, texture } = setup;

  const sizeToVideo = (): void => {
    const [w, h] = stackedFrameSize(video.videoWidth, video.videoHeight);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
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

  const draw = (): void => {
    if (failed || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    sizeToVideo();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    } catch (error) {
      // A tainted (cross-origin, no CORS) video: WebGL may never read it.
      // Stop for good and let the caller fall back to the raw clip rather
      // than throw out of a dims-manager listener every frame.
      failed = true;
      running = false;
      options.onFailure?.(error);
      return;
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    hasFrame = true;
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

  return {
    canvas,
    get hasFrame() {
      return hasFrame;
    },
    start() {
      if (running || failed) return;
      running = true;
      draw();
      if (running) schedule();
    },
    stop() {
      running = false;
      cancel();
    },
    dispose() {
      running = false;
      cancel();
      video.removeEventListener('loadedmetadata', sizeToVideo);
      gl.deleteTexture(texture);
      gl.deleteProgram(program);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}
