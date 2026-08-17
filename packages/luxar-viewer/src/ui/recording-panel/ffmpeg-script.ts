/**
 * The `encode_video.sh` bundled with every image / EXR frame sequence.
 *
 * Two very different jobs, depending on the frames in the ZIP:
 *
 * **LDR sequences (PNG / WebP / JPEG)** are already exactly what the
 * viewer showed — `renderToImageData` reads the framebuffer after the
 * mega-shader has applied exposure, offset, gamma, tone mapping and the
 * sRGB encode. The script must therefore apply *no* colour maths at all;
 * it just muxes frames into a container.
 *
 * **EXR sequences are scene-linear and pre-grade.** The EXR capture mode
 * (`hdr-effects-pre-tone`) deliberately bypasses the whole display
 * transform so the archive keeps unclipped HDR. Encoding those floats
 * without reversing that decision produces a dark, colour-shifted video:
 * measured on a real capture, mean RGB (26, 47, 39) against the viewer's
 * own (67, 67, 71). So for EXR the script reproduces the shader chain —
 * exposure → offset → gamma → tone map → sRGB — and reproduces it
 * *exactly*, as a `geq` expression carrying the same constants as
 * `three`'s tone-mapping functions, because ffmpeg's built-in `tonemap`
 * curves are not the same functions. Measured against the viewer's own
 * PNG of the same frame (PSNR, higher is closer):
 *
 *   viewer ACES      exact geq 40.3 dB · tonemap=hable 16.0 · nothing 23.8
 *   viewer Reinhard  exact geq (see tests) · tonemap=reinhard 34.8
 *
 * `hable`, the usual stand-in for ACES, is *worse than no tone mapping
 * at all*, which is why this module does not offer the approximations.
 *
 * AgX is the one mode with no practical closed form here (four matrix
 * stages around a log-space polynomial would expand to hundreds of terms
 * per channel), so its script says so and points at the LDR-sequence
 * route for a pixel-exact match.
 *
 * @module ui/recording-panel/ffmpeg-script
 */

/** Tone-mapping modes, mirroring the shader's `LUXAR_TONE_MAPPING_MODE`. */
export type ToneMapName = 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral';

/** The display transform to undo when the frames are scene-linear EXR. */
export interface GradeSettings {
  toneMapping: ToneMapName;
  /** `uExposure`, applied by the shader as `exp2(exposure)`. */
  exposure: number;
  /** `uGlobalOffset`, added after exposure and clamped at 0. */
  offset: number;
  /** `uGlobalGamma`, applied as `pow(c, 1 / gamma)`. */
  gamma: number;
}

export interface FfmpegScriptOptions {
  fps: number;
  frameCount: number;
  /** Extension of the frames in the ZIP: `png` | `jpg` | `webp` | `exr`. */
  frameExt: string;
  /** Recording mode — drives output naming and the looping note. */
  mode: 'video' | 'turntable';
  /** Base name for the encoded files, e.g. `luxar-capture-2026-08-17-120000`. */
  outputBase: string;
  /** Viewer grade at capture time. Only consulted for EXR input. */
  grade?: GradeSettings;
}

/** Human-facing labels for the tone-mapping modes. */
const TONE_MAP_LABEL: Record<ToneMapName, string> = {
  linear: 'Linear (clamp)',
  reinhard: 'Reinhard',
  cineon: 'Cineon (Hejl–Burgess-Dawson)',
  aces: 'ACES Filmic',
  agx: 'AgX',
  neutral: 'Khronos PBR Neutral',
};

/** `r(X,Y)`-style channel reads, pre-multiplied by the EOG chain. */
interface ExposedChannels {
  r: string;
  g: string;
  b: string;
}

/**
 * Exposure → offset → gamma, exactly as the shader's EOG block:
 *   `pow(max(c * exp2(exposure) + offset, 0), 1 / gamma)`
 * Emitted only where it changes the pixel, so the common all-default
 * case keeps the expression short.
 */
function applyEog(channel: string, grade: GradeSettings): string {
  let expr = channel;
  const scale = Math.pow(2, grade.exposure);
  if (scale !== 1) expr = `(${expr}*${fmt(scale)})`;
  if (grade.offset !== 0) expr = `max(${expr}+${fmt(grade.offset)},0)`;
  if (grade.gamma !== 1) expr = `pow(max(${expr},0),${fmt(1 / grade.gamma)})`;
  return expr;
}

/** Trim float noise out of the generated shell text. */
function fmt(v: number): string {
  return Number(v.toPrecision(8)).toString();
}

/** `saturate()` — the clamp every three.js tone-map function ends with. */
function sat(expr: string): string {
  return `clip(${expr},0,1)`;
}

/**
 * Build the exact per-channel expressions for a tone-mapping mode, or
 * `null` when the mode has no practical closed form (AgX).
 *
 * The constants are the ones in three's `tonemapping_pars_fragment`;
 * `luxar` applies exposure separately (the EOG block) and the functions
 * run with `toneMappingExposure = 1`.
 */
function toneMapExpressions(c: ExposedChannels, mode: ToneMapName): ExposedChannels | null {
  switch (mode) {
    case 'linear':
      // LinearToneMapping is a bare saturate once exposure is applied.
      return { r: sat(c.r), g: sat(c.g), b: sat(c.b) };

    case 'reinhard': {
      // saturate(c / (1 + c))
      const f = (x: string): string => sat(`(${x})/(1+(${x}))`);
      return { r: f(c.r), g: f(c.g), b: f(c.b) };
    }

    case 'cineon': {
      // OptimizedCineonToneMapping: subtract the 0.004 toe, apply the
      // Hejl/Burgess-Dawson rational, then pow(2.2) back to linear
      // (the formula bakes in a display gamma that Luxar re-applies
      // itself in the sRGB step).
      const f = (x: string): string => {
        const t = `max((${x})-0.004,0)`;
        return `pow(((${t})*(6.2*(${t})+0.5))/((${t})*(6.2*(${t})+1.7)+0.06),2.2)`;
      };
      return { r: f(c.r), g: f(c.g), b: f(c.b) };
    }

    case 'aces': {
      // colour × (1/0.6), ACES input matrix, RRTAndODTFit, output
      // matrix, saturate.
      const s = '(1/0.6)';
      const ar = `(0.59719*(${c.r})+0.35458*(${c.g})+0.04823*(${c.b}))*${s}`;
      const ag = `(0.07600*(${c.r})+0.90834*(${c.g})+0.01566*(${c.b}))*${s}`;
      const ab = `(0.02840*(${c.r})+0.13383*(${c.g})+0.83777*(${c.b}))*${s}`;
      const fit = (v: string): string =>
        `(((${v})*((${v})+0.0245786)-0.000090537)/((${v})*(0.983729*(${v})+0.432951)+0.238081))`;
      const fr = fit(ar);
      const fg = fit(ag);
      const fb = fit(ab);
      return {
        r: sat(`1.60475*${fr}-0.53108*${fg}-0.07367*${fb}`),
        g: sat(`-0.10208*${fr}+1.10813*${fg}-0.00605*${fb}`),
        b: sat(`-0.00327*${fr}-0.07276*${fg}+1.07602*${fb}`),
      };
    }

    case 'neutral': {
      // Khronos PBR Neutral. `startCompression = 0.8 - 0.04`,
      // `desaturation = 0.15`. Written with ffmpeg's `if()` because the
      // GLSL takes an early return below the compression knee.
      const startCompression = 0.8 - 0.04;
      const d = 1 - startCompression;
      const x = `min((${c.r}),min((${c.g}),(${c.b})))`;
      const offset = `if(lt(${x},0.08),(${x})-6.25*(${x})*(${x}),0.04)`;
      const cr = `((${c.r})-(${offset}))`;
      const cg = `((${c.g})-(${offset}))`;
      const cb = `((${c.b})-(${offset}))`;
      const peak = `max(${cr},max(${cg},${cb}))`;
      const newPeak = `(1-${fmt(d * d)}/((${peak})+${fmt(d - startCompression)}))`;
      const gDesat = `(1-1/(0.15*((${peak})-(${newPeak}))+1))`;
      // below the knee → untouched; above → scale to newPeak, then
      // desaturate toward it.
      const compress = (ch: string): string =>
        `if(lt(${peak},${fmt(startCompression)}),${ch},` +
        `(${ch})*(${newPeak})/(${peak})*(1-(${gDesat}))+(${newPeak})*(${gDesat}))`;
      return { r: sat(compress(cr)), g: sat(compress(cg)), b: sat(compress(cb)) };
    }

    case 'agx':
      // Four matrix stages around a log2-space 6th-order polynomial —
      // expanding that per channel is hundreds of terms. Not worth
      // emitting; the script explains the LDR-sequence alternative.
      return null;
  }
}

/**
 * The `-vf` chain that turns scene-linear EXR into display-referred
 * video, or `null` for LDR frames (already graded — no colour maths).
 */
function buildGradeFilter(
  frameExt: string,
  grade: GradeSettings | undefined
): { filter: string; exact: boolean } | null {
  if (frameExt !== 'exr' || !grade) return null;

  const channels: ExposedChannels = {
    r: applyEog('r(X,Y)', grade),
    g: applyEog('g(X,Y)', grade),
    b: applyEog('b(X,Y)', grade),
  };
  const mapped = toneMapExpressions(channels, grade.toneMapping);
  // zscale carries the linear→sRGB encode the shader does last — it is
  // what makes the EXR's linear floats display-referred at all.
  //
  // The matrix must be a YUV one (`m=bt709`): asking zscale for `m=gbr`
  // while the encoder wants yuv420p fails outright with "YUV color
  // family cannot have RGB matrix coefficients". Limited range is the
  // safe delivery default; `r=pc` measured 1.5 dB closer to the viewer
  // but relies on every player honouring the full-range flag.
  const srgb = 'zscale=tin=linear:t=iec61966-2-1:min=gbr:m=bt709:pin=bt709:p=bt709:r=tv';

  if (!mapped) {
    // AgX: clamp and encode. Not the viewer's curve — see the note the
    // script prints.
    return { filter: `format=gbrpf32le,${srgb},format=yuv420p`, exact: false };
  }
  return {
    filter:
      `format=gbrpf32le,geq=r='${mapped.r}':g='${mapped.g}':b='${mapped.b}',` +
      `${srgb},format=yuv420p`,
    exact: true,
  };
}

/**
 * Colour tags for the encoded stream. The frames are sRGB either way —
 * LDR frames were written that way by the shader, EXR frames are encoded
 * to sRGB above — so say so rather than letting players assume BT.709
 * (worth ~2 dB against the viewer, measured).
 *
 * Applied as a `setparams` filter rather than the `-color_*` output
 * options: those only reached the container for the EXR chain (where
 * zscale had already stamped the frames), leaving LDR encodes tagged
 * `unknown`. Marking the frames works for both.
 */
const COLOUR_TAGS = 'setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709';

/**
 * Generate the `encode_video.sh` bundled next to the frames.
 *
 * Pure: no I/O, deterministic given the inputs.
 */
export function generateFfmpegScript(opts: FfmpegScriptOptions): string {
  const { fps, frameCount, frameExt, mode, outputBase } = opts;
  const duration = (frameCount / fps).toFixed(1);
  const inputPattern = `frame_%06d.${frameExt}`;
  const isTurntable = mode === 'turntable';
  const name = isTurntable ? `${outputBase}-turntable` : outputBase;

  const grade = buildGradeFilter(frameExt, opts.grade);
  // `-vf` is shared by every encode below; LDR frames get the bare
  // pixel-format conversion H.264/H.265 need.
  const vf = `${grade ? grade.filter : 'format=yuv420p'},${COLOUR_TAGS}`;

  const header: string[] = [
    '#!/bin/bash',
    '# Luxar Image Sequence Encoder',
    `# ${frameCount} ${frameExt.toUpperCase()} frames at ${fps} FPS (${duration}s)`,
    '# Generated by Luxar Viewer — https://github.com/royerlab/luxar',
    '#',
    '# Requirements: ffmpeg',
    '# Usage: chmod +x encode_video.sh && ./encode_video.sh',
  ];

  if (isTurntable) {
    header.push(
      '#',
      '# Turntable: the frames cover a full 360° and stop one step short',
      '# of the start pose, so the encode loops seamlessly — no duplicate',
      '# frame at the wrap.'
    );
  }

  if (grade) {
    const label = TONE_MAP_LABEL[opts.grade!.toneMapping];
    header.push(
      '#',
      '# These EXR frames are SCENE-LINEAR and ungraded: the capture keeps',
      '# unclipped HDR by bypassing the viewer’s display transform. The',
      '# filter chain below re-applies that transform so the video matches',
      '# what you saw:',
      `#   exposure ${fmt(opts.grade!.exposure)} EV, offset ${fmt(opts.grade!.offset)}, ` +
        `gamma ${fmt(opts.grade!.gamma)}`,
      `#   tone mapping: ${label}`,
      '#'
    );
    if (grade.exact) {
      header.push(
        '# The tone map is the viewer’s own curve, written out as an exact',
        '# expression (ffmpeg’s built-in `tonemap` curves are different',
        '# functions — for ACES, `tonemap=hable` measures further from the',
        '# viewer than applying no tone mapping at all). `geq` evaluates it',
        '# per pixel on the CPU, so this is slower than a plain mux; drop',
        '# the -vf chain if you would rather grade the linear frames',
        '# yourself in a colour tool.'
      );
    } else {
      header.push(
        '# NOTE: AgX has no practical closed form for ffmpeg, so the frames',
        '# below are only clamped and sRGB-encoded — the AgX look is NOT',
        '# reproduced. For a pixel-exact match to the viewer, record a',
        '# PNG/WebP sequence instead (those frames are already graded).'
      );
    }
  }

  const lines: string[] = [
    ...header,
    '',
    'set -e',
    'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
    'cd "$SCRIPT_DIR"',
    '',
    '# --- MP4 (H.265, high quality) ---',
    'echo "Encoding MP4 (H.265)..."',
    `ffmpeg -y -framerate ${fps} -start_number 0 -i '${inputPattern}' \\`,
    '  -c:v libx265 -preset slow -crf 18 \\',
    `  -vf "${vf}" \\`,
    // hvc1 rather than the libx265 default hev1: QuickTime, Safari and
    // Final Cut refuse to play hev1-tagged MP4.
    '  -tag:v hvc1 \\',
    '  -movflags +faststart \\',
    `  "${name}.mp4"`,
    `echo "  -> ${name}.mp4"`,
    '',
    '# --- MP4 (H.264, widely compatible) ---',
    '# Uncomment if H.265 is not supported by your player:',
    '# echo "Encoding MP4 (H.264)..."',
    `# ffmpeg -y -framerate ${fps} -start_number 0 -i '${inputPattern}' \\`,
    '#   -c:v libx264 -preset slow -crf 18 \\',
    `#   -vf "${vf}" \\`,
    '#   -movflags +faststart \\',
    `#   "${name}-h264.mp4"`,
    `# echo "  -> ${name}-h264.mp4"`,
  ];

  if (frameExt === 'exr') {
    // HDR10 from the linear frames. Commented out because the peak
    // luminance the scene-linear values should map to is a judgement
    // call (`npl`), and because it needs an HDR display to evaluate.
    // Note this converts to PQ/BT.2020 rather than just TAGGING the
    // output as such — tagging alone, which this script used to do,
    // mislabels SDR pixels as HDR.
    lines.push(
      '',
      '# --- HDR10 MP4 (H.265, 10-bit PQ / BT.2020) ---',
      '# Uncomment for an HDR master. `npl` sets the nominal peak',
      '# luminance the linear values map to — tune it for your display.',
      '# echo "Encoding HDR MP4 (H.265 10-bit)..."',
      `# ffmpeg -y -framerate ${fps} -start_number 0 -i '${inputPattern}' \\`,
      '#   -vf "zscale=tin=linear:min=gbr:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020:npl=100:r=tv,format=yuv420p10le" \\',
      '#   -c:v libx265 -preset slow -crf 18 \\',
      '#   -x265-params "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc" \\',
      '#   -tag:v hvc1 -movflags +faststart \\',
      `#   "${name}-hdr.mp4"`,
      `# echo "  -> ${name}-hdr.mp4"`
    );
  }

  lines.push('', 'echo "Done!"', '');
  return lines.join('\n');
}
