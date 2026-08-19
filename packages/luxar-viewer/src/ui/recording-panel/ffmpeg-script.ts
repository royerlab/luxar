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
 *   viewer ACES      exact geq 38.2 dB · tonemap=hable 16.0 · nothing 23.8
 *                    (the same maths with PNG out, i.e. no codec, tops
 *                    out at 40.3 dB — that is the ceiling, not this chain)
 *   viewer Reinhard  tonemap=reinhard 34.8 dB, and the exact expression
 *                    measured closer still
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
 * The exposure multiply and the gamma `pow` are emitted only where they
 * change the pixel, so the common all-default case keeps the expression
 * short. The `max` is NOT optional: the shader clamps unconditionally,
 * and a scene-linear EXR carrying a negative component would otherwise
 * run a value through the tone curve that the viewer never saw
 * (Reinhard on −2 comes out at +2, i.e. white where the viewer is black).
 * It is emitted once, before the `pow`, whether or not an offset was
 * applied.
 */
function applyEog(channel: string, grade: GradeSettings): string {
  let expr = channel;
  const scale = Math.pow(2, grade.exposure);
  if (scale !== 1) expr = `(${expr}*${fmt(scale)})`;
  expr = grade.offset !== 0 ? `max(${expr}+${fmt(grade.offset)},0)` : `max(${expr},0)`;
  if (grade.gamma !== 1) {
    expr = `pow(${expr},${fmt(1 / grade.gamma)})`;
  }
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
 * Join `st()` assignments and a final expression with ffmpeg's `;`
 * sequencing operator, which evaluates each term and yields the last.
 *
 * Used by the cross-channel curves (ACES, Neutral) to compute a shared
 * intermediate ONCE per pixel instead of textually re-inlining it at
 * every use. Inlining is what a naive expansion does, and it blows up
 * multiplicatively: the Neutral curve reads `peak`, which reads all
 * three offset channels, which each read all three inputs. Measured on
 * the same 1920×1088 frames (`geq` is a per-pixel CPU interpreter, so
 * expression size is the whole cost):
 *
 *   ACES     8.0 KB → 11.0 s/frame   |  2.0 KB → 1.5 s/frame
 *   Neutral 63.2 KB → 26.1 s/frame   |  1.5 KB → 1.0 s/frame
 *
 * i.e. a 600-frame turntable went from ~2-4 h of encoding to ~10-15 min,
 * with bit-identical output (both forms verified against three's own
 * functions to < 1e-7). Those s/frame figures are wall clock on the
 * machine they were measured on; `geq` is slice-threaded, so they scale
 * with the core count and the header's own numbers (CPU time) are the
 * same measurements counted differently.
 *
 * `st(i, …)` / `ld(i)` have ten slots (0–9) and the registers belong to
 * the expression, not the filter, so each of the r/g/b expressions
 * needs its own preamble.
 */
function seq(...parts: string[]): string {
  return parts.join(';');
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
      // matrix, saturate. Slots 0-2 hold the graded input, 3-5 the
      // ACES-space colour, 6-8 its RRTAndODTFit.
      const s = '(1/0.6)';
      const fit = (v: string): string =>
        `((${v}*(${v}+0.0245786)-0.000090537)/(${v}*(0.983729*${v}+0.432951)+0.238081))`;
      const pre = seq(
        `st(0,${c.r})`,
        `st(1,${c.g})`,
        `st(2,${c.b})`,
        `st(3,(0.59719*ld(0)+0.35458*ld(1)+0.04823*ld(2))*${s})`,
        `st(4,(0.07600*ld(0)+0.90834*ld(1)+0.01566*ld(2))*${s})`,
        `st(5,(0.02840*ld(0)+0.13383*ld(1)+0.83777*ld(2))*${s})`,
        `st(6,${fit('ld(3)')})`,
        `st(7,${fit('ld(4)')})`,
        `st(8,${fit('ld(5)')})`
      );
      return {
        r: seq(pre, sat('1.60475*ld(6)-0.53108*ld(7)-0.07367*ld(8)')),
        g: seq(pre, sat('-0.10208*ld(6)+1.10813*ld(7)-0.00605*ld(8)')),
        b: seq(pre, sat('-0.00327*ld(6)-0.07276*ld(7)+1.07602*ld(8)')),
      };
    }

    case 'neutral': {
      // Khronos PBR Neutral. `startCompression = 0.8 - 0.04`,
      // `desaturation = 0.15`. Written with ffmpeg's `if()` because the
      // GLSL takes an early return below the compression knee.
      //
      // Slots: 0-2 graded input, 3 the channel minimum and then the
      // offset derived from it, 4-6 the offset colour, 7 its peak, 8 the
      // compressed peak, 9 the desaturation weight.
      const startCompression = 0.8 - 0.04;
      const d = 1 - startCompression;
      const pre = seq(
        `st(0,${c.r})`,
        `st(1,${c.g})`,
        `st(2,${c.b})`,
        'st(3,min(ld(0),min(ld(1),ld(2))))',
        'st(3,if(lt(ld(3),0.08),ld(3)-6.25*ld(3)*ld(3),0.04))',
        'st(4,ld(0)-ld(3))',
        'st(5,ld(1)-ld(3))',
        'st(6,ld(2)-ld(3))',
        'st(7,max(ld(4),max(ld(5),ld(6))))',
        // Below the knee these two are unused (the `if` below is lazy),
        // so a division by a zero peak never reaches the result.
        `st(8,1-${fmt(d * d)}/(ld(7)+${fmt(d - startCompression)}))`,
        'st(9,1-1/(0.15*(ld(7)-ld(8))+1))'
      );
      // below the knee → untouched; above → scale to newPeak, then
      // desaturate toward it.
      const compress = (slot: number): string =>
        `if(lt(ld(7),${fmt(startCompression)}),ld(${slot}),` +
        `ld(${slot})*ld(8)/ld(7)*(1-ld(9))+ld(8)*ld(9))`;
      return {
        r: seq(pre, sat(compress(4))),
        g: seq(pre, sat(compress(5))),
        b: seq(pre, sat(compress(6))),
      };
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
): { filter: string; exact: 'exact' | 'no-curve' | 'unknown-grade' } | null {
  if (frameExt !== 'exr') return null;

  const channels: ExposedChannels | null = grade
    ? {
        r: applyEog('r(X,Y)', grade),
        g: applyEog('g(X,Y)', grade),
        b: applyEog('b(X,Y)', grade),
      }
    : null;
  const mapped = channels && grade ? toneMapExpressions(channels, grade.toneMapping) : null;
  // zscale carries the linear→sRGB encode the shader does last — it is
  // what makes the EXR's linear floats display-referred at all.
  //
  // The matrix must be a YUV one (`m=bt709`): asking zscale for `m=gbr`
  // while the encoder wants yuv420p fails outright with "YUV color
  // family cannot have RGB matrix coefficients". Limited range is the
  // safe delivery default; `r=pc` measured 1.5 dB closer to the viewer
  // but relies on every player honouring the full-range flag.
  const srgb = 'zscale=tin=linear:t=iec61966-2-1:min=gbr:m=bt709:pin=bt709:p=bt709:r=tv';

  // AgX has no closed form for its curve, but exposure/offset/gamma and
  // the clamp still have to be applied — dropping them silently encoded
  // the frames at the wrong brightness, which is the failure this whole
  // chain exists to avoid. Falling back to a bare saturate is exactly
  // what the shader does for Linear; only the AgX look is missing, and
  // the script says so.
  const out =
    mapped ?? (channels ? { r: sat(channels.r), g: sat(channels.g), b: sat(channels.b) } : null);

  if (!out) {
    // The grade could not be read at all (no `getGradeSettings`). Even
    // then the linear floats must not go out untouched — that is the
    // 15 dB failure this chain exists to prevent — so still convert the
    // transfer, and let the script say the curve is missing.
    //
    // The clamp is part of that minimum. Without it the over-range
    // floats are clipped after the RGB→YUV matrix instead, per channel
    // and unevenly, which shifts hue rather than just crushing the
    // highlight: a linear (100, 0.5, 0.5) pixel measured (255, 237, 7)
    // through the unclamped chain against (254, 186, 186) with the
    // clamp in front. In-range and negative values are unaffected.
    const clamped = (channel: string): string => sat(`max(${channel},0)`);
    return {
      filter:
        `format=gbrpf32le,geq=interpolation=nearest:r='${clamped('r(X,Y)')}':` +
        `g='${clamped('g(X,Y)')}':b='${clamped('b(X,Y)')}',${srgb},format=yuv420p`,
      exact: 'unknown-grade',
    };
  }

  // `interpolation=nearest` is load-bearing, not a tuning knob. `geq`
  // samples with bilinear interpolation by default and clamps the sample
  // coordinate to [0, w-2] × [0, h-2], so the rightmost column and the
  // bottom row come out as copies of their neighbours — on an 8×2
  // gbrpf32le frame under an identity expression, a last column of 1.0
  // read back as 0.6. These expressions only ever read integer X, Y, so
  // nearest is both exact and cheaper.
  return {
    filter:
      `format=gbrpf32le,geq=interpolation=nearest:r='${out.r}':g='${out.g}':b='${out.b}',` +
      `${srgb},format=yuv420p`,
    exact: mapped !== null ? 'exact' : 'no-curve',
  };
}

/**
 * Exposure / offset / gamma on their own, for the HDR10 stanza: an HDR
 * master keeps the scene's dynamic range (no SDR tone curve) but must
 * still honour the grade, or it comes out at a different brightness
 * from the SDR encode beside it. Returns `null` when the grade is
 * neutral, so the common case adds no filter at all.
 */
function buildEogOnlyFilter(grade: GradeSettings | undefined): string | null {
  if (!grade) return null;
  if (grade.exposure === 0 && grade.offset === 0 && grade.gamma === 1) return null;
  const r = applyEog('r(X,Y)', grade);
  const g = applyEog('g(X,Y)', grade);
  const b = applyEog('b(X,Y)', grade);
  // `interpolation=nearest` for the same reason as the SDR chain above:
  // the default bilinear sampling duplicates the edge column and row.
  return `format=gbrpf32le,geq=interpolation=nearest:r='${r}':g='${g}':b='${b}',`;
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
 * LDR frames are sRGB and have to become YUV for H.264/H.265 — and the
 * matrix has to be named, because swscale's default for an RGB→YUV
 * conversion is BT.601. A bare `format=yuv420p` under the BT.709 tag
 * above therefore hands every player the wrong matrix, which is a visible
 * shift, not a rounding one: on a lossless round-trip of saturated
 * patches the worst channel error is 40/255 (pure green comes back 215)
 * against 3/255 once the conversion is told to use BT.709 too.
 *
 * The EXR chain needs none of this — its `zscale` already converts with
 * `m=bt709`, so its pixels and the tag agree (measured: ≤1/255 against
 * the shader's own sRGB output).
 */
const LDR_TO_YUV = 'scale=out_color_matrix=bt709,format=yuv420p';

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
  // `-vf` is shared by every encode below; LDR frames get the
  // pixel-format conversion H.264/H.265 need.
  const vf = `${grade ? grade.filter : LDR_TO_YUV},${COLOUR_TAGS}`;

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
    header.push(
      '#',
      '# These EXR frames are SCENE-LINEAR and ungraded: the capture keeps',
      '# unclipped HDR by bypassing the viewer’s display transform.'
    );
    // List the grade whenever it is known — it is worth reading even
    // when the curve cannot be reproduced — but only promise a MATCH
    // when the chain really carries the curve. Gating the whole block on
    // `opts.grade` alone let AgX announce that the video matches what
    // you saw six lines above the NOTE saying its look is not
    // reproduced. The unknown-grade branch names nothing at all, which
    // would leave a colon in front of an empty list.
    if (opts.grade) {
      header.push(
        ...(grade.exact === 'exact'
          ? [
              '# The filter chain below re-applies that transform so the video',
              '# matches what you saw:',
            ]
          : [
              '# The filter chain below re-applies the grade that was in',
              '# effect (the curve is a separate matter — see the NOTE):',
            ]),
        `#   exposure ${fmt(opts.grade.exposure)} EV, offset ${fmt(opts.grade.offset)}, ` +
          `gamma ${fmt(opts.grade.gamma)}`,
        `#   tone mapping: ${TONE_MAP_LABEL[opts.grade.toneMapping]}`
      );
    }
    header.push(
      '#',
      '# Vignette, detector noise and lens distortion are NOT in these',
      '# frames and cannot be put back here — the EXR capture turns them',
      '# off so the archive holds the scene, not a look.',
      '#'
    );
    if (grade.exact === 'unknown-grade') {
      header.push(
        '# NOTE: the viewer’s grade could not be read at capture time, so',
        '# the frames below are only clamped and sRGB-encoded — exposure and',
        '# the tone-mapping curve are NOT applied. Record a PNG/WebP',
        '# sequence for a pixel-exact match to what you saw.'
      );
    } else if (grade.exact === 'exact') {
      header.push(
        '# The tone map is the viewer’s own curve, written out as an exact',
        '# expression (ffmpeg’s built-in `tonemap` curves are different',
        '# functions — for ACES, `tonemap=hable` measures further from the',
        '# viewer than applying no tone mapping at all).'
      );
    } else {
      header.push(
        '# NOTE: AgX has no practical closed form for ffmpeg. The grade',
        '# above is applied, but the curve itself falls back to a plain',
        '# clamp — the AgX look is NOT reproduced. For a pixel-exact match',
        '# to the viewer, record a PNG/WebP sequence instead (those frames',
        '# are already graded).'
      );
    }

    // The cost warning belongs to every chain that carries a curve —
    // AgX applies exposure/offset/gamma through the same per-pixel
    // expression and is exactly as slow. The `unknown-grade` chain runs
    // `geq` too, but only a three-term clamp; the measurements below
    // were taken on the full expressions and would overstate it.
    if (grade.exact !== 'unknown-grade') {
      header.push(
        '#',
        '# SLOW: `geq` evaluates that expression per pixel. Measured CPU',
        '# time ~2.9 s/frame at 720p, ~7.5 s at 1080p and ~35 s at 4K. The',
        '# filter is slice-threaded, so the wall clock is shorter but not',
        '# proportionally shorter: the 1080p chain measured ~0.73 s/frame',
        '# on 16 threads, against ~0.02 s/frame for a plain mux. On a',
        `# many-core machine these ${frameCount} frames are minutes of encoding;`,
        '# on a small one, budget rather more. Faster options, in order of',
        '# convenience:',
        '#   1. Record a PNG/WebP sequence instead — those frames come out',
        '#      of the viewer already graded, so the encode is a plain mux',
        '#      and matches the viewer exactly. Prefer this unless you',
        '#      specifically need the unclipped HDR these EXRs carry.',
        '#   2. Drop the -vf chain here and grade the linear frames in a',
        '#      colour tool (Resolve, Nuke, oiiotool) that does it on the GPU.',
        '#   3. Keep it, and let it run — the output is the reference.'
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
      // Exposure/offset/gamma still apply (they are the grade, not the
      // display transform); the SDR tone curve deliberately does not.
      `#   -vf "${buildEogOnlyFilter(opts.grade) ?? ''}zscale=tin=linear:min=gbr:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020:npl=100:r=tv,format=yuv420p10le" \\`,
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
