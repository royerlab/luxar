/**
 * Unit tests for the bundled `encode_video.sh` generator.
 *
 * The colour maths in here was validated against the viewer itself:
 * a linear EXR and a PNG of the SAME frame were captured, the EXR was
 * encoded with the generated chain, and the result compared to the PNG
 * (PSNR). Those numbers are quoted where they pin a decision — the unit
 * tests below guard the structure that produced them.
 */

import { describe, it, expect } from 'vitest';
import {
  generateFfmpegScript,
  type FfmpegScriptOptions,
  type GradeSettings,
  type ToneMapName,
} from '../../../../ui/recording-panel/ffmpeg-script';

function opts(overrides: Partial<FfmpegScriptOptions> = {}): FfmpegScriptOptions {
  return {
    fps: 30,
    frameCount: 600,
    frameExt: 'png',
    mode: 'turntable',
    outputBase: 'luxar-capture-2026-05-05-103000',
    ...overrides,
  };
}

const NEUTRAL_GRADE = { exposure: 0, offset: 0, gamma: 1 } as const;

describe('generateFfmpegScript', () => {
  it('is a runnable bash script with the frame stats in the header', () => {
    const script = generateFfmpegScript(opts());
    expect(script.startsWith('#!/bin/bash')).toBe(true);
    expect(script).toContain('set -e');
    expect(script).toContain('600 PNG frames at 30 FPS (20.0s)');
    expect(script).toContain("'frame_%06d.png'");
    expect(script).toContain('-framerate 30');
    expect(script).toContain('chmod +x encode_video.sh');
  });

  it('names outputs after the capture, not a hardcoded "turntable"', () => {
    const turntable = generateFfmpegScript(opts());
    expect(turntable).toContain('"luxar-capture-2026-05-05-103000-turntable.mp4"');
    expect(turntable).not.toContain('"turntable.mp4"');

    const video = generateFfmpegScript(opts({ mode: 'video' }));
    expect(video).toContain('"luxar-capture-2026-05-05-103000.mp4"');
    expect(video).not.toContain('-turntable.mp4');
  });

  it('only mentions seamless looping for a turntable', () => {
    expect(generateFfmpegScript(opts())).toContain('loops seamlessly');
    expect(generateFfmpegScript(opts({ mode: 'video' }))).not.toContain('loops seamlessly');
  });

  it('tags H.265 as hvc1 so QuickTime and Safari will play it', () => {
    // libx265 defaults to hev1, which Apple players refuse.
    expect(generateFfmpegScript(opts())).toContain('-tag:v hvc1');
  });

  it('pins the frame numbering the ZIP actually uses', () => {
    expect(generateFfmpegScript(opts())).toContain('-start_number 0');
  });

  describe('LDR sequences', () => {
    it('applies no colour maths — the frames are already graded', () => {
      // PNG/WebP/JPEG come out of renderToImageData, i.e. after the
      // shader's exposure/tone map/sRGB. Touching them again would
      // double-grade.
      const script = generateFfmpegScript(
        opts({ frameExt: 'png', grade: { toneMapping: 'aces', ...NEUTRAL_GRADE } })
      );
      expect(script).not.toContain('geq');
      expect(script).not.toContain('zscale');
      // Pixel-format conversion and colour tagging only — no maths.
      expect(script).toContain('-vf "scale=out_color_matrix=bt709,format=yuv420p,setparams=');
    });

    it('converts to YUV with the matrix it tags', () => {
      // swscale's RGB→YUV default is BT.601, so tagging BT.709 over a bare
      // `format=yuv420p` makes every player decode with the wrong matrix:
      // measured on a lossless round-trip of saturated patches, pure green
      // came back 215 instead of 255 (worst error 40/255, vs 3/255 once the
      // conversion is told to use BT.709 as well).
      const script = generateFfmpegScript(opts({ frameExt: 'png' }));
      expect(script).toContain('colorspace=bt709');
      expect(script).toContain('scale=out_color_matrix=bt709');
      expect(script).not.toMatch(/-vf "format=yuv420p/);
    });

    it('has no HDR stanza', () => {
      const script = generateFfmpegScript(opts({ frameExt: 'jpg' }));
      expect(script).not.toContain('yuv420p10le');
      expect(script).not.toContain('smpte2084');
    });
  });

  describe('EXR sequences', () => {
    it('re-applies the viewer display transform (linear frames are pre-grade)', () => {
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'aces', ...NEUTRAL_GRADE } })
      );
      // sRGB encode — without it the video is dark and colour-shifted
      // (measured mean RGB 26,47,39 against the viewer's 67,67,71).
      expect(script).toContain('t=iec61966-2-1');
      expect(script).toContain('tin=linear');
      // ACES written out exactly: the input-matrix first row and the
      // RRTAndODTFit denominator constant.
      expect(script).toContain('0.59719');
      expect(script).toContain('0.238081');
      expect(script).toContain('geq=');
      expect(script).toContain('tone mapping: ACES Filmic');
    });

    it.each([
      ['linear', 'clip('],
      ['reinhard', '/(1+'],
      ['cineon', '6.2*'],
      ['aces', '0.983729'],
      ['neutral', '6.25*'],
    ] as const)('writes an exact expression for %s', (toneMapping, marker) => {
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping, ...NEUTRAL_GRADE } })
      );
      expect(script).toContain('geq=');
      expect(script).toContain(marker);
    });

    it('warns about geq cost on the AgX chain too — it runs the same filter', () => {
      // AgX still applies exposure/offset/gamma and the clamp through geq,
      // so it is exactly as slow as the exact curve. Only the no-grade
      // fallback skips geq, and only it should skip the warning.
      const agx = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'agx', ...NEUTRAL_GRADE } })
      );
      expect(agx).toContain('geq=');
      expect(agx).toContain('SLOW');

      const unreadable = generateFfmpegScript(opts({ frameExt: 'exr', grade: undefined }));
      expect(unreadable).not.toContain('geq=');
      expect(unreadable).not.toContain('SLOW');
    });

    it('says plainly that AgX is not reproduced rather than faking it', () => {
      // Four matrix stages around a log-space polynomial: no practical
      // closed form for geq, and ffmpeg has no AgX curve.
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'agx', ...NEUTRAL_GRADE } })
      );
      expect(script).toContain('AgX has no practical closed form');
      expect(script).toContain('PNG/WebP sequence instead');
      // Still sRGB-encoded — linear floats must not go straight out.
      expect(script).toContain('t=iec61966-2-1');
    });

    it('still applies the grade under AgX, only the curve falls back', () => {
      // The AgX fallback used to emit no geq at all, which silently threw
      // away exposure, offset and gamma — a +2 EV capture came out four
      // stops dark even though the header said the grade was applied.
      const script = generateFfmpegScript(
        opts({
          frameExt: 'exr',
          grade: { toneMapping: 'agx', exposure: 2, offset: 0.01, gamma: 1.5 },
        })
      );
      expect(script).toContain('geq=');
      expect(script).toContain('*4'); // exp2(2)
      expect(script).toContain('+0.01');
      expect(script).toContain('pow(');
      // The curve itself degrades to Linear's bare clamp.
      expect(script).toContain('clip(');
      expect(script).toContain('exposure 2 EV, offset 0.01, gamma 1.5');
    });

    it('factors the cross-channel curves through st()/ld() registers', () => {
      // Textually re-inlining `peak` (which reads all three offset
      // channels, which each read all three inputs) grew the Neutral
      // expression to 63 KB and 26 s per 1920×1088 frame — geq is a
      // per-pixel CPU interpreter, so expression size is the whole cost.
      // The registers compute each intermediate once per pixel.
      for (const toneMapping of ['aces', 'neutral'] as const) {
        const script = generateFfmpegScript(
          opts({ frameExt: 'exr', grade: { toneMapping, ...NEUTRAL_GRADE } })
        );
        expect(script).toContain('st(0,');
        expect(script).toContain('ld(0)');
        const geq = /geq=[^\n]*/.exec(script)?.[0] ?? '';
        expect(geq.length).toBeLessThan(4000);
        // Ten slots exist (0-9); anything higher is silently clipped.
        expect(geq).not.toMatch(/[sl][td]\(1\d/);
      }
    });

    it('folds exposure, offset and gamma into the expression when non-default', () => {
      const script = generateFfmpegScript(
        opts({
          frameExt: 'exr',
          grade: { toneMapping: 'reinhard', exposure: 1, offset: 0.02, gamma: 2 },
        })
      );
      // The shader does `pow(max(c·exp2(E) + O, 0), 1/G)`, so the emitted
      // exponent is the RECIPROCAL — 1/2 = 0.5, not 2. Asserting only
      // that `pow(` appears lets an inverted gamma through.
      expect(script).toContain('(r(X,Y)*2)'); // exp2(1) = 2
      expect(script).toContain('+0.02,0)'); // offset, clamped at 0
      expect(script).toContain(',0.5)'); // 1 / 2
      expect(script).not.toContain(',2)0'); // guards the reciprocal
      expect(script).toContain('exposure 1 EV, offset 0.02, gamma 2');
    });

    it('omits the exposure and gamma terms when they are at their defaults', () => {
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'linear', ...NEUTRAL_GRADE } })
      );
      // A neutral grade leaves the multiply and the `pow` unemitted. The
      // `max` stays: the shader's EOG clamp is unconditional, and the
      // numeric-parity tests below fail on a negative sample without it.
      // (This assertion used to read `clip(r(X,Y),0,1)`.)
      expect(script).toContain('clip(max(r(X,Y),0),0,1)');
      expect(script).not.toContain('pow(');
      expect(script).not.toMatch(/r\(X,Y\)\*/);
    });

    it('warns that the exact chain is slow and names the faster routes', () => {
      // Measured CPU time: ~2.9 s/frame at 720p, ~7.5 at 1080p, ~35 at 4K
      // against ~0.02 for a plain mux. `geq` is slice-threaded, so the
      // wall clock is that over the core count (~0.8 s/frame at 1080p on
      // 16 threads) — enough to be worth warning about, not enough to
      // describe as one core's worth of hours.
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'aces', ...NEUTRAL_GRADE } })
      );
      expect(script).toContain('SLOW');
      expect(script).toContain('s/frame at 720p');
      expect(script).toContain('slice-threaded');
      expect(script).toContain('PNG/WebP sequence instead');
      expect(script).not.toContain('one CPU core');
    });

    it('tells geq to sample with nearest interpolation, in both chains', () => {
      // `geq` defaults to BILINEAR sampling and clamps its sample
      // coordinate to [0, w-2] × [0, h-2], so the rightmost column and
      // bottom row of every frame come out as copies of their
      // neighbours: on an 8×2 gbrpf32le frame under an identity
      // expression, a last column of 1.0 read back as 0.6. The
      // expressions only ever read integer X, Y, so nearest is exact.
      const script = generateFfmpegScript(
        opts({
          frameExt: 'exr',
          grade: { toneMapping: 'aces', exposure: 1, offset: 0.01, gamma: 1.2 },
        })
      );
      const geqs = script.match(/geq=[^:]*/g) ?? [];
      // The shared `-vf` chain (H.265 and the commented H.264 alternate)
      // plus the HDR10 stanza's EOG-only filter.
      expect(geqs.length).toBe(3);
      expect(geqs.every((g) => g === 'geq=interpolation=nearest')).toBe(true);
    });

    it('still sRGB-encodes (and says so) when the grade could not be read', () => {
      // A missing grade must not silently fall back to the uncorrected
      // chain — that is the 15 dB failure. Linear floats always get at
      // least the transfer conversion.
      const script = generateFfmpegScript(opts({ frameExt: 'exr', grade: undefined }));
      expect(script).toContain('t=iec61966-2-1');
      expect(script).toContain('grade could not be read');
      expect(script).not.toContain('geq=');
    });

    it('offers an HDR10 stanza that converts rather than just tagging', () => {
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'aces', ...NEUTRAL_GRADE } })
      );
      expect(script).toContain('yuv420p10le');
      // The old script set bt2020/PQ metadata on unconverted pixels.
      expect(script).toContain('t=smpte2084');
      expect(script).toContain('p=bt2020');
    });
  });
});

// ── Numeric parity with three.js ──────────────────────────────────
//
// The assertions above are single-substring markers, which is a weak
// net for colour maths: transposing an ACES matrix row, swapping the
// `r:` and `g:` output expressions, mistyping 1.60475 as 1.06475 or
// pointing an `st()` at the wrong `ld()` slot all keep them green, and
// all are per-channel, hue-specific errors that a mean-RGB comparison
// cannot see either. So EVALUATE the emitted text and compare it,
// channel by channel, against three's own tone-mapping functions.

/** A linear RGB triple, as the EXR frames carry it. */
type RGB = [number, number, number];

/**
 * Evaluate the subset of ffmpeg's expression language this module
 * emits: `;` sequencing, `st(i,x)` / `ld(i)` over ten registers,
 * `clip`, `min`, `max`, `pow`, `if`, `lt`, the four arithmetic
 * operators with parentheses and unary minus, numeric literals, and
 * the channel reads `r(X,Y)` / `g(X,Y)` / `b(X,Y)`.
 *
 * A real recursive-descent evaluation of the generated string, not a
 * re-statement of what it is meant to say — that is the whole point of
 * the test. `X` and `Y` evaluate to 0 (the expressions are per-pixel
 * constants, so the coordinate is irrelevant), and `if()` evaluates
 * both branches before selecting: no emitted branch has a side effect,
 * and a division by a zero peak in the untaken branch is discarded.
 */
function evaluateFfmpegExpression(source: string, input: RGB): number {
  const s = source.replace(/\s+/g, '');
  const registers = new Array<number>(10).fill(0);
  let pos = 0;

  const peek = (): string | undefined => s[pos];

  const call = (name: string, args: number[]): number => {
    switch (name) {
      case 'r':
        return input[0];
      case 'g':
        return input[1];
      case 'b':
        return input[2];
      case 'st':
        registers[Math.trunc(args[0])] = args[1];
        return args[1];
      case 'ld':
        return registers[Math.trunc(args[0])];
      case 'clip':
        return Math.min(Math.max(args[0], args[1]), args[2]);
      case 'min':
        return Math.min(args[0], args[1]);
      case 'max':
        return Math.max(args[0], args[1]);
      case 'pow':
        return Math.pow(args[0], args[1]);
      case 'lt':
        return args[0] < args[1] ? 1 : 0;
      case 'if':
        return args[0] !== 0 ? args[1] : args[2];
      default:
        throw new Error(`unsupported function '${name}' at ${pos} in ${s}`);
    }
  };

  const parsePrimary = (): number => {
    if (peek() === '(') {
      pos++;
      const value = parseAdditive();
      if (peek() !== ')') throw new Error(`expected ')' at ${pos} in ${s}`);
      pos++;
      return value;
    }
    const number = /^[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?/.exec(s.slice(pos));
    if (number) {
      pos += number[0].length;
      return parseFloat(number[0]);
    }
    const ident = /^[A-Za-z_][A-Za-z_0-9]*/.exec(s.slice(pos));
    if (!ident) throw new Error(`unparsable at ${pos} in ${s}`);
    pos += ident[0].length;
    if (peek() !== '(') {
      if (ident[0] === 'X' || ident[0] === 'Y') return 0;
      throw new Error(`unknown variable '${ident[0]}' in ${s}`);
    }
    pos++;
    const args: number[] = [];
    if (peek() !== ')') {
      args.push(parseAdditive());
      while (peek() === ',') {
        pos++;
        args.push(parseAdditive());
      }
    }
    if (peek() !== ')') throw new Error(`expected ')' at ${pos} in ${s}`);
    pos++;
    return call(ident[0], args);
  };

  const parseUnary = (): number => {
    if (peek() === '-') {
      pos++;
      return -parseUnary();
    }
    if (peek() === '+') {
      pos++;
      return parseUnary();
    }
    return parsePrimary();
  };

  const parseMultiplicative = (): number => {
    let value = parseUnary();
    for (;;) {
      const op = peek();
      if (op !== '*' && op !== '/') return value;
      pos++;
      const rhs = parseUnary();
      value = op === '*' ? value * rhs : value / rhs;
    }
  };

  function parseAdditive(): number {
    let value = parseMultiplicative();
    for (;;) {
      const op = peek();
      if (op !== '+' && op !== '-') return value;
      pos++;
      const rhs = parseMultiplicative();
      value = op === '+' ? value + rhs : value - rhs;
    }
  }

  let result = parseAdditive();
  while (peek() === ';') {
    pos++;
    result = parseAdditive();
  }
  if (pos !== s.length) throw new Error(`trailing input at ${pos} in ${s}`);
  return result;
}

/** Pull the three per-channel expressions out of the generated script. */
function emittedExpressions(grade: GradeSettings): { r: string; g: string; b: string } {
  const script = generateFfmpegScript(opts({ frameExt: 'exr', grade }));
  const match = /geq=(?:interpolation=nearest:)?r='([^']*)':g='([^']*)':b='([^']*)'/.exec(script);
  if (!match) throw new Error('no geq chain in the generated script');
  return { r: match[1], g: match[2], b: match[3] };
}

// ── three.js reference, ported from
//    node_modules/three/src/renderers/shaders/ShaderChunk/
//    tonemapping_pars_fragment.glsl.js (toneMappingExposure = 1, which
//    is how the mega-shader runs them — exposure is the EOG block's).
const saturate = (x: number): number => Math.min(Math.max(x, 0), 1);
const perChannel =
  (f: (v: number) => number) =>
  (c: RGB): RGB => [f(c[0]), f(c[1]), f(c[2])];

const linearToneMapping = perChannel(saturate);

const reinhardToneMapping = perChannel((v) => saturate(v / (1 + v)));

const optimizedCineonToneMapping = perChannel((v) => {
  const c = Math.max(0, v - 0.004);
  return Math.pow((c * (6.2 * c + 0.5)) / (c * (6.2 * c + 1.7) + 0.06), 2.2);
});

function acesFilmicToneMapping(colour: RGB): RGB {
  // three stores both matrices transposed (column vectors), so a row of
  // the product below is a COLUMN of the GLSL literal.
  const mul = (m: RGB[], v: RGB): RGB => [
    m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
    m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
    m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
  ];
  const inputMat: RGB[] = [
    [0.59719, 0.076, 0.0284],
    [0.35458, 0.90834, 0.13383],
    [0.04823, 0.01566, 0.83777],
  ];
  const outputMat: RGB[] = [
    [1.60475, -0.10208, -0.00327],
    [-0.53108, 1.10813, -0.07276],
    [-0.07367, -0.00605, 1.07602],
  ];
  const fit = (v: number): number =>
    (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.432951) + 0.238081);
  const scaled: RGB = [colour[0] / 0.6, colour[1] / 0.6, colour[2] / 0.6];
  const aces = mul(inputMat, scaled);
  const fitted: RGB = [fit(aces[0]), fit(aces[1]), fit(aces[2])];
  return mul(outputMat, fitted).map(saturate) as RGB;
}

function pbrNeutralToneMapping(colour: RGB): RGB {
  const startCompression = 0.8 - 0.04;
  const desaturation = 0.15;
  const x = Math.min(colour[0], Math.min(colour[1], colour[2]));
  const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  const c: RGB = [colour[0] - offset, colour[1] - offset, colour[2] - offset];
  const peak = Math.max(c[0], Math.max(c[1], c[2]));
  if (peak < startCompression) return c;
  const d = 1 - startCompression;
  const newPeak = 1 - (d * d) / (peak + d - startCompression);
  const scaled: RGB = [(c[0] * newPeak) / peak, (c[1] * newPeak) / peak, (c[2] * newPeak) / peak];
  const g = 1 - 1 / (desaturation * (peak - newPeak) + 1);
  return scaled.map((v) => v * (1 - g) + newPeak * g) as RGB;
}

const REFERENCE_CURVES: Record<string, (c: RGB) => RGB> = {
  linear: linearToneMapping,
  reinhard: reinhardToneMapping,
  cineon: optimizedCineonToneMapping,
  aces: acesFilmicToneMapping,
  neutral: pbrNeutralToneMapping,
};

/** The mega-shader's EOG block, verbatim (shader.glsl.ts §4). */
function applyEogReference(colour: RGB, grade: Omit<GradeSettings, 'toneMapping'>): RGB {
  const scale = Math.pow(2, grade.exposure);
  return colour.map((v) => Math.pow(Math.max(v * scale + grade.offset, 0), 1 / grade.gamma)) as RGB;
}

/**
 * Colours chosen so a per-channel mistake cannot hide: primaries and
 * secondaries pin each matrix row on its own, the saturated HDR pairs
 * exercise the ACES/Neutral cross-channel terms asymmetrically, the
 * 0.52/0.56/0.76 triples straddle Neutral's compression knee, and the
 * negative components catch an EOG clamp that is not emitted (Reinhard
 * maps −2 to +2, i.e. white where the viewer is black).
 */
const PARITY_COLOURS: RGB[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 1],
  [1, 0, 1],
  [8, 0.2, 0.05],
  [0.05, 0.2, 8],
  [0.52, 0.56, 0.76],
  [0.76, 0.52, 0.56],
  [0.56, 0.76, 0.52],
  [0.0012, 0.0004, 0.002],
  [-2, 0.4, 0.4],
  [0.4, -1, 0.4],
  [0.4, 0.4, -0.5],
];

const PARITY_GRADES: Array<{ label: string; grade: Omit<GradeSettings, 'toneMapping'> }> = [
  { label: 'default grade', grade: { exposure: 0, offset: 0, gamma: 1 } },
  {
    label: 'exposure 1.5 / offset -0.02 / gamma 2.2',
    grade: { exposure: 1.5, offset: -0.02, gamma: 2.2 },
  },
];

describe.each(PARITY_GRADES)(
  'emitted expressions vs three.js tone mapping ($label)',
  ({ grade }) => {
    // three's Cineon and PBR-Neutral do not end in a saturate; the module
    // clamps every curve because the frame is about to become 8-bit YUV,
    // where the clamp is a no-op. Compare against the clamped form.
    it.each(['linear', 'reinhard', 'cineon', 'aces', 'neutral'] as const)(
      '%s agrees per channel to < 1e-6',
      (toneMapping: ToneMapName) => {
        const expressions = emittedExpressions({ toneMapping, ...grade });
        const mismatches: string[] = [];

        for (const colour of PARITY_COLOURS) {
          const expected = REFERENCE_CURVES[toneMapping](applyEogReference(colour, grade)).map(
            saturate
          ) as RGB;
          const actual: RGB = [
            evaluateFfmpegExpression(expressions.r, colour),
            evaluateFfmpegExpression(expressions.g, colour),
            evaluateFfmpegExpression(expressions.b, colour),
          ];
          for (const channel of [0, 1, 2] as const) {
            if (!(Math.abs(actual[channel] - expected[channel]) < 1e-6)) {
              mismatches.push(
                `${toneMapping} [${colour.join(', ')}] channel ${'rgb'[channel]}: ` +
                  `emitted ${actual[channel]} vs three ${expected[channel]}`
              );
            }
          }
        }

        expect(mismatches).toEqual([]);
      }
    );
  }
);
