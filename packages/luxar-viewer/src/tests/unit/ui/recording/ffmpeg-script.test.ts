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

    it('omits EOG terms entirely when they are at their defaults', () => {
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'linear', ...NEUTRAL_GRADE } })
      );
      // A neutral grade should leave a bare `clip(r(X,Y),0,1)`.
      expect(script).toContain('clip(r(X,Y),0,1)');
    });

    it('warns that the exact chain is slow and names the faster routes', () => {
      // Measured: ~2.9 s/frame at 720p, ~7.5 at 1080p, ~35 at 4K against
      // ~0.02 for a plain mux. A 4K turntable is hours, so the script has
      // to say so rather than looking hung.
      const script = generateFfmpegScript(
        opts({ frameExt: 'exr', grade: { toneMapping: 'aces', ...NEUTRAL_GRADE } })
      );
      expect(script).toContain('SLOW');
      expect(script).toContain('s/frame at 720p');
      expect(script).toContain('PNG/WebP sequence instead');
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
