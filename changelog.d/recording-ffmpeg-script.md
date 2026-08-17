#### The bundled encode script now produces the video you actually saw

Every image and EXR frame sequence ships with an `encode_video.sh`. For EXR it was
producing a badly wrong video, because EXR frames are captured *scene-linear and
pre-grade* — the capture mode deliberately bypasses exposure, tone mapping and the sRGB
encode so the archive keeps unclipped HDR — and the script fed those floats to the
encoder as if they were already display-referred. Measured on a real capture, that put
the video 15.2 dB from the viewer's own PNG of the same frame: mean RGB (26, 47, 39)
against (67, 67, 71). Dark and colour-shifted.

The script now re-applies the viewer's display transform: exposure, offset, gamma, the
tone-mapping curve, then the sRGB encode, with the values read from the renderer at
capture time. The tone map is written out as the viewer's own curve — three's exact
constants as an ffmpeg expression — rather than one of ffmpeg's built-in approximations,
which are different functions: for an ACES scene, `tonemap=hable` (the usual stand-in)
measures 16.0 dB, *worse than applying no tone mapping at all* (23.8 dB). The exact
expression lands at 38.2 dB, within 2 dB of what the same maths achieves with no video
codec in the way at all. Linear, Reinhard, Cineon, ACES and Neutral are all reproduced
exactly; AgX has no practical closed form here, so its curve degrades to a plain clamp
and the script says so, pointing at recording a PNG sequence instead of quietly producing
the wrong look — but exposure, offset and gamma are still applied there, since dropping
them would put the frames at the wrong brightness for the same reason the old script did.

Because `geq` interprets its expression per pixel on the CPU, expression size *is* the
encode cost, so the two cross-channel curves share their intermediates through
expression registers instead of being re-inlined at every use: ACES 8.0 KB → 2.0 KB
(11.0 → 1.5 s per 1920×1088 frame) and Neutral 63.2 KB → 1.5 KB (26.1 → 1.0 s), which
takes a 600-frame turntable from hours of encoding down to minutes with bit-identical
output.

Several smaller things in the same script were wrong or missing. Outputs were always
named `turntable.mp4` regardless of what you recorded — they now take the capture's own
name, and a turntable's script notes that its frames loop seamlessly while a video's does
not. H.265 output is tagged `hvc1`, without which QuickTime, Safari and Final Cut refuse
to play the file at all. The colour tags now reach the container (as frame parameters —
the `-color_*` output options silently failed to on the LDR path). Frame numbering is
pinned with `-start_number 0`. And the HDR10 variant now converts to PQ/BT.2020 instead
of merely tagging SDR pixels as HDR, though it stays commented out since its peak-luminance
mapping needs an HDR display to judge.

PNG/WebP/JPEG sequences are deliberately left alone by all of this: those frames come out
of the viewer already graded, so any colour maths in the script would double-apply it.
