#### KTX2 UASTC authoring uses measured RDO defaults (#2403)

UASTC texture authoring now exposes `texture_ktx2_rdo_l` and
`texture_ktx2_zcmp`, defaulting to RDO lambda 0.25 and zstd level 9.
Set `texture_ktx2_rdo_l=0` to keep the previous lossless zstd-only authoring. On
the two real 8193x8192 Blue Marble tiles used by the Earth demos, `toktx` 4.4.2
at UASTC quality 2 drops from 107.26 MiB without RDO to 88.80 MiB with those
defaults, a 17.2% wire reduction. Decoded level-0 quality moves from 48.99/47.70
dB PSNR and 0.9966/0.9954 SSIM to 48.16/47.08 dB and 0.9943/0.9932 respectively.

The more aggressive lambda 0.5 with zstd level 9 reached 84.76 MiB, but doubled
the PSNR loss for only 4.04 MiB more savings, so it is not the general default.
ETC1S quality 128 reached 10.97 MiB but only 34.13/33.19 dB and
0.9147/0.9141 SSIM; it remains opt-in and uses `toktx`'s `etc1s` mode.
