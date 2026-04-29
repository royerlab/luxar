# Launcher icon assets

These files are referenced by `cli/native_app.py` when producing native
bundles via `luxar export --native ...`:

- `luxar-logo.png` — 1024×1024 RGBA PNG; copied into Linux folder bundles
  as `<AppName>.png` (FreeDesktop convention).
- `AppIcon.icns` — multi-resolution macOS icon set; copied into
  `<App>.app/Contents/Resources/AppIcon.icns`.

Both are checked into the repository so that:

1. Wheel installs (`pip install luxar`) get a working icon out of the
   box — no system fonts or rendering dependencies required at install
   time.
2. Contributors on Linux/Windows can produce iconed bundles without
   installing macOS-specific tooling.

## Regenerating

The source render pipeline lives in `packages/luxar-launcher/assets/`:

```bash
hatch run python packages/luxar-launcher/assets/build_logo.py   # PNG (macOS only)
bash packages/luxar-launcher/assets/build_icons.sh              # ICNS (macOS only)
```

Both scripts write directly into this directory. Re-run only when the
source emoji or rendering pipeline changes.
