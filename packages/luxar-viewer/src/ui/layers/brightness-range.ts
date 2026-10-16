/**
 * The gsplats **Brightness** control (`display_gain`) and the data-unit labels
 * of the display window.
 *
 * A `luxar gsplat convert` scene stores splat amplitudes in SOURCE DATA UNITS
 * (detector counts above background, or the source's own float units), so the
 * display window reads in data units. How bright those values render is a
 * separate per-node factor, `display_gain`, which the shader multiplies into
 * every amplitude-linear term (emission and volumetric density) and never into
 * the colormap lookup (LUXAR_ZARR_FORMAT.md, "Amplitude units and display
 * gain"). The panel therefore shows two independent things: the window, in data
 * units, and the gain, as a multiplier.
 *
 * The gain spans the data's own scale (≈ 1/4000 for a 12-bit camera, 1e5 for a
 * float volume in [0, 1e-5]), so the track is LOGARITHMIC and decade-anchored:
 * three decades either side of the decade holding the live value. Anchoring on
 * the decade rather than the value keeps the track still while the user drags
 * within it.
 *
 * @module ui/layers/brightness-range
 */

import type { LayerInfo } from './layer-state';

/** Decades of gain the track spans either side of the live value's decade. */
export const BRIGHTNESS_TRACK_DECADES = 3;

/**
 * The smallest gain the control writes. Position 0 of a `log` track is an exact
 * 0, which would hide the layer through a control that is not the eye; it
 * clamps here instead (visually black, still a valid gain).
 */
export const BRIGHTNESS_FLOOR = 1e-12;

/** Log-track bounds for the Brightness slider at a given live gain. */
export function brightnessSliderRange(current: number): { min: number; max: number } {
  const gain = Number.isFinite(current) && current > 0 ? current : 1;
  const decade = Math.floor(Math.log10(gain));
  return {
    min: Math.pow(10, decade - BRIGHTNESS_TRACK_DECADES),
    max: Math.pow(10, decade + 1 + BRIGHTNESS_TRACK_DECADES),
  };
}

/** Readout text for a gain: a multiplier with three significant digits. */
export function formatBrightness(v: number): string {
  if (!(v > 0)) return '×0';
  if (v >= 1000 || v < 0.01) return `×${v.toExponential(2)}`;
  return `×${Number(v.toPrecision(3))}`;
}

/** Tooltip on the Brightness control. */
export const BRIGHTNESS_TOOLTIP =
  'Display gain: rendered brightness per data unit. Scales how bright the splats ' +
  'render (and, in volumetric mode, how dense they are) without changing the ' +
  'data values, the display range or the colormap.';

/**
 * Format a data-unit value for every readout of a unit-bearing layer — the
 * window's header values, its track bounds and the tooltip — so one value reads
 * the same everywhere: four significant digits below 1000 ("5.252", "208.9"),
 * whole numbers with thousands separators from 1000 ("1,597"), scientific
 * notation at the extremes. Scenes without a unit keep
 * {@link formatLegacyRangeValue}.
 */
export function formatDataValue(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const magnitude = Math.abs(v);
  if (magnitude === 0) return '0';
  if (magnitude >= 1e6 || magnitude < 1e-3) return v.toExponential(2);
  if (magnitude >= 1000) {
    return Math.round(v)
      .toString()
      .replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }
  return String(Number(v.toPrecision(4)));
}

const SCALAR_RANGE_LABEL = 'Display range';
const SPLAT_AMPLITUDE_LABEL = 'Splat amplitude';
const SCALAR_RANGE_TOOLTIP = 'Scalar data values in this range are mapped across the colormap.';
const PEAK_NOTE =
  "Each value is one splat's peak. Overlapping splats sum to the data value, " +
  'and the view also sums along each ray.';

/**
 * Format a value in the HISTORICAL display-range style (3 decimals, trailing
 * zeros stripped) — what every scene without an `amplitude_unit` keeps showing,
 * so an old scene's readouts are unchanged character for character.
 */
export function formatLegacyRangeValue(v: number): string {
  return v.toFixed(3).replace(/\.?0+$/, '');
}

/**
 * "Data range: 0 – 1,597 (floor 405)" — the SOURCE's span above the subtracted
 * floor (`source_data_range` / `amplitude_offset`), or `undefined` when the
 * scene records none. A tooltip line: the panel body stays as compact as any
 * other range control.
 */
export function dataRangeLine(
  layer: Pick<LayerInfo, 'sourceDataRange' | 'amplitudeOffset'>
): string | undefined {
  const range = layer.sourceDataRange;
  if (!range) return undefined;
  const floor =
    layer.amplitudeOffset !== undefined ? ` (floor ${formatDataValue(layer.amplitudeOffset)})` : '';
  return `Data range: ${formatDataValue(range[0])} \u2013 ${formatDataValue(range[1])}${floor}`;
}

/**
 * The window label and tooltip for a colormapped layer.
 *
 * A layer whose amplitudes carry a unit is windowing SPLAT AMPLITUDES — each one
 * Gaussian's PEAK in that unit, which sits below the data's own range wherever
 * Gaussians overlap. The label says so; everything else (the unit, the data
 * range with its floor, the one-line peak/sum explanation) lives in the label's
 * tooltip, a few short lines, so the control takes exactly the space a normal
 * range control takes. A layer without a unit keeps the historical label/text.
 */
export function scalarRangeLabel(
  layer: Pick<LayerInfo, 'amplitudeUnit' | 'amplitudeOffset' | 'sourceDataRange'>
): { label: string; tooltip: string } {
  const unit = layer.amplitudeUnit;
  if (!unit) return { label: SCALAR_RANGE_LABEL, tooltip: SCALAR_RANGE_TOOLTIP };
  const lines = [`Unit: ${unit}`];
  const data = dataRangeLine(layer);
  if (data) lines.push(data);
  lines.push(PEAK_NOTE);
  return { label: SPLAT_AMPLITUDE_LABEL, tooltip: lines.join('\n') };
}

/**
 * The colour-range tooltip, with the data range appended when the scene records
 * one (an RGB / multichannel gsplats layer still has a source range).
 */
export function colourRangeTooltip(
  base: string,
  layer: Pick<LayerInfo, 'amplitudeUnit' | 'amplitudeOffset' | 'sourceDataRange'>
): string {
  const data = dataRangeLine(layer);
  if (!data) return base;
  return [base, layer.amplitudeUnit ? `${data} ${layer.amplitudeUnit}` : data].join('\n');
}

/**
 * The Brightness slider shows the gain RELATIVE to the authored one (×1 = the
 * opening exposure); the stored `display_gain` stays absolute.
 */
export function relativeBrightness(
  layer: Pick<LayerInfo, 'brightness' | 'authoredBrightness'>
): number {
  const authored = layer.authoredBrightness > 0 ? layer.authoredBrightness : 1;
  return layer.brightness / authored;
}

/** Tooltip for the Brightness control, carrying the absolute gain. */
export function brightnessTooltip(
  layer: Pick<LayerInfo, 'brightness' | 'authoredBrightness'>
): string {
  return (
    `${BRIGHTNESS_TOOLTIP} ×1 is the brightness the scene opened with; ` +
    `double-click to return to it. Absolute display gain: ${Number(layer.brightness.toPrecision(4))} ` +
    `(authored ${Number(layer.authoredBrightness.toPrecision(4))}).`
  );
}

/**
 * Re-express a window chosen on `primary` for another selected `layer`, as the
 * same FRACTIONS of each layer's own reference range — the scalar data range
 * of a colormapped layer, `[0, 1]` for a direct-colour one. A multi-selection
 * of layers in different units would otherwise receive the primary's absolute
 * numbers (a 0–4,000-count window on a 0–1 layer). Falls back to the absolute
 * window when either reference is unusable.
 */
export function remapWindowForLayer(
  primary: Pick<LayerInfo, 'scalarWindow' | 'scalarDataRange'>,
  layer: Pick<LayerInfo, 'scalarWindow' | 'scalarDataRange'>,
  low: number,
  high: number
): [number, number] {
  const reference = (l: typeof primary): [number, number] | undefined =>
    l.scalarWindow ? l.scalarDataRange : [0, 1];
  const from = reference(primary);
  const to = reference(layer);
  if (!from || !to) return [low, high];
  const fromSpan = from[1] - from[0];
  const toSpan = to[1] - to[0];
  if (!(fromSpan > 0) || !(toSpan > 0) || !Number.isFinite(fromSpan) || !Number.isFinite(toSpan)) {
    return [low, high];
  }
  const t0 = (low - from[0]) / fromSpan;
  const t1 = (high - from[0]) / fromSpan;
  return [to[0] + t0 * toSpan, to[0] + t1 * toSpan];
}
