/**
 * Axis-aligned opening-view presets (`viewer_config.camera.view_axis`,
 * `?viewAxis=`): the plane the opening camera faces, horizontal axis first. The
 * letters are scene DIMENSION names. Pose resolution (which needs three.js and
 * the displayed dims) lives in `scene/scene-manager/camera/view-axis.ts`; this
 * module is the dependency-free vocabulary the config and URL layers parse with.
 *
 * @module types/view-axis
 */

import { VIEW_AXES as CONTRACT_VIEW_AXES, type ViewAxisName } from './format-contract';

/** One of the contract's `view_axes` presets. */
export type ViewAxis = ViewAxisName;

/** Narrow an untrusted value (URL / zarr attrs) to a {@link ViewAxis}; null if invalid. */
export function parseViewAxis(raw: unknown): ViewAxis | null {
  if (typeof raw !== 'string') return null;
  const value = raw.trim().toLowerCase();
  return CONTRACT_VIEW_AXES.find((axis) => axis === value) ?? null;
}
