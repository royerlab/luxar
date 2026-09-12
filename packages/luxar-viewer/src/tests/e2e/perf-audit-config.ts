/** Pure configuration and snapshot helpers for the viewer audit benchmark. */

import type {
  DebugState,
  DrawOrderEntry,
  LODGroupDebugInfo,
} from '../../core/app/debug/debug-state';

export interface LodBiasArm {
  value: number | null;
  scenarioSuffix: string;
  query: string;
}

export function parseLodBiasArms(raw: string | undefined): LodBiasArm[] {
  const entries = (raw ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) return [{ value: null, scenarioSuffix: '', query: '' }];

  const values = [...new Set(entries.map(Number))];
  for (const value of values) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(
        `LUXAR_PERF_AUDIT_LOD_BIASES must contain positive finite numbers; received '${raw}'`
      );
    }
  }
  return values.map((value) => ({
    value,
    scenarioSuffix: value === 1 ? '' : `-lod-bias-${value}`,
    query: value === 1 ? '' : `&lod-bias=${value}`,
  }));
}

export function biasArmsForScene(hasLodLadder: boolean, arms: LodBiasArm[]): LodBiasArm[] {
  return hasLodLadder ? arms : [{ value: null, scenarioSuffix: '', query: '' }];
}

export interface SelectionSnapshot {
  visibleElements: number | null;
  activeLevels: string | null;
}

function formatLodGroup(group: LODGroupDebugInfo): string {
  const footprint = group.footprintStamped ? 'stamps-present' : 'no-stamps';
  return `${group.name}:${group.activeLevel}/${group.levelCount - 1}[${group.selector},${footprint}]`;
}

export function hasSubstitutiveSelection(values: Array<string | null>): boolean {
  return values.some((value) => value !== null && value.length > 0);
}

export function summarizeSelection(
  state: Pick<DebugState, 'lodGroups'> | null | undefined,
  drawOrder: DrawOrderEntry[] | null | undefined
): SelectionSnapshot {
  if (!state || !drawOrder) return { visibleElements: null, activeLevels: null };
  const lodPaths = state.lodGroups.map((group) => group.name);
  const measuredEntries =
    lodPaths.length === 0
      ? drawOrder
      : drawOrder.filter((entry) =>
          lodPaths.some((lodPath) => entry.path === lodPath || entry.path.startsWith(`${lodPath}/`))
        );
  return {
    visibleElements: measuredEntries.reduce((sum, entry) => sum + entry.elements, 0),
    activeLevels: state.lodGroups.map(formatLodGroup).sort().join(','),
  };
}
