import type { AppConfig } from '../../types';

/**
 * Validate depth-shard configuration.
 *
 * Errors flag values that break the split outright (a shard count below 2 is
 * not a split; a budget below 2 cannot admit even one); warnings flag values
 * that defeat the design intent — either forfeiting the ordering the feature
 * exists to provide, or spending draws the measured cost curve says buy nothing.
 */
export function validateDepthShards(config: AppConfig, errors: string[], warnings: string[]): void {
  const ds = config.depthShards;

  if (!Number.isInteger(ds.shardsPerNode) || ds.shardsPerNode < 2) {
    errors.push(
      `depthShards.shardsPerNode must be an integer >= 2 (got ${ds.shardsPerNode}); ` +
        'a count of 1 is not a split — disable the feature instead'
    );
  } else if (ds.shardsPerNode > 256) {
    warnings.push(
      `depthShards.shardsPerNode (${ds.shardsPerNode}) is very high; the measured cost ` +
        'saturates (128 interleaved draws cost +5.5 ms at 2M splats and 512 cost +6.4 ms), ' +
        'so beyond ~32 the extra draws buy ordering accuracy nothing has asked for'
    );
  }

  if (!Number.isInteger(ds.maxInterleavedDraws) || ds.maxInterleavedDraws < 2) {
    errors.push(
      `depthShards.maxInterleavedDraws must be an integer >= 2 (got ${ds.maxInterleavedDraws}); ` +
        'below 2 no node could be split at all'
    );
  } else if (ds.maxInterleavedDraws > 2048) {
    warnings.push(
      `depthShards.maxInterleavedDraws (${ds.maxInterleavedDraws}) is far past where the ` +
        'per-draw cost was measured (512); the linear term is small but unbounded here'
    );
  } else if (ds.shardsPerNode > ds.maxInterleavedDraws) {
    warnings.push(
      `depthShards.shardsPerNode (${ds.shardsPerNode}) exceeds maxInterleavedDraws ` +
        `(${ds.maxInterleavedDraws}), so even a single qualifying node is scaled down; ` +
        'the per-node count is effectively the budget'
    );
  }

  if (!Number.isInteger(ds.minElements) || ds.minElements < 0) {
    errors.push(`depthShards.minElements must be a non-negative integer (got ${ds.minElements})`);
  } else if (ds.minElements === 0) {
    warnings.push(
      'depthShards.minElements is 0, so even a handful of elements can be split across ' +
        'several draws; a node that small is already a thin depth interval, so the draws ' +
        'cost frame time for no ordering gain'
    );
  }
}
