import { readFileSync } from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { describe, expect, it } from 'vitest';
import { resolveGalleryOnly } from '../screenshots/gallery-selection';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '../../../../..');
const README_SOURCE = readFileSync(path.join(REPO_ROOT, 'README.md'), 'utf-8');
const MANIFEST = JSON.parse(
  readFileSync(path.join(REPO_ROOT, 'scripts/gallery/manifest.json'), 'utf-8')
) as { demos: Array<{ id: string }> };
const MANIFEST_IDS = MANIFEST.demos.map((demo) => demo.id);

const README_GALLERY_IDS = [
  'atp_synthase',
  'caida_as_topology',
  'collision',
  'dipc_3d_genome',
  'dmri_tractography',
  'earthquakes',
  'flywire_connectome',
  'galaxy_simulation',
  'global_rivers_earth',
  'gsplats_3d_blastocyst_multichannel',
  'gsplats_3d_cells3d_multichannel',
  'gsplats_3d_ct_totalsegmentator',
  'gsplats_3d_milky_way_dust',
  'gsplats_3d_tribolium_embryo',
  'gsplats_4d_celegans_tracking',
  'gsplats_4d_neuromast_2ch',
  'hilbert_curve_3d',
  'human_multiome_peak_umap',
  'huri_interactome',
  'lorenz',
  'mesh_isosurface_cells3d',
  'ocean',
  'protein_landscape',
  'quantum_orbitals',
  'rainbow_sphere',
  'spiral_galaxy',
  'spotify_tracks',
  'tabula_sapiens',
  'zebrahub_multiome',
];

describe('gallery selection', () => {
  it('resolves readme to the exact front-page media set', () => {
    const selection = resolveGalleryOnly('readme', README_SOURCE, MANIFEST_IDS);

    expect([...selection.wantedIds].sort()).toEqual(README_GALLERY_IDS);
    expect(selection.unknownTokens).toEqual([]);
  });

  it('combines explicit ids with the readme alias without duplicates', () => {
    const selection = resolveGalleryOnly(
      'readme, lorenz, desi_galaxies',
      README_SOURCE,
      MANIFEST_IDS
    );

    expect(selection.wantedIds.size).toBe(README_GALLERY_IDS.length + 1);
    expect(selection.wantedIds).toContain('lorenz');
    expect(selection.wantedIds).toContain('desi_galaxies');
  });

  it('reports ordinary tokens that do not match a manifest id', () => {
    const selection = resolveGalleryOnly('lorenz, readmes, missing', README_SOURCE, MANIFEST_IDS);

    expect([...selection.wantedIds]).toEqual(['lorenz']);
    expect(selection.unknownTokens).toEqual(['readmes', 'missing']);
  });

  it('rejects a README media stem that is absent from the manifest', () => {
    expect(() =>
      resolveGalleryOnly('readme', 'docs/images/readme/gallery/not_a_demo.webp', MANIFEST_IDS)
    ).toThrow(/manifest.*not_a_demo/);
  });

  it('rejects a readme alias that resolves to no media', () => {
    expect(() => resolveGalleryOnly('readme', '# no gallery media', MANIFEST_IDS)).toThrow(
      /no gallery media/
    );
  });
});
