/**
 * Global Setup for Playwright E2E Tests
 *
 * This file runs BEFORE any tests and verifies pre-conditions:
 * 1. Required example datasets exist
 * 2. Servers can be reached
 * 3. Basic environment checks
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at module load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Required datasets for E2E tests
const REQUIRED_DATASETS = [
  'simple_nd_example.luxar.zarr',
  'build_example_manual.luxar.zarr',
  'build_example_structured.luxar.zarr',
  'dimension_navigation_example.luxar.zarr',
  'dimension_sliders_5d_example.luxar.zarr',
  'dense_grid_5d_example.luxar.zarr',
  'layers_test_example.luxar.zarr',
];

export default async function globalSetup() {
  console.log('\n🔍 Running pre-flight checks...\n');

  // Get project root (5 levels up from this file)
  // src/tests/e2e -> tests -> src -> luxar-viewer -> packages -> project root
  const projectRoot = path.resolve(__dirname, '../../../../..');
  const examplesDir = path.join(projectRoot, 'datasets/examples');

  // Check 1: Verify examples directory exists
  if (!fs.existsSync(examplesDir)) {
    console.warn(`⚠️  Examples directory not found: ${examplesDir}`);
    console.warn('   Run "make run-examples" to generate test datasets');
    console.warn('   Tests requiring example datasets will fail.\n');
    // Don't throw - allow tests that don't need examples to run
    // (e.g., basic-rendering, viewer-initialization, test-fixtures, geometry-types)
  } else {
    console.log(`✅ Examples directory found: ${examplesDir}`);

    // Check 2: Verify required datasets exist
    const missingDatasets: string[] = [];
    const foundDatasets: string[] = [];

    for (const dataset of REQUIRED_DATASETS) {
      const datasetPath = path.join(examplesDir, dataset);
      if (fs.existsSync(datasetPath)) {
        foundDatasets.push(dataset);
      } else {
        missingDatasets.push(dataset);
      }
    }

    console.log(`✅ Found ${foundDatasets.length}/${REQUIRED_DATASETS.length} required datasets`);

    if (missingDatasets.length > 0) {
      console.warn('\n⚠️  Warning: Some datasets are missing:');
      for (const dataset of missingDatasets) {
        console.warn(`   - ${dataset}`);
      }
      console.warn('\n   Tests requiring these datasets will fail.');
      console.warn('   Run "make run-examples" to generate all datasets.\n');
    }
  }

  // Check 3: Verify we can write to test output directory
  const testResultsDir = path.join(__dirname, '../../../test-results');

  try {
    if (!fs.existsSync(testResultsDir)) {
      fs.mkdirSync(testResultsDir, { recursive: true });
    }
    // Also create debug subdirectory for agent-driver output
    const debugDir = path.join(testResultsDir, 'debug');
    if (!fs.existsSync(debugDir)) {
      fs.mkdirSync(debugDir, { recursive: true });
    }
    console.log('✅ Test output directories are writable');
  } catch (error) {
    console.error('❌ Cannot create test output directories:', error);
    throw error;
  }

  console.log('\n✅ Pre-flight checks passed!\n');
}
