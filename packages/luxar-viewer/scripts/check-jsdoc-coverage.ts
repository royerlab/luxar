#!/usr/bin/env node
/**
 * JSDoc Coverage Checker
 *
 * Analyzes TypeScript files to ensure adequate JSDoc documentation coverage.
 * Reports coverage percentage and identifies files needing improvement.
 *
 * Usage:
 *   npx tsx scripts/check-jsdoc-coverage.ts [--threshold=70] [--verbose]
 */

import * as fs from 'fs';
import * as path from 'path';
import { glob } from 'glob';

interface CoverageResult {
  file: string;
  exports: number;
  documented: number;
  coverage: number;
}

interface Summary {
  totalExports: number;
  totalDocumented: number;
  overallCoverage: number;
  fileResults: CoverageResult[];
  passedFiles: number;
  failedFiles: number;
}

class JSDocChecker {
  private threshold: number;
  private verbose: boolean;

  constructor(threshold: number = 70, verbose: boolean = false) {
    this.threshold = threshold;
    this.verbose = verbose;
  }

  /**
   * Check JSDoc coverage for all TypeScript files in src/
   */
  async checkCoverage(): Promise<Summary> {
    const srcDir = path.join(process.cwd(), 'src');
    const files = await glob('**/*.ts', {
      cwd: srcDir,
      ignore: ['**/*.test.ts', '**/*.spec.ts', '**/__tests__/**'],
    });

    const results: CoverageResult[] = [];
    let totalExports = 0;
    let totalDocumented = 0;

    for (const file of files) {
      const filePath = path.join(srcDir, file);
      const result = this.checkFile(filePath);
      results.push(result);
      totalExports += result.exports;
      totalDocumented += result.documented;
    }

    const overallCoverage = totalExports > 0 ? (totalDocumented / totalExports) * 100 : 100;
    const passedFiles = results.filter((r) => r.coverage >= this.threshold).length;
    const failedFiles = results.filter((r) => r.coverage < this.threshold).length;

    return {
      totalExports,
      totalDocumented,
      overallCoverage,
      fileResults: results,
      passedFiles,
      failedFiles,
    };
  }

  /**
   * Check JSDoc coverage for a single file
   */
  private checkFile(filePath: string): CoverageResult {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');

    let exports = 0;
    let documented = 0;

    // Pattern to match exports
    const exportPattern = /^export\s+(function|class|interface|type|const|enum)/;
    // Pattern to match JSDoc comments
    const jsdocPattern = /\/\*\*/;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (exportPattern.test(line)) {
        exports++;

        // Check previous 15 lines for JSDoc comment
        let hasJSDoc = false;
        for (let j = Math.max(0, i - 15); j < i; j++) {
          if (jsdocPattern.test(lines[j])) {
            hasJSDoc = true;
            break;
          }
        }

        // A JSDoc block longer than the 15-line window is still adjacent
        // documentation: when a block comment closes on the line directly
        // above the export, walk up to its opener and accept it if it is
        // a `/**` (stop at a plain `/*` — that's not JSDoc).
        if (!hasJSDoc && i > 0 && lines[i - 1].trim().endsWith('*/')) {
          for (let j = i - 1; j >= 0; j--) {
            if (jsdocPattern.test(lines[j])) {
              hasJSDoc = true;
              break;
            }
            if (lines[j].includes('/*')) {
              break;
            }
          }
        }

        if (hasJSDoc) {
          documented++;
        }
      }
    }

    const coverage = exports > 0 ? (documented / exports) * 100 : 100;

    return {
      file: path.relative(path.join(process.cwd(), 'src'), filePath),
      exports,
      documented,
      coverage,
    };
  }

  /**
   * Print coverage summary and detailed results
   */
  printResults(summary: Summary): boolean {
    console.log('\n' + '='.repeat(70));
    console.log('📊 JSDOC COVERAGE REPORT');
    console.log('='.repeat(70) + '\n');

    console.log(`📈 Overall Coverage: ${summary.overallCoverage.toFixed(1)}%`);
    console.log(`   Total Exports: ${summary.totalExports}`);
    console.log(`   Documented: ${summary.totalDocumented}`);
    console.log(`   Missing JSDoc: ${summary.totalExports - summary.totalDocumented}\n`);

    console.log(`✅ Files passing (≥${this.threshold}%): ${summary.passedFiles}`);
    console.log(`❌ Files failing (<${this.threshold}%): ${summary.failedFiles}\n`);

    // Show files below threshold
    const failedFiles = summary.fileResults
      .filter((r) => r.coverage < this.threshold && r.exports > 0)
      .sort((a, b) => a.coverage - b.coverage);

    if (failedFiles.length > 0) {
      console.log('Files needing improvement:\n');
      failedFiles.forEach((result) => {
        console.log(`  ❌ ${result.file}`);
        console.log(
          `     Coverage: ${result.coverage.toFixed(0)}% (${result.documented}/${result.exports})`
        );
        console.log(
          `     Need ${Math.ceil((this.threshold / 100) * result.exports) - result.documented} more JSDoc comments\n`
        );
      });
    }

    // Show top performers if verbose
    if (this.verbose) {
      const topFiles = summary.fileResults
        .filter((r) => r.coverage === 100 && r.exports > 0)
        .sort((a, b) => b.exports - a.exports)
        .slice(0, 10);

      if (topFiles.length > 0) {
        console.log('\n📚 Top documented files (100% coverage):\n');
        topFiles.forEach((result) => {
          console.log(`  ✅ ${result.file} (${result.exports} exports)`);
        });
      }
    }

    // Package-level breakdown
    const byPackage = this.groupByPackage(summary.fileResults);
    console.log('\n📦 Coverage by package:\n');

    Object.entries(byPackage)
      .sort(([, a], [, b]) => b.coverage - a.coverage)
      .forEach(([pkg, stats]) => {
        const icon = stats.coverage >= this.threshold ? '✅' : '❌';
        console.log(
          `  ${icon} ${pkg}: ${stats.coverage.toFixed(0)}% (${stats.documented}/${stats.exports})`
        );
      });

    return summary.overallCoverage >= this.threshold;
  }

  /**
   * Group results by package (top-level directory)
   */
  private groupByPackage(
    results: CoverageResult[]
  ): Record<string, { exports: number; documented: number; coverage: number }> {
    const packages: Record<string, { exports: number; documented: number }> = {};

    results.forEach((result) => {
      const pkg = result.file.split('/')[0] || 'root';

      if (!packages[pkg]) {
        packages[pkg] = { exports: 0, documented: 0 };
      }

      packages[pkg].exports += result.exports;
      packages[pkg].documented += result.documented;
    });

    // Calculate coverage for each package
    const packagesWithCoverage: Record<
      string,
      { exports: number; documented: number; coverage: number }
    > = {};
    Object.entries(packages).forEach(([pkg, stats]) => {
      packagesWithCoverage[pkg] = {
        ...stats,
        coverage: stats.exports > 0 ? (stats.documented / stats.exports) * 100 : 100,
      };
    });

    return packagesWithCoverage;
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const thresholdArg = args.find((arg) => arg.startsWith('--threshold='));
  const threshold = thresholdArg ? parseInt(thresholdArg.split('=')[1], 10) : 70;
  const verbose = args.includes('--verbose');

  const checker = new JSDocChecker(threshold, verbose);
  const summary = await checker.checkCoverage();
  const passed = checker.printResults(summary);

  if (!passed) {
    console.log(
      `\n⚠️  JSDoc coverage (${summary.overallCoverage.toFixed(1)}%) is below threshold (${threshold}%)`
    );
    console.log('   Add JSDoc comments to exported functions, classes, and interfaces.\n');
    process.exit(1);
  } else {
    console.log(`\n✅ JSDoc coverage meets threshold (${threshold}%)\n`);
    process.exit(0);
  }
}

main().catch((err) => {
  console.error('Error checking JSDoc coverage:', err);
  process.exit(1);
});
