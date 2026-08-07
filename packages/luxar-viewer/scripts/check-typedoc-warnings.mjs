#!/usr/bin/env node
/**
 * Baseline-driven TypeDoc warning ratchet.
 *
 * TypeDoc's validation is useful, but the viewer currently has known warning
 * debt. This checker records normalized warning messages, tolerates the
 * checked-in baseline, and fails only when a warning is newly introduced.
 */

import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

import {
  Application,
  Logger,
  LogLevel,
  PackageJsonReader,
  TSConfigReader,
  TypeDocReader,
} from 'typedoc';

const DEFAULT_BASELINE = 'typedoc-warnings-baseline.json';
const ANSI_ESCAPE = /\u001b\[[0-?]*[ -/]*[@-~]/g;
const BASELINE_COMMENT =
  'TypeDoc warning baseline. Regenerate with: pnpm run typedoc:check-warnings -- --update-baseline. ' +
  'Warnings are normalized and compared as a multiset; newly introduced warnings fail the check.';

/** Return the severity tag from TypeDoc's formatted console diagnostic. */
export function diagnosticLevel(message) {
  const match = String(message)
    .replace(ANSI_ESCAPE, '')
    .match(/^\s*\[(warning|error)\]\s*/i);
  return match?.[1].toLowerCase();
}

/** Normalize machine-specific paths and incidental whitespace in a diagnostic. */
export function normalizeDiagnostic(message, viewerRoot = process.cwd()) {
  const normalizedRoot = path.resolve(viewerRoot).replaceAll('\\', '/').replace(/\/$/, '');
  return String(message)
    .replace(ANSI_ESCAPE, '')
    .replace(/^\[(?:warning|error)\]\s*/i, '')
    .replaceAll('\\', '/')
    .replaceAll(normalizedRoot, '<viewer>')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Add an explicit error when TypeDoc conversion produced no project. */
export function ensureConversionError(converted, errors) {
  if (!converted && errors.length === 0) {
    return ['TypeDoc conversion produced no project'];
  }
  return errors;
}

/** Return multiset additions/removals while preserving duplicate warning counts. */
export function diffWarningMultisets(currentWarnings, baselineWarnings) {
  const subtract = (left, right) => {
    const remaining = new Map();
    for (const item of right) {
      remaining.set(item, (remaining.get(item) ?? 0) + 1);
    }

    const difference = [];
    for (const item of left) {
      const count = remaining.get(item) ?? 0;
      if (count > 0) {
        remaining.set(item, count - 1);
      } else {
        difference.push(item);
      }
    }
    return difference.sort();
  };

  return {
    newWarnings: subtract(currentWarnings, baselineWarnings),
    fixedWarnings: subtract(baselineWarnings, currentWarnings),
  };
}

/** Read and validate a TypeDoc warning baseline. */
export async function loadWarningBaseline(baselinePath) {
  let payload;
  try {
    payload = JSON.parse(await readFile(baselinePath, 'utf8'));
  } catch (error) {
    throw new Error(
      `Could not read TypeDoc warning baseline ${baselinePath}: ${error.message}. ` +
        'Create or repair it with: pnpm run typedoc:check-warnings -- --update-baseline'
    );
  }

  if (
    !payload ||
    typeof payload !== 'object' ||
    !Array.isArray(payload.warnings) ||
    payload.warnings.some((warning) => typeof warning !== 'string')
  ) {
    throw new Error(
      `Malformed TypeDoc warning baseline ${baselinePath}: expected a string array at "warnings"`
    );
  }

  return [...payload.warnings].sort();
}

/** Write the normalized, sorted warning baseline. */
export async function saveWarningBaseline(baselinePath, warnings) {
  const payload = {
    _comment: BASELINE_COMMENT,
    warnings: [...warnings].sort(),
  };
  await writeFile(baselinePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

export class RecordingLogger extends Logger {
  constructor(viewerRoot) {
    super();
    this.viewerRoot = viewerRoot;
    this.warnings = [];
    this.errors = [];
  }

  log(message, level) {
    super.log(message, level);
    const normalized = normalizeDiagnostic(message, this.viewerRoot);
    if (level === LogLevel.Warn) {
      this.warnings.push(normalized);
    } else if (level === LogLevel.Error) {
      this.errors.push(normalized);
    }
  }
}

/** Run TypeDoc conversion/validation without emitting documentation. */
async function collectTypeDocDiagnostics(viewerRoot) {
  // TypeDoc does not accept a logger until after bootstrap. Capture its scoped
  // console diagnostics so option/configuration warnings cannot bypass the
  // baseline merely because they occur before the recording logger is set.
  const bootstrapWarnings = [];
  const bootstrapErrors = [];
  const originalWarn = console.warn;
  const originalError = console.error;
  console.warn = (...parts) => {
    const message = parts.join(' ');
    if (diagnosticLevel(message) === 'warning') {
      bootstrapWarnings.push(normalizeDiagnostic(message, viewerRoot));
    } else {
      originalWarn(...parts);
    }
  };
  console.error = (...parts) => {
    const message = parts.join(' ');
    if (diagnosticLevel(message) === 'error') {
      bootstrapErrors.push(normalizeDiagnostic(message, viewerRoot));
    } else {
      originalError(...parts);
    }
  };

  let application;
  try {
    application = await Application.bootstrapWithPlugins({ emit: 'none' }, [
      new TypeDocReader(),
      new PackageJsonReader(),
      new TSConfigReader(),
    ]);
  } finally {
    console.warn = originalWarn;
    console.error = originalError;
  }

  const logger = new RecordingLogger(viewerRoot);
  application.logger = logger;

  const project = await application.convert();
  if (project) {
    await application.validate(project);
  }

  return {
    converted: Boolean(project),
    warnings: [...bootstrapWarnings, ...logger.warnings].sort(),
    errors: [...bootstrapErrors, ...logger.errors].sort(),
  };
}

export function parseArgs(argv) {
  const options = {
    baseline: DEFAULT_BASELINE,
    json: false,
    updateBaseline: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--') {
      continue;
    } else if (argument === '--json') {
      options.json = true;
    } else if (argument === '--update-baseline') {
      options.updateBaseline = true;
    } else if (argument === '--baseline') {
      index += 1;
      if (index >= argv.length || argv[index] === '--' || argv[index].startsWith('--')) {
        throw new Error('--baseline requires a path');
      }
      options.baseline = argv[index];
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }

  return options;
}

function printHumanReport(report, baselinePath) {
  console.log(
    `TypeDoc warning ratchet: ${report.summary.current} current, ` +
      `${report.summary.baselined} baselined, ${report.summary.new} new, ` +
      `${report.summary.fixed} fixed`
  );

  if (report.errors.length > 0) {
    console.error('\nTypeDoc errors:');
    for (const error of report.errors) {
      console.error(`  - ${error}`);
    }
  }

  if (report.newWarnings.length > 0) {
    console.error('\nNew TypeDoc warnings:');
    for (const warning of report.newWarnings) {
      console.error(`  - ${warning}`);
    }
  }

  if (report.fixedWarnings.length > 0) {
    console.log('\nFixed baseline warnings:');
    for (const warning of report.fixedWarnings) {
      console.log(`  - ${warning}`);
    }
    console.log(
      `\nTighten the baseline with: pnpm run typedoc:check-warnings -- --update-baseline`
    );
  }

  if (report.newWarnings.length === 0 && report.errors.length === 0) {
    if (report.fixedWarnings.length === 0) {
      console.log(`TypeDoc warnings match ${baselinePath}; no regression detected.`);
    } else {
      console.log('No new TypeDoc warnings; no regression detected.');
    }
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const viewerRoot = process.cwd();
  const baselinePath = path.resolve(viewerRoot, options.baseline);
  const diagnostics = await collectTypeDocDiagnostics(viewerRoot);
  const errors = ensureConversionError(diagnostics.converted, diagnostics.errors);

  if (errors.length > 0) {
    const report = {
      summary: {
        current: diagnostics.warnings.length,
        baselined: 0,
        new: 0,
        fixed: 0,
      },
      errors,
      newWarnings: [],
      fixedWarnings: [],
    };
    if (options.json) {
      console.log(JSON.stringify(report, null, 2));
    } else {
      printHumanReport(report, baselinePath);
    }
    return 2;
  }

  if (options.updateBaseline) {
    await saveWarningBaseline(baselinePath, diagnostics.warnings);
    if (options.json) {
      console.log(
        JSON.stringify(
          { baseline: baselinePath, warnings: diagnostics.warnings, updated: true },
          null,
          2
        )
      );
    } else {
      console.log(`Wrote ${diagnostics.warnings.length} TypeDoc warnings to ${baselinePath}`);
    }
    return 0;
  }

  const baselineWarnings = await loadWarningBaseline(baselinePath);
  const { newWarnings, fixedWarnings } = diffWarningMultisets(
    diagnostics.warnings,
    baselineWarnings
  );
  const report = {
    summary: {
      current: diagnostics.warnings.length,
      baselined: baselineWarnings.length,
      new: newWarnings.length,
      fixed: fixedWarnings.length,
    },
    errors: [],
    newWarnings,
    fixedWarnings,
  };

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    printHumanReport(report, baselinePath);
  }
  return newWarnings.length > 0 ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main()
    .then((exitCode) => {
      process.exitCode = exitCode;
    })
    .catch((error) => {
      console.error(`TypeDoc warning check failed: ${error.message}`);
      process.exitCode = 2;
    });
}
