import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import ts from 'typescript';

const LONG_DEADLINE_THRESHOLD_MS = 30_000;

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SPEC_ROOT = join(PACKAGE_ROOT, 'src/tests/e2e');
const BASELINE_PATH = join(PACKAGE_ROOT, 'scripts/e2e-timeout-budget-baseline.json');
const EXCEPTIONS_PATH = join(PACKAGE_ROOT, 'scripts/e2e-timeout-budget-exceptions.json');
const PLAYWRIGHT_CONFIG_PATH = join(PACKAGE_ROOT, 'playwright.config.ts');
const DEADLINE_NAME = /(timeout|deadline)/i;
// Statements that DECLARE a budget rather than wait for anything.
const BUDGET_CALL_PATHS = new Set(['test.setTimeout', 'test.slow', 'test.describe.configure']);

function propertyName(node) {
  if (ts.isIdentifier(node) || ts.isStringLiteral(node)) return node.text;
  return undefined;
}

function constantDeclarations(sourceFile) {
  const declarations = new Map();
  for (const statement of sourceFile.statements) {
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (ts.isIdentifier(declaration.name) && declaration.initializer) {
        declarations.set(declaration.name.text, declaration.initializer);
      }
    }
  }
  return declarations;
}

function evaluateNumber(node, constants, resolving = new Set()) {
  if (!node) return undefined;
  if (ts.isNumericLiteral(node)) return Number(node.text);
  if (ts.isParenthesizedExpression(node))
    return evaluateNumber(node.expression, constants, resolving);
  if (ts.isPrefixUnaryExpression(node)) {
    const value = evaluateNumber(node.operand, constants, resolving);
    if (value === undefined) return undefined;
    if (node.operator === ts.SyntaxKind.MinusToken) return -value;
    if (node.operator === ts.SyntaxKind.PlusToken) return value;
    return undefined;
  }
  if (ts.isBinaryExpression(node)) return evaluateBinary(node, constants, resolving);
  if (!ts.isIdentifier(node) || resolving.has(node.text)) return undefined;
  const initializer = constants.get(node.text);
  if (!initializer) return undefined;
  const nextResolving = new Set(resolving).add(node.text);
  return evaluateNumber(initializer, constants, nextResolving);
}

function evaluateBinary(node, constants, resolving) {
  const left = evaluateNumber(node.left, constants, resolving);
  const right = evaluateNumber(node.right, constants, resolving);
  if (left === undefined || right === undefined) return undefined;
  if (node.operatorToken.kind === ts.SyntaxKind.PlusToken) return left + right;
  if (node.operatorToken.kind === ts.SyntaxKind.MinusToken) return left - right;
  if (node.operatorToken.kind === ts.SyntaxKind.AsteriskToken) return left * right;
  if (node.operatorToken.kind === ts.SyntaxKind.SlashToken) return left / right;
  return undefined;
}

function localHelpers(sourceFile, constants) {
  const helpers = new Map();
  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name) {
      helpers.set(statement.name.text, deadlineParameters(statement.parameters, constants));
    }
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (
        ts.isIdentifier(declaration.name) &&
        declaration.initializer &&
        (ts.isArrowFunction(declaration.initializer) ||
          ts.isFunctionExpression(declaration.initializer))
      ) {
        helpers.set(
          declaration.name.text,
          deadlineParameters(declaration.initializer.parameters, constants)
        );
      }
    }
  }
  return helpers;
}

function parsedSource(source, file) {
  return typeof source === 'string'
    ? ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true)
    : source;
}

function exportedHelpers(source, file, cache) {
  const sourceFile = parsedSource(source, file);
  const cacheKey = sourceFile.fileName;
  const cached = cache.get(cacheKey);
  if (cached) return cached;
  const constants = constantDeclarations(sourceFile);
  const helpers = localHelpers(sourceFile, constants);
  const exported = new Map();
  for (const statement of sourceFile.statements) {
    if (ts.isExportDeclaration(statement)) {
      const elements =
        statement.exportClause && ts.isNamedExports(statement.exportClause)
          ? statement.exportClause.elements
          : [];
      const isTypeOnly =
        statement.isTypeOnly ||
        (elements.length > 0 && elements.every((element) => element.isTypeOnly));
      if (isTypeOnly) continue;
      if (statement.moduleSpecifier) {
        throw new Error(`Shared helper module ${file} uses an unsupported re-export.`);
      }
      for (const element of elements) {
        if (element.isTypeOnly) continue;
        const localName = element.propertyName?.text ?? element.name.text;
        const parameters = helpers.get(localName);
        if (parameters) exported.set(element.name.text, parameters);
      }
      continue;
    }
    const modifiers = ts.canHaveModifiers(statement) ? ts.getModifiers(statement) : undefined;
    if (!modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword)) continue;
    if (ts.isFunctionDeclaration(statement) && statement.name) {
      exported.set(statement.name.text, helpers.get(statement.name.text) ?? []);
    }
    if (!ts.isVariableStatement(statement)) continue;
    for (const declaration of statement.declarationList.declarations) {
      if (ts.isIdentifier(declaration.name) && helpers.has(declaration.name.text)) {
        exported.set(declaration.name.text, helpers.get(declaration.name.text));
      }
    }
  }
  cache.set(cacheKey, exported);
  return exported;
}

function importedHelpers(sourceFile, helperSources, exportedHelperCache) {
  const helpers = new Map();
  for (const statement of sourceFile.statements) {
    if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) {
      continue;
    }
    const specifier = statement.moduleSpecifier.text;
    if (!/^\.\/helpers(?:\/|$)/.test(specifier)) continue;
    const importClause = statement.importClause;
    if (!importClause) continue;
    if (importClause.isTypeOnly) continue;
    if (importClause.name) {
      throw new Error(`Shared helper module ${specifier} uses an unsupported default import.`);
    }
    const bindings = importClause.namedBindings;
    if (bindings && ts.isNamespaceImport(bindings)) {
      throw new Error(`Shared helper module ${specifier} uses an unsupported namespace import.`);
    }
    if (!bindings) continue;
    if (bindings.elements.every((element) => element.isTypeOnly)) continue;
    const helperSource = helperSources.get(specifier);
    if (helperSource === undefined) {
      throw new Error(`Could not resolve shared helper module ${specifier}.`);
    }
    const exported = exportedHelpers(helperSource, specifier, exportedHelperCache);
    for (const element of bindings.elements) {
      if (element.isTypeOnly) continue;
      const importedName = element.propertyName?.text ?? element.name.text;
      const parameters = exported.get(importedName);
      if (parameters) helpers.set(element.name.text, parameters);
    }
  }
  return helpers;
}

function deadlineParameters(parameters, constants) {
  return parameters.flatMap((parameter, index) => {
    if (!ts.isIdentifier(parameter.name) || !DEADLINE_NAME.test(parameter.name.text)) return [];
    return [{ defaultMs: evaluateNumber(parameter.initializer, constants), index }];
  });
}

function callPath(node) {
  if (ts.isIdentifier(node)) return [node.text];
  if (!ts.isPropertyAccessExpression(node)) return [];
  return [...callPath(node.expression), node.name.text];
}

function isBudgetStatement(node, infoName = '') {
  if (!ts.isExpressionStatement(node) || !ts.isCallExpression(node.expression)) return false;
  const path = callPath(node.expression.expression).join('.');
  return (
    BUDGET_CALL_PATHS.has(path) ||
    (infoName !== '' && (path === `${infoName}.setTimeout` || path === `${infoName}.slow`))
  );
}

function callbackArgument(call) {
  return [...call.arguments]
    .reverse()
    .find((argument) => ts.isArrowFunction(argument) || ts.isFunctionExpression(argument));
}

function testTitle(call) {
  const title = call.arguments[0];
  if (ts.isStringLiteral(title) || ts.isNoSubstitutionTemplateLiteral(title)) return title.text;
  return '<dynamic title>';
}

function budgetCalls(statements) {
  return statements.flatMap((statement) =>
    ts.isExpressionStatement(statement) && ts.isCallExpression(statement.expression)
      ? [statement.expression]
      : []
  );
}

// A conditional modifier declares nothing we can model: Playwright bails on a
// falsy condition, and the `() => …` form is resolved from fixtures at run time.
function isUnconditionalSlow(call) {
  return call.arguments.length === 0;
}

function configuredTimeout(call, constants) {
  const options = call.arguments[0];
  if (!options || !ts.isObjectLiteralExpression(options)) return undefined;
  const timeout = options.properties.find(
    (property) => ts.isPropertyAssignment(property) && propertyName(property.name) === 'timeout'
  );
  return timeout && ts.isPropertyAssignment(timeout)
    ? normalizedBudget(evaluateNumber(timeout.initializer, constants))
    : undefined;
}

// Suite scope — a file's top level or a describe body. Nothing executes here:
// `test.slow()` becomes a static annotation while setTimeout/configure set the
// suite's own `_timeout`, and the worker resolves the timeout FIRST and applies
// the annotations after. So textual order is irrelevant, the declared timeout
// always wins, and the x3 is deferred to the test boundary. `_timeout` resolves
// to the nearest ancestor that declared one.
function suiteScope(statements, constants, enclosing) {
  let declaredMs;
  let { slowApplied } = enclosing;
  for (const call of budgetCalls(statements)) {
    const path = callPath(call.expression).join('.');
    if (path === 'test.slow') {
      slowApplied ||= isUnconditionalSlow(call);
    } else if (path === 'test.setTimeout') {
      declaredMs = normalizedBudget(evaluateNumber(call.arguments[0], constants));
    } else if (path === 'test.describe.configure') {
      const timeout = configuredTimeout(call, constants);
      if (timeout !== undefined) declaredMs = timeout;
    }
  }
  return { declaredMs: declaredMs ?? enclosing.declaredMs, slowApplied };
}

// An undeclared budget stays undefined — the whole point of the gate is that a
// long deadline riding on the bare project timeout is the thing to flag.
function materializedBudget(scope, projectTimeoutMs) {
  if (scope.declaredMs === undefined && !scope.slowApplied) return undefined;
  return (scope.declaredMs ?? projectTimeoutMs) * (scope.slowApplied ? 3 : 1);
}

// A test body DOES execute, so here order matters: TimeoutManager.setTimeout
// assigns the slot outright and slow() triples whatever it finds, once per test
// (a suite-level slow already spent that flag).
function testBudget(statements, constants, enclosing, projectTimeoutMs) {
  let budgetMs = materializedBudget(enclosing, projectTimeoutMs);
  let { slowApplied } = enclosing;
  for (const call of budgetCalls(statements)) {
    const path = callPath(call.expression).join('.');
    if (path === 'test.setTimeout') {
      const directBudgetMs = normalizedBudget(evaluateNumber(call.arguments[0], constants));
      if (directBudgetMs !== undefined) budgetMs = directBudgetMs;
    } else if (path === 'test.slow' && !slowApplied && isUnconditionalSlow(call)) {
      slowApplied = true;
      budgetMs = (budgetMs ?? projectTimeoutMs) * 3;
    }
  }
  return budgetMs;
}

// beforeAll/afterAll run in their own slot, initially set to the project
// timeout. Suite and per-test declarations do not change that slot; hook-local
// slow() triples its current value at most once.
function allHookBudget(callback, constants, projectTimeoutMs, infoName) {
  let budgetMs;
  let slowApplied = false;
  for (const call of budgetCalls(blockStatements(callback))) {
    const path = callPath(call.expression).join('.');
    if (path === 'test.setTimeout' || path === `${infoName}.setTimeout`) {
      const directBudgetMs = normalizedBudget(evaluateNumber(call.arguments[0], constants));
      if (directBudgetMs !== undefined) budgetMs = directBudgetMs;
    } else if (
      (path === 'test.slow' || path === `${infoName}.slow`) &&
      !slowApplied &&
      isUnconditionalSlow(call)
    ) {
      slowApplied = true;
      budgetMs = (budgetMs ?? projectTimeoutMs) * 3;
    }
  }
  return budgetMs;
}

function normalizedBudget(value) {
  if (value === 0) return Number.POSITIVE_INFINITY;
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : undefined;
}

function blockStatements(callback) {
  return callback && ts.isBlock(callback.body) ? callback.body.statements : [];
}

function deadlineCandidates(callback, constants, helpers, infoName = '') {
  const deadlines = [];
  const visit = (node) => {
    // A budget declaration is not a wait: test.setTimeout(MESH_TIMEOUT_MS)
    // would otherwise resolve its own constant into the deadline it has to beat.
    if (isBudgetStatement(node, infoName)) return;
    if (ts.isPropertyAssignment(node) && propertyName(node.name) === 'timeout') {
      deadlines.push(evaluateNumber(node.initializer, constants));
    }
    if (ts.isIdentifier(node) && DEADLINE_NAME.test(node.text)) {
      deadlines.push(evaluateNumber(node, constants));
    }
    if (ts.isCallExpression(node)) collectCallDeadlines(node, constants, helpers, deadlines);
    ts.forEachChild(node, visit);
  };
  visit(callback.body);
  return deadlines.filter((value) => value !== undefined && Number.isFinite(value));
}

function hookDeadline(statements, constants, helpers) {
  const deadlines = statements.flatMap((statement) => {
    if (!ts.isExpressionStatement(statement) || !ts.isCallExpression(statement.expression)) {
      return [];
    }
    const path = callPath(statement.expression.expression).join('.');
    if (path !== 'test.beforeEach' && path !== 'test.afterEach') return [];
    const callback = callbackArgument(statement.expression);
    return callback ? deadlineCandidates(callback, constants, helpers) : [];
  });
  return Math.max(0, ...deadlines);
}

function collectCallDeadlines(call, constants, helpers, deadlines) {
  const path = callPath(call.expression);
  if (path.at(-1) === 'waitForTimeout')
    deadlines.push(evaluateNumber(call.arguments[0], constants));
  if (path.at(-1)?.startsWith('wait')) {
    for (const argument of call.arguments) deadlines.push(evaluateNumber(argument, constants));
  }
  if (path.length !== 1) return;
  for (const parameter of helpers.get(path[0]) ?? []) {
    const argument = call.arguments[parameter.index];
    deadlines.push(argument ? evaluateNumber(argument, constants) : parameter.defaultMs);
  }
}

function findTests(sourceFile, constants, helpers, file, projectTimeoutMs) {
  const violations = [];
  const visit = (node, enclosing, inheritedDeadlineMs = 0) => {
    if (ts.isCallExpression(node)) {
      const path = callPath(node.expression);
      const callback = callbackArgument(node);
      const statements = blockStatements(callback);
      if (callback && path[0] === 'test' && path[1] === 'describe') {
        visit(
          callback.body,
          suiteScope(statements, constants, enclosing),
          Math.max(inheritedDeadlineMs, hookDeadline(statements, constants, helpers))
        );
        return;
      }
      if (
        callback &&
        path.length === 2 &&
        path[0] === 'test' &&
        ['beforeAll', 'afterAll'].includes(path[1])
      ) {
        const testInfo = callback.parameters[1]?.name;
        const infoName = testInfo && ts.isIdentifier(testInfo) ? testInfo.text : '';
        const candidates = deadlineCandidates(callback, constants, helpers, infoName);
        const deadlineMs = Math.max(0, ...candidates);
        const budgetMs = allHookBudget(callback, constants, projectTimeoutMs, infoName);
        if (deadlineMs > 0 && (budgetMs === undefined || budgetMs <= deadlineMs)) {
          const title = node.arguments[0];
          const suffix =
            ts.isStringLiteral(title) || ts.isNoSubstitutionTemplateLiteral(title)
              ? `: ${title.text}`
              : '';
          violations.push({
            deadlineMs,
            file,
            line: sourceFile.getLineAndCharacterOfPosition(node.getStart()).line + 1,
            test: `${path[1]}${suffix}`,
          });
        }
        return;
      }
      if (callback && (path.join('.') === 'test' || path.join('.') === 'test.only')) {
        const deadlines = deadlineCandidates(callback, constants, helpers);
        const deadlineMs = Math.max(inheritedDeadlineMs, ...deadlines);
        const budgetMs = testBudget(statements, constants, enclosing, projectTimeoutMs);
        if (deadlineMs > 0 && (budgetMs === undefined || budgetMs <= deadlineMs)) {
          violations.push({
            deadlineMs,
            file,
            line: sourceFile.getLineAndCharacterOfPosition(node.getStart()).line + 1,
            test: testTitle(node),
          });
        }
        return;
      }
    }
    ts.forEachChild(node, (child) => visit(child, enclosing, inheritedDeadlineMs));
  };
  visit(
    sourceFile,
    suiteScope(sourceFile.statements, constants, { declaredMs: undefined, slowApplied: false }),
    hookDeadline(sourceFile.statements, constants, helpers)
  );
  return violations;
}

export function analyzeSpec(
  source,
  file,
  thresholdMs = LONG_DEADLINE_THRESHOLD_MS,
  projectTimeoutMs = thresholdMs * 2,
  helperSources = new Map(),
  exportedHelperCache = new Map()
) {
  const sourceFile = parsedSource(source, file);
  const constants = constantDeclarations(sourceFile);
  const helpers = new Map([
    ...localHelpers(sourceFile, constants),
    ...importedHelpers(sourceFile, helperSources, exportedHelperCache),
  ]);
  return findTests(sourceFile, constants, helpers, file, projectTimeoutMs).filter(
    (violation) => violation.deadlineMs > thresholdMs
  );
}

export function projectTestTimeout(sourceText) {
  const sourceFile = ts.createSourceFile(
    'playwright.config.ts',
    sourceText,
    ts.ScriptTarget.Latest,
    true
  );
  const assignment = sourceFile.statements.find(ts.isExportAssignment);
  const configCall = assignment?.expression;
  if (!configCall || !ts.isCallExpression(configCall)) {
    throw new Error('Could not find exported Playwright config.');
  }
  const config = configCall.arguments[0];
  if (!config || !ts.isObjectLiteralExpression(config)) {
    throw new Error('Playwright config must be an object literal.');
  }
  const timeout = config.properties.find(
    (property) => ts.isPropertyAssignment(property) && propertyName(property.name) === 'timeout'
  );
  const value =
    timeout && ts.isPropertyAssignment(timeout)
      ? evaluateNumber(timeout.initializer, constantDeclarations(sourceFile))
      : undefined;
  if (!value) throw new Error('Could not read the Playwright test timeout.');
  return value;
}

function exceptionKey(entry) {
  return `${entry.file}\0${entry.line}\0${entry.test}`;
}

export function compareViolationsToExceptions(violations, exceptions) {
  const exceptionMap = new Map();
  for (const exception of exceptions) {
    if (!exception.reason?.trim())
      throw new Error('Every timeout-budget exception needs a non-empty reason.');
    const key = exceptionKey(exception);
    if (exceptionMap.has(key))
      throw new Error(
        `Duplicate timeout-budget exception: ${exception.file}:${exception.line} :: ${exception.test}`
      );
    exceptionMap.set(key, exception);
  }
  const violationKeys = new Set(violations.map(exceptionKey));
  return {
    newViolations: violations.filter((violation) => !exceptionMap.has(exceptionKey(violation))),
    staleExceptions: exceptions.filter((exception) => !violationKeys.has(exceptionKey(exception))),
  };
}

export function compareViolationCountsToBaseline(violations, baseline) {
  const actual = new Map();
  for (const violation of violations)
    actual.set(violation.file, (actual.get(violation.file) ?? 0) + 1);
  const files = new Set([...Object.keys(baseline), ...actual.keys()]);
  const regressions = [];
  const improvements = [];
  for (const file of [...files].sort()) {
    const allowed = baseline[file] ?? 0;
    if (!Number.isInteger(allowed) || allowed < 0) {
      throw new Error(`Timeout-budget baseline count for ${file} must be a non-negative integer.`);
    }
    const found = actual.get(file) ?? 0;
    if (found > allowed) regressions.push({ allowed, file, found });
    if (found < allowed) improvements.push({ allowed, file, found });
  }
  return { improvements, regressions };
}

export function isDefaultConfigSpec(path) {
  const normalized = path.split(sep).join('/');
  return !normalized.endsWith('perf-bench.spec.ts') && !normalized.includes('/mobile/');
}

function normalizedRelative(path, root = PACKAGE_ROOT) {
  return relative(root, path).split(sep).join('/');
}

export function specFiles(directory = SPEC_ROOT, root = PACKAGE_ROOT) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return specFiles(path, root);
    // Mobile and perf specs share this testDir but have their own, larger project budgets.
    // Match on the package-relative path: an absolute one drags the checkout's own
    // ancestors (a /home/mobile/... clone) through the exclusion patterns.
    return entry.name.endsWith('.spec.ts') && isDefaultConfigSpec(normalizedRelative(path, root))
      ? [path]
      : [];
  });
}

function loadExceptions(path = EXCEPTIONS_PATH) {
  const payload = JSON.parse(readFileSync(path, 'utf8'));
  if (!Array.isArray(payload.exceptions))
    throw new Error(`${normalizedRelative(path)} must contain an exceptions array.`);
  return payload.exceptions;
}

function loadBaseline(path = BASELINE_PATH) {
  const payload = JSON.parse(readFileSync(path, 'utf8'));
  if (!payload.files || Array.isArray(payload.files) || typeof payload.files !== 'object') {
    throw new Error(`${normalizedRelative(path)} must contain a files object.`);
  }
  return payload.files;
}

export function helperSourcesForSpec(source, path, helperModuleCache = new Map()) {
  const sourceFile = parsedSource(source, path);
  const sources = new Map();
  for (const statement of sourceFile.statements) {
    if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) {
      continue;
    }
    const specifier = statement.moduleSpecifier.text;
    if (!/^\.\/helpers(?:\/|$)/.test(specifier)) continue;
    const importClause = statement.importClause;
    if (
      importClause?.isTypeOnly ||
      (importClause?.namedBindings &&
        ts.isNamedImports(importClause.namedBindings) &&
        importClause.namedBindings.elements.every((element) => element.isTypeOnly))
    ) {
      continue;
    }
    const base = resolve(dirname(path), specifier);
    const modulePath = [`${base}.ts`, join(base, 'index.ts'), base].find(
      (candidate) => existsSync(candidate) && statSync(candidate).isFile()
    );
    if (!modulePath) throw new Error(`Could not resolve shared helper module ${specifier}.`);
    let helperSource = helperModuleCache.get(modulePath);
    if (!helperSource) {
      helperSource = ts.createSourceFile(
        modulePath,
        readFileSync(modulePath, 'utf8'),
        ts.ScriptTarget.Latest,
        true
      );
      helperModuleCache.set(modulePath, helperSource);
    }
    sources.set(specifier, helperSource);
  }
  return sources;
}

function violationCounts(violations) {
  const counts = new Map();
  for (const violation of violations) {
    counts.set(violation.file, (counts.get(violation.file) ?? 0) + 1);
  }
  return Object.fromEntries([...counts].sort(([left], [right]) => left.localeCompare(right)));
}

export function saveBaseline(violations, path = BASELINE_PATH) {
  const payload = JSON.parse(readFileSync(path, 'utf8'));
  payload.files = violationCounts(violations);
  writeFileSync(path, `${JSON.stringify(payload, null, 2)}\n`);
}

export function parseArgs(argv) {
  const options = { updateBaseline: false };
  for (const argument of argv) {
    if (argument === '--') continue;
    if (argument === '--update-baseline') options.updateBaseline = true;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  return options;
}

function formatViolation(violation) {
  return `${violation.file}:${violation.line} ${violation.test} (${violation.deadlineMs} ms)`;
}

export function runCheck({ updateBaseline = false } = {}) {
  const projectTimeoutMs = projectTestTimeout(readFileSync(PLAYWRIGHT_CONFIG_PATH, 'utf8'));
  const thresholdMs = projectTimeoutMs / 2;
  const paths = specFiles();
  const helperModuleCache = new Map();
  const exportedHelperCache = new Map();
  const violations = paths.flatMap((path) => {
    const sourceText = readFileSync(path, 'utf8');
    const sourceFile = ts.createSourceFile(path, sourceText, ts.ScriptTarget.Latest, true);
    return analyzeSpec(
      sourceFile,
      normalizedRelative(path),
      thresholdMs,
      projectTimeoutMs,
      helperSourcesForSpec(sourceFile, path, helperModuleCache),
      exportedHelperCache
    );
  });
  const exceptions = compareViolationsToExceptions(violations, loadExceptions());
  if (updateBaseline && !exceptions.staleExceptions.length) {
    saveBaseline(exceptions.newViolations);
    console.log(
      `Wrote ${Object.keys(violationCounts(exceptions.newViolations)).length} files to ${normalizedRelative(BASELINE_PATH)}.`
    );
    return true;
  }
  const baseline = compareViolationCountsToBaseline(exceptions.newViolations, loadBaseline());
  for (const regression of baseline.regressions) {
    console.error(
      `Timeout-budget baseline regression: ${regression.file} has ${regression.found}, allowed ${regression.allowed}. Regenerate with pnpm update:e2e-timeout-budget-baseline.`
    );
    for (const violation of exceptions.newViolations.filter(
      ({ file }) => file === regression.file
    )) {
      console.error(`  Missing timeout budget: ${formatViolation(violation)}`);
    }
  }
  for (const improvement of baseline.improvements) {
    console.error(
      `Stale timeout-budget baseline: ${improvement.file} has ${improvement.found}, recorded ${improvement.allowed}. Regenerate with pnpm update:e2e-timeout-budget-baseline.`
    );
  }
  for (const exception of exceptions.staleExceptions) {
    console.error(
      `Stale timeout-budget exception: ${exception.file}:${exception.line} :: ${exception.test}`
    );
  }
  if (baseline.regressions.length) {
    console.error(
      'Declare test.setTimeout(...), test.slow(), or describe.configure({ timeout }) for tests; beforeAll/afterAll need test.setTimeout(...), testInfo.setTimeout(...), or slow() inside the hook. Use an exact reasoned exception only when the heuristic is wrong; regenerate intentional count changes with pnpm update:e2e-timeout-budget-baseline.'
    );
  }
  if (
    baseline.regressions.length ||
    baseline.improvements.length ||
    exceptions.staleExceptions.length
  ) {
    return false;
  }
  console.log(`E2E timeout budgets checked: ${paths.length} specs.`);
  return true;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (!runCheck(options)) process.exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
