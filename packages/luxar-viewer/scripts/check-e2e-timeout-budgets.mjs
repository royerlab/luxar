import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import ts from 'typescript';

const LONG_DEADLINE_THRESHOLD_MS = 30_000;

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SPEC_ROOT = join(PACKAGE_ROOT, 'src/tests/e2e');
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

function isBudgetStatement(node) {
  return (
    ts.isExpressionStatement(node) &&
    ts.isCallExpression(node.expression) &&
    BUDGET_CALL_PATHS.has(callPath(node.expression.expression).join('.'))
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

// Returns the budget these statements declare plus the running `slow` flag, so
// a caller can hand both down to the scope inside.
function budgetScopeForStatements(
  statements,
  constants,
  enclosing,
  projectTimeoutMs,
  describeScope
) {
  let budgetMs;
  let { slowApplied } = enclosing;
  for (const statement of statements) {
    if (!ts.isExpressionStatement(statement) || !ts.isCallExpression(statement.expression)) {
      continue;
    }
    const call = statement.expression;
    const path = callPath(call.expression).join('.');
    if (path === 'test.slow') {
      // TimeoutManager.slow() triples the slot timeout AS RESOLVED AT THAT
      // POINT and is guarded by a per-test flag, so a preceding setTimeout or
      // describe.configure wins and a suite-level slow makes an in-test one a
      // no-op — the total is 3x, never 9x.
      if (slowApplied) continue;
      slowApplied = true;
      budgetMs = (budgetMs ?? enclosing.budgetMs ?? projectTimeoutMs) * 3;
      continue;
    }
    if (path === 'test.setTimeout') {
      budgetMs = normalizedBudget(evaluateNumber(call.arguments[0], constants));
      continue;
    }
    if (!describeScope || path !== 'test.describe.configure') continue;
    const options = call.arguments[0];
    if (!options || !ts.isObjectLiteralExpression(options)) continue;
    const timeout = options.properties.find(
      (property) => ts.isPropertyAssignment(property) && propertyName(property.name) === 'timeout'
    );
    if (timeout && ts.isPropertyAssignment(timeout)) {
      budgetMs = normalizedBudget(evaluateNumber(timeout.initializer, constants));
    }
  }
  return { budgetMs, slowApplied };
}

function normalizedBudget(value) {
  if (value === 0) return Number.POSITIVE_INFINITY;
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : undefined;
}

function directBudgetScope(callback, constants, enclosing, projectTimeoutMs, describeScope) {
  return ts.isBlock(callback.body)
    ? budgetScopeForStatements(
        callback.body.statements,
        constants,
        enclosing,
        projectTimeoutMs,
        describeScope
      )
    : { budgetMs: undefined, slowApplied: enclosing.slowApplied };
}

function deadlineCandidates(callback, constants, helpers) {
  const deadlines = [];
  const visit = (node) => {
    // A budget declaration is not a wait: test.setTimeout(MESH_TIMEOUT_MS)
    // would otherwise resolve its own constant into the deadline it has to beat.
    if (isBudgetStatement(node)) return;
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
      if (callback && path[0] === 'test' && path[1] === 'describe') {
        const statements = ts.isBlock(callback.body) ? callback.body.statements : [];
        const scope = directBudgetScope(callback, constants, enclosing, projectTimeoutMs, true);
        visit(
          callback.body,
          { budgetMs: scope.budgetMs ?? enclosing.budgetMs, slowApplied: scope.slowApplied },
          Math.max(inheritedDeadlineMs, hookDeadline(statements, constants, helpers))
        );
        return;
      }
      if (callback && (path.join('.') === 'test' || path.join('.') === 'test.only')) {
        const deadlines = deadlineCandidates(callback, constants, helpers);
        const deadlineMs = Math.max(inheritedDeadlineMs, ...deadlines);
        const scope = directBudgetScope(callback, constants, enclosing, projectTimeoutMs, false);
        const budgetMs = scope.budgetMs ?? enclosing.budgetMs;
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
    budgetScopeForStatements(
      sourceFile.statements,
      constants,
      { budgetMs: undefined, slowApplied: false },
      projectTimeoutMs,
      true
    ),
    hookDeadline(sourceFile.statements, constants, helpers)
  );
  return violations;
}

export function analyzeSpec(
  sourceText,
  file,
  thresholdMs = LONG_DEADLINE_THRESHOLD_MS,
  projectTimeoutMs = thresholdMs * 2
) {
  const sourceFile = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true);
  const constants = constantDeclarations(sourceFile);
  return findTests(
    sourceFile,
    constants,
    localHelpers(sourceFile, constants),
    file,
    projectTimeoutMs
  ).filter((violation) => violation.deadlineMs > thresholdMs);
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

function formatViolation(violation) {
  return `${violation.file}:${violation.line} ${violation.test} (${violation.deadlineMs} ms)`;
}

export function runCheck() {
  const projectTimeoutMs = projectTestTimeout(readFileSync(PLAYWRIGHT_CONFIG_PATH, 'utf8'));
  const thresholdMs = projectTimeoutMs / 2;
  const paths = specFiles();
  const violations = paths.flatMap((path) =>
    analyzeSpec(readFileSync(path, 'utf8'), normalizedRelative(path), thresholdMs, projectTimeoutMs)
  );
  const result = compareViolationsToExceptions(violations, loadExceptions());
  for (const violation of result.newViolations)
    console.error(`Missing timeout budget: ${formatViolation(violation)}`);
  for (const exception of result.staleExceptions) {
    console.error(
      `Stale timeout-budget exception: ${exception.file}:${exception.line} :: ${exception.test}`
    );
  }
  if (result.newViolations.length) {
    console.error(
      'Declare test.setTimeout(...), test.slow(), or describe.configure({ timeout }), or add a reasoned entry to scripts/e2e-timeout-budget-exceptions.json.'
    );
  }
  if (result.newViolations.length || result.staleExceptions.length) return false;
  console.log(`E2E timeout budgets checked: ${paths.length} specs.`);
  return true;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href && !runCheck()) {
  process.exitCode = 1;
}
