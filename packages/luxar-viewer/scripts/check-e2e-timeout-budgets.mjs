import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import ts from 'typescript';

export const LONG_DEADLINE_THRESHOLD_MS = 30_000;

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SPEC_ROOT = join(PACKAGE_ROOT, 'src/tests/e2e');
const EXCEPTIONS_PATH = join(PACKAGE_ROOT, 'scripts/e2e-timeout-budget-exceptions.json');
const DEADLINE_NAME = /(timeout|deadline)/i;

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

function hasBudgetStatements(statements, describeScope) {
  return statements.some((statement) => {
    if (!ts.isExpressionStatement(statement) || !ts.isCallExpression(statement.expression)) {
      return false;
    }
    const path = callPath(statement.expression.expression).join('.');
    if (path === 'test.slow' || path === 'test.setTimeout') return true;
    return describeScope && path === 'test.describe.configure';
  });
}

function hasDirectBudget(callback, describeScope) {
  return ts.isBlock(callback.body) && hasBudgetStatements(callback.body.statements, describeScope);
}

function deadlineCandidates(callback, constants, helpers) {
  const deadlines = [];
  const visit = (node) => {
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

function findTests(sourceFile, constants, helpers, file) {
  const violations = [];
  const visit = (node, inheritedBudget = false) => {
    if (ts.isCallExpression(node)) {
      const path = callPath(node.expression);
      const callback = callbackArgument(node);
      if (callback && path[0] === 'test' && path[1] === 'describe') {
        visit(callback.body, inheritedBudget || hasDirectBudget(callback, true));
        return;
      }
      if (callback && (path.join('.') === 'test' || path.join('.') === 'test.only')) {
        const deadlines = deadlineCandidates(callback, constants, helpers);
        const deadlineMs = Math.max(0, ...deadlines);
        if (
          deadlineMs > LONG_DEADLINE_THRESHOLD_MS &&
          !inheritedBudget &&
          !hasDirectBudget(callback, false)
        ) {
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
    ts.forEachChild(node, (child) => visit(child, inheritedBudget));
  };
  visit(sourceFile, hasBudgetStatements(sourceFile.statements, true));
  return violations;
}

export function analyzeSpec(sourceText, file) {
  const sourceFile = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true);
  const constants = constantDeclarations(sourceFile);
  return findTests(sourceFile, constants, localHelpers(sourceFile, constants), file);
}

function exceptionKey(entry) {
  return `${entry.file}\0${entry.test}`;
}

export function compareViolationsToExceptions(violations, exceptions) {
  const exceptionMap = new Map();
  for (const exception of exceptions) {
    if (!exception.reason?.trim())
      throw new Error('Every timeout-budget exception needs a non-empty reason.');
    const key = exceptionKey(exception);
    if (exceptionMap.has(key))
      throw new Error(`Duplicate timeout-budget exception: ${exception.file} :: ${exception.test}`);
    exceptionMap.set(key, exception);
  }
  const violationKeys = new Set(violations.map(exceptionKey));
  return {
    newViolations: violations.filter((violation) => !exceptionMap.has(exceptionKey(violation))),
    staleExceptions: exceptions.filter((exception) => !violationKeys.has(exceptionKey(exception))),
  };
}

function specFiles(directory = SPEC_ROOT) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return specFiles(path);
    return entry.name.endsWith('.spec.ts') ? [path] : [];
  });
}

function normalizedRelative(path) {
  return relative(PACKAGE_ROOT, path).split(sep).join('/');
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
  const violations = specFiles().flatMap((path) =>
    analyzeSpec(readFileSync(path, 'utf8'), normalizedRelative(path))
  );
  const result = compareViolationsToExceptions(violations, loadExceptions());
  for (const violation of result.newViolations)
    console.error(`Missing timeout budget: ${formatViolation(violation)}`);
  for (const exception of result.staleExceptions) {
    console.error(`Stale timeout-budget exception: ${exception.file} :: ${exception.test}`);
  }
  if (result.newViolations.length || result.staleExceptions.length) return false;
  console.log(
    `E2E timeout budgets checked: ${specFiles().length} specs, ${violations.length} exceptions.`
  );
  return true;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href && !runCheck()) {
  process.exitCode = 1;
}
