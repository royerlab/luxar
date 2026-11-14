import { expect, it } from 'vitest';
import { optionHelp, parseOptions } from './cli-options.mjs';

const specs = [
  { name: 'chrome-args', takesValue: true, defaultValue: '', allowLeadingDashes: true },
  { name: 'suite', takesValue: true, defaultValue: 'exact' },
  { name: 'headed', takesValue: false, defaultValue: false },
];
const fail = (message) => {
  throw new Error(message);
};

it('accepts equals-form values, including values that begin with dashes or contain equals', () => {
  expect(parseOptions(specs, ['--chrome-args=--a,--b=x', '--suite=perf'], fail)).toEqual({
    'chrome-args': '--a,--b=x',
    suite: 'perf',
    headed: false,
  });
  expect(optionHelp('gate.mjs', specs)).toContain('--name=value');
});

it('accepts Chrome flags as the following value without swallowing a known option', () => {
  expect(
    parseOptions(specs, ['--chrome-args', '--disable-domain-blocking-for-3d-apis'], fail)
  ).toMatchObject({ 'chrome-args': '--disable-domain-blocking-for-3d-apis' });
  expect(() => parseOptions(specs, ['--chrome-args', '--headed'], fail)).toThrow(
    /--chrome-args requires a value/
  );
});

it.each([
  [['--headed=x'], /--headed does not take a value/],
  [['--suite='], /--suite requires a value/],
  [['--suite', '--headed'], /--suite requires a value/],
  [['--suite=perf', '--suite', 'exact'], /duplicate option --suite/],
])('rejects invalid invocation %j', (argv, error) => {
  expect(() => parseOptions(specs, argv, fail)).toThrow(error);
});

it('does not continue if an error callback returns', () => {
  expect(() => parseOptions(specs, ['--bogus'], () => {})).toThrow(/unknown option --bogus/);
  expect(() => parseOptions(specs, ['--suite'], () => {})).toThrow(/--suite requires a value/);
});
