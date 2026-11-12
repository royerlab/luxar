/** Parse a command's declared options before it starts any work. */
export function parseOptions(specs, argv, configError) {
  const fail = (message) => {
    configError(message);
    throw new Error(message);
  };
  const known = new Map(specs.map((spec) => [`--${spec.name}`, spec]));
  const options = Object.fromEntries(specs.map(({ name, defaultValue }) => [name, defaultValue]));
  const seen = new Set();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const equals = arg.indexOf('=');
    const name = equals < 0 ? arg : arg.slice(0, equals);
    const spec = known.get(name);
    if (!spec) fail(`unknown option ${arg}`);
    if (seen.has(name)) fail(`duplicate option ${name}`);
    seen.add(name);
    if (spec.takesValue) {
      const value = equals < 0 ? argv[++i] : arg.slice(equals + 1);
      if (value === undefined || value === '' || (equals < 0 && value.startsWith('--'))) {
        fail(`${name} requires a value`);
      }
      options[spec.name] = value;
    } else {
      if (equals >= 0) fail(`${name} does not take a value`);
      options[spec.name] = true;
    }
  }
  return options;
}

export function optionHelp(command, specs) {
  const lines = specs.map(({ name, takesValue, defaultValue }) => {
    const flag = `--${name}${takesValue ? ' <value>' : ''}`;
    return `  ${flag}${defaultValue == null || defaultValue === false || defaultValue === '' ? '' : ` (default: ${defaultValue})`}`;
  });
  return `Usage: node ${command} [options]\nValue options also accept --name=value (required for values starting with --).\nOptions:\n${lines.join('\n')}\n`;
}
