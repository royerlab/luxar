/** Parse a command's declared options before it starts any work. */
export function parseOptions(specs, argv, configError) {
  const known = new Map(specs.map((spec) => [`--${spec.name}`, spec]));
  const options = Object.fromEntries(specs.map(({ name, defaultValue }) => [name, defaultValue]));
  const seen = new Set();
  for (let i = 0; i < argv.length; i++) {
    const name = argv[i];
    const spec = known.get(name);
    if (!spec) configError(`unknown option ${name}`);
    if (seen.has(name)) configError(`duplicate option ${name}`);
    seen.add(name);
    if (spec.takesValue) {
      const value = argv[++i];
      if (value === undefined || value === '' || value.startsWith('--')) {
        configError(`${name} requires a value`);
      }
      options[spec.name] = value;
    } else {
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
  return `Usage: node ${command} [options]\nOptions:\n${lines.join('\n')}\n`;
}
