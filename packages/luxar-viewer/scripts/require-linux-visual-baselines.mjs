if (process.platform !== 'linux') {
  console.error('Visual baselines are maintained on Linux only.');
  process.exit(1);
}
