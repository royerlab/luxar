#### Match gsplat amplitude bits to source dtype

- Standalone gsplat writers now accept `amplitude_bits="auto"`, using 8-bit
  geometric-log amplitude codes for 8-bit integer sources recorded by the
  fitter while retaining the existing AUTO policy for unknown, floating-point,
  and wider source dtypes. Direct saves remain unchanged by default, while CLI
  fit outputs opt into source matching.
