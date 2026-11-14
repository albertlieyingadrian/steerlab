"""circuitkit: tools on top of circuit-tracer.

- `circuitkit.diff`: compare attribution graphs before and after fine-tuning or RL;
- `circuitkit.validate`: test whether graph features are a switch or a readout, against random and matched-norm
  controls;
- `circuitkit.labels`: cheap automatic feature labels with detection and simulation scores;
- `circuitkit.tracer`: the circuit-tracer adapter (needs torch and circuit-tracer);
- `circuitkit.report`: HTML pages for all three.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("steerlab-circuitkit")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"
