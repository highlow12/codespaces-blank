"""Project-level clustering defaults selected by reproducible benchmarks."""

DEFAULT_PIPELINE_FUZZIFIER = 1.2
# Try the benchmark winner first, then progressively fuzzier fallbacks.
DEFAULT_FAST_FUZZIFIER_VALUES = (1.2, 1.4, 1.6, 1.8, 2.0)
