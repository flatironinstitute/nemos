"""Regression tests for scripts/benchmarking/benchmarking_glm.py.

These exercise the benchmarking harness itself (not just nemos), since the
harness calls private solver internals directly and can drift out of sync
with what model.fit() actually does.
"""

import sys
from pathlib import Path

import jax
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarking")
)
from benchmarking_glm import (  # noqa: E402
    benchmark_fit,
    generate_data,
    model_from_config,
)


@pytest.mark.solver_related
def test_benchmark_newton_population_glm_synthetic():
    """Newton[nemos] must benchmark successfully on a PopulationGLM (pop_size > 1).

    _benchmark_nemos used to call the solver's private init/run methods on the
    full parameter pytree instead of partitioning active/frozen params the way
    model.fit() does. That's a no-op for a single-neuron GLM, but
    PopulationGLM's Newton Hessian vmaps over the frozen params, so skipping
    the partition raised a `vmap in_axes` pytree-structure ValueError for
    every PopulationGLM fit -- including, unnoticed, every real-data (NWB)
    fit, since those are always fit as a PopulationGLM.
    """
    jax.config.update("jax_enable_x64", True)
    config = {
        "package": "nemos",
        "input_shapes": {"X": [50, 2], "y": [50, 3]},
        "model_conf": {
            "solver_name": "Newton[nemos]",
            "solver_kwargs": {"maxiter": 100, "tol": 1e-6},
            "regularizer": "Ridge",
            "regularizer_strength": 0.001,
        },
        "device": "cpu",
        "file_name": "cpu_test_newton_population.json",
    }
    model = model_from_config(config)
    X, y = generate_data(model, config)

    result = benchmark_fit(config, X, y, n_reps=1)

    assert result["results"]["converged"][0]
