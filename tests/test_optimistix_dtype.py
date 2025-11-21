import jax
import jax.numpy as jnp
import pytest

import nemos.solvers as solvers


@pytest.mark.solver_related
def test_optimistix_f_struct_dtype_matches_precision(request):
    """Test that the dtype for f_struct in the optimistix params is set correctly."""
    # get the current value
    original = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)

        X, y, _, params, loss = request.getfixturevalue("linear_regression")
        f_struct, aux_struct = solvers._optimistix_solvers._make_f_and_aux_struct(
            lambda params, args: loss(params, *args),
            False,
            # jax.tree_util.tree_map(jnp.zeros_like, params),
            params,
            (X, y),
        )
        assert f_struct.dtype == jnp.float32

        jax.config.update("jax_enable_x64", True)

        X, y, _, params, loss = request.getfixturevalue("linear_regression")
        f_struct, aux_struct = solvers._optimistix_solvers._make_f_and_aux_struct(
            lambda params, args: loss(params, *args),
            False,
            # jax.tree_util.tree_map(jnp.zeros_like, params),
            params,
            (X, y),
        )
        assert f_struct.dtype == jnp.float64
    finally:
        # restore to the original value
        jax.config.update("jax_enable_x64", original)
