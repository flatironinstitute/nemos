import jax
import jax.numpy as jnp

import nemos.solvers as solvers


def test_optimistix_config_f_struct_dtype_matches_precision():
    """Test that the dtype for f_struct in the optimistix params is set correctly."""
    # get the current value
    original = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        cfg = solvers._optimistix_solvers.OptimistixConfig()
        assert cfg.f_struct.dtype == jnp.float32

        jax.config.update("jax_enable_x64", True)
        cfg = solvers._optimistix_solvers.OptimistixConfig()
        assert cfg.f_struct.dtype == jnp.float64
    finally:
        # restore to the original value
        jax.config.update("jax_enable_x64", original)
