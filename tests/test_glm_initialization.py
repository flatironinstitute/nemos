import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos.glm.initialize_parameters import initialize_intercept_matching_mean_rate


@pytest.mark.parametrize(
    "non_linearity",
    [
        jnp.exp,
        jax.nn.softplus,
        lambda x: jnp.exp(x),
        jax.nn.sigmoid,
        jax.lax.logistic,
        lambda x: jax.lax.logistic(x),
        jax.scipy.special.expit,
        jax.scipy.stats.norm.cdf,
    ],
)
@pytest.mark.parametrize(
    "output_y",
    [np.random.uniform(0, 1, size=(10,)), np.random.uniform(0, 1, size=(10, 2))],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_invert_non_linearity(non_linearity, output_y):
    inv_y = initialize_intercept_matching_mean_rate(
        inverse_link_function=non_linearity, y=output_y
    )
    assert jnp.allclose(non_linearity(inv_y), jnp.mean(output_y, axis=0), rtol=10**-5)


@pytest.mark.parametrize(
    "non_linearity, expectation",
    [
        (jnp.exp, pytest.raises(ValueError, match=".+The mean firing rate has")),
        (
            jax.nn.softplus,
            pytest.raises(ValueError, match=".+The mean firing rate has"),
        ),
        (
            lambda x: jnp.exp(x),
            pytest.raises(
                ValueError, match=".+Please, provide initial parameters instead"
            ),
        ),
        (
            jax.nn.sigmoid,
            pytest.raises(
                ValueError, match=".+Please, provide initial parameters instead"
            ),
        ),
        (
            jax.lax.logistic,
            pytest.raises(ValueError, match=".+The mean firing rate has"),
        ),
        (
            lambda x: jax.lax.logistic(x),
            pytest.raises(
                ValueError, match=".+Please, provide initial parameters instead"
            ),
        ),
        (
            jax.scipy.stats.norm.cdf,
            pytest.raises(ValueError, match=".+The mean firing rate has"),
        ),
    ],
)
def test_initialization_error_nan_input(non_linearity, expectation):
    """Initialize invalid."""
    output_y = np.full((10, 2), np.nan)
    with expectation:
        initialize_intercept_matching_mean_rate(
            inverse_link_function=non_linearity, y=output_y
        )


def test_initialization_error_non_invertible():
    """Initialize invalid."""
    output_y = np.random.uniform(size=100)
    inv_link = lambda x: jax.nn.softplus(x) + 10
    with pytest.raises(
        ValueError, match="Failed to initialize the model intercept.+Please, provide"
    ):
        with warnings.catch_warnings():
            # ignore the warning raised by the root-finder (there is no root)
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Tolerance of"
            )
            initialize_intercept_matching_mean_rate(
                inverse_link_function=inv_link, y=output_y
            )
