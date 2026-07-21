import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos.basis import FourierGP
from nemos.basis._fourier_basis import _get_nodes_weights


def se_kernel(x1, x2, length_scale, variance):
    diff = x1 - x2
    return variance * jnp.exp(-0.5 * diff**2 / length_scale**2)


@pytest.mark.requires_x64
@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("variance", [1e-1, 1e0, 1e1])
@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2, 1e-1])
@pytest.mark.parametrize("bounds", [(0.0, 1.0)])
def test_covariance_approximation_accuracy(length_scale, variance, eps, bounds):
    basis = FourierGP(
        lengthscale=length_scale, bounds=bounds, eps=eps, variance=variance
    )
    x = jnp.linspace(bounds[0], bounds[1], 20)
    Phi = basis.evaluate(x)
    K_approx = Phi @ Phi.T

    x1, x2 = jnp.meshgrid(x, x)
    K_true = se_kernel(x1, x2, length_scale, variance)
    decimal = int(round(-np.log10(eps)))
    np.testing.assert_array_almost_equal(K_approx, K_true, decimal=decimal)


@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("variance", [1e0])
@pytest.mark.parametrize("eps", [1e-4])
@pytest.mark.parametrize("L", [1.0])
def test_real_weights(length_scale, variance, eps, L):
    _, weights, _, _ = _get_nodes_weights(length_scale, variance, eps, L)
    assert np.all(np.isreal(weights))

@pytest.mark.parametrize("length_scale", [1e-2])
@pytest.mark.parametrize("variance", [1e-1])
@pytest.mark.parametrize("eps", [1e-4])
@pytest.mark.parametrize("bounds", [(0.0, 1.0)])
def test_scalar_property_setting(length_scale, variance, eps, bounds):
    basis = FourierGP(lengthscale=length_scale, bounds=bounds, eps=eps, variance=variance)
    assert basis.lengthscale == length_scale
    assert basis.variance == variance
    assert basis.eps == eps


@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("variance", [1e0])
@pytest.mark.parametrize("eps", [1e-4])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-2.0, 3.0)])
def test_xis_property_setting(length_scale, variance, eps, bounds):
    L = bounds[1] - bounds[0]
    expected_xis, _, _, _ = _get_nodes_weights(length_scale, variance, eps, L)

    basis = FourierGP(lengthscale=length_scale, bounds=bounds, eps=eps, variance=variance)
    np.testing.assert_array_almost_equal(basis.xis, expected_xis)


@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-2.0, 3.0)])
def test_equispaced_grid(length_scale, eps, bounds):
    basis = FourierGP(lengthscale=length_scale, bounds=bounds, eps=eps)
    gaps = np.diff(basis.xis)
    np.testing.assert_allclose(gaps, basis.frequency_spacing, rtol=1e-4)


@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-2.0, 3.0)])
def test_n_frequencies_matches_shapes(length_scale, eps, bounds):
    basis = FourierGP(lengthscale=length_scale, bounds=bounds, eps=eps)
    assert basis.n_frequencies == len(basis.xis) - 1
    assert basis.n_frequencies == len(basis.frequencies[0]) - 1