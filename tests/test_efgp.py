import jax.numpy as jnp
import numpy as np
import pytest

from nemos.basis import FourierSEBasis
from nemos.basis._fourier_basis import _se_quadrature


def se_kernel(x1, x2, length_scale, variance):
    diff = x1 - x2
    return variance * jnp.exp(-0.5 * diff**2 / length_scale**2)


@pytest.mark.requires_x64
@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("variance", [1e-1, 1e0, 1e1])
@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2, 1e-1])
@pytest.mark.parametrize("domain", [(0.0, 1.0)])
def test_covariance_approximation_accuracy(length_scale, variance, eps, domain):
    basis = FourierSEBasis(
        lengthscale=length_scale, domain=domain, eps=eps, variance=variance
    )
    x = jnp.linspace(domain[0], domain[1], 20)
    Phi = basis.evaluate(x)
    K_approx = Phi @ Phi.T

    x1, x2 = jnp.meshgrid(x, x)
    K_true = se_kernel(x1, x2, length_scale, variance)
    max_err = jnp.max(jnp.abs(K_approx - K_true))
    assert max_err < eps


@pytest.mark.parametrize("length_scale", [1e-2, 1e-1, 1e0])
@pytest.mark.parametrize("variance", [1e0])
@pytest.mark.parametrize("eps", [1e-4])
@pytest.mark.parametrize("L", [1.0])
def test_real_weights(length_scale, variance, eps, L):
    _, weights, _, _ = _se_quadrature(length_scale, variance, eps, L)
    assert np.all(np.isreal(weights))
