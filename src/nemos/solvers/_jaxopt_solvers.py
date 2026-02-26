"""Adapters wrapping JAXopt solvers, if available."""

from importlib.util import find_spec as _find_spec

from ._jaxopt_adapter import JaxoptAdapter

JAXOPT_AVAILABLE = _find_spec("jaxopt") is not None

if JAXOPT_AVAILABLE:
    from jaxopt import BFGS, LBFGS, GradientDescent, NonlinearCG, ProximalGradient

if JAXOPT_AVAILABLE:

    class JaxoptGradientDescent(JaxoptAdapter):
        """Adapter for jaxopt.GradientDescent."""

        _solver_cls = GradientDescent

    class JaxoptProximalGradient(JaxoptAdapter):
        """
        Adapter for jaxopt.ProximalGradient.

        The `prox` argument passed to `jaxopt.ProximalGradient`
        is read from the regularizer.
        """

        _solver_cls = ProximalGradient
        _proximal = True

    class JaxoptBFGS(JaxoptAdapter):
        """Adapter for jaxopt.BFGS."""

        _solver_cls = BFGS

    class JaxoptLBFGS(JaxoptAdapter):
        """Adapter for jaxopt.LBFGS."""

        _solver_cls = LBFGS

    class JaxoptNonlinearCG(JaxoptAdapter):
        """Adapter for jaxopt.NonlinearCG."""

        _solver_cls = NonlinearCG
