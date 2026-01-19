"""Adapters wrapping JAXopt solvers, if available."""

from ._jaxopt_adapter import JaxoptAdapter

JAXOPT_AVAILABLE = False

try:
    from jaxopt import BFGS, LBFGS, GradientDescent, NonlinearCG, ProximalGradient
except ModuleNotFoundError:  # pragma: no cover - exercised when jaxopt is optional
    pass
else:
    JAXOPT_AVAILABLE = True

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
