"""Adapters wrapping JAXopt solvers, if available."""

from typing import TYPE_CHECKING

from ._jaxopt_adapter import JaxoptAdapter
from ..typing import Params, StepResult

if TYPE_CHECKING:
    from ..batching import DataLoader

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
        _supports_stochastic = True

        def _stochastic_run_impl(
            self,
            init_params: Params,
            data_loader: "DataLoader",
            num_epochs: int,
        ) -> StepResult:
            """Run gradient descent over mini-batches from a data loader."""
            sample_X, sample_y = data_loader.sample_batch()
            state = self.init_state(init_params, sample_X, sample_y)
            params = init_params
            aux = None

            for _ in range(num_epochs):
                for X_batch, y_batch in data_loader:
                    params, state, aux = self.update(params, state, X_batch, y_batch)

            return (params, state, aux)

    class JaxoptProximalGradient(JaxoptAdapter):
        """
        Adapter for jaxopt.ProximalGradient.

        The `prox` argument passed to `jaxopt.ProximalGradient`
        is read from the regularizer.
        """

        _solver_cls = ProximalGradient
        _proximal = True
        _supports_stochastic = True

        def _stochastic_run_impl(
            self,
            init_params: Params,
            data_loader: "DataLoader",
            num_epochs: int,
        ) -> StepResult:
            """Run proximal gradient descent over mini-batches from a data loader."""
            sample_X, sample_y = data_loader.sample_batch()
            state = self.init_state(init_params, sample_X, sample_y)
            params = init_params
            aux = None

            for _ in range(num_epochs):
                for X_batch, y_batch in data_loader:
                    params, state, aux = self.update(params, state, X_batch, y_batch)

            return (params, state, aux)

    class JaxoptBFGS(JaxoptAdapter):
        """Adapter for jaxopt.BFGS."""

        _solver_cls = BFGS

    class JaxoptLBFGS(JaxoptAdapter):
        """Adapter for jaxopt.LBFGS."""

        _solver_cls = LBFGS

    class JaxoptNonlinearCG(JaxoptAdapter):
        """Adapter for jaxopt.NonlinearCG."""

        _solver_cls = NonlinearCG
