"""Adapters wrapping JAXopt solvers, if available."""

from importlib.util import find_spec as _find_spec
from typing import TYPE_CHECKING

from ..typing import Params, StepResult
from ._jaxopt_adapter import JaxoptAdapter

if TYPE_CHECKING:
    from ..batching import DataLoader

JAXOPT_AVAILABLE = _find_spec("jaxopt") is not None

if JAXOPT_AVAILABLE:
    from jaxopt import BFGS, LBFGS, GradientDescent, NonlinearCG, ProximalGradient

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
