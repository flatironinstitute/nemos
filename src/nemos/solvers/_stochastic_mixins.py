"""Mixins providing standard stochastic optimization implementations."""

from typing import TYPE_CHECKING

from ..typing import Params, StepResult

if TYPE_CHECKING:
    from ..batching import DataLoader


class StochasticSolverMixin:
    """Mixin providing standard stochastic optimization loop.

    This mixin provides a default implementation of `_stochastic_run_impl`
    that iterates over the data loader for the specified number of epochs,
    calling `update` on each batch.

    Classes using this mixin must implement `init_state` and `update` methods.
    """

    _supports_stochastic = True

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader: "DataLoader",
        num_epochs: int,
    ) -> StepResult:
        """Run optimization over mini-batches from a data loader.

        Parameters
        ----------
        init_params : Params
            Initial parameter values.
        data_loader : DataLoader
            Data loader providing batches and metadata.
        num_epochs : int
            Number of passes over the data.

        Returns
        -------
        StepResult
            Final (params, state, aux) tuple.
        """
        sample_X, sample_y = data_loader.sample_batch()
        state = self.init_state(init_params, sample_X, sample_y)
        params = init_params
        aux = None

        for _ in range(num_epochs):
            for X_batch, y_batch in data_loader:
                params, state, aux = self.update(params, state, X_batch, y_batch)

        return (params, state, aux)


class OptimistixStochasticSolverMixin(StochasticSolverMixin):
    """Mixin for Optimistix solvers that updates stats after optimization.

    Extends `StochasticSolverMixin` to also update `self.stats` with
    the number of steps taken after the optimization loop completes.
    """

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader: "DataLoader",
        num_epochs: int,
    ) -> StepResult:
        """Run optimization and update stats.

        Parameters
        ----------
        init_params : Params
            Initial parameter values.
        data_loader : DataLoader
            Data loader providing batches and metadata.
        num_epochs : int
            Number of passes over the data.

        Returns
        -------
        StepResult
            Final (params, state, aux) tuple.
        """
        result = super()._stochastic_run_impl(init_params, data_loader, num_epochs)
        _, state, _ = result
        num_steps = self._extract_num_steps(state)
        self.stats = {"num_steps": num_steps, "max_steps": self.maxiter}
        return result
