"""Mixins providing standard stochastic optimization implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..callbacks import Callback, TrainingContext
from ..typing import Params, SolverState, StepResult

if TYPE_CHECKING:
    from ..batching import DataLoader

from optimistix._misc import cauchy_termination

from ..tree_utils import tree_l2_norm, tree_sub


def _params_only_cauchy_criterion(
    params: Params,
    prev_params: Params,
    atol: float,
    rtol: float = 0.0,
):  # bool
    return cauchy_termination(
        rtol,
        atol,
        tree_l2_norm,
        params,
        tree_sub(params, prev_params),
        0.0,
        0.0,
    )


class StochasticSolverMixin:
    """
    Mixin providing standard stochastic optimization loop.

    This mixin provides a default implementation of ``_stochastic_run_impl``
    that iterates over the data loader for the specified number of epochs,
    calling ``update`` on each batch.

    Supports optional callbacks for per-epoch and per-batch hooks.

    Classes using this mixin must implement ``init_state`` and ``update`` methods.
    """

    _supports_stochastic = True

    def _validate_stochastic_options(self):
        """Make sure acceleration and linesearches are turned off if available."""
        if "acceleration" in self.get_accepted_arguments():
            if self.acceleration:
                raise ValueError(
                    "Turn off acceleration option for stochastic optimization."
                )

        if "stepsize" in self.get_accepted_arguments():
            if self.stepsize is None or self.stepsize <= 0.0:
                raise ValueError(
                    "Turn off linesearch and set explicit stepsizes for stochastic optimization."
                )

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader: DataLoader,
        num_epochs: int,
        callback: Callback,
    ) -> StepResult:
        """
        Run optimization over mini-batches from a data loader.

        Parameters
        ----------
        init_params :
            Initial parameter values.
        data_loader :
            Data loader providing batches and metadata.
        num_epochs :
            Number of passes over the data (maximum if convergence monitoring
            is enabled).
        callback :
            Training callback.

        Returns
        -------
        StepResult :
            Final (params, state, aux) tuple.
        """
        self._validate_stochastic_options()

        sample_batch = data_loader.sample_batch()
        state = self.init_state(init_params, *sample_batch)
        params = init_params
        aux = None

        ctx = TrainingContext(
            solver=self, params=params, state=state, num_epochs=num_epochs
        )

        callback.on_train_begin(ctx)

        for epoch in range(num_epochs):
            prev_params = params
            prev_state = state
            ctx.epoch, ctx.prev_params, ctx.prev_state = epoch, prev_params, prev_state

            callback.on_epoch_begin(ctx)

            for batch_idx, batch_data in enumerate(data_loader):
                ctx.batch_idx = batch_idx
                callback.on_batch_begin(ctx)

                params, state, aux = self.update(params, state, *batch_data)
                ctx.params, ctx.state, ctx.aux = params, state, aux

                callback.on_batch_end(ctx)
                if ctx.should_stop:
                    callback.on_train_end(ctx)
                    return (params, state, aux)

            callback.on_epoch_end(ctx)
            if ctx.should_stop:
                callback.on_train_end(ctx)
                return (params, state, aux)

        callback.on_train_end(ctx)

        return (params, state, aux)


class OptimistixStochasticSolverMixin(StochasticSolverMixin):
    """
    Mixin for Optimistix solvers that updates stats after optimization.

    Extends ``StochasticSolverMixin`` to also update ``self.stats`` with
    the number of steps taken after the optimization loop completes.

    Defines per-epoch convergence criterion as the Cauchy criterion
    on the parameters only (i.e. ignoring function value).
    """

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader: DataLoader,
        num_epochs: int,
        callback: Callback,
    ) -> StepResult:
        """
        Run optimization and update stats.

        Parameters
        ----------
        init_params :
            Initial parameter values.
        data_loader :
            Data loader providing batches and metadata.
        num_epochs :
            Number of passes over the data.
        callback :
            See ``StochasticSolverMixin._stochastic_run_impl``.

        Returns
        -------
        StepResult :
            Final (params, state, aux) tuple.
        """
        params, state, aux = super()._stochastic_run_impl(
            init_params,
            data_loader,
            num_epochs,
            callback=callback,
        )
        self.stats = {
            "num_steps": self._extract_num_steps(state),
            "max_steps": self.maxiter,
        }
        return (params, state, aux)

    def stochastic_convergence_criterion(
        self,
        params: Params,
        prev_params: Params,
        state: SolverState,
        prev_state: SolverState,
        aux: Any,
        epoch: int,
    ):
        """Cauchy criterion on parameters using the solver's atol and rtol."""
        del state, prev_state, aux, epoch
        # solver needs to have tol and rtol
        # function evaluation on the whole data might be too expensive
        return _params_only_cauchy_criterion(params, prev_params, self.tol, self.rtol)


def _stepsize_normalized_convergence(
    params: Params,
    prev_params: Params,
    stepsize: float,
    tol: float,
) -> bool:
    """
    Step-size-normalized parameter convergence: ||params - prev_params|| / stepsize <= tol.

    Parameters
    ----------
    params :
        Parameter values at end of current epoch.
    prev_params :
        Parameter values at end of previous epoch.
    stepsize :
        Step size used for the gradient updates.
    tol :
        Convergence tolerance.

    Returns
    -------
    bool
        True if the criterion is met.
    """
    return tree_l2_norm(tree_sub(params, prev_params)) / stepsize <= tol


class JaxoptStochasticSolverMixin(StochasticSolverMixin):
    """
    Mixin for JAXopt solvers.

    Defines stochastic convergence criterion as |params - prev_params| / stepsize <= tol
    """

    def stochastic_convergence_criterion(
        self,
        params: Params,
        prev_params: Params,
        state: SolverState,
        prev_state: SolverState,
        aux: Any,
        epoch: int,
    ):
        """Step-size-normalized parameter change: ||params - prev_params|| / stepsize <= tol."""
        del prev_state, aux, epoch
        return _stepsize_normalized_convergence(
            params, prev_params, state.stepsize, self.tol
        )
