"""Mixins providing standard stochastic optimization implementations."""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

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


def _as_stop_flag(value: Any, callback_name: str) -> bool:
    """
    Validate and convert callback return value to a stop flag.

    Parameters
    ----------
    value :
        Return value from ``batch_callback`` or ``convergence_criterion``.
    callback_name :
        Name used in error messages.

    Returns
    -------
    bool
        Converted stop flag.

    Raises
    ------
    TypeError
        If ``value`` is not a scalar boolean.
    """
    if isinstance(value, bool):
        return value

    arr = np.asarray(value)
    if arr.shape == () and np.issubdtype(arr.dtype, np.bool_):
        return bool(arr.item())

    raise TypeError(
        f"{callback_name} must return a scalar boolean; got "
        f"type={type(value).__name__}, shape={arr.shape}, dtype={arr.dtype}."
    )


class StochasticSolverMixin:
    """
    Mixin providing standard stochastic optimization loop.

    This mixin provides a default implementation of ``_stochastic_run_impl``
    that iterates over the data loader for the specified number of epochs,
    calling ``update`` on each batch.

    Supports optional per-epoch convergence monitoring and per-batch callbacks.
    If ``convergence_criterion`` is ``None``, no convergence check is performed.

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
        data_loader: "DataLoader",
        num_epochs: int,
        convergence_criterion: Callable | None = None,
        batch_callback: Callable | None = None,
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
        convergence_criterion :
            Per-epoch convergence criterion.
            ``None`` disables convergence monitoring.
            A callable with signature
            ``(params, prev_params, state, prev_state, aux, epoch) -> bool``
            stops optimization when it returns ``True``.
        batch_callback :
            Optional per-batch callback with signature
            ``(params, state, aux, batch_idx, epoch) -> bool``.
            Returning ``True`` stops optimization after that batch.

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

        for epoch in range(num_epochs):
            prev_params = params
            prev_state = state

            for batch_idx, batch_data in enumerate(data_loader):
                params, state, aux = self.update(params, state, *batch_data)
                if batch_callback is not None:
                    stop_on_batch = batch_callback(params, state, aux, batch_idx, epoch)
                    if _as_stop_flag(stop_on_batch, "batch_callback"):
                        return (params, state, aux)

            if convergence_criterion is not None:
                stop_on_epoch = convergence_criterion(
                    params, prev_params, state, prev_state, aux, epoch
                )
                if _as_stop_flag(stop_on_epoch, "convergence_criterion"):
                    return (params, state, aux)

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
        data_loader: "DataLoader",
        num_epochs: int,
        convergence_criterion: Callable | None = None,
        batch_callback: Callable | None = None,
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
        convergence_criterion :
            See ``StochasticSolverMixin._stochastic_run_impl``.
        batch_callback :
            See ``StochasticSolverMixin._stochastic_run_impl``.

        Returns
        -------
        StepResult :
            Final (params, state, aux) tuple.
        """
        result = super()._stochastic_run_impl(
            init_params,
            data_loader,
            num_epochs,
            convergence_criterion=convergence_criterion,
            batch_callback=batch_callback,
        )
        _, state, _ = result
        num_steps = self._extract_num_steps(state)
        self.stats = {"num_steps": num_steps, "max_steps": self.maxiter}
        return result

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
