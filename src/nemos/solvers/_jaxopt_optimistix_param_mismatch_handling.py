import inspect
import warnings
from typing import Any, Type


def _clean_solver_kwargs(
    solver_class: Type, solver_init_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Clean up the arguments passed to the solver accounting for some mismatches between JAXopt and Optimistix.

    The maximum number of iterations: JAXopt solvers use `maxiter`, `Optimistix` uses `max_steps.`
    If the unexpected parameter name is given, just rename it in `solver_init_kwargs`.

    The tolerances for the convergence criterion are defined differently.
    JAXopt solvers typically have a single `tol` parameter, while Optimistix passes `atol` and `rtol`
    to their Cauchy criterion.
    If `atol` and `rtol` are passed to a JAXopt solver, `atol` is used for `tol`.
    If `tol` is passed to an Optimistix solver, it is used to set `atol` and `rtol` is set to 0.

    In all cases a warning is raise about the unexpected argument.
    """
    try:
        accepted_args = solver_class.get_accepted_arguments()
    except AttributeError:
        # NOTE might want to take the union with solver_class.__init__'s args?
        accepted_args = set(inspect.getfullargspec(solver_class).args)

    solver_init_kwargs = _replace_param(
        solver_init_kwargs, accepted_args, "maxiter", "max_steps"
    )

    solver_init_kwargs = _replace_param(
        solver_init_kwargs, accepted_args, "max_steps", "maxiter"
    )

    solver_init_kwargs = _replace_tol(solver_init_kwargs, accepted_args)
    solver_init_kwargs = _replace_atol_rtol(solver_init_kwargs, accepted_args)

    return solver_init_kwargs


def _replace_param(
    solver_init_kwargs: dict[str, Any],
    accepted_arguments: set[str],
    unaccepted_name: str,
    accepted_name: str,
) -> dict[str, Any]:
    """Replace a parameter name with another and warn about it."""
    if (
        unaccepted_name in solver_init_kwargs
        and unaccepted_name not in accepted_arguments
    ):
        warnings.warn(
            f"Solver does not accept `{unaccepted_name}`. "
            f"Using its value for `{accepted_name}`."
        )
        solver_init_kwargs[accepted_name] = solver_init_kwargs.pop(unaccepted_name)

    return solver_init_kwargs


def _replace_tol(
    solver_init_kwargs: dict[str, Any],
    accepted_arguments: set[str],
) -> dict[str, Any]:
    """Handle an unexpected `tol` argument."""
    if "tol" in solver_init_kwargs and "tol" not in accepted_arguments:
        warnings.warn(
            "Solver does not accept `tol`. "
            "Using its value for `atol` and setting `rtol` to zero."
        )
        if "atol" in solver_init_kwargs:
            raise ValueError("tol and atol can't both be given.")
        if "rtol" in solver_init_kwargs:
            raise ValueError("tol and rtol can't both be given.")

        solver_init_kwargs["atol"] = solver_init_kwargs.pop("tol")
        solver_init_kwargs["rtol"] = 0.0

    return solver_init_kwargs


def _replace_atol_rtol(
    solver_init_kwargs: dict[str, Any],
    accepted_arguments: set[str],
) -> dict[str, Any]:
    """Handle unexpected `atol` and `rtol` arguments."""
    if "atol" in solver_init_kwargs and "atol" not in accepted_arguments:
        warnings.warn(
            "solver does not accept `atol` and `rtol`. "
            "Discarding `rtol` and using `atol` as `tol`."
        )
        solver_init_kwargs["tol"] = solver_init_kwargs.pop("atol")
        solver_init_kwargs.pop("rtol", None)

    return solver_init_kwargs
