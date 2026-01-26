import inspect
import warnings
from typing import Any, Callable, Type

import jax.numpy as jnp
import numpy as np

from nemos.regularizer import Ridge

from ._abstract_solver import AbstractSolver, OptimizationInfo

# Notes
# We could enforce adherence to the API with type checkers
# https://github.com/agronholm/typeguard
# https://github.com/beartype/beartype

METHOD_NAMES = AbstractSolver.__abstractmethods__
AUX_VAL = -1


def _get_params(
    fun: Callable,
    first_n_params: int = None,
    names_only: bool = True,
) -> list[str] | list[inspect.Parameter]:
    """
    Get the (names of the) parameters of a function.

    Parameters
    ----------
    fun :
        Function to inspect.
    first_n_params :
        Number of arguments to include.
    names_only :
        Whether to return only the names or the inspect.Parameter
        with extra info.
    """
    signature = inspect.signature(fun)
    params = list(signature.parameters.values())

    if names_only:
        params = [p.name for p in params]

    if first_n_params is not None:
        params = params[:first_n_params]

    return params


def _validate_method_signature(
    solver_class: Type, method_name: str
) -> tuple[bool, str]:
    """
    Check that the arguments of the method are the same.

    For __init__ only check the first arguments that are needed,
    the following ones (**solver_kwargs) can be anything.

    Returns (True, None) if there are no problems, and (False, error_message)
    if there are.
    """
    n_params_to_check = None
    if method_name == "__init__":
        n_params_to_check = sum(
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            for p in _get_params(getattr(AbstractSolver, method_name), names_only=False)
        )

    reference = _get_params(getattr(AbstractSolver, method_name), n_params_to_check)
    got = _get_params(getattr(solver_class, method_name), n_params_to_check)

    # enforcing names, not just the number of parameters
    if got != reference:
        problem = (
            f"Incompatible signature for {method_name}. Got {got}. Expected {reference}"
        )
        return False, problem

    return True, None


def _check_all_signatures_match(solver_class: Type) -> None:
    """
    Check that the signature of all required methods matches AbstractSolver's.

    They must have the same argument names.
    In __init__ only the required ones are checked, the rest (**solver_kwargs)
    can be anything.
    """
    # collect mismatches in signatures
    success_dict, problem_dict = {}, {}
    for method_name in METHOD_NAMES:
        success_dict[method_name], problem_dict[method_name] = (
            _validate_method_signature(solver_class, method_name)
        )

    # raise one error with all the problems found
    if not all(success_dict.values()):
        error_msg = "\n".join(
            problem_dict[method] for method in METHOD_NAMES if not success_dict[method]
        )
        raise ValueError(error_msg)


def _check_required_methods_exist(solver_class: Type):
    """Check that all abstractmethods of AbstractSolver are implemented."""
    # a bit more detailed than issubclass(solver_class, SolverProtocol)
    for method_name in METHOD_NAMES:
        try:
            getattr(solver_class, method_name)
        except AttributeError as e:
            raise AttributeError(
                f"{solver_class.__name__}.{method_name} does not exist. Please implement it."
            ) from e


def _assert_step_result(step_result: Any, method_name: str) -> tuple[Any, Any, Any]:
    """Make sure step_result is a tuple of length 3."""
    if not isinstance(step_result, tuple):
        raise TypeError(
            f"{method_name} must return a tuple of (params, state, aux), "
            f"got {type(step_result)!r}."
        )
    if len(step_result) != 3:
        raise TypeError(
            f"{method_name} must return a tuple of (params, state, aux), "
            f"got a tuple of length {len(step_result)}."
        )
    return step_result


def _tiny_ridge_regression_problem(
    has_aux: bool,
    seed: int = 123,
    n_samples: int = 100,
    n_features: int = 3,
):
    """Create a tiny ridge regression problem to quickly test solver implementations with."""

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=(n_features,))
    y = X.dot(coef) + 0.1 * rng.normal(size=(n_samples,))

    def _loss(params, XX, yy):
        return jnp.power(yy - jnp.dot(XX, params), 2).mean()

    if not has_aux:
        loss = _loss
    else:
        # return (loss_val, aux)
        def loss(params, XX, yy):
            return (_loss(params, XX, yy), AUX_VAL)

    init_params = jnp.zeros((n_features,))
    return jnp.asarray(X), jnp.asarray(y), init_params, loss


def _validate_solver_class_on_ridge(
    solver_class: type,
    has_aux: bool,
    solver_kwargs: dict[str, Any] | None = None,
):
    """
    Validate a custom solver by running a tiny ridge regression problem.

    This checks that required methods can be called with the expected inputs,
    and that they return sensible outputs.
    """

    if solver_kwargs is None:
        solver_kwargs = {}
    regularizer = Ridge()
    regularizer_strength = 1e-2
    X, y, init_params, unregularized_loss = _tiny_ridge_regression_problem(has_aux)

    solver = solver_class(
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params=init_params,
        **solver_kwargs,
    )

    # init_state works
    _ = solver.init_state(init_params, X, y)

    # run can be called as intended
    run_params, run_state, run_aux = _assert_step_result(
        solver.run(init_params, X, y), "run"
    )
    # update can proceed from run's output
    update_params, update_state, update_aux = _assert_step_result(
        solver.update(run_params, run_state, X, y), "update"
    )
    # update can proceed from its own output
    update_params, update_state, update_aux = _assert_step_result(
        solver.update(update_params, update_state, X, y), "update"
    )

    if has_aux:
        assert run_aux == AUX_VAL
        assert update_aux == AUX_VAL
    else:
        assert run_aux is None
        assert update_aux is None

    optim_info = solver.get_optim_info(run_state)
    if not isinstance(optim_info, OptimizationInfo):
        raise TypeError(
            f"get_optim_info must return OptimizationInfo, got {type(optim_info)!r}."
        )

    penalized_loss = regularizer.penalized_loss(
        unregularized_loss, regularizer_strength, init_params=init_params
    )
    init_loss = penalized_loss(init_params, X, y)
    run_loss = penalized_loss(run_params, X, y)
    update_loss = penalized_loss(update_params, X, y)

    # only look at the function value
    if has_aux:
        init_loss = init_loss[0]
        run_loss = run_loss[0]
        update_loss = update_loss[0]

    if not jnp.all(jnp.isfinite(jnp.array([init_loss, run_loss, update_loss]))):
        raise ValueError("Loss values must be finite for the validation problem.")

    if run_loss > init_loss:
        warnings.warn(f"{solver_class.__name__} increases loss on ridge problem")


def validate_solver_class(
    solver_class: Type,
    test_ridge: bool,
    loss_has_aux: bool,
) -> None:
    """
    Validate required methods against AbstractSolver and optionally run a quick ridge regression.

    1. Check if all required methods are there
    2. Check their signatures and make sure they have the same argument names.
       In __init__ only the required ones are checked.
    3. If `test_ridge` is True, run a ridge regression toy problem to see if
       the solver actually works.
       If `loss_has_aux` is True, the ridge loss will carry an aux variable,
       otherwise it's a scalar loss value.
    """
    _check_required_methods_exist(solver_class)
    _check_all_signatures_match(solver_class)
    if test_ridge:
        _validate_solver_class_on_ridge(solver_class, has_aux=loss_has_aux)
