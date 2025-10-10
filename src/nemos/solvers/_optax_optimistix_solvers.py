"""Solvers wrapping Optax solvers with Optimistix for use with NeMoS."""

import abc
import inspect
from typing import Any, Callable, ClassVar, NamedTuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx

from ..regularizer import Regularizer
from ..tree_utils import tree_sub
from ..typing import Pytree
from ._optimistix_solvers import (
    DEFAULT_ATOL,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    OptimistixAdapter,
    OptimistixStepResult,
    Params,
)


class AbstractOptimistixOptaxSolver(OptimistixAdapter, abc.ABC):
    """Adapter for optimistix.OptaxMinimiser which is an adapter for Optax solvers."""

    _solver_cls = optx.OptaxMinimiser
    # if defined, the docstring is extended to include the documentation of the wrapped Optax solver
    _optax_solver: ClassVar[Callable[..., optax.GradientTransformation]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # only append things if the _optax_solver class attribute is defined
        if not hasattr(cls, "_optax_solver"):
            return

        doc_so_far = inspect.cleandoc(inspect.getdoc(cls))
        # delete the part about OptaxMinimiser
        doc_so_far = doc_so_far.split("\n\nOptaxMinimiser's documentation:", 1)[0]

        init_header = inspect.cleandoc(f"More info from {cls.__name__}.__init__'s doc")
        init_header += "\n" + "-" * len(init_header)
        init_doc = inspect.cleandoc(
            inspect.getdoc(cls.__init__)
            or f"No documentation found for {cls.__name__}.init"
        )
        init_doc = init_header + "\n" + init_doc

        optax_header = inspect.cleandoc(
            f"""
            More info from Optax's {cls._optax_solver.__name__} documentation:
            """
        )
        optax_header += "\n" + "-" * len(optax_header)
        optax_doc = inspect.cleandoc(
            inspect.getdoc(cls._optax_solver) or "No documentation found in Optax."
        )
        optax_doc = optax_header + "\n" + optax_doc

        full_doc = "\n\n".join(
            (
                doc_so_far,
                init_doc,
                optax_doc,
            )
        )

        cls.__doc__ = inspect.cleandoc(full_doc)


class ScaleByLearningRateState(NamedTuple):
    learning_rate: Union[float, jax.Array]


def stateful_scale_by_learning_rate(
    stepsize: float, flip_sign: bool = True
) -> optax.GradientTransformation:
    """
    Reimplementation of optax.scale_by_learning_rate, just storing the learning rate in the state.

    Required for setting the scaling appropriately when used with
    proximal gradient descent.
    """
    m = -1 if flip_sign else 1

    def init_fn(params):
        del params
        return ScaleByLearningRateState(jnp.array(stepsize))

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda g: m * stepsize * g, updates)

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)  # pyright: ignore


def _make_rate_scaler(
    stepsize: float | None,
    linesearch_kwargs: dict[str, Any] | None,
) -> optax.GradientTransformation:
    """
    Make an Optax transformation for setting the learning rate.

    If `stepsize` is not None and larger than 0, use it as a constant learning rate.
    Otherwise `optax.scale_by_zoom_linesearch` is used with `linesearch_kwargs`.
    By default the curvature condition is disabled, which reduces to a backtracking
    linesearch where the stepsizes are chosen using the cubic or quadratic interpolation
    used in zoom linesearch.
    By default, 15 linesearch steps are used, which can be overwritten with
    `max_linesearch_steps` in `linesearch_kwargs`.
    """
    if stepsize is None or stepsize <= 0.0:
        if linesearch_kwargs is None:
            linesearch_kwargs = {}

        if "max_linesearch_steps" not in linesearch_kwargs:
            linesearch_kwargs["max_linesearch_steps"] = 15

        if "curv_rtol" not in linesearch_kwargs:
            linesearch_kwargs["curv_rtol"] = jnp.inf

        return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
    else:
        if linesearch_kwargs:
            raise ValueError("Only provide stepsize or linesearch_kwargs.")
        # GradientDescent works with optax.scale_by_learning_rate as well
        # but for ProximalGradient we need to be able to extract the current learning rate
        return stateful_scale_by_learning_rate(stepsize)


class OptimistixOptaxGradientDescent(AbstractOptimistixOptaxSolver):
    """
    Gradient descent implementation combining Optax and Optimistix.

    Uses Optax's SGD combined with Optax's zoom linesearch or a constant learning rate.

    The full optimization loop is handled by the `optimistix.OptaxMinimiser` wrapper.
    """

    _optax_solver = optax.sgd

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        maxiter: int = DEFAULT_MAX_STEPS,
        momentum: float | None = None,
        acceleration: bool = True,
        stepsize: float | None = None,
        linesearch_kwargs: dict | None = None,
        **solver_init_kwargs,
    ):
        """
        Create a solver wrapping `optax.sgd`.

        If `acceleration` is True, use the Nesterov acceleration as defined by Sutskever et al. 2013.
        Note that this is different from the Nesterov acceleration implemented by JAXopt and
        only has an effect if `momentum` is used as well.

        If `stepsize` is not None and larger than 0, use it as a constant learning rate.
        Otherwise `optax.scale_by_zoom_linesearch` is used with the curvature condition
        disabled, which reduces to a backtracking linesearch where the stepsizes are chosen
        using the cubic or quadratic interpolation used in zoom linesearch.
        By default, 15 linesearch steps are used, which can be overwritten with
        `max_linesearch_steps` in `linesearch_kwargs`.

        References
        ----------
        [1] [Sutskever, I., Martens, J., Dahl, G. &amp; Hinton, G.. (2013).
        "On the importance of initialization and momentum in deep learning."
        Proceedings of the 30th International Conference on Machine Learning, PMLR 28(3):1139-1147, 2013.
        ](https://proceedings.mlr.press/v28/sutskever13.html)
        """
        _sgd = optax.chain(
            optax.sgd(learning_rate=1.0, momentum=momentum, nesterov=acceleration),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )
        solver_init_kwargs["optim"] = _sgd

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            tol=tol,
            rtol=rtol,
            maxiter=maxiter,
            **solver_init_kwargs,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        arguments = super().get_accepted_arguments()

        arguments.discard("optim")  # we create this, it can't be passed

        return arguments

    @classmethod
    def _note_about_accepted_arguments(cls) -> str:
        note = super()._note_about_accepted_arguments()
        accel_nesterov = inspect.cleandoc(
            """
            `acceleration` is passed to `optax.sgd` as the `nesterov` parameter.
            Note that this only has an effect if `momentum` is used as well.
            """
        )
        return inspect.cleandoc(note + "\n" + accel_nesterov)


class OptimistixOptaxLBFGS(AbstractOptimistixOptaxSolver):
    """
    L-BFGS implementation using optax.lbfgs wrapped by optimistix.OptaxMinimiser.

    Convergence criterion is implemented by Optimistix, so the Cauchy criterion is used.
    """

    fun: Callable
    fun_with_aux: Callable

    # stats: dict[str, PyTree[ArrayLike]]
    stats: dict[str, Pytree]

    _optax_solver = optax.lbfgs

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        maxiter: int = DEFAULT_MAX_STEPS,
        stepsize: float | None = None,
        memory_size: int = 10,
        scale_init_precond: bool = True,
        **solver_init_kwargs,
    ):
        """
        Create a solver wrapping `optax.lbfgs`.

        `stepsize` is passed for `learning_rate` to `optax.lbfgs`.
          If None, a zoom linesearch is used (recommended).
        `memory_size` and `scale_init_precond` are passed to `optax.lbfgs`.

        For more information on these parameters, see `get_solver_documentation`.
        """
        solver_init_kwargs["optim"] = optax.lbfgs(
            learning_rate=stepsize,
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        )

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            tol=tol,
            rtol=rtol,
            maxiter=maxiter,
            **solver_init_kwargs,
        )


# fix optax.lbfgs doctest failing due to x64 being set in pytest
OptimistixOptaxLBFGS.__doc__ = (OptimistixOptaxLBFGS.__doc__ or "").replace(
    "jnp.array([1., 2., 3.])", "jnp.array([1., 2., 3.], dtype=jnp.float32)"
)
