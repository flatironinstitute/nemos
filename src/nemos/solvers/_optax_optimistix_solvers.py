"""Solvers wrapping Optax solvers with Optimistix for use with NeMoS."""

import abc
from typing import Any, Callable, NamedTuple, Union, ClassVar
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import ArrayLike
from ..typing import Pytree as PyTree
from ..regularizer import Regularizer
from ..tree_utils import tree_sub
from ._optimistix_solvers import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    OptimistixStepResult,
    Params,
    OptimistixAdapter,
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


# NOTE This might be solved in a simpler way using
# https://optax.readthedocs.io/en/latest/getting_started.html#accessing-learning-rate


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

    If `stepsize` is not None, use it as a constant learning rate.
    If `stepsize` is None, create a zoom linesearch with `linesearch_kwargs`.
    """
    if stepsize is None or stepsize <= 0.0:
        if linesearch_kwargs is None:
            linesearch_kwargs = {
                # "approx_dec_rtol" : None, # setting this to none might be useful
            }

        if "max_linesearch_steps" not in linesearch_kwargs:
            linesearch_kwargs["max_linesearch_steps"] = 15

        return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
    else:
        if linesearch_kwargs:
            raise ValueError("Only provide stepsize or linesearch_kwargs.")
        # NOTE GradientDescent works with optax.scale_by_learning_rate as well
        # but for ProximalGradient we need to be able to extract the current learning rate
        return stateful_scale_by_learning_rate(stepsize)


class OptimistixOptaxGradientDescent(AbstractOptimistixOptaxSolver):
    """
    Gradient descent implementation combining Optax and Optimistix.

    Uses Optax's SGD with Nesterov acceleration combined with Optax's
    zoom linesearch or a constant learning rate.

    The full optimization loop is handled by the `optimistix.OptaxMinimiser` wrapper.
    """

    _optax_solver = optax.sgd

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        acceleration: bool = True,
        stepsize: float | None = None,
        linesearch_kwargs: dict | None = None,
        **solver_init_kwargs,
    ):
        """
        Create a solver wrapping `optax.sgd`.

        If `acceleration` is True, use Nesterov acceleration.

        Use either `stepsize` or `linesearch_kwargs`.
        If `stepsize` is not None and larger than 0, it is used as a fixed stepsize,
        otherwise `optax.zoom_linesearch` is used.
        """
        _sgd = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=acceleration),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )
        solver_init_kwargs["optim"] = _sgd

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            atol=atol,
            rtol=rtol,
            **solver_init_kwargs,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        arguments = super().get_accepted_arguments()

        arguments.discard("optim")  # we create this, it can't be passed

        return arguments


class OptimistixOptaxProximalGradient(AbstractOptimistixOptaxSolver):
    """
    ProximalGradient implementation combining Optax and Optimistix.

    Uses Optax's SGD with Nesterov acceleration combined with Optax's
    zoom linesearch or a constant learning rate.
    Then uses the learning rate given by Optax to scale the proximal
    operator's update and check for convergence using Optimistix's criterion.

    Works with the same proximal operator functions as JAXopt did.
    """

    fun: Callable
    fun_with_aux: Callable
    prox: Callable

    stats: dict[str, PyTree[ArrayLike]]

    _proximal: ClassVar[bool] = True

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        acceleration: bool = True,
        stepsize: float | None = None,
        linesearch_kwargs: dict | None = None,
        **solver_init_kwargs,
    ):
        """
        Create a proximal gradient solver using `optax.sgd` and applying the proximal operator `prox` on each update step.

        If `acceleration` is True, use Nesterov acceleration.

        Use either `stepsize` or `linesearch_kwargs`.
        If `stepsize` is not None and larger than 0, it is used as a fixed stepsize,
        otherwise `optax.zoom_linesearch` is used.
        """
        if linesearch_kwargs is None:
            linesearch_kwargs = {}

        # disable the curvature test
        if "curv_rtol" not in linesearch_kwargs:
            linesearch_kwargs["curv_rtol"] = jnp.inf

        _sgd = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=acceleration),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )
        solver_init_kwargs["optim"] = _sgd

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            atol=atol,
            rtol=rtol,
            **solver_init_kwargs,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        arguments = super().get_accepted_arguments()

        arguments.discard("optim")  # we create this, it can't be passed

        return arguments

    def get_learning_rate(self, state: optx._solver.optax._OptaxState) -> float:
        """
        Read out the learning rate for scaling within the proximal operator.

        This learning rate is either a static learning rate or was found by a linesearch.
        """
        return state.opt_state[-1].learning_rate

    def step(
        self,
        fn: Callable,
        y: Params,
        args: PyTree,
        options: dict[str, Any],
        state: optx._solver.optax._OptaxState,
        tags: frozenset[object],
    ):
        # take gradient step
        new_params, new_state, new_aux = self._solver.step(
            fn, y, args, options, state, tags
        )

        # apply the proximal operator
        new_params = self.prox(
            new_params,
            self.regularizer_strength,
            self.get_learning_rate(new_state),
        )

        # reevaluate function value at the new point
        new_state = eqx.tree_at(lambda s: s.f, new_state, fn(new_params, args)[0])

        # recheck convergence criteria with the projected point
        updates = tree_sub(new_params, y)

        terminate = optx._misc.cauchy_termination(
            self._solver.rtol,
            self._solver.atol,
            self._solver.norm,
            y,
            updates,
            new_state.f,
            new_state.f - state.f,
        )
        # we could also replicate the jaxopt stopping criterion
        # terminate = (
        #    optx.two_norm(updates) / self.get_learning_rate(new_state)
        #    < self._solver.atol
        # )

        new_state = eqx.tree_at(lambda s: s.terminate, new_state, terminate)

        return new_params, new_state, new_aux

    def run(
        self,
        init_params: Params,
        *args,
    ) -> OptimistixStepResult:
        solution = optx.minimise(
            fn=self.fun,
            solver=self,  # pyright: ignore
            y0=init_params,
            args=args,
            options=self.config.options,
            has_aux=self.config.has_aux,
            max_steps=self.config.max_steps,
            adjoint=self.config.adjoint,
            throw=self.config.throw,
            tags=self.config.tags,
        )

        self.stats.update(solution.stats)

        return solution.value, solution.state

    def init(self, *args, **kwargs):
        # so that when passing self to optx.minimise, init can be called
        return self._solver.init(*args, **kwargs)

    def terminate(self, *args, **kwargs):
        # so that when passing self to optx.minimise, terminate can be called
        return self._solver.terminate(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        # so that when passing self to optx.minimise, postprocess can be called
        return self._solver.postprocess(*args, **kwargs)


class OptimistixOptaxLBFGS(AbstractOptimistixOptaxSolver):
    """
    L-BFGS implementation using optax.lbfgs wrapped by optimistix.OptaxMinimiser.

    Convergence criterion is implemented by Optimistix, so it's their Cauchy criterion.
    """

    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

    _optax_solver = optax.lbfgs

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        stepsize: float | None = None,
        **solver_init_kwargs,
    ):
        """
        Create a solver wrapping `optax.lbfgs`.

        `stepsize` is passed for `learning_rate` to `optax.lbfgs`.
        """
        # TODO might want to expose some more parameters?
        solver_init_kwargs["optim"] = optax.lbfgs(learning_rate=stepsize)

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            atol=atol,
            rtol=rtol,
            **solver_init_kwargs,
        )
