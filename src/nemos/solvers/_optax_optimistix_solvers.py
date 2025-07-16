"""Solvers wrapping Optax solvers with Optimistix for use with NeMoS."""

import dataclasses
from typing import Any, Callable, NamedTuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import ArrayLike, PyTree

from ..regularizer import Regularizer
from ..tree_utils import tree_sub
from ._optimistix_solvers import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    OptimistixConfig,
    OptimistixOptaxSolver,
    OptimistixStepResult,
    Params,
)

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
    if stepsize is None:
        if linesearch_kwargs is None:
            linesearch_kwargs = {
                # "approx_dec_rtol" : None, # setting this to none might be useful
            }

        if "max_linesearch_steps" not in linesearch_kwargs:
            linesearch_kwargs["max_linesearch_steps"] = 15

        return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
    else:
        # NOTE GradientDescent works with optax.scale_by_learning_rate as well
        # but for ProximalGradient we need to be able to extract the current learning rate
        return stateful_scale_by_learning_rate(stepsize)


class OptaxOptimistixGradientDescent(OptimistixOptaxSolver):
    """
    Gradient descent implementation combining Optax and Optimistix.

    Uses Optax's SGD with Nesterov acceleration combined with Optax's
    zoom linesearch or a constant learning rate.

    The full optimization loop is handled by the `optimistix.OptaxMinimiser` wrapper.
    """

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        acceleration: bool = True,
        **solver_init_kwargs,
    ):
        stepsize = solver_init_kwargs.get("stepsize", None)
        linesearch_kwargs = solver_init_kwargs.get("linesearch_kwargs", {})

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
        arguments.add("stepsize")
        arguments.add("linesearch_kwargs")

        return arguments


class OptaxOptimistixProximalGradient(OptimistixOptaxSolver):
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

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        acceleration: bool = True,
        **solver_init_kwargs,
    ):
        loss_fn = unregularized_loss
        self.fun = lambda params, args: loss_fn(params, *args)
        self.fun_with_aux = lambda params, args: (loss_fn(params, *args), None)

        self.prox = regularizer.get_proximal_operator()
        self.regularizer_strength = regularizer_strength

        # take out the arguments that go into minimise, init, terminate and so on
        # and only pass the actually needed things to __init__
        user_args = {}
        for f in dataclasses.fields(OptimistixConfig):
            kw = f.name
            if kw in solver_init_kwargs:
                user_args[kw] = solver_init_kwargs.pop(kw)
        self.config = OptimistixConfig(**user_args)

        stepsize = solver_init_kwargs.get("stepsize", None)
        linesearch_kwargs = solver_init_kwargs.get("linesearch_kwargs", {})

        # disable the curvature test
        if "curv_rtol" not in linesearch_kwargs:
            linesearch_kwargs["curv_rtol"] = jnp.inf

        _sgd = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=acceleration),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )

        self._solver = optx.OptaxMinimiser(
            optim=_sgd,
            rtol=rtol,
            atol=atol,
            norm=self.config.norm,
        )

        self.stats = {}

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        arguments = super().get_accepted_arguments()

        arguments.discard("optim")  # we create this, it can't be passed
        arguments.add("stepsize")
        arguments.add("linesearch_kwargs")

        return arguments

    @property
    def maxiter(self):
        return self.config.max_steps

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


class OptaxOptimistixLBFGS(OptimistixOptaxSolver):
    """
    L-BFGS implementation using optax.lbfgs wrapped by optimistix.OptaxMinimiser.

    Convergence criterion is implemented by Optimistix, so it's their Cauchy criterion.
    """

    fun: Callable
    fun_with_aux: Callable

    stats: dict[str, PyTree[ArrayLike]]

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
