import equinox as eqx
import optax
import optimistix as optx
from typing import NamedTuple, Union, Any, Callable, Optional, cast

import dataclasses

from .optimistix_solvers import (
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    DEFAULT_ATOL,
    OptimistixAdapter,
    OptimistixOptaxSolver,
    OptimistixConfig,
)

from jaxtyping import PyTree, Scalar, ArrayLike, Array

import jax
import jax.numpy as jnp

from optimistix._solver.optax import _OptaxState
from ..tree_utils import tree_sub

from .abstract_solver import AbstractSolver

# FIXME This might be solved in a simpler way using
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
            linesearch_kwargs["max_linesearch_steps"] = 30

        return optax.scale_by_zoom_linesearch(**linesearch_kwargs)
    else:
        # NOTE GradientDescent works with optax.scale_by_learning_rate as well
        # but for ProximalGradient we need to be able to extract the current learning rate
        return stateful_scale_by_learning_rate(stepsize)


class OptaxOptimistixGradientDescent(OptimistixOptaxSolver):
    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        **solver_init_kwargs,
    ):
        stepsize = solver_init_kwargs.get("stepsize", None)
        linesearch_kwargs = solver_init_kwargs.get("linesearch_kwargs", {})

        _sgd = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=True),
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
    def get_accepted_arguments(cls) -> list[str]:
        arguments = set(super().get_accepted_arguments())

        arguments.discard("optim")  # we create this, it can't be passed
        arguments.add("stepsize")
        arguments.add("linesearch_kwargs")

        return list(arguments)


class OptaxOptimistixProximalGradient(OptimistixOptaxSolver):
    """
    ProximalGradient implementation combining Optax and Optimistix.

    Uses Optax's SGD with Nesterov acceleration combined with Optax's
    zoom linesearch or a constant learning rate.
    Then uses the learning rate given by Optax to scale the proximal
    operator's update and check for convergence using Optimistix's.

    Works with the same proximal operator functions as JAXopt did.

    Passes the regularizer strength (aka. `hyperparams_prox` following
    the JAXopt naming) to .step through the `options` dict as
    `options["regularizer_strength"]`.
    """

    fun: Callable
    fun_with_aux: Callable
    prox: Callable

    stats: dict[str, PyTree[ArrayLike]]

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        **solver_init_kwargs,
    ):
        loss_fn = unregularized_loss
        self.fun = lambda params, args: loss_fn(params, *args)
        self.fun_with_aux = lambda params, args: (loss_fn(params, *args), None)

        self.prox = regularizer.get_proximal_operator()
        self.regularizer_strength = regularizer_strength

        atol = solver_init_kwargs.pop("atol", atol)
        rtol = solver_init_kwargs.pop("rtol", rtol)

        solver_init_kwargs = self._replace_maxiter(solver_init_kwargs)

        user_args = {
            f.name: solver_init_kwargs.pop(f.name)
            for f in dataclasses.fields(OptimistixConfig)
            if f.name in solver_init_kwargs
        }
        self.config = OptimistixConfig(**user_args)

        stepsize = solver_init_kwargs.get("stepsize", None)
        linesearch_kwargs = solver_init_kwargs.get("linesearch_kwargs", {})

        _sgd = optax.chain(
            optax.sgd(learning_rate=1.0, nesterov=True),
            _make_rate_scaler(stepsize, linesearch_kwargs),
        )

        # TODO aren't these already handled in user_args?
        optax_minim_kwargs = {}
        if "verbose" in solver_init_kwargs:
            optax_minim_kwargs["verbose"] = solver_init_kwargs["verbose"]
        if "norm" in solver_init_kwargs:
            optax_minim_kwargs["norm"] = solver_init_kwargs["norm"]

        self._solver = optx.OptaxMinimiser(_sgd, rtol, atol, **optax_minim_kwargs)

        self.stats = {}

    @classmethod
    def get_accepted_arguments(cls) -> list[str]:
        arguments = set(super().get_accepted_arguments())

        arguments.discard("optim")  # we create this, it can't be passed
        arguments.add("stepsize")
        arguments.add("linesearch_kwargs")

        return list(arguments)

    @property
    def maxiter(self):
        return self.config.max_steps

    def get_learning_rate(self, state) -> float:
        """
        Read out the learning rate for scaling within the proximal operator.

        This learning rate is either a static learning rate or was found by a linesearch.
        """
        return state.opt_state[-1].learning_rate

    def step(
        self,
        fn,
        y,
        args: PyTree,
        options: dict[str, Any],
        state,
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

        new_state = eqx.tree_at(lambda s: s.terminate, new_state, terminate)

        return new_params, new_state, new_aux

    # def __getattr__(self, name: str):
    #    return getattr(self._solver, name)

    def init(self, *args, **kwargs):
        # so that when passing self to optx.minimise, init can be called
        return self._solver.init(*args, **kwargs)

    def terminate(self, *args, **kwargs):
        return self._solver.terminate(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        return self._solver.postprocess(*args, **kwargs)

    def run(
        self,
        init_params,
        *args,
    ):
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
