"""Implementation of the FISTA algorithm as an Optimistix IterativeSolver. Adapted from JAXopt."""

from typing import Any, Callable, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool, Float, Int, PyTree
from optimistix._custom_types import Aux, Y

from ..tree_utils import tree_add_scalar_mul, tree_sub
from ._optimistix_solvers import OptimistixAdapter


def prox_none(x: PyTree, hyperparams=None, scaling: float = 1.0):
    """Identity proximal operator."""
    del hyperparams, scaling
    return x


def tree_nan_like(x: PyTree):
    return jax.tree.map(lambda arr: jnp.full_like(arr, jnp.nan), x)


class ProxGradState(eqx.Module):
    """ProximalGradient (FISTA) solver state."""

    iter_num: Int[Array, ""]
    stepsize: Float[Array, ""]
    velocity: PyTree
    t: Float[Array, ""]
    f: Float[Array, ""]

    terminate: Bool[Array, ""]


class FISTA(optx.AbstractMinimiser[Y, Aux, ProxGradState]):
    """
    Accelerated Proximal Gradient (FISTA) [1] as an Optimistix minimiser. Adapted from JAXopt.

    Parameters
    ----------
    atol:
        Absolute tolerance for Cauchy termination.
    rtol:
        Relative tolerance for Cauchy termination.
    norm:
        Norm to use in Cauchy termination.
    prox:
        Proximal operator function.
    regularizer_strength:
        Regularizer strength passed to the proximal operator.
    stepsize:
        If None (default), use backtracking linesearch to determine an
        appropriate stepsize on each iteration.
        If a float, value for a constant stepsize.
    maxls:
        Maximum number of linesearch iterations.
    decrease_factor:
        Backtracking linesearch's decrease factor.
    max_stepsize:
        Maximum allowed stepsize.
        If None, no maximum is used.
    acceleration:
        Whether to use Nesterov acceleration.


    References
    ----------
    .. [1] Beck, A., & Teboulle, M. (2009).
    "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems."
    *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
    https://doi.org/10.1137/080716542
    """

    atol: float
    rtol: float
    norm: Callable

    prox: Callable
    regularizer_strength: float | None

    stepsize: float | None = None
    maxls: int = 15
    decrease_factor: float = 0.5
    max_stepsize: float | None = 1.0

    acceleration: bool = True

    while_loop_kind: Literal["lax", "checkpointed", "bounded"] | None = None

    def init(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> ProxGradState:
        del options, f_struct, aux_struct, tags
        fun_val, _ = fn(y, args)

        if self.acceleration:
            vel = y
            t = jnp.asarray(1.0)
        else:
            vel = tree_nan_like(y)
            t = jnp.asarray(jnp.nan)

        return ProxGradState(
            iter_num=jnp.asarray(0),
            velocity=vel,
            t=t,
            stepsize=jnp.asarray(1.0),
            terminate=jnp.asarray(False),
            f=fun_val,
        )

    def step(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
        tags: frozenset[object],
    ) -> tuple[Y, ProxGradState, Aux]:
        del tags

        # Some clarification on variable names because Optimistix's and the FISTA paper's / JAXopt's
        # notation are different:
        #
        # In the paper x_{i} are the parameters, y_{i} the points after the momentum step.
        # The updates:
        #   x_{k} = prox(y_{k} - stepsize_{k} * gradient_at_y_{k})
        #
        #   t_{k+1} = (1 + sqrt(1 + 4 t_{k}^2)) / 2
        #   y_{k+1} = x_{k} + ((t_{k} - 1) / t_{k+1}) * (x_{k} - x_{k-1})
        #
        #   Where we run a linesearch to find the stepsize_{k} starting with stepsize_{k-1} / decrease_factor or 1.
        #   Note that instead of x_{k}, the current parameter values, the linsearch is done "looking out" from y_{k}
        #   in the direction of the gradient at that point.
        #   Also, the new t_{k+1} and y_{k+1} are precalculated to be used on the next iteration.
        #
        # In Optimistix the parameter values are denoted by y,
        # so what is x in the formula above will be y in the code,
        # and what is y in the formula will be called velocity or vel.
        #
        # In order of appearance in the code:
        #   y = x_{k-1}
        #   state.velocity = y_{k}          # note that it was precalculated at the previous step
        #   state.stepsize = stepsize_{k-1}
        #   new_y = x_{k}
        #   new_stepsize = stepsize_{k}
        #   state.t = t_{k}                 # also precalculated at the previous step
        #   next_t = t_{k+1}
        #   next_vel = y_{k+1}

        if self.acceleration:
            update_point = state.velocity
        else:
            update_point = y

        new_y, new_stepsize = self._update_at_point(
            fn, update_point, args, options, state
        )
        # TODO: These could be returned from _update_at_point
        # because the linesearch already calculates it
        # so a function evaluation could be saved here.
        new_fun_val, new_aux = fn(new_y, args)
        diff_y = tree_sub(new_y, y)

        if self.acceleration:
            next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * state.t**2))
            next_vel = tree_add_scalar_mul(new_y, (state.t - 1) / next_t, diff_y)
        else:
            next_t = state.t
            next_vel = state.velocity

        # use Cauchy for consistency with other solvers
        # terminate = optx._misc.cauchy_termination(
        #     self.rtol,
        #     self.atol,
        #     self.norm,
        #     y,
        #     diff_y,
        #     state.f,
        #     new_fun_val - state.f,
        # )
        # TODO: Adapt this to terminate on nan with Cauchy
        # or use the same termination as JAXopt
        # terminate = ~continue_iteration assures that it stops if anything is NaN
        continue_iteration = (optx.two_norm(diff_y) / new_stepsize) > self.atol
        terminate = ~continue_iteration

        next_state = ProxGradState(
            iter_num=state.iter_num + 1,
            velocity=next_vel,
            t=next_t,
            stepsize=jnp.asarray(new_stepsize),
            terminate=terminate,
            f=new_fun_val,
        )

        return new_y, next_state, new_aux

    def _update_at_point(
        self,
        fn: Callable,
        update_point: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
    ):
        """
        Perform the update with or without linesearch around `update_point`.

        If acceleration is used (FISTA), `update_point` is state.velocity ~ y_{k}.
        Without acceleration (ISTA) `update_point` is `y` ~ x_{k-1}.
        """
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_at_point, lin_fn, _ = jax.linearize(
            lambda _y: fn(_y, args), update_point, has_aux=True
        )
        grad_at_point = optx._misc.lin_to_grad(
            lin_fn, update_point, autodiff_mode=autodiff_mode
        )

        if self.stepsize is None or self.stepsize <= 0.0:
            # do linesearch to find the new stepsize
            new_y, new_stepsize = self.fista_line_search(
                lambda params, args: fn(params, args)[0],
                update_point,
                f_at_point,
                grad_at_point,
                state.stepsize,
                args,
            )

            # attempt to increase the stepsize for the new linesearch
            # or reset it if it's very small
            new_stepsize = jnp.where(
                new_stepsize <= 1e-6,
                jnp.array(1.0),
                new_stepsize / self.decrease_factor,
            )
            # B: in my experience, this guard helps stabilize and reduce the number of iterations
            # For some reason, without it this implementation sometimes needs more iterations than the
            # original JAXopt implementation, which in theory should be mathematically identical.
            if self.max_stepsize is not None:
                new_stepsize = jnp.minimum(new_stepsize, self.max_stepsize)
        else:
            # use the fixed stepsize
            new_stepsize = self.stepsize
            new_y = tree_add_scalar_mul(update_point, -new_stepsize, grad_at_point)
            new_y = self.prox(new_y, self.regularizer_strength, new_stepsize)

        return new_y, new_stepsize

    # adapted from JAXopt
    def fista_line_search(
        self,
        fun: Callable,
        x: Y,
        x_fun_val: Float[Array, ""],
        grad: Y,
        stepsize: Float[Array, ""],
        args: PyTree[Any],
    ) -> tuple[Y, Float[Array, ""]]:
        # epsilon of current dtype for robust checking of
        # sufficient decrease condition
        eps = jnp.finfo(x_fun_val.dtype).eps

        def cond_fun(carry):
            next_x, stepsize = carry

            new_fun_val = fun(next_x, args)

            diff_x = tree_sub(next_x, x)
            sqdist = optx._misc.sum_squares(diff_x)

            # verbatim from JAXopt
            # The expression below checks the sufficient decrease condition
            # f(next_x) < f(x) + dot(grad_f(x), diff_x) + (0.5/stepsize) ||diff_x||^2
            # where the terms have been reordered for numerical stability.
            fun_decrease = stepsize * (new_fun_val - x_fun_val)
            expected_decrease = (
                stepsize * optx._misc.tree_dot(diff_x, grad) + 0.5 * sqdist
            )

            return fun_decrease > expected_decrease + eps

        def body_fun(carry):
            stepsize = carry[1]
            new_stepsize = stepsize * self.decrease_factor
            next_x = tree_add_scalar_mul(x, -new_stepsize, grad)
            next_x = self.prox(next_x, self.regularizer_strength, new_stepsize)
            return next_x, new_stepsize

        init_x = tree_add_scalar_mul(x, -stepsize, grad)
        init_x = self.prox(init_x, self.regularizer_strength, stepsize)
        init_val = (init_x, stepsize)

        return eqx.internal.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=init_val,
            max_steps=self.maxls,
            kind=self.while_loop_kind,
        )

    def terminate(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx._solution.RESULTS]:
        del fn, y, args, options, tags

        return state.terminate, optx._solution.RESULTS.successful

    def postprocess(
        self,
        fn: Callable,
        y: Y,
        aux: Aux,
        args: PyTree[Any],
        options: dict[str, Any],
        state: ProxGradState,
        tags: frozenset[object],
        result: optx._solution.RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        del fn, args, options, state, tags, result
        return y, aux, {}


class GradientDescent(FISTA):
    """Gradient descent with Nesterov acceleration. Adapted from JAXopt."""

    prox: ClassVar[Callable] = staticmethod(prox_none)
    regularizer_strength: float | None = None


class OptimistixFISTA(OptimistixAdapter):
    """Port of JAXopt's ProximalGradient to the Optimistix API."""

    _solver_cls = FISTA
    _proximal = True

    def adjust_solver_init_kwargs(
        self, solver_init_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Derive the "kind" parameter of the linesearch's while_loop based on the adjoint."""
        if isinstance(self.config.adjoint, optx.ImplicitAdjoint):
            kind = "lax"
        elif isinstance(self.config.adjoint, optx.RecursiveCheckpointAdjoint):
            kind = "checkpointed"
        else:
            raise ValueError(
                "adjoint has to be ImplicitAdjoint or RecursiveCheckpointAdjoint"
            )

        return {"while_loop_kind": kind, **solver_init_kwargs}


class OptimistixNAG(OptimistixAdapter):
    """Port of Nesterov's accelerated gradient descent from JAXopt to the Optimistix API."""

    _solver_cls = GradientDescent
    _proximal = False

    adjust_solver_init_kwargs = OptimistixFISTA.adjust_solver_init_kwargs
