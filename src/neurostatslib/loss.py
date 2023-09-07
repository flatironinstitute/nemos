import abc
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import jaxopt

from .base_class import _Base
from .proximal_operator import prox_group_lasso


class Solver(
    jaxopt.GradientDescent,
    jaxopt.BFGS,
    jaxopt.LBFGS,
    jaxopt.ScipyMinimize,
    jaxopt.NonlinearCG,
    jaxopt.ScipyBoundedMinimize,
    jaxopt.LBFGSB,
    jaxopt.ProximalGradient,
):
    """Class grouping any solver we allow.

    We allow the following solvers:
        - Unconstrained: GradientDescent, BFGS, LBFGS, ScipyMinimize, NonlinearCG
        - Box-Bounded: ScipyBoundedMinimize, LBFGSB
        - Non-Smooth: ProximalGradient

    """


class Regularizer(_Base, abc.ABC):
    allowed_solvers = []

    def __init__(self, alpha: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _check_solver(self, solver_name: str):
        if solver_name not in self.allowed_solvers:
            raise ValueError(
                f"Solver `{solver_name}` not allowed for "
                f"{self.__class__} regularization. "
                f"Allowed solvers are {self.allowed_solvers}."
            )

    @abc.abstractmethod
    def instantiate_solver(
        self,
        solver_name: str,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
        solver_kwargs: Optional[dict] = None,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        pass


class UnRegularized(Regularizer):
    allowed_solvers = [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG" "ScipyBoundedMinimize",
        "LBFGSB",
    ]

    def __init__(self):
        super().__init__()

    def instantiate_solver(
        self,
        solver_name: str,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
        solver_kwargs: Optional[dict] = None,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        self._check_solver(solver_name)
        solver = getattr(jaxopt, solver_name)(fun=loss, **solver_kwargs)

        def solver_run(
            init_params: Tuple[jnp.ndarray, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, X=X, y=y)

        return solver_run


class Ridge(Regularizer):
    allowed_solvers = [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG" "ScipyBoundedMinimize",
        "LBFGSB",
    ]

    def __init__(self, alpha: float):
        super().__init__()

    def penalization(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        return 0.5 * self.alpha * jnp.sum(jnp.power(params[0], 2)) / params[1].shape[0]

    def instantiate_solver(
        self,
        solver_name: str,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
        solver_kwargs: Optional[dict] = None,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        self._check_solver(solver_name)

        def penalized_loss(params, X, y):
            return loss(params, X, y) + self.penalization(params)

        solver = getattr(jaxopt, solver_name)(fun=penalized_loss, **solver_kwargs)

        def solver_run(
            init_params: Tuple[jnp.ndarray, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, X=X, y=y)

        return solver_run


class ProxGradientRegularizer(Regularizer, abc.ABC):
    allowed_solvers = ["ProximalGradient"]

    def __init__(self, alpha, mask: jnp.ndarray):
        super().__init__(alpha)
        if not jnp.all((mask == 1) | (mask == 0)):
            raise ValueError("mask must be an jnp.ndarray of 0s and 1s!")
        if jnp.any(jnp.sum(mask, axis=0) != 1):
            raise ValueError("Each feature must be assigned to a group!")
        self.mask = mask

    @abc.abstractmethod
    def prox_operator(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        l2reg: float,
        scaling: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass

    def instantiate_solver(
        self,
        solver_name: str,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
        solver_kwargs: Optional[dict] = None,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        self._check_solver(solver_name)

        if solver_kwargs is None:
            solver_kwargs = dict()

        solver = getattr(jaxopt, solver_name)(
            fun=loss, prox=self.prox_operator, **solver_kwargs
        )

        def solver_run(
            init_params: Tuple[jnp.ndarray, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, X=X, y=y, hyperparams_prox=self.alpha)

        return solver_run


class LassoRegularizer(ProxGradientRegularizer):
    def __init__(self, alpha, mask: jnp.ndarray):
        super().__init__(alpha=alpha, mask=mask)

    def prox_operator(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        alpha: float,
        scaling: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        Ws, bs = params
        return jaxopt.prox.prox_lasso(Ws, l1reg=alpha, scaling=scaling), bs


class GroupLassoRegularizer(ProxGradientRegularizer):
    def __init__(self, alpha, mask: jnp.ndarray):
        super().__init__(alpha=alpha, mask=mask)

    def prox_operator(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        alpha: float,
        scaling: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return prox_group_lasso(params, alpha=alpha, mask=self.mask, scaling=scaling)
