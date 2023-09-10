import abc
import inspect
from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray

from .base_class import _Base
from .proximal_operator import prox_group_lasso


class Solver(_Base, abc.ABC):
    allowed_solvers = []

    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._check_solver(solver_name)
        self.alpha = alpha
        self.solver_name = solver_name
        if solver_kwargs is None:
            self.solver_kwargs = dict()
        else:
            self.solver_kwargs = solver_kwargs
        self._check_solver_kwargs(self.solver_name, self.solver_kwargs)

    def _check_solver(self, solver_name: str):
        if solver_name not in self.allowed_solvers:
            raise ValueError(
                f"Solver `{solver_name}` not allowed for "
                f"{self.__class__} regularization. "
                f"Allowed solvers are {self.allowed_solvers}."
            )

    @staticmethod
    def _check_solver_kwargs(solver_name, solver_kwargs):
        solver_args = inspect.getfullargspec(getattr(jaxopt, solver_name)).args
        undefined_kwargs = set(solver_kwargs.keys()).difference(solver_args)
        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for jaxopt.{solver_name}!"
            )

    @staticmethod
    def _check_is_callable(func):
        if not callable(func):
            raise TypeError("The loss function must a Callable!")

    @abc.abstractmethod
    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        pass

    def get_runner(
        self,
        solver_kwargs: dict,
        run_kwargs: dict,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        solver = getattr(jaxopt, self.solver_name)(**solver_kwargs)

        def solver_run(
            init_params: Tuple[jnp.ndarray, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, X=X, y=y, **run_kwargs)

        return solver_run


class UnRegularizedSolver(Solver):
    allowed_solvers = [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
    ]

    def __init__(self, solver_name: str, solver_kwargs: Optional[dict] = None):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)

    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        solver_kwargs = self.solver_kwargs.copy()
        solver_kwargs["fun"] = loss
        return self.get_runner(solver_kwargs, {})


class RidgeSolver(Solver):
    allowed_solvers = [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
    ]

    def __init__(
        self, solver_name: str, solver_kwargs: Optional[dict] = None, alpha: float = 1.0
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs, alpha=alpha)

    def penalization(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        return 0.5 * self.alpha * jnp.sum(jnp.power(params[0], 2)) / params[1].shape[0]

    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        self._check_is_callable(loss)

        def penalized_loss(params, X, y):
            return loss(params, X, y) + self.penalization(params)

        solver_kwargs = self.solver_kwargs.copy()
        solver_kwargs["fun"] = penalized_loss
        return self.get_runner(solver_kwargs, {})


class ProxGradientSolver(Solver, abc.ABC):
    allowed_solvers = ["ProximalGradient"]

    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        alpha: float = 1.0,
        mask: Optional[Union[NDArray, jnp.ndarray]] = None,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs, alpha=alpha)
        self.mask = mask

    @abc.abstractmethod
    def get_prox_operator(
        self,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], float, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        pass

    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        self._check_is_callable(loss)

        solver_kwargs = self.solver_kwargs.copy()
        solver_kwargs["fun"] = loss
        solver_kwargs["prox"] = self.get_prox_operator()

        run_kwargs = dict(hyperparams_prox=self.alpha)

        return self.get_runner(solver_kwargs, run_kwargs)


class LassoSolver(ProxGradientSolver):
    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        alpha: float = 1.0,
        mask: Optional[Union[NDArray, jnp.ndarray]] = None,
    ):
        super().__init__(
            solver_name, solver_kwargs=solver_kwargs, alpha=alpha, mask=mask
        )

    def get_prox_operator(
        self,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], float, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        def prox_op(params, l1reg, scaling=1.0):
            Ws, bs = params
            return jaxopt.prox.prox_lasso(Ws, l1reg, scaling=scaling), bs

        return prox_op


class GroupLassoSolver(ProxGradientSolver):
    def __init__(
        self,
        solver_name: str,
        mask: Union[jnp.ndarray, NDArray],
        solver_kwargs: Optional[dict] = None,
        alpha: float = 1.0,
    ):
        super().__init__(
            solver_name, solver_kwargs=solver_kwargs, alpha=alpha, mask=mask
        )
        self._check_mask()

    def _check_mask(self):
        if self.mask.ndim != 2:
            raise ValueError(
                "`mask` must be 2-dimensional. "
                f"{self.mask.ndim} dimensional mask provided instead!"
            )

        if self.mask.shape[0] == 0:
            raise ValueError(f"Empty mask provided! Mask has shape {self.mask.shape}.")

        if jnp.any((self.mask != 1) & (self.mask != 0)):
            raise ValueError("Mask elements be 0s and 1s!")

        if self.mask.sum() == 0:
            raise ValueError("Empty mask provided!")

        if jnp.any(self.mask.sum(axis=0) > 1):
            raise ValueError(
                "Incorrect group assignment. Some of the features are assigned "
                "to more then one group."
            )

        if not jnp.issubdtype(self.mask.dtype, jnp.floating):
            raise ValueError(
                "Mask should be a floating point jnp.ndarray. "
                f"Data type {self.mask.dtype} provided instead!"
            )

    def get_prox_operator(
        self,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], float, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        def prox_op(params, alpha, scaling=1.0):
            return prox_group_lasso(params, alpha, mask=self.mask, scaling=scaling)

        return prox_op
