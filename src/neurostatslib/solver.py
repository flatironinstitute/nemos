"""
## A Module for Optimization with Various Regularizations.

This module provides a series of classes that facilitate the optimization of models
with different types of regularizations. Each solver class in this module interfaces
with various optimization methods, and they can be applied depending on the model's requirements.
"""
import abc
import inspect
from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray

from .base_class import Base
from .proximal_operator import prox_group_lasso

__all__ = ["UnRegularizedSolver", "RidgeSolver", "LassoSolver", "GroupLassoSolver"]


def __dir__() -> list[str]:
    return __all__


class Solver(Base, abc.ABC):
    """
    Abstract base class for optimization solvers.

    This class is designed to provide a consistent interface for optimization solvers,
    enabling users to easily switch between different solvers and ensure compatibility
    with various loss functions and regularization schemes.

    Attributes
    ----------
    allowed_optimizers :
        List of optimizer names that are allowed for use with this solver.
    solver_name :
        Name of the solver being used.
    solver_kwargs :
        Additional keyword arguments to be passed to the solver during instantiation.

    Methods
    -------
    instantiate_solver(loss) :
        Abstract method to instantiate a solver with a given loss function.
    get_runner(solver_kwargs, run_kwargs) :
        Get the solver runner with provided arguments.
    """

    allowed_optimizers: List[str] = []

    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._check_solver(solver_name)
        self._solver_name = solver_name
        if solver_kwargs is None:
            self._solver_kwargs = dict()
        else:
            self._solver_kwargs = solver_kwargs
        self._check_solver_kwargs(self.solver_name, self.solver_kwargs)

    @property
    def solver_name(self):
        return self._solver_name

    @solver_name.setter
    def solver_name(self, solver_name: str):
        self._check_solver(solver_name)
        self._solver_name = solver_name

    @property
    def solver_kwargs(self):
        return self._solver_kwargs

    @solver_kwargs.setter
    def solver_kwargs(self, solver_kwargs: dict):
        self._check_solver_kwargs(self.solver_name, solver_kwargs)
        return self._solver_kwargs

    def _check_solver(self, solver_name: str):
        """
        Ensure the provided solver name is allowed.

        Parameters
        ----------
        solver_name :
            Name of the solver to be checked.

        Raises
        ------
        ValueError
            If the provided solver name is not in the list of allowed optimizers.
        """
        if solver_name not in self.allowed_optimizers:
            raise ValueError(
                f"Solver `{solver_name}` not allowed for "
                f"{self.__class__} regularization. "
                f"Allowed solvers are {self.allowed_optimizers}."
            )

    @staticmethod
    def _check_solver_kwargs(solver_name, solver_kwargs):
        """
        Check if provided solver keyword arguments are valid.

        Parameters
        ----------
        solver_name :
            Name of the solver.
        solver_kwargs :
            Additional keyword arguments for the solver.

        Raises
        ------
        NameError
            If any of the solver keyword arguments are not valid.
        """
        solver_args = inspect.getfullargspec(getattr(jaxopt, solver_name)).args
        undefined_kwargs = set(solver_kwargs.keys()).difference(solver_args)
        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for jaxopt.{solver_name}!"
            )

    @staticmethod
    def _check_is_callable_from_jax(func: Callable):
        """
        Check if the provided function is callable and from the jax namespace.

        Ensures that the given function is not only callable, but also belongs to
        the `jax` namespace, ensuring compatibility and safety when using jax-based
        operations.

        Parameters
        ----------
        func :
            The function to check.

        Raises
        ------
        TypeError
            If the provided function is not callable.
        ValueError
            If the function does not belong to the `jax` or `neurostatslib.glm` namespaces.
        """
        if not callable(func):
            raise TypeError("The loss function must a Callable!")

        if (not hasattr(func, "__module__")) or (
            not (
                func.__module__.startswith("jax")
                or func.__module__.startswith("neurostatslib.glm")
            )
        ):
            raise ValueError(
                f"The function {func.__name__} is not from the jax namespace. "
                "Only functions from the jax namespace are allowed."
            )

    @abc.abstractmethod
    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        """Abstract method to instantiate a solver with a given loss function."""
        pass

    def get_runner(
        self,
        solver_kwargs: dict,
        run_kwargs: dict,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        """
        Get the solver runner with provided arguments.

        Parameters
        ----------
        solver_kwargs :
            Additional keyword arguments for the solver instantiation.
        run_kwargs :
            Additional keyword arguments for the solver run.

        Returns
        -------
        :
            The solver runner.
        """
        solver = getattr(jaxopt, self.solver_name)(**solver_kwargs)

        def solver_run(
            init_params: Tuple[jnp.ndarray, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, X=X, y=y, **run_kwargs)

        return solver_run


class UnRegularizedSolver(Solver):
    """
    Solver class for optimizing unregularized models.

    This class provides an interface to various optimization methods for models that
    do not involve regularization. The optimization methods that are allowed for this
    class are defined in the `allowed_optimizers` attribute.

    Attributes
    ----------
    allowed_optimizers : list of str
        List of optimizer names that are allowed for this solver class.

    Methods
    -------
    instantiate_solver(loss)
        Instantiates the optimization algorithm with the given loss function.

    See Also
    --------
    [Solver](./#neurostatslib.solver.Solver) : Base solver class from which this class inherits.
    """

    allowed_optimizers = [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
    ]

    def __init__(
        self, solver_name: str = "GradientDescent", solver_kwargs: Optional[dict] = None
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)

    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        """
        Instantiate the optimization algorithm for a given loss function.

        Parameters
        ----------
        loss :
            The loss function that needs to be minimized.

        Returns
        -------
        :
            A runner function that uses the specified optimization algorithm
            to minimize the given loss function.
        """
        self._check_is_callable_from_jax(loss)
        solver_kwargs = self.solver_kwargs.copy()
        solver_kwargs["fun"] = loss
        return self.get_runner(solver_kwargs, {})


class RidgeSolver(Solver):
    """
    Solver for Ridge regularization using various optimization algorithms.

    This class uses `jaxopt` optimizers to perform Ridge regularization. It extends
    the base Solver class, with the added feature of Ridge penalization.

    Attributes
    ----------
    allowed_optimizers : List[..., str]
        A list of optimizer names that are allowed to be used with this solver.
    """

    allowed_optimizers = [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
    ]

    def __init__(
        self,
        solver_name: str = "GradientDescent",
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)
        self.regularizer_strength = regularizer_strength

    def penalization(self, params: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute the Ridge penalization for given parameters.

        Parameters
        ----------
        params :
            Model parameters for which to compute the penalization.

        Returns
        -------
        float
            The Ridge penalization value.
        """
        return (
            0.5
            * self.regularizer_strength
            * jnp.sum(jnp.power(params[0], 2))
            / params[1].shape[0]
        )

    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        """
        Instantiate the solver with a penalized loss function.

        Parameters
        ----------
        loss :
            The original loss function to be optimized.

        Returns
        -------
        Callable
            A function that runs the solver with the penalized loss.
        """
        self._check_is_callable_from_jax(loss)

        def penalized_loss(params, X, y):
            return loss(params, X, y) + self.penalization(params)

        solver_kwargs = self.solver_kwargs.copy()
        solver_kwargs["fun"] = penalized_loss
        return self.get_runner(solver_kwargs, {})


class ProxGradientSolver(Solver, abc.ABC):
    """
    Solver for optimization using the Proximal Gradient method.

    This class utilizes the `jaxopt` library's Proximal Gradient optimizer. It extends
    the base Solver class, with the added functionality of a proximal operator.

    Attributes
    ----------
    allowed_optimizers : List[...,str]
        A list of optimizer names that are allowed to be used with this solver.
    mask : Optional[Union[NDArray, jnp.ndarray]]
        An optional mask array for element-wise operations. Shape (n_groups, n_features)
    """

    allowed_optimizers = ["ProximalGradient"]

    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
        mask: Optional[Union[NDArray, jnp.ndarray]] = None,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)
        self.mask = mask
        self.regularizer_strength = regularizer_strength

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask: jnp.ndarray):
        self._check_mask(mask)
        self._mask = mask

    @staticmethod
    def _check_mask(mask: jnp.ndarray):
        """
        Validate the mask array.

        This method ensures the mask adheres to requirements:
        - It should be 2-dimensional.
        - Each element must be either 0 or 1.
        - Each feature should belong to only one group.
        - The mask should not be empty.
        - The mask is an array of float type.

        Raises
        ------
        ValueError
            If any of the above conditions are not met.
        """
        if mask.ndim != 2:
            raise ValueError(
                "`mask` must be 2-dimensional. "
                f"{mask.ndim} dimensional mask provided instead!"
            )

        if mask.shape[0] == 0:
            raise ValueError(f"Empty mask provided! Mask has shape {mask.shape}.")

        if jnp.any((mask != 1) & (mask != 0)):
            raise ValueError("Mask elements be 0s and 1s!")

        if mask.sum() == 0:
            raise ValueError("Empty mask provided!")

        if jnp.any(mask.sum(axis=0) > 1):
            raise ValueError(
                "Incorrect group assignment. Some of the features are assigned "
                "to more then one group."
            )

        if not jnp.issubdtype(mask.dtype, jnp.floating):
            raise ValueError(
                "Mask should be a floating point jnp.ndarray. "
                f"Data type {mask.dtype} provided instead!"
            )

    @abc.abstractmethod
    def get_prox_operator(
        self,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], float, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        """
        Abstract method to retrieve the proximal operator for this solver.

        Returns
        -------
        :
            The proximal operator, which typically applies a form of regularization.
        """
        pass

    def instantiate_solver(
        self,
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jaxopt.OptStep
    ]:
        """
        Instantiate the solver with the provided loss function and proximal operator.

        Parameters
        ----------
        loss :
            The original loss function to be optimized.

        Returns
        -------
        :
            A function that runs the solver with the provided loss and proximal operator.
        """
        self._check_is_callable_from_jax(loss)

        solver_kwargs = self.solver_kwargs.copy()
        solver_kwargs["fun"] = loss
        solver_kwargs["prox"] = self.get_prox_operator()

        run_kwargs = dict(hyperparams_prox=self.regularizer_strength)

        return self.get_runner(solver_kwargs, run_kwargs)


class LassoSolver(ProxGradientSolver):
    """
    Solver for optimization using the Lasso (L1 regularization) method with Proximal Gradient.

    This class is a specialized version of the ProxGradientSolver with the proximal operator
    set for L1 regularization (Lasso). It utilizes the `jaxopt` library's proximal gradient optimizer.
    """

    def __init__(
        self,
        solver_name: str = "ProximalGradient",
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
        mask: Optional[Union[NDArray, jnp.ndarray]] = None,
    ):
        super().__init__(
            solver_name,
            solver_kwargs=solver_kwargs,
            mask=mask,
        )
        self.regularizer_strength = regularizer_strength

    def get_prox_operator(
        self,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], float, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        """
        Retrieve the proximal operator for Lasso regularization (L1 penalty).

        Returns
        -------
        :
            The proximal operator, applying L1 regularization to the provided parameters. The intercept
            term is not regularized.
        """

        def prox_op(params, l1reg, scaling=1.0):
            Ws, bs = params
            return jaxopt.prox.prox_lasso(Ws, l1reg, scaling=scaling), bs

        return prox_op


class GroupLassoSolver(ProxGradientSolver):
    """
    Solver for optimization using the Group Lasso regularization method with Proximal Gradient.

    This class is a specialized version of the ProxGradientSolver with the proximal operator
    set for Group Lasso regularization. The Group Lasso regularization induces sparsity on groups
    of features rather than individual features.

    Attributes
    ----------
    mask : Union[jnp.ndarray, NDArray]
        A mask array indicating groups of features for regularization.
        Each row represents a group of features.
        Each column corresponds to a feature, where a value of 1 indicates that the feature belongs
        to the group, and a value of 0 indicates it doesn't.

    Methods
    -------
    _check_mask():
        Validate the mask array to ensure it meets the requirements for Group Lasso regularization.
    get_prox_operator():
        Retrieve the proximal operator for Group Lasso regularization.
    """

    def __init__(
        self,
        solver_name: str,
        mask: Union[NDArray, jnp.ndarray],
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
    ):
        super().__init__(
            solver_name,
            solver_kwargs=solver_kwargs,
            mask=mask,
        )
        self.regularizer_strength = regularizer_strength
        mask = jnp.asarray(mask)
        self._check_mask(mask)

    def get_prox_operator(
        self,
    ) -> Callable[
        [Tuple[jnp.ndarray, jnp.ndarray], float, float], Tuple[jnp.ndarray, jnp.ndarray]
    ]:
        """
        Retrieve the proximal operator for Group Lasso regularization.

        Returns
        -------
        :
            The proximal operator, applying Group Lasso regularization to the provided parameters. The
            intercept term is not regularized.
        """

        def prox_op(params, regularizer_strength, scaling=1.0):
            return prox_group_lasso(
                params, regularizer_strength, mask=self.mask, scaling=scaling
            )

        return prox_op
