"""
A Module for Optimization with Various Regularization Schemes.

This module provides a series of classes that facilitate the optimization of models
with different types of regularizations. Each `Regularizer` class in this module interfaces
with various optimization methods, and they can be applied depending on the model's requirements.
"""

import abc
import inspect
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray

from . import tree_utils, utils
from .base_class import DESIGN_INPUT_TYPE, Base
from .proximal_operator import prox_group_lasso
from .pytrees import FeaturePytree

SolverRunner = Callable[
    [
        Tuple[
            jnp.ndarray, jnp.ndarray
        ],  # Model parameters (for now tuple, eventually pytree)
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    jaxopt.OptStep,
]

ProximalOperator = Callable[
    [
        Tuple[
            jnp.ndarray, jnp.ndarray
        ],  # Model parameters (for now tuple, eventually pytree)
        float,  # Regularizer strength (for now float, eventually pytree)
        float,
    ],  # Step-size for optimization (must be a float)
    Tuple[jnp.ndarray, jnp.ndarray],
]

__all__ = ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]


def __dir__() -> list[str]:
    return __all__


class Regularizer(Base, abc.ABC):
    """
    Abstract base class for regularized solvers.

    This class is designed to provide a consistent interface for optimization solvers,
    enabling users to easily switch between different regularizers, ensuring compatibility
    with various loss functions and optimization algorithms.

    Attributes
    ----------
    allowed_solvers :
        Tuple of solver names that are allowed for use with this regularizer.
    solver_name :
        Name of the solver being used.
    solver_kwargs :
        Additional keyword arguments to be passed to the solver during instantiation.
    """

    _allowed_solvers: Tuple[str] = tuple()

    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.solver_name = solver_name
        if solver_kwargs is None:
            solver_kwargs = dict()
        self.solver_kwargs = solver_kwargs

    @property
    def allowed_solvers(self):
        return self._allowed_solvers

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
        self._solver_kwargs = solver_kwargs

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
        if solver_name not in self.allowed_solvers:
            raise ValueError(
                f"Solver `{solver_name}` not allowed for "
                f"{self.__class__} regularization. "
                f"Allowed solvers are {self.allowed_solvers}."
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

    def instantiate_solver(
        self, loss: Callable, *args: Any, prox: Optional[Callable] = None, **kwargs: Any
    ) -> SolverRunner:
        """
        Instantiate the solver with the provided loss function.

        Parameters
        ----------
        loss :
            The loss function to be optimized.

        *args:
            Positional arguments for the jaxopt `solver.run` method, e.g. the regularizing
            strength for proximal gradient methods.

        prox:
            Optional, the proximal projection operator.

        *kwargs:
            Keyword arguments for the jaxopt `solver.run` method.

        Returns
        -------
        :
            A function that runs the solver with the provided loss and proximal operator.
        """
        # check that the loss is Callable
        utils.assert_is_callable(loss, "loss")

        # get the solver with given arguments.
        # The "fun" argument is not always the first one, but it is always KEYWORD
        # see jaxopt.EqualityConstrainedQP for example. The most general way is to pass it as keyword.
        # The proximal gradient is added to the kwargs if passed. This avoids issues with over-writing
        # the proximal operator.
        if "prox" in self.solver_kwargs:
            if prox is None:
                raise ValueError(
                    f"Regularizer of type {self.__class__.__name__} "
                    f"does not require a proximal operator!"
                )
            else:
                warnings.warn(
                    "Overwritten the user-defined proximal operator! "
                    "There is only one valid proximal operator for each regularizer type.",
                    UserWarning,
                )
        # update the kwargs if prox is passed
        if prox is not None:
            solver_kwargs = self.solver_kwargs.copy()
            solver_kwargs.update(prox=prox)
        else:
            solver_kwargs = self.solver_kwargs
        solver = getattr(jaxopt, self.solver_name)(fun=loss, **solver_kwargs)

        def solver_run(
            init_params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], *run_args: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, *args, *run_args, **kwargs)

        return solver_run


class UnRegularized(Regularizer):
    """
    Solver class for optimizing unregularized models.

    This class provides an interface to various optimization methods for models that
    do not involve regularization. The optimization methods that are allowed for this
    class are defined in the `allowed_solvers` attribute.

    Attributes
    ----------
    allowed_solvers : list of str
        List of solver names that are allowed for this regularizer class.

    See Also
    --------
    [Regularizer](./#nemos.regularizer.Regularizer) : Base solver class from which this class inherits.
    """

    _allowed_solvers = (
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
    )

    def __init__(
        self, solver_name: str = "GradientDescent", solver_kwargs: Optional[dict] = None
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)


class Ridge(Regularizer):
    """
    Solver for Ridge regularization using various optimization algorithms.

    This class uses `jaxopt` optimizers to perform Ridge regularization. It extends
    the base Solver class, with the added feature of Ridge penalization.

    Attributes
    ----------
    allowed_solvers : List[..., str]
        A list of solver names that are allowed to be used with this regularizer.
    """

    _allowed_solvers = (
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
    )

    def __init__(
        self,
        solver_name: str = "GradientDescent",
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)
        self.regularizer_strength = regularizer_strength

    def _penalization(
        self, params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]
    ) -> jnp.ndarray:
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

        def l2_penalty(coeff: jnp.ndarray, intercept: jnp.ndarray) -> jnp.ndarray:
            return (
                0.5
                * self.regularizer_strength
                * jnp.sum(jnp.power(coeff, 2))
                / intercept.shape[0]
            )

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            lambda x: l2_penalty(x, params[1]), sum, params[0]
        )

    def instantiate_solver(
        self, loss: Callable, *args: Any, **kwargs: Any
    ) -> SolverRunner:
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
        # this check has be performed here because the penalized loss will
        # always be a callable independently of which loss is passed!
        utils.assert_is_callable(loss, "loss")

        def penalized_loss(params, X, y):
            return loss(params, X, y) + self._penalization(params)

        return super().instantiate_solver(penalized_loss, *args, **kwargs)


class ProxGradientRegularizer(Regularizer, abc.ABC):
    """
    Abstract class for ptimization solvers using the Proximal Gradient method.

    This class utilizes the `jaxopt` library's Proximal Gradient optimizer. It extends
    the base Solver class, with the added functionality of a proximal operator.

    Attributes
    ----------
    allowed_solvers : List[...,str]
        A list of solver names that are allowed to be used with this regularizer.
    """

    _allowed_solvers = ("ProximalGradient",)

    def __init__(
        self,
        solver_name: str,
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)
        self.regularizer_strength = regularizer_strength

    @abc.abstractmethod
    def _get_proximal_operator(
        self,
    ) -> ProximalOperator:
        """
        Abstract method to retrieve the proximal operator for this solver.

        Returns
        -------
        :
            The proximal operator, which typically applies a form of regularization.
        """
        pass

    def instantiate_solver(
        self, loss: Callable, *args: Any, **kwargs: Any
    ) -> SolverRunner:
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
        return super().instantiate_solver(
            loss,
            self.regularizer_strength,
            *args,
            prox=self._get_proximal_operator(),
            **kwargs,
        )


class Lasso(ProxGradientRegularizer):
    """
     Optimization solver using the Lasso (L1 regularization) method with Proximal Gradient.

    This class is a specialized version of the ProxGradientSolver with the proximal operator
    set for L1 regularization (Lasso). It utilizes the `jaxopt` library's proximal gradient optimizer.
    """

    def __init__(
        self,
        solver_name: str = "ProximalGradient",
        solver_kwargs: Optional[dict] = None,
        regularizer_strength: float = 1.0,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)
        self.regularizer_strength = regularizer_strength

    def _get_proximal_operator(
        self,
    ) -> ProximalOperator:
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
            l1reg /= bs.shape[0]
            # if Ws is a pytree, l1reg needs to be a pytree with the same
            # structure
            if isinstance(Ws, (dict, FeaturePytree)):
                struct = jax.tree_util.tree_structure(Ws)
                l1reg = jax.tree_util.tree_unflatten(
                    struct, [l1reg] * struct.num_leaves
                )
            return jaxopt.prox.prox_lasso(Ws, l1reg, scaling=scaling), bs

        return prox_op


class GroupLasso(ProxGradientRegularizer):
    """
    Optimization solver using the Group Lasso regularization method with Proximal Gradient.

    This class is a specialized version of the ProxGradientSolver with the proximal operator
    set for Group Lasso regularization. The Group Lasso regularization induces sparsity on groups
    of features rather than individual features.

    Attributes
    ----------
    mask : Union[jnp.ndarray, NDArray]
        A 2d mask array indicating groups of features for regularization.
        Each row represents a group of features.
        Each column corresponds to a feature, where a value of 1 indicates that the feature belongs
        to the group, and a value of 0 indicates it doesn't.
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
        )
        self.regularizer_strength = regularizer_strength
        self.mask = jnp.asarray(mask)

    @property
    def mask(self):
        """Getter for the mask attribute."""
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

    def _get_proximal_operator(
        self,
    ) -> ProximalOperator:
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
