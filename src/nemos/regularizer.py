"""
A Module for Optimization with Various Regularization Schemes.

This module provides a series of classes that facilitate the optimization of models
with different types of regularizations. Each `Regularizer` class in this module interfaces
with various optimization methods, and they can be applied depending on the model's requirements.
"""

import abc
import inspect
import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray

from . import tree_utils, utils
from .base_class import DESIGN_INPUT_TYPE, Base
from .proximal_operator import prox_group_lasso
from .pytrees import FeaturePytree

ProximalOperator = Callable[
    [
        Any,  # parameters, could be any pytree
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

    Parameters
    ----------
    regularizer_strength
        Float representing the strength of the regularization being applied.
        Default 1.0.

    Attributes
    ----------
    allowed_solvers :
        Tuple of solver names that are allowed for use with this regularizer.
    default_solver :
        String of the default solver name allowed for use with this regularizer.
    """

    _allowed_solvers: Tuple[str] = tuple()
    _default_solver: str = None

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # default regularizer strength
        self.regularizer_strength = 1.0

    @property
    def allowed_solvers(self):
        return self._allowed_solvers

    @property
    def default_solver(self):
        return self._default_solver

    @property
    def regularizer_strength(self) -> float:
        return self._regularizer_strength

    @regularizer_strength.setter
    def regularizer_strength(self, strength: float):
        self._regularizer_strength = strength

    def penalized_loss(self, loss: Callable) -> Callable:
        """
        Abstract method to penalize loss functions.

        Parameters
        ----------
        loss :
            Callable loss function.

        Returns
        -------
        :
            A modified version of the loss function including any relevant penalization based on the regularizer
            type.
        """
        pass

    def get_proximal_operator(
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
    default_solver :
        Default solver for this regularizer is GradientDescent.

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
        "ProximalGradient"
    )

    _default_solver = "GradientDescent"

    def __init__(
            self,
    ):
        super().__init__()

    def penalized_loss(self, loss: Callable):
        """Unregularized method does not add any penalty."""
        return loss

    def get_proximal_operator(self, ) -> ProximalOperator:
        """Unregularized method has no proximal operator."""

        def prox_op(params, hyperparams, scaling=1.0):
            Ws, bs = params
            return jaxopt.prox.prox_none(Ws, hyperparams, scaling=scaling), bs

        return prox_op


class Ridge(Regularizer):
    """
    Solver for Ridge regularization using various optimization algorithms.

    This class uses `jaxopt` optimizers to perform Ridge regularization. It extends
    the base Solver class, with the added feature of Ridge penalization.

    Parameters
    ----------
    regularizer_strength :
        Indicates the strength of the penalization being applied.
        Float with default value of 1.0.

    Attributes
    ----------
    allowed_solvers : List[..., str]
        A list of solver names that are allowed to be used with this regularizer.
    default_solver :
        Default solver for this regularizer is GradientDescent.
    """

    _allowed_solvers = (
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
        "ProximalGradient"
    )

    _default_solver = "GradientDescent"

    def __init__(
            self,
            regularizer_strength: float = 1.0,
    ):
        super().__init__()
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

    def penalized_loss(self, loss: Callable) -> Callable:
        def _penalized_loss(params, X, y):
            return loss(params, X, y) + self._penalization(params)

        return _penalized_loss

    def get_proximal_operator(
            self,
    ) -> ProximalOperator:
        def prox_op(params, l2reg, scaling=1.0):
            Ws, bs = params
            l2reg /= bs.shape[0]
            # if Ws is a pytree, l2reg needs to be a pytree with the same
            # structure
            if isinstance(Ws, (dict, FeaturePytree)):
                struct = jax.tree_util.tree_structure(Ws)
                l2reg = jax.tree_util.tree_unflatten(
                    struct, [l2reg] * struct.num_leaves
                )
            return jaxopt.prox.prox_lasso(Ws, l2reg, scaling=scaling), bs

        return prox_op


class Lasso(Regularizer):
    """
    Optimization solver using the Lasso (L1 regularization) method with Proximal Gradient.

    This class is a specialized version of the ProxGradientSolver with the proximal operator
    set for L1 regularization (Lasso). It utilizes the `jaxopt` library's proximal gradient optimizer.
    """

    _allowed_solvers = ("ProximalGradient,")

    _default_solver = "ProximalGradient"

    def __init__(
            self,
            regularizer_strength: float = 1.0,
    ):
        super().__init__()
        self.regularizer_strength = regularizer_strength

    def get_proximal_operator(
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

    def penalized_loss(self, loss: Callable) -> Callable:
        return loss


class GroupLasso(Regularizer):
    """
    Optimization solver using the Group Lasso regularization method with Proximal Gradient.

    This class is a specialized version of the ProxGradientSolver with the proximal operator
    set for Group Lasso regularization. The Group Lasso regularization induces sparsity on groups
    of features rather than individual features.

    Attributes
    ----------
    mask : Union[jnp.ndarray, NDArray]
        A 2d mask array indicating groups of features for regularization, shape (num_groups, num_features).
        Each row represents a group of features.
        Each column corresponds to a feature, where a value of 1 indicates that the feature belongs
        to the group, and a value of 0 indicates it doesn't.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.regularizer import GroupLasso  # Assuming the module is named group_lasso
    >>> from nemos.glm import GLM

    >>> # simulate some counts
    >>> num_samples, num_features, num_groups = 1000, 5, 3
    >>> X = np.random.normal(size=(num_samples, num_features)) # design matrix
    >>> w = [0, 0.5, 1, 0, -0.5] # define some weights
    >>> y = np.random.poisson(np.exp(X.dot(w))) # observed counts

    >>> # Define a mask for 3 groups and 5 features
    >>> mask = np.zeros((num_groups, num_features))
    >>> mask[0] = [1, 0, 0, 1, 0]  # Group 0 includes features 0 and 3
    >>> mask[1] = [0, 1, 0, 0, 0]  # Group 1 includes features 1
    >>> mask[2] = [0, 0, 1, 0, 1]  # Group 2 includes features 2 and 4

    >>> # Create the GroupLasso regularizer instance
    >>> group_lasso = GroupLasso(regularizer_strength=0.1, mask=mask)
    >>> # fit a group-lasso glm
    >>> model = GLM(regularizer=group_lasso).fit(X, y)
    >>> print(f"coeff: {model.coef_}")
    """

    _allowed_solvers = ("ProximalGradient,")

    _default_solver = "ProximalGradient"

    def __init__(
            self,
            mask: Union[NDArray, jnp.ndarray] = None,
            regularizer_strength: float = 1.0,
    ):
        super().__init__()
        self.regularizer_strength = regularizer_strength

        if mask is not None:
            self.mask = jnp.asarray(mask)
        else:
            # default mask if None is a singular group
            self.mask = jnp.asarray([[1.0]])

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

    def penalized_loss(self, loss: Callable) -> Callable:
        return loss

    def get_proximal_operator(
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
