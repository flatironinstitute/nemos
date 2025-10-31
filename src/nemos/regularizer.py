"""
A Module for Optimization with Various Regularization Schemes.

This module provides a series of classes that facilitate the optimization of models
with different types of regularizations. Each `Regularizer` class in this module interfaces
with various optimization methods, and they can be applied depending on the model's requirements.
"""

import abc
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from nemos.third_party.jaxopt import jaxopt

from . import tree_utils
from .base_class import Base
from .proximal_operator import prox_elastic_net, prox_group_lasso
from .typing import DESIGN_INPUT_TYPE, ProximalOperator
from .utils import format_repr

__all__ = ["UnRegularized", "Ridge", "Lasso", "GroupLasso", "ElasticNet"]


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
    allowed_solvers : Tuple[str]
        Tuple of solver names that are allowed for use with this regularizer.
    default_solver : str
        String of the default solver name allowed for use with this regularizer.
    """

    _allowed_solvers: Tuple[str] = tuple()
    _default_solver: str = None

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @property
    def allowed_solvers(self) -> Tuple[str]:
        return self._allowed_solvers

    @property
    def default_solver(self) -> str:
        return self._default_solver

    @abc.abstractmethod
    def penalized_loss(self, loss: Callable, regularizer_strength: float) -> Callable:
        """
        Abstract method to penalize loss functions.

        Parameters
        ----------
        loss :
            Callable loss function.
        regularizer_strength :
            Float the indicates the regularization strength.

        Returns
        -------
        :
            A modified version of the loss function including any relevant penalization based on the regularizer
            type.
        """
        pass

    @abc.abstractmethod
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

    def check_solver(self, solver_name: str):
        """Raise an error if the given solver is not allowed."""
        if solver_name not in self._allowed_solvers:
            raise ValueError(
                f"The solver: {solver_name} is not allowed for "
                f"{self.__class__.__name__} regularization. Allowed solvers are "
                f"{self.allowed_solvers}."
            )

    def __repr__(self):
        return format_repr(self)

    def _validate_regularizer_strength(self, strength: Union[None, float]):
        if strength is None:
            strength = 1.0
        else:
            try:
                # force conversion to float to prevent weird GPU issues
                strength = float(strength)
            except ValueError:
                # raise a more detailed ValueError
                raise ValueError(
                    f"Could not convert the regularizer strength: {strength} to a float."
                )
        return strength


class UnRegularized(Regularizer):
    """
    Regularizer class for unregularized models.

    This class equips models with the identity proximal operator (no shrinkage) and the
    unpenalized loss function.
    """

    _allowed_solvers = (
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "NonlinearCG",
        "ProximalGradient",
        "SVRG",
        "ProxSVRG",
    )

    _default_solver = "GradientDescent"

    def __init__(
        self,
    ):
        super().__init__()

    def penalized_loss(self, loss: Callable, regularizer_strength: float):
        """
        Return the original loss function unpenalized.

        Unregularized regularization method does not add any penalty.
        """
        return loss

    def get_proximal_operator(
        self,
    ) -> ProximalOperator:
        """
        Return the identity operator.

        Unregularized method corresponds to an identity proximal operator, since no
        shrinkage factor is applied.
        """
        return jaxopt.prox.prox_none

    def _validate_regularizer_strength(self, strength: None):
        return None


class Ridge(Regularizer):
    """
    Regularizer class for Ridge (L2 regularization).

    This class equips models with the Ridge proximal operator and the
    Ridge penalized loss function.
    """

    _allowed_solvers = (
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "NonlinearCG",
        "ProximalGradient",
        "SVRG",
        "ProxSVRG",
    )

    _default_solver = "GradientDescent"

    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def _penalization(
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], regularizer_strength: float
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
                * regularizer_strength
                * jnp.sum(jnp.power(coeff, 2))
                / intercept.shape[0]
            )

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            lambda x: l2_penalty(x, params[1]), sum, params[0]
        )

    def penalized_loss(self, loss: Callable, regularizer_strength: float) -> Callable:
        """Return the penalized loss function for Ridge regularization."""

        def _penalized_loss(params, X, y):
            return loss(params, X, y) + self._penalization(params, regularizer_strength)

        return _penalized_loss

    def get_proximal_operator(
        self,
    ) -> ProximalOperator:
        """
        Retrieve the proximal operator for Ridge regularization (L2 penalty).

        Returns
        -------
        :
            The proximal operator, applying L2 regularization to the provided parameters. The intercept
            term is not regularized.
        """

        def prox_op(params, l2reg, scaling=1.0):
            Ws, bs = params
            l2reg /= bs.shape[0]
            return jaxopt.prox.prox_ridge(Ws, l2reg, scaling=scaling), bs

        return prox_op


class Lasso(Regularizer):
    """
    Regularizer class for Lasso (L1 regularization).

    This class equips models with the Lasso proximal operator and the
    Lasso penalized loss function.
    """

    _allowed_solvers = (
        "ProximalGradient",
        "ProxSVRG",
    )

    _default_solver = "ProximalGradient"

    def __init__(
        self,
    ):
        super().__init__()

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
            l1reg = jax.tree_util.tree_map(lambda x: l1reg * jnp.ones_like(x), Ws)
            return jaxopt.prox.prox_lasso(Ws, l1reg, scaling=scaling), bs

        return prox_op

    @staticmethod
    def _penalization(
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], regularizer_strength: float
    ) -> jnp.ndarray:
        """
        Compute the Lasso penalization for given parameters.

        Parameters
        ----------
        params :
            Model parameters for which to compute the penalization.

        Returns
        -------
        float
            The Lasso penalization value.
        """

        def l1_penalty(coeff: jnp.ndarray, intercept: jnp.ndarray) -> jnp.ndarray:
            return regularizer_strength * jnp.sum(jnp.abs(coeff)) / intercept.shape[0]

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            lambda x: l1_penalty(x, params[1]), sum, params[0]
        )

    def penalized_loss(self, loss: Callable, regularizer_strength: float) -> Callable:
        """Return a function for calculating the penalized loss using Lasso regularization."""

        def _penalized_loss(params, X, y):
            return loss(params, X, y) + self._penalization(params, regularizer_strength)

        return _penalized_loss


class ElasticNet(Regularizer):
    r"""
    Regularizer class for Elastic Net (L1 + L2 regularization).

    The Elasitc Net penalty [3]_ [4]_ is defined as:

    .. math::
        P(\beta) = \alpha \left((1 - \lambda) \frac{1}{2} ||\beta||_{\ell_2}^2 +
        \lambda ||\beta||_{\ell_1} \right)

    where :math:`\alpha` is the regularizer strength, and :math:`\lambda` is the regularizer ratio.
    The regularizer ratio controls the balance between L1 (Lasso) and L2 (Ridge)
    regularization, where :math:`\lambda = 0` is equivalent to Ridge regularization and
    :math:`\lambda = 1` is equivalent to Lasso regularization.

    This class equips models with the Elastic Net proximal operator and the
    Elastic Net penalized loss function.

    References
    ----------
    .. [3] Zou, H., & Hastie, T. (2005).
        Regularization and variable selection via the elastic net.
        Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301-320.
        https://doi.org/10.1111/j.1467-9868.2005.00503.x

    .. [4] https://en.wikipedia.org/wiki/Elastic_net_regularization
    """

    _allowed_solvers = (
        "ProximalGradient",
        "ProxSVRG",
    )

    _default_solver = "ProximalGradient"

    def __init__(
        self,
    ):
        super().__init__()

    def get_proximal_operator(
        self,
    ) -> ProximalOperator:
        """
        Retrieve the proximal operator for Elastic Net regularization (L1 + L2 penalty).

        Returns
        -------
        :
            The proximal operator, applying L1 + L2 regularization to the provided parameters. The intercept
            term is not regularized.
        """

        def prox_op(params, netreg, scaling=1.0):
            Ws, bs = params
            # since we do not allow array regularization assume we pass a tuple
            regularizer_strength, regularizer_ratio = netreg
            regularizer_strength /= bs.shape[0]
            lam = regularizer_strength * regularizer_ratio  # hyperparams[0]
            gam = (1 - regularizer_ratio) / regularizer_ratio  # hyperparams[1]
            # if Ws is a pytree, netreg needs to be a pytree with the same
            # structure
            lam = jax.tree_util.tree_map(lambda x: lam * jnp.ones_like(x), Ws)
            gam = jax.tree_util.tree_map(lambda x: gam * jnp.ones_like(x), Ws)
            return prox_elastic_net(Ws, (lam, gam), scaling=scaling), bs

        return prox_op

    @staticmethod
    def _penalization(
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        net_regularization: Tuple[float, float],
    ) -> jnp.ndarray:
        r"""
        Compute the Elastic Net penalization for given parameters.

        The elastic net penalty is defined as:

        .. math::
            P(\beta) = \alpha ((1 - \lambda) \frac{1}{2} ||\beta||_{\ell_2}^2 +
            \lambda ||\beta||_{\ell_1}

        where :math:`\alpha` is the regularizer strength, and :math:`\lambda` is the regularizer ratio.
        The regularizer ratio controls the balance between L1 (Lasso) and L2 (Ridge)
        regularization, where :math:`\lambda = 0` is equivalent to Ridge regularization and
        :math:`\lambda = 1` is equivalent to Lasso regularization.

        Parameters
        ----------
        params :
            Model parameters for which to compute the penalization.

        Returns
        -------
        :
            The Elastic Net penalization value.
        """

        def net_penalty(coeff: jnp.ndarray, intercept: jnp.ndarray) -> jnp.ndarray:
            regularizer_strength, regularizer_ratio = net_regularization
            return (
                regularizer_strength
                * (
                    0.5 * (1 - regularizer_ratio) * jnp.sum(jnp.power(coeff, 2))
                    + regularizer_ratio * jnp.sum(jnp.abs(coeff))
                )
                / intercept.shape[0]
            )

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            lambda x: net_penalty(x, params[1]), sum, params[0]
        )

    def penalized_loss(
        self, loss: Callable, regularizer_strength: Tuple[float, float]
    ) -> Callable:
        """Return a function for calculating the penalized loss using Elastic Net regularization."""

        def _penalized_loss(params, X, y):
            return loss(params, X, y) + self._penalization(params, regularizer_strength)

        return _penalized_loss

    def _validate_regularizer_strength(
        self, strength: Union[None, float, Tuple[float, float]]
    ):
        if strength is None:
            strength = (1.0, 0.5)
        elif hasattr(strength, "__len__") is False:
            try:
                # force conversion to float to prevent weird GPU issues
                strength = (float(strength), 0.5)
            except ValueError:
                # raise a more detailed ValueError
                raise ValueError(
                    f"Could not convert the regularizer strength: {strength} to a float."
                )
        else:
            try:
                # force conversion to float to prevent weird GPU issues
                strength = jax.tree_util.tree_map(float, tuple(strength))
            except ValueError:
                # raise a more detailed ValueError
                raise ValueError(
                    f"Could not convert the regularizer strength and regularizer ratio: {strength} to a tuple of "
                    "floats."
                )
            if len(strength) != 2:
                raise ValueError(
                    f"Invalid regularization strength and regularizer ratio: {strength}. regularizer_strength must "
                    "be a tuple of two floats."
                )
            if (strength[1] > 1) | (strength[1] < 0):
                raise ValueError(
                    f"Invalid regularization ratio: {strength[1]}. Regularization ratio must be a number between "
                    "0 and 1."
                )
            elif strength[1] == 0:
                raise ValueError(
                    "Regularization ratio of 0 is not supported. Use Ridge regularization instead."
                )

        return strength


class GroupLasso(Regularizer):
    """
    Regularizer class for Group Lasso (group-L1) regularized models.

    This class equips models with the group-lasso proximal operator and the
    group-lasso penalized loss function.

    Attributes
    ----------
    mask :
        A 2d mask array indicating groups of features for regularization, shape ``(num_groups, num_features)``.
        Each row represents a group of features.
        Each column corresponds to a feature, where a value of 1 indicates that the feature belongs
        to the group, and a value of 0 indicates it doesn't.
        Default is ``mask = np.ones((1, num_features))``, grouping all features in a single group.

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
    >>> group_lasso = GroupLasso(mask=mask)
    >>> # fit a group-lasso glm
    >>> model = GLM(regularizer=group_lasso, regularizer_strength=0.1).fit(X, y)
    >>> print(f"coeff shape: {model.coef_.shape}")
    coeff shape: (5,)
    """

    _allowed_solvers = (
        "ProximalGradient",
        "ProxSVRG",
    )

    _default_solver = "ProximalGradient"

    def __init__(
        self,
        mask: Union[NDArray, jnp.ndarray] = None,
    ):
        super().__init__()

        self.mask = mask

    @property
    def mask(self):
        """Getter for the mask attribute."""
        return self._mask

    @mask.setter
    def mask(self, mask: Union[jnp.ndarray, None]):
        """Setter for the mask attribute."""
        # check mask if passed by user, else will be initialized later
        if mask is not None:
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
                "to more than one group."
            )

        if not jnp.issubdtype(mask.dtype, jnp.floating):
            raise ValueError(
                "Mask should be a floating point jnp.ndarray. "
                f"Data type {mask.dtype} provided instead!"
            )

    def _penalization(
        self, params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], regularizer_strength: float
    ) -> jnp.ndarray:
        r"""
        Calculate the penalization.

        Note: the penalty is being calculated according to the following formula:

        .. math::

            \\text{loss}(\beta_1,...,\beta_g) + \alpha \cdot \sum _{j=1...,g} \sqrt{\dim(\beta_j)} || \beta_j||_2

        where :math:`g` is the number of groups, :math:`\dim(\cdot)` is the dimension of the vector,
        i.e. the number of coefficient in each :math:`\beta_j`, and :math:`||\cdot||_2` is the euclidean norm.
        """
        # conform to shape (1, n_features) if param is (n_features,) or (n_neurons, n_features) if
        # param is (n_features, n_neurons)
        param_with_extra_axis = jnp.atleast_2d(params[0].T)

        vec_prod = jax.vmap(
            lambda x: self.mask * x, in_axes=0, out_axes=2
        )  # this vectorizes the product over the neurons, and adds the neuron axis as the last axis

        masked_param = vec_prod(
            param_with_extra_axis
        )  # this masks the param, (group, feature, neuron)

        penalty = jax.numpy.sum(
            jax.numpy.linalg.norm(masked_param, axis=1).T
            * jax.numpy.sqrt(self.mask.sum(axis=1))
        )

        # divide regularization strength by number of neurons
        regularizer_strength = regularizer_strength / params[1].shape[0]

        return penalty * regularizer_strength

    def penalized_loss(self, loss: Callable, regularizer_strength: float) -> Callable:
        """Return a function for calculating the penalized loss using Group Lasso regularization."""

        def _penalized_loss(params, X, y):
            return loss(params, X, y) + self._penalization(params, regularizer_strength)

        return _penalized_loss

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
