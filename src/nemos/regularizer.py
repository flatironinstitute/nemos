"""
A Module for Optimization with Various Regularization Schemes.

This module provides a series of classes that facilitate the optimization of models
with different types of regularizations. Each `Regularizer` class in this module interfaces
with various optimization methods, and they can be applied depending on the model's requirements.
"""

import abc
import math
from typing import Any, Callable, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import tree_utils
from .base_class import Base
from .proximal_operator import (
    compute_normalization,
    masked_norm_2,
    prox_elastic_net,
    prox_group_lasso,
    prox_lasso,
    prox_none,
    prox_ridge,
)
from .tree_utils import pytree_map_and_reduce
from .typing import (
    DESIGN_INPUT_TYPE,
    ElasticNetRegularizerStrength,
    ModelParamsT,
    ProximalOperator,
    RegularizerStrength,
)
from .utils import format_repr
from .validation import convert_tree_leaves_to_jax_array

__all__ = ["UnRegularized", "Ridge", "Lasso", "GroupLasso", "ElasticNet"]


def __dir__() -> list[str]:
    return __all__


def apply_operator(func, params, *args, filter_kwargs=None, **kwargs):
    """
    Apply an operator to all regularizable subtrees of a parameter pytree.

    This function iterates over all locations returned by
    ``params.regularizable_subtrees()`` and applies ``func`` to each selected
    subtree. The updated values are written back into ``params`` using
    :func:`equinox.tree_at`. Typical use cases include applying proximal
    operators or other transformations to parameter tensors while leaving
    non-regularized fields (e.g., intercepts or structural metadata) unchanged.

    Parameters
    ----------
    func :
        A callable with signature ``func(x, *args, **kwargs) -> Any``.
        It receives each regularizable subtree ``x`` and must return a value
        with the same pytree structure that should replace that subtree.
    params :
       params any parameter object. If it implements ``regularizable_subtrees()``, the
       method is used to return an iterable of selector functions (
       suitable for ``eqx.tree_at``) that identify the leaves/subtrees to be transformed.
    *args :
        Additional positional arguments passed directly to ``func``.
    filter_kwargs :
        Optional keyword-only dictionary of keyword arguments with PyTree values
        that should be filtered per subtree. For each regularizable subtree, the
        subtree selector is applied to each value in this dict, extracting only
        the portion relevant to that subtree. These extracted kwargs are then
        passed to ``func`` along with the subtree. This is useful for operators
        that need PyTree-structured metadata (e.g., masks) aligned with the
        parameter structure. Must be passed as a keyword argument. Default is
        None, which results in no filtering.
    **kwargs :
        Additional keyword arguments passed directly to ``func``.

    Returns
    -------
    params_new : same type as ``params``
        A new pytree/module with ``func`` applied to all regularizable
        subtrees. Non-regularized fields are preserved unchanged.

    Notes
    -----
    - ``regularizable_subtrees()`` must return a sequence of callables
      compatible with ``eqx.tree_at``. Each callable should extract a subtree
      from ``params``.
    - ``func`` must be pure and JAX-compatible if this function is used inside
      JIT-compiled code.
    - When ``filter_kwargs`` is provided, each value in the dict must be a PyTree
      with the same structure as ``params`` (or compatible with the subtree selectors).

    Examples
    --------
    A minimal working example with a fake ``Params`` object:

    >>> import equinox as eqx
    >>> class Params(eqx.Module):
    ...     w: float
    ...     b: float
    ...
    ...     # Only `w` is regularizable
    ...     def regularizable_subtrees(self):
    ...         return [lambda p: p.w]

    >>> p = Params(w=3.0, b=10.0)

    Define an operator that halves the value:

    >>> def halve(x):
    ...     return x / 2

    Apply it only to the regularizable subtree (`w`):

    >>> p2 = apply_operator(halve, p)
    >>> p2.w
    1.5
    >>> p2.b
    10.0

    The bias `b` is unchanged because it is not listed in
    `regularizable_subtrees`.

    Example with ``filter_kwargs`` for PyTree-structured metadata:

    >>> def masked_op(x, mask=None):
    ...     if mask is not None:
    ...         return x * mask
    ...     return x

    >>> # Create a mask with same structure as params
    >>> mask_tree = Params(w=0.5, b=1.0)

    >>> # Apply operator with filtered kwargs - only the relevant mask piece
    >>> # is passed to each subtree
    >>> p3 = apply_operator(masked_op, p, filter_kwargs={"mask": mask_tree})
    >>> p3.w  # w was multiplied by mask.w (0.5)
    1.5
    >>> p3.b  # b is not regularizable, so unchanged
    10.0
    """
    filter_kwargs = filter_kwargs or {}
    # if there is a list of regularizable sub-trees use that
    if hasattr(params, "regularizable_subtrees"):
        regularizable_subtrees = params.regularizable_subtrees()
    # otherwise regularize all the tree
    else:
        regularizable_subtrees = [lambda x: x]

    for where in regularizable_subtrees:
        # Extract subtree-specific kwargs by applying the selector to each value
        subtree_kwargs = {key: where(val) for key, val in filter_kwargs.items()}
        params = eqx.tree_at(
            where,
            params,
            func(where(params), *args, **kwargs, **subtree_kwargs),
        )

    return params


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
    _proximal_operator: Callable = None

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
    def get_proximal_operator(self, init_params: Any = None) -> ProximalOperator:
        """
        Retrieve the proximal operator for this solver.

        Parameters
        ----------
        init_params:
            The parameters to be regularized.

        Returns
        -------
        :
            The proximal operator, applying regularization to the provided parameters.
        """

        def prox_op(params, strength, scaling=1.0):
            for where, substrength in zip(params.regularizable_subtrees(), strength):
                params = eqx.tree_at(
                    where,
                    params,
                    self._proximal_operator(
                        where(params), substrength, scaling=scaling
                    ),
                )
            return params

        return prox_op

    def check_solver(self, solver_name: str):
        """Raise an error if the given solver is not allowed."""
        # Temporary parsing until an improved registry is implemented.
        algo_name = solver_name.split("[", 1)[0]
        if (
            solver_name not in self._allowed_solvers
            and algo_name not in self._allowed_solvers
        ):
            raise ValueError(
                f"The solver: {solver_name} is not allowed for "
                f"{self.__class__.__name__} regularization. Allowed solvers are "
                f"{self.allowed_solvers}."
            )

    def __repr__(self):
        return format_repr(self)

    def __str__(self):
        return format_repr(self)

    @staticmethod
    def _check_loss_output_tuple(output: tuple):
        if len(output) != 2:
            n_out = len(output)
            word = "value" if n_out == 1 else "values"
            raise ValueError(
                f"Invalid loss function return. The loss function returns a tuple with {n_out} {word}.\n"
                "A valid loss function can return either a single value (float or a 0-dim array), the loss, "
                "or a tuple with two values, the loss and an auxiliary variable."
            )

    def penalized_loss(
        self, loss: Callable, strength: float, init_params: Any
    ) -> Callable:
        """Return a function for calculating the penalized loss using Lasso regularization."""

        filter_kwargs = self._get_filter_kwargs(init_params)

        def _penalized_loss(params, *args, **kwargs):
            result = loss(params, *args, **kwargs)
            penalty = self._penalization(params, strength, filter_kwargs=filter_kwargs)
            if isinstance(result, tuple):
                self._check_loss_output_tuple(result)
                loss_value, aux = result
                return loss_value + penalty, aux

            return result + penalty

        return _penalized_loss

    def _penalization(
        self,
        params: ModelParamsT,
        strength: RegularizerStrength,
        filter_kwargs: dict,
    ) -> jnp.ndarray:
        penalty = jnp.array(0.0)

        if hasattr(params, "regularizable_subtrees"):
            for where, substrength in zip(params.regularizable_subtrees(), strength):
                subtree = where(params)
                subtree_kwargs = {key: where(val) for key, val in filter_kwargs.items()}
                penalty = penalty + self._penalty_on_subtree(
                    subtree, strength, **subtree_kwargs
                )
        else:
            penalty = penalty + self._penalty_on_subtree(
                params, strength, **filter_kwargs
            )
        return penalty

    @abc.abstractmethod
    def _penalty_on_subtree(
        self, sub_params, strength: RegularizerStrength, **kwargs
    ) -> jnp.ndarray:
        pass

    def _validate_regularizer_strength(
        self, strength: Union[None, RegularizerStrength]
    ):
        if strength is None:
            strength = 1.0
        else:

            def _validate(substrength, extra_msg=""):
                if substrength is None:
                    substrength = 1.0
                if not isinstance(
                    substrength, (int, float, jnp.ndarray, np.ndarray, dict)
                ):
                    raise TypeError(
                        "Regularizer strength should be either a float or a pytree of floats, "
                        f"you passed {substrength} of type {type(substrength)}"
                        + extra_msg
                    )
                return substrength

            if isinstance(strength, list):
                strength = [
                    _validate(s, extra_msg=f", at index {i} of {strength}")
                    for i, s in enumerate(strength)
                ]
            else:
                strength = _validate(strength)

        return strength

    def _validate_regularizer_strength_structure(
        self,
        params: ModelParamsT,
        strength: Union[None, RegularizerStrength],
    ):
        regularizable_subtrees = params.regularizable_subtrees()

        # validate length
        if isinstance(strength, list):
            if len(strength) != len(regularizable_subtrees):
                raise ValueError(
                    f"Length of regularizer strength ({len(strength)}) should match number "
                    f"of regularizable parameter sets: {len(regularizable_subtrees)}"
                )
            else:
                _strength = strength
        else:
            _strength = [strength] * len(regularizable_subtrees)

        # validate structure
        for i, where in enumerate(regularizable_subtrees):
            subtree = where(params)

            # wrap in tree if not a tree already
            if not hasattr(_strength[i], "__len__"):
                _strength[i] = tree_utils.tree_full_like(subtree, _strength[i])

            # compare structure to params
            if jax.tree_util.tree_structure(
                _strength[i]
            ) != jax.tree_util.tree_structure(subtree):
                raise ValueError(
                    f"Regularizer strength must conform to parameters: {strength}"
                )

            # if arrays are leaves, also check structure match
            def check_and_broadcast(param_leaf, strength_leaf):
                if isinstance(strength_leaf, jnp.ndarray):
                    if param_leaf.shape != strength_leaf.shape:
                        raise ValueError(
                            f"Shape mismatch between regularizer strength and parameters. "
                            f"Expected shape {param_leaf.shape}, but got {strength_leaf.shape}."
                        )
                else:
                    strength_leaf = jnp.broadcast_to(strength_leaf, param_leaf.shape)
                return strength_leaf

            _strength[i] = jax.tree_util.tree_map(
                check_and_broadcast, subtree, _strength[i]
            )

            # Force conversion to float to prevent weird GPU issues
            try:
                _strength[i] = jax.tree_util.tree_map(
                    lambda x: jnp.asarray(x, dtype=float), _strength[i]
                )
            except ValueError:
                raise ValueError(
                    f"Could not convert regularizer strength to floats: {_strength[i]}"
                )

        return _strength

    @staticmethod
    def _get_filter_kwargs(init_params: Any) -> dict:
        """Return kwargs that need subtree filtering."""
        return {}


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

    def get_proximal_operator(self, init_params=None) -> ProximalOperator:
        """
        Return the identity operator.

        Unregularized method corresponds to an identity proximal operator, since no
        shrinkage factor is applied.

        Parameters
        ----------
        init_params
        """
        return prox_none

    def _validate_regularizer_strength(self, strength: Any):
        return None

    def _validate_regularizer_strength_structure(
        self,
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        strength: Union[None, float],
    ):
        return None

    def _penalization(self, params: ModelParamsT, strength: Any) -> jnp.ndarray:
        return jnp.array(0.0)

    def _penalty_on_subtree(self, subtree, substrength: Any) -> jnp.ndarray:
        return jnp.array(0.0)


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

    _proximal_operator = staticmethod(prox_ridge)

    def __init__(
        self,
    ):
        super().__init__()

    def _penalty_on_subtree(
        self,
        subtree,
        substrength: RegularizerStrength,
    ) -> jnp.ndarray:
        """
        Compute the Ridge penalization for given parameters.

        Parameters
        ----------
        subtree :
            Model parameter subtree for which to compute the penalization.
        strength :
            Regularization strength.

        Returns
        -------
        float
            The Ridge penalization value.
        """

        def l2_penalty(coeff: jnp.ndarray, strength: jnp.ndarray):
            return 0.5 * jnp.sum(strength * jnp.square(coeff))

        return tree_utils.pytree_map_and_reduce(
            l2_penalty,
            sum,
            subtree,
            substrength,
        )

    def get_proximal_operator(self, init_params=None) -> ProximalOperator:
        """
        Retrieve the proximal operator for Ridge regularization (L2 penalty).

        Parameters
        ----------
        init_params

        Returns
        -------
        :
            The proximal operator, applying L2 regularization to the provided parameters. The intercept
            term is not regularized.
        """

        def prox_op(params, l2reg, scaling=1.0):
            return apply_operator(prox_ridge, params, l2reg, scaling=scaling)

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

    _proximal_operator = staticmethod(prox_lasso)

    def __init__(
        self,
    ):
        super().__init__()

    def get_proximal_operator(self, init_params=None) -> ProximalOperator:
        """
        Retrieve the proximal operator for Lasso regularization (L1 penalty).

        Parameters
        ----------
        init_params

        Returns
        -------
        :
            The proximal operator, applying L1 regularization to the provided parameters. The intercept
            term is not regularized.
        """

        def prox_op(params, l1reg, scaling=1.0):
            return apply_operator(prox_lasso, params, l1reg, scaling=scaling)

        return prox_op

    def _penalty_on_subtree(
        self,
        subtree: ModelParamsT,
        substrength: RegularizerStrength,
    ) -> jnp.ndarray:
        """
        Compute the Lasso penalization for given parameters.

        Parameters
        ----------
        subtree :
            Model parameters for which to compute the penalization.
        substrength :
            Regularization strength.

        Returns
        -------
        float
            The Lasso penalization value.
        """

        def l1_penalty(coeff: jnp.ndarray, strength: jnp.ndarray):
            return jnp.sum(strength * jnp.abs(coeff))

        return tree_utils.pytree_map_and_reduce(
            l1_penalty,
            sum,
            subtree,
            substrength,
        )


class ElasticNet(Regularizer):
    r"""
    Regularizer class for Elastic Net (L1 + L2 regularization).

    The Elastic Net penalty [3]_ [4]_ is defined as:

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

    _proximal_operator = staticmethod(prox_elastic_net)

    def __init__(
        self,
    ):
        super().__init__()

    def get_proximal_operator(self, init_params=None) -> ProximalOperator:
        """
        Retrieve the proximal operator ElasticNet regularization.

        Parameters
        ----------
        init_params

        Returns
        -------
        :
            The proximal operator, applying regularization to the provided parameters.
        """

        def prox_op(params, strength, scaling=1.0):
            for where, substrength, subratio in zip(
                params.regularizable_subtrees(), *strength
            ):
                params = eqx.tree_at(
                    where,
                    params,
                    self._proximal_operator(
                        where(params), substrength, subratio, scaling=scaling
                    ),
                )
            return params

        return prox_op

    def _penalization(
        self, params: ModelParamsT, strength: RegularizerStrength
    ) -> jnp.ndarray:
        penalty = jnp.array(0.0)

        for where, substrength, subratio in zip(
            params.regularizable_subtrees(), *strength
        ):
            subtree = where(params)
            penalty = penalty + self._penalty_on_subtree(subtree, substrength, subratio)

        return penalty

    def _penalty_on_subtree(
        self,
        subtree,
        substrength: ElasticNetRegularizerStrength,
        subratio: ElasticNetRegularizerStrength,
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
        subtree :
            Model parameters for which to compute the penalization.
        substrength :
            Regularization strength.

        Returns
        -------
        :
            The Elastic Net penalization value.
        """

        def net_penalty(
            coeff: jnp.ndarray, strength: jnp.ndarray, ratio: jnp.ndarray
        ) -> jnp.ndarray:
            quad = 0.5 * (1.0 - ratio) * jnp.power(coeff, 2)
            l1 = ratio * jnp.abs(coeff)
            return jnp.sum(strength * (quad + l1))

        # tree map the computation and sum over leaves
        return tree_utils.pytree_map_and_reduce(
            net_penalty,
            sum,
            subtree,
            substrength,
            subratio,
        )

    def _validate_regularizer_strength(
        self, strength: Union[None, ElasticNetRegularizerStrength]
    ):
        if strength is None:
            _strength, _ratio = (1.0, 0.5)
        elif isinstance(strength, Tuple):
            if len(strength) != 2:
                raise TypeError(
                    f"ElasticNet regularizer strength should be a tuple of two floats, you passed: {strength}"
                )
            else:
                _strength = super()._validate_regularizer_strength(strength[0])
                try:
                    _ratio = super()._validate_regularizer_strength(strength[1])
                except TypeError as e:
                    raise TypeError(str(e).replace("strength", "ratio")) from None

                def _verify_ratio(ratio, extra_msg=""):
                    if jnp.any((ratio > 1) | (ratio < 0)):
                        raise ValueError(
                            f"Regularization ratio must be between 0 and 1: {ratio}"
                            + extra_msg
                        )
                    if jnp.any(ratio == 0):
                        raise ValueError(
                            f"Regularization ratio of 0 is not supported: {ratio}"
                            + extra_msg
                        )

                if isinstance(_ratio, list):
                    for i, r in enumerate(_ratio):
                        jax.tree_util.tree_map(
                            lambda r: _verify_ratio(
                                r, extra_msg=f", at index {i} of {strength[1]}"
                            ),
                            r,
                        )
                else:
                    jax.tree_util.tree_map(_verify_ratio, _ratio)
        else:
            _strength = super()._validate_regularizer_strength(strength)
            _ratio = 0.5

        return _strength, _ratio

    def _validate_regularizer_strength_structure(
        self,
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        strength: Union[None, ElasticNetRegularizerStrength],
    ):
        _strength = super()._validate_regularizer_strength_structure(
            params, strength[0]
        )
        try:
            _ratio = super()._validate_regularizer_strength_structure(
                params, strength[1]
            )
        except ValueError as e:
            raise ValueError(str(e).replace("strength", "ratio")) from None

        return _strength, _ratio


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

    _proximal_operator = staticmethod(prox_group_lasso)

    def __init__(
        self,
        mask: Any = None,
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
            mask = self._cast_and_check_mask(mask)
        self._mask = mask

    def initialize_mask(self, x: Any) -> Any:
        """
        Initialize a default group mask for a PyTree of parameters.

        Creates a mask where each leaf array and each of its trailing dimensions
        (beyond the first) are assigned to separate groups. This default grouping
        treats:
        - Each leaf in the PyTree as a distinct parameter set
        - The first dimension (axis 0) as the feature dimension
        - Each trailing dimension as a separate group of features

        For a leaf with shape (n_features, d1, d2, ..., dk), this creates
        (d1 * d2 * ... * dk) groups, where each group's mask is 1.0 for all
        features in that specific trailing dimension combination and 0.0 elsewhere.

        Parameters
        ----------
        x : Any
            PyTree of parameter arrays. Each leaf should have shape
            (n_features, ...) where n_features is the number of features
            and trailing dimensions define additional grouping structure.

        Returns
        -------
        mask : Any
            PyTree with the same structure as x, where each leaf has shape
            (n_groups, n_features, ...) matching the original leaf shape.
            The mask contains 1.0 for elements in each group and 0.0 elsewhere.

        """
        reg_subtrees = (
            x.regularizable_subtrees()
            if hasattr(x, "regularizable_subtrees")
            else [lambda z: z]
        )
        struct = jax.tree_util.tree_structure(x)
        mask = jax.tree_util.tree_unflatten(struct, [None] * struct.num_leaves)
        for where in reg_subtrees:
            mask = eqx.tree_at(
                where,
                mask,
                self._initialize_subtree_mask(where(x)),
                is_leaf=lambda m: m is None,
            )
        return mask

    @staticmethod
    def _initialize_subtree_mask(x_subtree: Any) -> Any:
        """Initialize individual subtree mask matching structure."""
        flat_x, struct = jax.tree_util.tree_flatten(x_subtree)

        # Calculate total number of groups across all leaves
        n_groups_per_leaf = [math.prod(leaf.shape[1:]) for leaf in flat_x]
        total_groups = sum(n_groups_per_leaf)

        # Build mask for each leaf
        mask_flat = []
        group_offset = 0

        for leaf, n_groups in zip(flat_x, n_groups_per_leaf):
            # Create mask: (total_groups, n_features, *extra_dims)
            mask_shape = (total_groups, *leaf.shape)
            mask = jnp.zeros(mask_shape, dtype=float)

            # Set 1.0 for this leaf's groups along the flattened extra dimensions
            for i in range(n_groups):
                # Use reshape to map linear index to multi-dimensional index
                extra_shape = leaf.shape[1:]
                multi_idx = jnp.unravel_index(i, extra_shape)
                # Build index tuple: (group_id, slice(:), *multi_idx)
                # When dropping support for Python < 3.11, replace with
                # mask = mask.at[group_offset + i, :, *multi_idx].set(1.0)
                full_idx = (group_offset + i, slice(None)) + multi_idx
                mask = mask.at[full_idx].set(1.0)

            mask_flat.append(mask)
            group_offset += n_groups

        return jax.tree_util.tree_unflatten(struct, mask_flat)

    @staticmethod
    def _cast_and_check_mask(mask: Any) -> Any:
        """
        Cast to jax array of floats and validate the mask.

        This method ensures the mask adheres to requirements:
        - The mask should be castable to a PyTree of arrays of float type.
        - Each element must be either 0 or 1.
        - Each feature should belong to only one group.
        - The mask should not be empty.

        Raises
        ------
        ValueError
            If any of the above conditions are not met.
        """
        mask = convert_tree_leaves_to_jax_array(
            mask,
            "Unable to convert mask to a tree ``jax.ndarray`` leaves.",
        )

        flat_mask = jax.tree_util.tree_leaves(mask)
        n_groups = flat_mask[0].shape[0]
        if not all(f.shape[0] == n_groups for f in flat_mask[1:]):
            n_groups = {f.shape[0] == n_groups for f in flat_mask[1:]}
            raise ValueError(
                "The length of the first dimension array leaves in the mask PyTree "
                "should be equal to ``n_groups``. "
                f"Leaves of the mask tree have inconsistent first dimension lengths: {n_groups}."
            )

        if any(m.ndim < 2 for m in flat_mask):
            raise ValueError(
                "Mask arrays should have at least 2 dimensions ``(n_groups, n_features, ...)``."
            )

        if n_groups == 0:
            raise ValueError("Empty mask provided!")

        has_invalid_entries = pytree_map_and_reduce(
            lambda m: jnp.any((m != 1) & (m != 0)), any, mask
        )
        if has_invalid_entries:
            raise ValueError("Mask elements be 0s and 1s!")

        all_zeros = pytree_map_and_reduce(lambda m: jnp.all(m == 0), all, mask)
        if all_zeros:
            raise ValueError("Empty mask provided!")

        multi_group_assignment = pytree_map_and_reduce(
            lambda m: jnp.any(m.sum(axis=0) > 1), any, mask
        )
        if multi_group_assignment:
            raise ValueError(
                "Incorrect group assignment. Some of the features are assigned "
                "to more than one group."
            )
        return mask

    def _penalty_on_subtree(
        self, subtree, strength: jnp.ndarray, mask: None = Any
    ) -> jnp.ndarray:
        r"""
        Note: the penalty is being calculated according to the following formula:

        .. math::

            \\text{loss}(\beta_1,...,\beta_g) + \alpha \cdot \sum _{j=1...,g} \sqrt{\dim(\beta_j)} || \beta_j||_2

        where :math:`g` is the number of groups, :math:`\dim(\cdot)` is the dimension of the vector,
        i.e. the number of coefficient in each :math:`\beta_j`, and :math:`||\cdot||_2` is the euclidean norm.
        """
        # l2_norms = masked_norm_2(sub_params, mask, normalize=False)
        # norm = compute_normalization(mask)
        # return jnp.sum(norm * l2_norms) * strength

        # conform to shape (1, n_features) if param is (n_features,) or (n_neurons, n_features) if
        # param is (n_features, n_neurons)
        param_with_extra_axis = jnp.atleast_2d(subtree.T)

        vec_prod = jax.vmap(
            lambda x: self.mask * x, in_axes=0, out_axes=2
        )  # this vectorizes the product over the neurons, and adds the neuron axis as the last axis

        masked_param = vec_prod(param_with_extra_axis)

        # ||β_g||₂ per group, summed over neurons
        # (group, neuron) -> (group,)
        group_norms = jnp.linalg.norm(masked_param, axis=1).sum(axis=1)

        # sqrt(p_g)
        group_sizes = jnp.sqrt(self.mask.sum(axis=1))

        return jnp.sum(strength * group_sizes * group_norms)

    def _validate_regularizer_strength_structure(
        self,
        params: ModelParamsT,
        strength: Union[None, RegularizerStrength],
    ):
        n_groups = self.mask.shape[0]
        regularizable_subtrees = params.regularizable_subtrees()

        # normalize to per-subtree list
        if strength is None:
            strength = 1.0

        if isinstance(strength, list):
            if len(strength) != len(regularizable_subtrees):
                raise ValueError(
                    f"Length of regularizer strength ({len(strength)}) must match "
                    f"number of regularizable parameter sets ({len(regularizable_subtrees)})"
                )
            strengths = strength
        else:
            strengths = [strength] * len(regularizable_subtrees)

        _strengths = []

        for s in strengths:
            if isinstance(s, (float, int)):
                s = jnp.ones(n_groups) * float(s)

            elif isinstance(s, (jnp.ndarray, np.ndarray)):
                if s.ndim != 1 or s.shape[0] != n_groups:
                    raise ValueError(
                        f"GroupLasso regularizer strength must match number of groups: {s.shape} vs. {n_groups}"
                    )
            else:
                raise TypeError(
                    "GroupLasso regularizer strength must be a float or "
                    f"a 1D array of length {n_groups}"
                )

            _strengths.append(jnp.asarray(s, dtype=float))

        return _strengths

    def get_proximal_operator(self, init_params=None) -> ProximalOperator:
        """
        Retrieve the proximal operator for Group Lasso regularization.

        Parameters
        ----------
        init_params

        Returns
        -------
        :
            The proximal operator, applying Group Lasso regularization to the provided parameters. The
            intercept term is not regularized.
        """
        filter_kwargs = self._get_filter_kwargs(init_params=init_params)

        def prox_op(params, strength, scaling=1.0):
            for where, substrength in zip(params.regularizable_subtrees(), strength):
                params = eqx.tree_at(
                    where,
                    params,
                    self._proximal_operator(
                        where(params), substrength, scaling=scaling, mask=self.mask
                    ),
                )
            return params
            # return apply_operator(
            #    prox_group_lasso,
            #    params,
            #    strength,
            #    filter_kwargs=filter_kwargs,
            #    scaling=scaling,
            # )

        return prox_op

    def _get_filter_kwargs(self, init_params: Any) -> dict:
        """Return kwargs that need subtree filtering."""
        if self.mask is None:
            mask = self.initialize_mask(init_params)
        else:
            mask = self.mask
        return {"mask": mask}
