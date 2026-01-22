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
from .typing import ProximalOperator
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

    _allowed_solvers: Tuple[str]
    _default_solver: str
    _proximal_operator: Callable

    @property
    def allowed_solvers(self) -> Tuple[str]:
        return self._allowed_solvers

    @property
    def default_solver(self) -> str:
        return self._default_solver

    def check_solver(self, solver_name: str) -> None:
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

    def __repr__(self) -> str:
        return format_repr(self)

    def __str__(self) -> str:
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

    def get_proximal_operator(self, params: Any, strength: Any) -> ProximalOperator:
        """
        Retrieve the proximal operator.

        Parameters
        ----------
        init_params:
            The parameters to be regularized.

        Returns
        -------
        :
            The proximal operator, applying regularization to the provided parameters.
        """
        filter_kwargs = self._get_filter_kwargs(strength=strength, params=params)

        def prox_op(params, hyperparams, scaling=1.0, *args):
            return apply_operator(
                self._proximal_operator,
                params,
                filter_kwargs=filter_kwargs,
                scaling=scaling,
            )

        return prox_op

    def penalized_loss(self, loss: Callable, params: Any, strength: Any) -> Callable:
        """Return a function for calculating the penalized loss."""

        filter_kwargs = self._get_filter_kwargs(strength=strength, params=params)

        def _penalized_loss(params, *args, **kwargs):
            result = loss(params, *args, **kwargs)
            penalty = self._penalization(params, filter_kwargs=filter_kwargs)
            if isinstance(result, tuple):
                self._check_loss_output_tuple(result)
                loss_value, aux = result
                return loss_value + penalty, aux

            return result + penalty

        return _penalized_loss

    def _penalization(self, params: Any, filter_kwargs: dict) -> jnp.ndarray:
        penalty = jnp.array(0.0)

        if hasattr(params, "regularizable_subtrees"):
            for where in params.regularizable_subtrees():
                subtree = where(params)
                subtree_kwargs = {key: where(val) for key, val in filter_kwargs.items()}
                penalty = penalty + self._penalty_on_subtree(subtree, **subtree_kwargs)
        else:
            penalty = penalty + self._penalty_on_subtree(params, **filter_kwargs)
        return penalty

    @abc.abstractmethod
    def _penalty_on_subtree(self, subtree, **kwargs) -> jnp.ndarray:
        pass

    def _validate_strength(self, strength: Any):
        """
        Normalize regularizer strength into a PyTree of JAX float arrays.

        Parameters
        ----------
        strength : Any
            Regularizer strength specified as a scalar, array-like, or PyTree.

        Returns
        -------
        Any
            PyTree with `jnp.ndarray` float leaves.

        Raises
        ------
        ValueError
            If conversion to float arrays fails.
        """
        if strength is None:
            return 1.0

        return convert_tree_leaves_to_jax_array(
            strength, f"Could not convert regularizer strength to floats: {strength}"
        )

    def _validate_strength_structure(self, params: Any, strength: Any):
        """
        Validate and broadcast regularizer strength to match regularizable parameters.

        This function aligns the provided regularizer strength with the structure of
        `params`, filling only the regularizable subtrees and broadcasting strength
        values to match parameter leaf shapes.

        Parameters
        ----------
        params : Any
            Model parameters structured as a PyTree. Regularizable subtrees are
            determined via `params.regularizable_subtrees()` if present; otherwise
            the entire parameter tree is treated as regularizable.

        strength : Any
            Regularizer strength specification. Accepted forms:

            - None
                Defaults to a scalar strength of 1.0 for all regularizable parameters.
            - scalar or 0-D array
                Broadcast to every regularizable parameter leaf.
            - PyTree
                Must match the structure of the regularizable subtrees. Each leaf
                may be a scalar or an array broadcastable to the corresponding
                parameter leaf shape.

        Returns
        -------
        structured_strength : Any
            PyTree with the same structure as `params`. Regularizable parameter leaves
            contain `jnp.ndarray` strengths broadcast to the parameter shapes; all
            non-regularizable leaves are `None`.

        Raises
        ------
        ValueError
            If:
            - The number of provided strength subtrees does not match the number of
              regularizable subtrees.
            - A strength PyTree does not have the same number of leaves as the
              corresponding parameter subtree.
            - A strength leaf cannot be broadcast to the shape of the corresponding
              parameter leaf.
        """

        wheres = getattr(params, "regularizable_subtrees", lambda: [lambda x: x])()
        struct = jax.tree_util.tree_structure(params)
        structured_strength = jax.tree_util.tree_unflatten(
            struct, [None] * struct.num_leaves
        )

        # handle scalar or None strength
        if (
            strength is None
            or isinstance(strength, (int, float))
            or (isinstance(strength, (np.ndarray, jnp.ndarray)) and strength.ndim == 0)
        ):
            scalar = 1.0 if strength is None else float(strength)
            for where in wheres:
                subtree = where(params)
                structured_strength = eqx.tree_at(
                    where,
                    structured_strength,
                    jax.tree_util.tree_map(
                        lambda p: jnp.full(p.shape, scalar, dtype=float), subtree
                    ),
                    is_leaf=lambda x: x is None,
                )
            return structured_strength

        # handle PyTree-aligned strength
        strengths = (
            [strength]
            if len(wheres) == 1
            else (strength if len(strength) == len(wheres) else None)
        )
        if strengths is None:
            raise ValueError(f"Expected {len(wheres)} strength values, got {strength}")

        for s, where in zip(strengths, wheres):
            subtree = where(params)
            param_leaves, treedef = jax.tree_util.tree_flatten(subtree)

            strength_leaves = (
                [s] * len(param_leaves)
                if isinstance(s, (np.ndarray, jnp.ndarray)) and s.ndim == 0
                else jax.tree_util.tree_leaves(s)
            )
            if len(strength_leaves) != len(param_leaves):
                raise ValueError(
                    f"Strength tree has {len(strength_leaves)} leaves, "
                    f"but parameter subtree has {len(param_leaves)} leaves"
                )

            validated = [
                jnp.broadcast_to(jnp.asarray(sl, dtype=float), p.shape)
                for p, sl in zip(param_leaves, strength_leaves)
            ]
            structured_strength = eqx.tree_at(
                where,
                structured_strength,
                jax.tree_util.tree_unflatten(treedef, validated),
            )

        return structured_strength

    def _get_filter_kwargs(self, params: Any, strength: Any):
        strength = self._validate_strength_structure(params, strength)
        return {"strength": strength}


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
    _proximal_operator = staticmethod(prox_none)

    def _penalty_on_subtree(self, subtree, **kwargs) -> jnp.ndarray:
        return jnp.array(0.0)

    def _validate_strength(self, strength: Any):
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

    _proximal_operator = staticmethod(prox_ridge)

    def _penalty_on_subtree(self, subtree, strength: Any, **kwargs) -> jnp.ndarray:
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
            strength,
        )


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

    def _penalty_on_subtree(self, subtree, strength: Any, **kwargs) -> jnp.ndarray:
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
            strength,
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

    def _penalty_on_subtree(self, subtree: Any, strength: Any, **kwargs) -> jnp.ndarray:
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
        strength :
            Regularization strength.
        ratio :
            Regularization ratio.

        Returns
        -------
        :
            The Elastic Net penalization value.
        """

        def net_penalty(coeff, strength):
            strength, ratio = strength
            quad = 0.5 * (1.0 - ratio) * jnp.square(coeff)
            l1 = ratio * jnp.abs(coeff)
            return jnp.sum(strength * (quad + l1))

        return tree_utils.pytree_map_and_reduce(
            net_penalty,
            sum,
            subtree,
            strength,
        )

    def _validate_strength(self, strength: Any):
        if strength is None:
            strength, ratio = 1.0, 0.5

        elif isinstance(strength, tuple):
            if len(strength) != 2:
                raise TypeError(
                    "ElasticNet regularizer strength must be a tuple (strength, ratio)"
                )
            strength, ratio = strength

        else:
            strength, ratio = strength, 0.5

        strength = super()._validate_strength(strength)
        ratio = super()._validate_strength(ratio)

        def check_ratio(r):
            if jnp.any((r <= 0) | (r > 1)):
                raise ValueError(
                    f"ElasticNet regularization ratio must be in (0, 1], got {r}"
                )
            return r

        ratio = jax.tree_util.tree_map(check_ratio, ratio)

        return strength, ratio

    def _validate_strength_structure(self, params: Any, strength: Any):
        _strength = super()._validate_strength_structure(params, strength[0])
        ratio = super()._validate_strength_structure(params, strength[1])

        def zip_leaves(s, r):
            if s is None:
                return None
            return (s, r)

        return jax.tree_util.tree_map(zip_leaves, _strength, ratio)


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

    Notes
    -----
    For GroupLasso, the regularizer strength is defined **per group**, not per parameter.
    It must be either a scalar or a 1D array of length ``n_groups``.

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
            "Unable to convert mask to a tree with ``jax.ndarray`` leaves.",
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
            raise ValueError("Mask elements must be 0s and 1s!")

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
        self, subtree, strength: Any, mask: Any = None, **kwargs
    ) -> jnp.ndarray:
        r"""
        Apply the Group Lasso penaly to a subtree.

        Note: the penalty is being calculated according to the following formula:

        .. math::

            \\text{loss}(\beta_1,...,\beta_g) + \alpha \cdot \sum _{j=1...,g} \sqrt{\dim(\beta_j)} || \beta_j||_2

        where :math:`g` is the number of groups, :math:`\dim(\cdot)` is the dimension of the vector,
        i.e. the number of coefficient in each :math:`\beta_j`, and :math:`||\cdot||_2` is the euclidean norm.
        """

        def penalty_leaf(leaf, leaf_mask, leaf_strength):
            leaf_l2_norm = masked_norm_2(leaf, leaf_mask, normalize=False)
            leaf_norm = compute_normalization(leaf_mask)
            return jnp.sum(leaf_strength * leaf_norm * leaf_l2_norm)

        penalties = jax.tree_util.tree_map(
            penalty_leaf,
            subtree,
            mask,
            strength,
        )

        return jnp.sum(jnp.array(jax.tree_util.tree_leaves(penalties)))

    def _validate_strength_structure(self, params: Any, strength: Any):
        flat_mask = jax.tree_util.tree_leaves(self.mask)
        n_groups = flat_mask[0].shape[0]

        if isinstance(strength, (int, float)) or strength.ndim == 0:
            per_group_strength = jnp.full(n_groups, strength, dtype=float)
        else:
            strength = jnp.asarray(strength, dtype=float)
            if strength.ndim != 1 or strength.shape[0] != n_groups:
                raise ValueError(
                    f"GroupLasso strength must be a scalar or shape ({n_groups},), "
                    f"got shape {strength.shape}"
                )
            per_group_strength = strength

        return jax.tree_util.tree_map(lambda _: per_group_strength, self.mask)

    def _get_filter_kwargs(self, params: Any, strength: Any) -> dict:
        if self.mask is None:
            self.mask = self.initialize_mask(params)
        return {"mask": self.mask, **super()._get_filter_kwargs(params, strength)}
