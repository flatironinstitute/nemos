"""Mixin classes for regressor models."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from numbers import Number
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._hess import LeafClaim, claim_nothing
from .label_encoder import LabelEncoder
from .observation_models import OBSERVATION_MODELS, CategoricalObservations
from .type_casting import is_numpy_array_like
from .typing import (
    DESIGN_INPUT_TYPE,
    ModelParamsT,
    SolverState,
    UserProvidedParamsT,
)


class ClassifierMixin:
    """Additional methods for classification models."""

    # observation model inferred
    _invalid_observation_types = list(
        set(OBSERVATION_MODELS) - set([CategoricalObservations])
    )

    def _hess_leaf_claims(
        self, params: ModelParamsT[jnp.ndarray], active_spec: ModelParamsT[bool]
    ) -> ModelParamsT[LeafClaim]:
        """Certify nothing, unlike the non-classification models this inherits from.

        Adding the same constant to every class's intercept leaves the softmax
        probabilities unchanged, so the intercept block is singular along that direction
        rather than definite.

        Parameters
        ----------
        params :
            The parameters being fitted.
        active_spec :
            The filter spec ``params`` was partitioned with. Unused: nothing is certified
            whether or not a leaf is being fitted.

        Returns
        -------
        :
            A tree shaped like ``params`` carrying ``LeafClaim.UNCLAIMED``
            everywhere.
        """
        return claim_nothing(params)

    def set_classes(self, y: ArrayLike | None) -> ClassifierMixin:
        """
        Infer unique class labels and set the ``classes`` attribute.

        This method infers class labels from ``y`` and sets up the internal
        encoding/decoding machinery. When labels are the default ``[0, 1, ..., n_classes-1]``,
        encoding is skipped for performance.

        Parameters
        ----------
        y
            An array that must contain all the class labels,
            i.e. ``len(np.unique(y)) == n_classes``. If None, it resets the labels.

        Raises
        ------
        ValueError
            If the number of unique class labels in ``y`` does not match ``n_classes``.

        Notes
        -----
        If ``set_classes`` is used to declare class labels, :meth:`fit` keeps them instead of
        inferring labels from the input ``y``. Manually setting class labels allows the input
        ``y`` to fit to hold any subset of possible classes. A label outside the set raises an
        error. Passing ``None`` clears the labels, and the next :meth:`fit` infers the labels
        from ``y`` again.

        :meth:`initialize_optimizer_and_state` and :meth:`update` do not infer labels. Declare
        them with ``set_classes`` before an :meth:`update` loop, since a single batch need not
        contain every class.

        Examples
        --------
        When fitting in batches with :meth:`update`, use ``set_classes`` to define
        all class labels before initialization. This is necessary when individual
        batches may not contain all classes.

        >>> import nemos as nmo
        >>> import numpy as np
        >>> model = nmo.glm.ClassifierGLM(3)

        Generate sample data where the first batch only contains 2 of 3 classes:

        >>> X = np.random.randn(100, 5)
        >>> y_all_classes = np.array(["a", "b", "c"])  # all possible classes
        >>> y_batch1 = np.array(["a", "b", "a", "b", "a"])  # first batch missing "c"
        >>> X_batch1 = X[:5]

        Without ``set_classes``, the model expects the default integer labels:

        >>> init_params = model.initialize_params(X_batch1, y_batch1)
        Traceback (most recent call last):
        ValueError: Expected integer labels when classes are the default...

        Call ``set_classes`` first to define all labels, then initialize:

        >>> model.set_classes(y_all_classes)
        ClassifierGLM(...)
        >>> init_params = model.initialize_params(X_batch1, y_batch1)
        >>> state = model.initialize_optimizer_and_state(
        ...     init_params, X_batch1, y_batch1
        ... )

        Now batches with any subset of classes work with :meth:`update`:

        >>> result = model.update(init_params, state, X_batch1, y_batch1)

        """
        if y is None:
            self._label_encoder.reset()
            return self
        self._label_encoder.set_classes(y, declared=True)
        return self

    @property
    def classes(self) -> NDArray | int:
        """Declared class labels, or the number of classes when they are not declared."""
        return (
            self.n_classes
            if not self._label_encoder.declared
            else self._label_encoder.classes_
        )

    @classes.setter
    def classes(self, value: NDArray | None) -> None:
        if value is not None:
            self._label_encoder.set_classes(value, declared=True)
        else:
            self._label_encoder.reset()
    
    @property
    def classes_(self) -> NDArray | None:
        """Class labels in use, declared or inferred by :meth:`fit`, or None if not set."""
        return self._label_encoder.classes_

    def compute_loss(
        self,
        params,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Compute the loss function for the model.

        This method validates inputs, encodes class labels to internal indices,
        and computes the loss (negative log-likelihood).

        Parameters
        ----------
        params
            Model parameters in the format expected by the specific model.
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target class labels in the same format as ``classes_``.
        *args
            Additional positional arguments passed to the model-specific loss function.
        **kwargs
            Additional keyword arguments passed to the model-specific loss function.

        Returns
        -------
        loss
            The loss value (negative log-likelihood).

        Raises
        ------
        RuntimeError
            If ``classes_`` has not been set.
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        self._label_encoder.check_classes_is_set("compute_loss")
        y = self._label_encoder.encode(y)
        return super().compute_loss(params, X, y, *args, **kwargs)

    @property
    def n_classes(self):
        """Number of classes."""
        return self._label_encoder.n_classes

    @n_classes.setter
    def n_classes(self, value: int):
        # extract item from scalar arrays
        if is_numpy_array_like(value)[1] and value.size == 1:
            value = value.item()

        if not isinstance(value, Number) or value < 2 or not int(value) == value:
            raise ValueError(
                "The number of classes must be an integer greater than or equal to 2.\n"
                f"Classes ``{value}`` provided instead."
            )

        self._label_encoder = LabelEncoder(int(value))

        # reset validator.
        self._validator = self._validator_class(
            extra_params=self._get_validator_extra_params()
        )

    def _get_validator_extra_params(self) -> dict:
        """Get validator extra parameters."""
        return {"n_classes": self._label_encoder.n_classes}

    def _preprocess_inputs(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Optional[jnp.ndarray] = None,
        *args: jnp.ndarray,
        drop_nans: bool = True,
    ) -> Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray | None]:
        """Preprocess inputs before initializing state."""
        X, y, *args = super()._preprocess_inputs(X, y, *args, drop_nans=drop_nans)
        if y is not None:
            y = self._validator.check_and_cast_y_to_integer(y)
            y = jax.nn.one_hot(y, self._label_encoder.n_classes)
        return (X, y, *args)

    def initialize_optimizer_and_state(
        self,
        init_params: UserProvidedParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        **kwargs,
    ) -> SolverState:
        """Initialize the solver and its state for running fit and update.

        This method must be called before using :meth:`update` for iterative optimization.
        It sets up the solver with the provided initial parameters and data.

        Parameters
        ----------
        init_params
            Initial parameter tuple of (coefficients, intercept).
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target labels, array of shape ``(n_time_bins,)`` for single neuron/subject models or
            ``(n_time_bins, n_neurons)`` for population models.

        Returns
        -------
        state
            Initial solver state.

        Raises
        ------
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        self._label_encoder.check_classes_is_set("initialize_optimizer_and_state")
        y = self._label_encoder.encode(y)
        return super().initialize_optimizer_and_state(init_params, X, y, **kwargs)

    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> UserProvidedParamsT:
        """
        Initialize model parameters for categorical GLM.

        Initialize coefficients with zeros and intercept by matching the mean class
        proportions. Class labels are automatically converted to one-hot encoding.

        Parameters
        ----------
        X :
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y :
            Class labels, array of shape ``(n_time_bins,)`` for single neuron
            models or ``(n_time_bins, n_neurons)`` for population models. Labels
            must be a subset of ``classes``.

        Returns
        -------
        :
            Initial parameter tuple of (coefficients, intercept).

        Notes
        -----
        All labels in ``y`` must be present in ``classes``. Passing labels not
        in ``classes`` will raise an error.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(classes=2)
        >>> model.set_classes(y)
        ClassifierGLM(...)
        >>> coef, intercept = model.initialize_params(X, y)
        >>> coef.shape
        (2, 2)
        """
        self._label_encoder.check_classes_is_set("initialize_params")
        y = self._label_encoder.encode(y)
        y = self._validator.check_and_cast_y_to_integer(y)
        y = jax.nn.one_hot(y, self.n_classes)
        return super().initialize_params(X, y)

    def _setup_classes(self, classes: int | ArrayLike):
        classes_array = np.asarray(classes)
        # ``np.asarray`` casts a dict or a string to a 0-D array too, so the dtype
        # is what separates a count from them
        if classes_array.ndim == 0 and np.issubdtype(classes_array.dtype, np.number):
            self.n_classes = classes
            self._label_encoder.set_classes(np.arange(self.n_classes), declared=False)
        elif classes_array.ndim == 1 and not np.issubdtype(
            classes_array.dtype, np.object_
        ):
            self.n_classes = len(classes_array)
            self.set_classes(classes_array)
        else:
            raise ValueError(
                "``classes`` must be a class count or a 1-D array-like of class "
                f"labels. The provided {classes} was cast to a "
                f"{classes_array.ndim}-D array of dtype {classes_array.dtype} instead."
            )
