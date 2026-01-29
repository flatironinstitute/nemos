"""GLM for Classification."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from numbers import Number
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .. import tree_utils
from ..regularizer import ElasticNet, GroupLasso, Lasso, Regularizer, Ridge
from ..type_casting import is_numpy_array_like, support_pynapple
from ..typing import (
    DESIGN_INPUT_TYPE,
    RegularizerStrength,
    SolverState,
    StepResult,
    UserProvidedParamsT,
)
from .glm import GLM, PopulationGLM
from .params import GLMUserParams
from .validation import (
    ClassifierGLMValidator,
    PopulationClassifierGLMValidator,
)

__all__ = ["ClassifierGLM", "ClassifierPopulationGLM"]


class ClassifierMixin:
    """GLM for classification."""

    # observation model inferred
    _invalid_observation_types = ()

    def _check_classes_is_set(self, method_name: str, y=None):
        if self._classes_ is None:
            raise RuntimeError(
                f"Classes are not set. Must call ``set_classes`` before calling ``{method_name}``."
            )

    def set_classes(self, y: ArrayLike) -> ClassifierMixin:
        """
        Infer unique class labels and set the ``classes_`` attribute.

        This method infers class labels from ``y`` and sets up the internal
        encoding/decoding machinery. When labels are the default ``[0, 1, ..., n_classes-1]``,
        encoding is skipped for performance.

        Parameters
        ----------
        y
            An array that must contain all the class labels,
            i.e. ``len(np.unique(y)) == n_classes``.

        Raises
        ------
        ValueError
            If the number of unique class labels in ``y`` does not match ``n_classes``.

        Notes
        -----
        :meth:`fit` and :meth:`initialize_solver_and_state` call ``set_classes`` internally,
        making sure that the ``classes_`` attribute matches the provided input.
        If you are fitting in batches by calling :meth:`update`, make sure that the ``classes_``
        are correctly set by calling ``set_classes`` before starting the :meth:`update` loop.

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
        >>> y_all_classes = np.array([0, 1, 2])  # all possible classes
        >>> y_batch1 = np.array([0, 1, 0, 1, 0])  # first batch missing class 2
        >>> X_batch1 = X[:5]

        Without ``set_classes``, initialization fails if batch lacks all classes:

        >>> _ = model.initialize_solver_and_state(X_batch1, y_batch1, init_params=None)
        Traceback (most recent call last):
        RuntimeError: Classes are not set. Must call ``set_classes`` before calling...

        Call ``set_classes`` first to define all labels, then initialize:

        >>> model.set_classes(y_all_classes)
        ClassifierGLM(...)
        >>> init_params = model.initialize_params(X_batch1, y_batch1)
        >>> state = model.initialize_solver_and_state(X_batch1, y_batch1, init_params)

        Now batches with any subset of classes work with :meth:`update`:

        >>> result = model.update(init_params, state, X_batch1, y_batch1)

        """
        # note that we must use NumPy, Jax does not allow non-numeric types
        classes = np.unique(y)
        n_unique = len(classes)

        # Validation
        if n_unique > self.n_classes:
            raise ValueError(
                f"Found {n_unique} unique class labels in y, but n_classes={self.n_classes}. "
                f"Increase n_classes or check your data."
            )
        elif n_unique < self.n_classes:
            raise ValueError(
                f"Found only {n_unique} unique class labels in y, but n_classes={self.n_classes}. "
                f"To correctly set the ``classes_`` attribute, provide an array containing all the "
                f"unique class labels.",
            )

        # Always store the actual classes array
        self._classes_ = classes

        # Check if classes are the default [0, 1, ..., n_classes-1]
        # If so, we can skip encoding/decoding for performance
        is_default = np.array_equal(classes, np.arange(self.n_classes))
        self._skip_encoding = is_default

        # Create dict lookup only when needed (non-default classes)
        self._class_to_index_ = (
            None if is_default else {label: i for i, label in enumerate(classes)}
        )
        return self

    def _encode_labels(self, y: ArrayLike) -> NDArray[int]:
        """Convert user-provided class labels to internal indices [0, n_classes-1]."""
        if self._skip_encoding:
            return y
        # use dict lookup instead of `np.searchsorted`
        # this approach will fail for label mismatches
        try:
            y = np.asarray(y)
            original_shape = y.shape
            y = np.fromiter(
                (self._class_to_index_[label] for label in y.ravel()),
                dtype=int,
                count=y.size,
            ).reshape(original_shape)
        except KeyError as e:
            unq_labels = np.unique(y)
            valid = list(self._class_to_index_.keys())
            invalid = [lab for lab in unq_labels if lab not in valid]
            raise ValueError(
                f"Unrecognized label(s) {invalid}. " f"Valid labels are {valid}."
            ) from e
        return y

    def _decode_labels(self, indices: NDArray[int]) -> NDArray:
        """Convert internal indices [0, n_classes-1] back to user-provided class labels."""
        if self._skip_encoding:
            return indices
        return self._classes_[indices]

    @property
    def classes_(self) -> NDArray | None:
        """Class labels, or None if not set."""
        return self._classes_

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
            Parameter tuple of (coefficients, intercept).
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
        ValueError
            If ``classes_`` has not been set, or if inputs/parameters have
            incompatible shapes or invalid values.
        """
        self._check_classes_is_set("compute_loss")
        y = self._encode_labels(y)
        return super().compute_loss(params, X, y, *args, **kwargs)

    @property
    def n_classes(self):
        """Number of classes."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value: int):
        # extract item from scalar arrays
        if is_numpy_array_like(value)[1] and value.size == 1:
            value = value.item()

        if not isinstance(value, Number) or value < 2 or not int(value) == value:
            raise ValueError(
                "The number of classes must be an integer greater than or equal to 2."
            )
        self._n_classes = int(value)
        # reset validator.
        self._validator = self._validator_class(
            extra_params=self._get_validator_extra_params()
        )
        # reset classes cache
        self._classes_ = None
        self._skip_encoding = False
        self._class_to_index_ = None

    def _get_validator_extra_params(self) -> dict:
        """Get validator extra parameters."""
        return {"n_classes": self._n_classes}

    def _preprocess_inputs(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Optional[jnp.ndarray] = None,
        drop_nans: bool = True,
    ) -> Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray | None]:
        """Preprocess inputs before initializing state."""
        X, y = super()._preprocess_inputs(X, y=y, drop_nans=drop_nans)
        if y is not None:
            y = self._validator.check_and_cast_y_to_integer(y)
            y = jax.nn.one_hot(y, self._n_classes)
        return X, y

    # Note: necessary double decorator. The super().predict is decorated as well,
    # but the pynapple metadata would be dropped if we do not decorate here.
    # This happens because super().predict returns the log-proba which have the same
    # shape of one_hot(y), not matching the original y.shape.
    @support_pynapple(conv_type="jax")
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X :
            The input samples. Can be an array of shape ``(n_samples, n_features)``
            or a ``FeaturePytree`` with arrays as leaves.

        Returns
        -------
        :
            Predicted class labels for each sample.
            Returns an integer array of shape  ``(n_samples, )`` with values in
            ``[0, n_classes - 1]``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> predictions = model.predict(X)
        >>> predictions.shape
        (4,)
        """
        # Below will raise if user set manually coef and intercept
        # and calls predict.
        # One could assume default labels 0,...,n-1
        # but requiring to be explicit is safer
        self._check_classes_is_set("predict")
        log_proba = super().predict(X)
        return self._decode_labels(jnp.argmax(log_proba, axis=-1))

    def predict_proba(
        self,
        X: DESIGN_INPUT_TYPE,
        return_type: Literal["log-proba", "proba"] = "log-proba",
    ) -> jnp.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X :
            The input samples. Can be an array of shape ``(n_samples, n_features)``
            or a ``FeaturePytree`` with arrays as leaves.
        return_type :
            The format of the returned probabilities. If ``"log-proba"``, returns
            log-probabilities. If ``"proba"``, returns probabilities. Defaults to
            ``"log-proba"``.

        Returns
        -------
        :
            Predicted class probabilities. Returns an array of shape ``(n_samples, n_classes)``
            where each row sums to 1 (for probabilities) or to 0 in log-space (for log-probabilities).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> proba = model.predict_proba(X, return_type="proba")
        >>> proba.shape
        (4, 2)
        """
        # Below will raise if user set manually coef and intercept
        # and calls predict without setting the class label mapping.
        # One could assume default labels 0,...,n-1
        # but requiring to be explicit makes the mapping between
        # the class labels and the probability index less ambiguous:
        #   `log_proba[:, i]` is the log-proba of class `self.classes_[i]`
        self._check_classes_is_set("predict_proba")
        # log-proba for categorical, proba for Bernoulli
        log_proba = super().predict(X)
        if return_type == "log-proba":
            return log_proba
        elif return_type == "proba":
            exp = support_pynapple(conv_type="jax")(jnp.exp)
            proba = exp(log_proba)
            # renormalize (sum to 1 constraint)
            proba /= proba.sum(axis=-1, keepdims=True)
            return proba
        else:
            raise ValueError(f"Unrecognized return type ``'{return_type}'``")

    def _estimate_resid_degrees_of_freedom(
        self, X: DESIGN_INPUT_TYPE, n_samples: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Estimate the degrees of freedom of the residuals for classifier GLM.

        Parameters
        ----------
        X :
            The design matrix.
        n_samples :
            The number of samples observed. If not provided, n_samples is set to
            ``X.shape[0]``. If the fit is batched, n_samples could be larger than
            ``X.shape[0]``.

        Returns
        -------
        :
            An estimate of the degrees of freedom of the residuals.
        """
        # Convert a pytree to a design-matrix
        x_leaf = jax.tree_util.tree_leaves(X)

        if n_samples is None:
            n_samples = x_leaf[0].shape[0]
        else:
            if not isinstance(n_samples, int):
                raise TypeError(
                    f"`n_samples` must be `None` or of type `int`. "
                    f"Type {type(n_samples)} provided instead!"
                )

        n_features = sum(x.shape[1] for x in x_leaf)
        # Effective degrees of freedom is n_classes - 1 due to probability simplex constraint
        n_m1_classes = self._n_classes - 1
        params = self._get_model_params()

        # Infer n_neurons from coef shape:
        # ClassifierGLM: coef is (n_features, n_classes) -> n_neurons = 1
        # ClassifierPopulationGLM: coef is (n_features, n_neurons, n_classes) -> n_neurons = shape[1]
        coef_leaf = jax.tree_util.tree_leaves(params.coef)[0]
        n_neurons = 1 if coef_leaf.ndim == 2 else coef_leaf.shape[1]

        # For Lasso-type regularizers, use the non-zero coefficients as DOF estimate
        # see https://arxiv.org/abs/0712.0881
        if isinstance(self.regularizer, (GroupLasso, Lasso, ElasticNet)):
            # Sum over features (axis 0) and classes (axis -1)
            # This leaves shape (n_neurons,) for ClassifierPopulationGLM
            # or scalar for ClassifierGLM
            resid_dof = tree_utils.pytree_map_and_reduce(
                lambda x: ~jnp.isclose(x, jnp.zeros_like(x)),
                lambda x: sum([jnp.sum(i, axis=(0, -1)) for i in x]),
                params.coef,
            )
            return jnp.atleast_1d(n_samples - resid_dof - n_m1_classes)

        elif isinstance(self.regularizer, Ridge):
            # For Ridge, use total parameters
            return (n_samples - n_m1_classes * n_features - n_m1_classes) * jnp.ones(
                n_neurons
            )

        else:
            # For UnRegularized, use the rank
            rank = jnp.linalg.matrix_rank(jnp.concatenate(x_leaf, axis=1))
            return (n_samples - rank * n_m1_classes - n_m1_classes) * jnp.ones(
                n_neurons
            )

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate categorical responses from the model.

        Parameters
        ----------
        random_key :
            A JAX random key used to generate the simulated responses.
        feedforward_input :
            The input samples used to generate the responses. Can be an array of
            shape ``(n_samples, n_features)`` or a ``FeaturePytree`` with arrays
            as leaves.

        Returns
        -------
        :
            A tuple ``(y, log_prob)`` where:
            - ``y`` is an array of shape ``(n_samples,)`` containing the
              simulated class labels (in the same format as ``classes_``).
            - ``log_prob`` is an array of shape ``(n_samples,)`` containing the
              log-probability of the simulated responses under the model.

        Raises
        ------
        ValueError
            If ``classes_`` has not been set. Call :meth:`set_classes` or :meth:`fit`
            before calling this method.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> key = jax.random.key(0)
        >>> simulated_y, log_prob = model.simulate(key, X)
        >>> simulated_y.shape
        (4,)
        """
        self._check_classes_is_set("simulate")
        y, log_prob = super().simulate(random_key, feedforward_input)
        argmax = support_pynapple(conv_type="jax")(lambda x: jnp.argmax(x, axis=-1))
        y = self._decode_labels(argmax(y))
        return y, log_prob

    def initialize_solver_and_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params: UserProvidedParamsT,
    ) -> SolverState:
        """Initialize the solver and its state for running fit and update.

        This method must be called before using :meth:`update` for iterative optimization.
        It sets up the solver with the provided initial parameters and data.

        Parameters
        ----------
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target labels, array of shape ``(n_time_bins,)`` for single neuron/subject models or
            ``(n_time_bins, n_neurons)`` for population models.
        init_params
            Initial parameter tuple of (coefficients, intercept).

        Returns
        -------
        state
            Initial solver state.

        Raises
        ------
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        self._check_classes_is_set("initialize_solver_and_state")
        y = self._encode_labels(y)
        return super().initialize_solver_and_state(X, y, init_params)

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
            Class labels as integers, array of shape ``(n_time_bins,)`` for single neuron
            models or ``(n_time_bins, n_neurons)`` for population models. Values should be
            in the range ``[0, n_classes - 1]``.

        Returns
        -------
        :
            Initial parameter tuple of (coefficients, intercept).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2)
        >>> model.set_classes(y)
        ClassifierGLM(...)
        >>> coef, intercept = model.initialize_params(X, y)
        >>> coef.shape
        (2, 2)
        """
        self._check_classes_is_set("initialize_params")
        y = self._encode_labels(y)
        y = self._validator.check_and_cast_y_to_integer(y)
        y = jax.nn.one_hot(y, self.n_classes)
        return super().initialize_params(X, y)

    def update(
        self,
        params: GLMUserParams,
        opt_state: SolverState,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """
        Update the model parameters and solver state.

        Performs a single optimization step using the model's solver. Class labels
        are automatically encoded to internal indices and converted to one-hot
        encoding before the update.

        **Important**: Labels of any dtype (integers, floats, strings, etc.) are
        supported and will be encoded using the ``classes_`` attribute set via
        :meth:`set_classes`. For best performance, use integer labels ``[0, n_classes - 1]``.

        Parameters
        ----------
        params :
            The current model parameters, typically a tuple of coefficients and intercepts.
        opt_state :
            The current state of the optimizer, encapsulating information necessary for the
            optimization algorithm to continue from the current state.
        X :
            The predictors used in the model fitting process. Shape ``(n_time_bins, n_features)``
            or a ``FeaturePytree``.
        y :
            Class labels, array of shape ``(n_time_bins,)`` for single neuron
            models or ``(n_time_bins, n_neurons)`` for population models. Labels must
            match those defined in ``classes_``.
        *args :
            Additional positional arguments to be passed to the solver's update method.
        n_samples :
            The total number of samples. Usually larger than the samples of an individual batch,
            used to estimate the scale parameter of the GLM.
        **kwargs :
            Additional keyword arguments to be passed to the solver's update method.

        Returns
        -------
        params :
            Updated model parameters (coefficients, intercepts).
        state :
            Updated optimizer state.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2)
        >>> model.set_classes(y)
        ClassifierGLM(...)
        >>> params = model.initialize_params(X, y)
        >>> opt_state = model.initialize_solver_and_state(X, y, params)
        >>> new_params, new_state = model.update(params, opt_state, X, y)
        """
        self._check_classes_is_set("update")
        y = self._encode_labels(y)
        # note: do not check and cast here. Risky but the performance of
        # the update has priority.
        y = jax.nn.one_hot(jnp.asarray(y, dtype=int), self._n_classes)
        return super().update(
            params, opt_state, X, y, *args, n_samples=n_samples, **kwargs
        )


class ClassifierGLM(ClassifierMixin, GLM):
    """
    Generalized Linear Model for multi-class classification.

    This model predicts discrete class labels from input features using a
    softmax (multinomial logistic) model. It uses an over-parameterized
    representation with one set of coefficients per class, resulting in
    coefficient shape ``(n_features, n_classes)`` and intercept shape ``(n_classes,)``.

    Parameters
    ----------
    n_classes
        The number of classes. Must be >= 2.
    inverse_link_function
        The inverse link function. Default is ``log_softmax``.
    regularizer
        The regularization scheme. Default is ``Ridge``. Note that the
        model is over-parameterized: one set of coefficients for each class.
        Regularization makes the parameters identifiable. Setting ``UnRegularized``
        will result in non-identifiable coefficients, see note below.
    regularizer_strength
        The strength of the regularization.
    solver_name
        The solver to use for optimization.
    solver_kwargs
        Additional keyword arguments for the solver.

    Attributes
    ----------
    coef_
        Fitted coefficients of shape ``(n_features, n_classes)`` after calling :meth:`fit`.
    intercept_
        Fitted intercepts of shape ``(n_classes,)`` after calling :meth:`fit`.

    Notes
    -----
    **Identifiability**

    This model uses an over-parameterized (symmetric) representation where each class
    has its own set of coefficients. Since probabilities from softmax are invariant to
    adding a constant to all linear predictors, the parameters are not uniquely
    identifiable without regularization. For example, if ``(coef, intercept)`` is a
    solution, so is ``(coef + c, intercept + c)`` for any constant ``c``.

    Using regularization (default is ``Ridge``) resolves this ambiguity by penalizing
    the parameter magnitudes, effectively centering the solution. If you use
    ``UnRegularized``, the optimization may converge to different equivalent solutions
    depending on initialization, though predictions will be identical.

    **Class Labels**

    The target array ``y`` can contain any hashable class labels that can be stored
    in a NumPy array, including integers, strings, or other hashable types. The model
    internally maps these labels to indices ``[0, n_classes - 1]`` for computation
    and maps them back when returning predictions.

    **Performance Considerations**

    For optimal performance, use integer labels ``[0, 1, ..., n_classes - 1]``. When
    labels follow this convention, the model skips the encoding/decoding steps entirely.
    Using other label formats (e.g., ``["cat", "dog"]`` or ``[5, 10, 15]``) incurs a
    small overhead for label translation.

    **Setting Class Labels**

    The :meth:`fit` and :meth:`initialize_solver_and_state` methods automatically infer
    class labels from the provided ``y``. If you set ``coef_`` and ``intercept_`` manually,
    you must call :meth:`set_classes` before using :meth:`predict`, :meth:`predict_proba`,
    :meth:`simulate`, :meth:`score`, or :meth:`compute_loss`.

    See Also
    --------
    ClassifierPopulationGLM : Multi-class classification for multiple neurons.
    GLM : Generalized Linear Model for continuous/count responses.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import nemos as nmo
    >>> # Binary classification with integer labels (most efficient)
    >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    >>> y = jnp.array([0, 0, 1, 1])
    >>> model = nmo.glm.ClassifierGLM(n_classes=2)
    >>> model = model.fit(X, y)
    >>> predictions = model.predict(X)  # Returns class labels
    >>> probabilities = model.predict_proba(X, return_type="proba")
    """

    _validator_class = ClassifierGLMValidator

    def __init__(
        self,
        n_classes: Optional[int] = 2,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[RegularizerStrength] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        self.n_classes = n_classes
        observation_model = obs.CategoricalObservations()
        if regularizer is None:
            regularizer = "Ridge"
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self._classes_ = None
        self._class_to_index_ = None
        self._skip_encoding = False

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[GLMUserParams] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        X
            Training input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            Target class labels of shape ``(n_samples,)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        init_params
            Initial parameter values as tuple of ``(coef, intercept)``. If None,
            parameters are initialized automatically.

        Returns
        -------
        :
            The fitted model.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2)
        >>> model = model.fit(X, y)
        >>> model.coef_.shape
        (2, 2)
        """
        self.set_classes(y)
        y = self._encode_labels(y)
        return super().fit(X, y, init_params)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Optional[Callable] = jnp.mean,
    ) -> jnp.ndarray:
        """
        Score the model on test data.

        Parameters
        ----------
        X
            Test input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            True class labels of shape ``(n_samples,)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        score_type
            The type of score to compute.
        aggregate_sample_scores
            Function to aggregate per-sample scores.

        Returns
        -------
        :
            The computed score.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> score = model.score(X, y)
        """
        # check if classes are not set, aka user set the coef and intercept
        # manually, raise otherwise there may be ambiguity in interpreting
        # the labels.
        self._check_classes_is_set("score")
        y = self._encode_labels(y)
        return super().score(X, y, score_type, aggregate_sample_scores)


class ClassifierPopulationGLM(ClassifierMixin, PopulationGLM):
    """
    Population Generalized Linear Model for multi-class classification.

    This model predicts discrete class labels from input features using a
    softmax (multinomial logistic) model for multiple neurons simultaneously.
    It uses an over-parameterized representation with one set of coefficients
    per class, resulting in coefficient shape ``(n_features, n_neurons, n_classes)``
    and intercept shape ``(n_neurons, n_classes)``.

    Parameters
    ----------
    n_classes
        The number of classes. Must be >= 2.
    inverse_link_function
        The inverse link function. Default is ``log_softmax``.
    regularizer
        The regularization scheme. Default is ``Ridge``. Note that the
        model is over-parameterized: one set of coefficients for each class.
        Regularization makes the parameters identifiable. Setting ``UnRegularized``
        will result in non-identifiable coefficients, see note below.
    regularizer_strength
        The strength of the regularization.
    solver_name
        The solver to use for optimization.
    solver_kwargs
        Additional keyword arguments for the solver.
    feature_mask
        Mask indicating which features are used for each neuron.

    Attributes
    ----------
    coef_
        Fitted coefficients of shape ``(n_features, n_neurons, n_classes)``
        after calling :meth:`fit`.
    intercept_
        Fitted intercepts of shape ``(n_neurons, n_classes)`` after calling :meth:`fit`.

    Notes
    -----
    **Identifiability**

    This model uses an over-parameterized (symmetric) representation where each class
    has its own set of coefficients. Since probabilities from softmax are invariant to
    adding a constant to all linear predictors, the parameters are not uniquely
    identifiable without regularization. For example, if ``(coef, intercept)`` is a
    solution, so is ``(coef + c, intercept + c)`` for any constant ``c``.

    Using regularization (default is ``Ridge``) resolves this ambiguity by penalizing
    the parameter magnitudes, effectively centering the solution. If you use
    ``UnRegularized``, the optimization may converge to different equivalent solutions
    depending on initialization, though predictions will be identical.

    **Class Labels**

    The target array ``y`` can contain any hashable class labels that can be stored
    in a NumPy array, including integers, strings, or other hashable types. The model
    internally maps these labels to indices ``[0, n_classes - 1]`` for computation
    and maps them back when returning predictions.

    **Performance Considerations**

    For optimal performance, use integer labels ``[0, 1, ..., n_classes - 1]``. When
    labels follow this convention, the model skips the encoding/decoding steps entirely.
    Using other label formats (e.g., ``["cat", "dog"]`` or ``[5, 10, 15]``) incurs a
    small overhead for label translation.

    **Setting Class Labels**

    The :meth:`fit` and :meth:`initialize_solver_and_state` methods automatically infer
    class labels from the provided ``y``. If you set ``coef_`` and ``intercept_`` manually,
    you must call :meth:`set_classes` before using :meth:`predict`, :meth:`predict_proba`,
    :meth:`simulate`, :meth:`score`, or :meth:`compute_loss`.

    See Also
    --------
    ClassifierGLM : Multi-class classification for a single neuron.
    PopulationGLM : Population GLM for continuous/count responses.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import nemos as nmo
    >>> # Multi-class classification for 2 neurons (integer labels, most efficient)
    >>> X = jnp.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
    >>> y = jnp.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
    >>> model = nmo.glm.ClassifierPopulationGLM(n_classes=3)
    >>> model = model.fit(X, y)
    >>> predictions = model.predict(X)  # Returns class labels, shape (n_samples, n_neurons)
    """

    _validator_class = PopulationClassifierGLMValidator

    def __init__(
        self,
        n_classes: Optional[int] = 2,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[RegularizerStrength] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
        feature_mask: Optional[jnp.ndarray] = None,
    ):
        self.n_classes = n_classes
        observation_model = obs.CategoricalObservations()
        if regularizer is None:
            regularizer = "Ridge"
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
            feature_mask=feature_mask,
        )
        self._classes_ = None
        self._class_to_index_ = None
        self._skip_encoding = False

    @property
    def feature_mask(self) -> Union[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Mask indicating which weights are used, matching the coefficients shape.

        The feature mask has the same structure and shape as the coefficients (``coef_``):

        - **Array input**: Shape ``(n_features, n_neurons, n_classes)``.
          Each entry ``[i, j, k]`` indicates whether the weight for feature ``i``,
          neuron ``j``, and category ``k`` is used (1 = used, 0 = masked).

        - **Dict/FeaturePytree input**: A dict with keys matching ``coef_``.
          Each leaf array has the same shape as the corresponding coefficient leaf
          ``(n_features_per_key, n_neurons, n_classes)``.

        Returns
        -------
        :
            The feature mask, or None if not set.
        """
        return self._feature_mask

    @feature_mask.setter
    def feature_mask(self, feature_mask: Union[DESIGN_INPUT_TYPE, dict]):
        # do not allow reassignment after fit
        if (self.coef_ is not None) and (self.intercept_ is not None):
            raise AttributeError(
                "property 'feature_mask' of 'populationGLM' cannot be set after fitting."
            )

        self._feature_mask = self._validator.validate_and_cast_feature_mask(
            feature_mask
        )

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[GLMUserParams] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        X
            Training input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            Target class labels of shape ``(n_samples, n_neurons)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        init_params
            Initial parameter values as tuple of ``(coef, intercept)``. If None,
            parameters are initialized automatically.

        Returns
        -------
        :
            The fitted model.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
        >>> y = jnp.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
        >>> model = nmo.glm.ClassifierPopulationGLM(n_classes=3)
        >>> model = model.fit(X, y)
        >>> model.coef_.shape
        (2, 2, 3)
        """
        self.set_classes(y)
        y = self._encode_labels(y)
        return super().fit(X, y, init_params)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Optional[Callable] = jnp.mean,
    ) -> jnp.ndarray:
        """
        Score the model on test data.

        Parameters
        ----------
        X
            Test input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            True class labels of shape ``(n_samples, n_neurons)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        score_type
            The type of score to compute.
        aggregate_sample_scores
            Function to aggregate per-sample scores.

        Returns
        -------
        :
            The computed score.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
        >>> y = jnp.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
        >>> model = nmo.glm.ClassifierPopulationGLM(n_classes=3).fit(X, y)
        >>> score = model.score(X, y)
        """
        self._check_classes_is_set("score")
        y = self._encode_labels(y)
        return super().score(X, y, score_type, aggregate_sample_scores)
