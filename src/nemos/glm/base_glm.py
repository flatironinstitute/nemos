"""Abstract base class shared by all GLM variants."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from sklearn.utils import InputTags, TargetTags

from .. import observation_models as obs
from .. import validation
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor
from ..exceptions import NotFittedError
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..solvers._compute_defaults import glm_compute_optimal_stepsize_configs
from ..type_casting import cast_to_jax, support_pynapple
from ..typing import DESIGN_INPUT_TYPE, ModelParamsT, UserProvidedParamsT
from ..utils import format_repr

__all__ = ["BaseGLM"]


class BaseGLM(BaseRegressor[UserProvidedParamsT, ModelParamsT]):
    """Abstract base for all GLM variants.

    Provides observation model / inverse link function handling and the generic
    implementations of ``predict``, ``score``, and ``simulate``, which dispatch
    through ``_get_model_params`` and ``_predict``. Subclasses that change the
    parameter structure only need to override those two methods and supply a
    matching validator.

    Subclasses must define the following class-level attribute:

    - ``_validator_class``: a concrete :class:`~nemos.base_validator.RegressorValidator`
      subclass used to validate and convert user-provided parameters.

    And the following abstract methods (in addition to those required by
    :class:`~nemos.base_regressor.BaseRegressor`):

    - ``_predict``
    - ``fit``
    - ``_compute_loss``
    - ``_get_model_params``
    - ``_set_model_params``
    - ``_initialize_optimizer_and_state``
    - ``_model_specific_initialization``
    """

    _invalid_observation_types: tuple = ()
    _validator_class: type  # must be set by each concrete subclass

    def __init__(
        self,
        observation_model: Union[
            obs.Observations,
            Literal["Poisson", "Gamma", "Gaussian", "Bernoulli", "NegativeBinomial"],
        ] = "Poisson",
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str]] = None,
        regularizer_strength: Any = None,
        solver_name: Optional[str] = None,
        solver_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function
        self._validator = self._validator_class(
            extra_params=self._get_validator_extra_params()
        )
        # fitted-output attributes
        self.coef_ = None
        self.intercept_ = None
        self.scale_ = None
        self.solver_state_ = None
        self.dof_resid_ = None
        self.aux_ = None
        self._solver = None

    @property
    def solver(self):
        """Getter for the solver class."""
        return self._solver

    @classmethod
    def _validate_observation_class(cls, observation: obs.Observations) -> None:
        """Raise TypeError if the observation type is not supported for this class."""
        if observation.__class__ in cls._invalid_observation_types:
            raise TypeError(
                f"The ``{observation.__class__.__name__}`` observation type is not "
                f"supported for ``{cls.__name__}`` models."
            )

    def __sklearn_tags__(self):
        """Return GLM-specific estimator tags."""
        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(allow_nan=True, two_d_array=True)
        tags.target_tags = TargetTags(
            required=True, one_d_labels=True, two_d_labels=False
        )
        return tags

    @property
    def inverse_link_function(self) -> Callable:
        """Getter for the inverse link function."""
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Setter for the inverse link function."""
        self._inverse_link_function = resolve_inverse_link_function(
            inverse_link_function, self._observation_model
        )

    @property
    def observation_model(self) -> Union[None, obs.Observations]:
        """Getter for the ``observation_model`` attribute."""
        return self._observation_model

    @observation_model.setter
    def observation_model(self, observation: Union[obs.Observations, str]):
        """Setter for the ``observation_model`` attribute."""
        if isinstance(observation, str):
            self._observation_model = instantiate_observation_model(observation)
            self._validate_observation_class(self.observation_model)
            return
        obs.check_observation_model(observation)
        self._observation_model = observation
        self._validate_observation_class(self.observation_model)

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        if (self.coef_ is None) or (self.intercept_ is None):
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    @support_pynapple(conv_type="jax")
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """Predict rates based on fit parameters.

        Parameters
        ----------
        X :
            Predictors, array of shape ``(n_time_bins, n_features)`` or a pytree
            of arrays of the same shape.

        Returns
        -------
        :
            The predicted rates with shape ``(n_time_bins, )``.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If ``X`` is not two-dimensional.
        ValueError
            If there's an inconsistent number of features between spike basis
            coefficients and ``X``.
        """
        self._check_is_fit()
        params = self._get_model_params()
        data, _ = self._preprocess_inputs(X, drop_nans=False)
        self._validator.validate_inputs(data)
        self._validator.validate_consistency(params, X=data)
        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), params
        )
        return self._predict(params, data)

    @cast_to_jax
    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Evaluate the goodness-of-fit of the model to the observed neural data.

        This method computes the goodness-of-fit score, which can either be the mean
        log-likelihood or of two versions of the pseudo-:math:`R^2`.
        The scoring process includes validation of input compatibility with the model's
        parameters, ensuring that the model has been previously fitted and the input data
        are appropriate for scoring. A higher score indicates a better fit of the model
        to the observed data.

        Parameters
        ----------
        X :
            Predictors, array of shape ``(n_time_bins, n_features)`` or a pytree
            of arrays of the same shape.
        y :
            Neural activity. Shape ``(n_time_bins, )``.
        score_type :
            Type of scoring: either log-likelihood or pseudo-:math:`R^2`.
        aggregate_sample_scores :
            Function that aggregates the score of all samples.

        Returns
        -------
        score :
            The log-likelihood or the pseudo-:math:`R^2` of the current model.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If X structure doesn't match the params, and if X and y have different
            number of samples.
        """
        self._check_is_fit()
        params = self._get_model_params()
        self._validator.validate_inputs(X, y)
        X, y = self._preprocess_inputs(X, y, drop_nans=True)
        self._validator.validate_consistency(params, X, y)
        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), params
        )
        if score_type == "log-likelihood":
            return self._observation_model.log_likelihood(
                y,
                self._predict(params, X),
                self.scale_,
                aggregate_sample_scores=aggregate_sample_scores,
            )
        elif score_type.startswith("pseudo-r2"):
            return self._observation_model.pseudo_r2(
                y,
                self._predict(params, X),
                score_type=score_type,
                scale=self.scale_,
                aggregate_sample_scores=aggregate_sample_scores,
            )
        else:
            raise NotImplementedError(
                f"Scoring method {score_type} not implemented! "
                "`score_type` must be either 'log-likelihood', 'pseudo-r2-McFadden', "
                "or 'pseudo-r2-Cohen'."
            )

    @support_pynapple(conv_type="jax")
    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate neural activity in response to a feed-forward input.

        Parameters
        ----------
        random_key :
            jax.random.key for seeding the simulation.
        feedforward_input :
            External input predictors to the model. Array of shape
            ``(n_time_bins, n_basis_input)`` or pytree with leaves of the same shape.

        Returns
        -------
        simulated_activity :
            Simulated activity for the neuron over time. Shape: ``(n_time_bins, )``.
        firing_rates :
            Simulated rates for the neuron over time. Shape: ``(n_time_bins, )``.

        Raises
        ------
        NotFittedError
            If the model hasn't been fitted prior to calling this method.
        """
        self._check_is_fit()
        params = self._get_model_params()
        validation.error_all_invalid(feedforward_input)
        self._validator.validate_inputs(X=feedforward_input)
        self._validator.validate_consistency(params, X=feedforward_input)
        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), params
        )
        feedforward_input, _ = self._preprocess_inputs(
            X=feedforward_input, drop_nans=False
        )
        predicted_rate = self._predict(params, feedforward_input)
        return (
            self._observation_model.sample_generator(
                key=random_key, predicted_rate=predicted_rate, scale=self.scale_
            ),
            predicted_rate,
        )

    def _get_optimal_solver_params_config(self):
        """Return functions for computing default step and batch size for the solver."""
        return glm_compute_optimal_stepsize_configs(self)

    def __repr__(self):
        """Representation of the model."""
        return format_repr(
            self, multiline=True, use_name_keys=["inverse_link_function"]
        )

    def __sklearn_clone__(self):
        """Clone the model."""
        params = self.get_params(deep=False)
        return self.__class__(**params)
