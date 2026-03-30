from typing import Any, Callable, Optional, Union

import jax.numpy as jnp
from numpy._typing import ArrayLike

from tests.test_observation_models import observation_model_rate_and_samples

from ..glm import GLM
from ..observation_models import NegativeBinomialObservations
from ..regularizer import Regularizer
from ..typing import DESIGN_INPUT_TYPE
from .params import GLMParams, GLMUserParams, NBGLMParams, NBGLMUserParams


def _joint_fit():
    pass


class NBGLM(GLM):
    def __init__(
        self,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Any = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        observation_model = NegativeBinomialObservations()
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[NBGLMUserParams] = None,
    ):
        self._validator.validate_inputs(X, y)

        # filter for non-nans, grab data if needed
        data, y = self._preprocess_inputs(X, y)
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )

        self._initialize_solver_and_state(data, y, init_params)
        params, state, aux = self.solver_run(init_params, data, y, self.scale)

    def _compute_loss(
        self,
        params: GLMParams,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        scale: float,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        predicted_rate = self._predict(params, X)
        return self._observation_model._negative_log_likelihood(
            y, predicted_rate, scale=scale
        )
