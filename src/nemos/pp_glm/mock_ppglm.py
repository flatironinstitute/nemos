from typing import Any, Callable, Optional, Union

import equinox as eqx
import jax.numpy as jnp

from ..base_regressor import BaseRegressor
from ..glm.params import GLMParams

from ..regularizer import Regularizer
from ..typing import DESIGN_INPUT_TYPE, UserProvidedParamsT
from .params import PPGLMParamsWithKey
from .data import X_ppglm, y_ppglm
from .validation import (
    PopulationPPGLMValidator,
    PPGLMValidator,
)


class MockPPGLM(BaseRegressor):
    """
    Minimal PP-GLM stand-in for testing PPGLMValidator logic.

    """

    def __init__(
        self,
        n_basis_funcs: int,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Any = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        self.n_basis_funcs = n_basis_funcs

        # validator is a frozen dataclass — constructed explicitly with n_basis_funcs
        self._validator = PPGLMValidator(n_basis_funcs=self.n_basis_funcs)

        self.coef_: Optional[jnp.ndarray] = None
        self.intercept_: Optional[jnp.ndarray] = None

    def _get_model_params(self) -> GLMParams:
        return self._validator.to_model_params(
            (
                self.coef_,
                self.intercept_,
            )
        )

    def _set_model_params(self, params):
        coef, intercept = self._validator.from_model_params(params)
        self.coef_ = coef
        self.intercept_ = intercept

    def _check_model_is_fit(self):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")

    def _model_specific_initialization(
        self,
        X: X_ppglm,
        y: y_ppglm,
        **kwargs,
    ) -> GLMParams:

        empty_params = self._validator.get_empty_params(X, y)

        n_neurons = jnp.unique(X.ids).size
        n_features = int(n_neurons * self.n_basis_funcs)

        init_params = eqx.tree_at(
            lambda p: (p.coef, p.intercept),
            empty_params,
            (jnp.zeros(n_features), jnp.ones(n_neurons)),
        )

        return init_params

    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> UserProvidedParamsT:
        pass

    def fit(self, X: X_ppglm, y: y_ppglm, init_params=None):
        """Validate inputs and set zero params; no actual optimization."""
        fit_params = self._model_specific_initialization(X, y)
        self._set_model_params(fit_params)

    def predict(self, X):
        self._check_model_is_fit()
        return jnp.array(0.0)

    def score(self, X, y, **kwargs):
        self._check_model_is_fit()
        return jnp.array(0.0)

    def simulate(self, *args, **kwargs):
        pass

    def save_params(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def _log_likelihood(self, params, X, y):
        return jnp.zeros(0.0)

    def _compute_loss(
        self,
        params: PPGLMParamsWithKey,
        X: X_ppglm,
        y: y_ppglm,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return jnp.array(0.0)

    def _get_optimal_solver_params_config(self, *args, **kwargs):
        pass


class MockPopulationPPGLM(MockPPGLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validator = PopulationPPGLMValidator(n_basis_funcs=self.n_basis_funcs)

