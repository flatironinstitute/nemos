import inspect
import warnings
from contextlib import nullcontext as does_not_raise
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sklearn
import statsmodels.api as sm
from pynapple import Tsd, TsdFrame
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import GridSearchCV

import nemos as nmo
from nemos.pytrees import FeaturePytree
from nemos.tree_utils import pytree_map_and_reduce, tree_l2_norm, tree_slice, tree_sub


class TestPopulationGLM:
    """
    Unit tests specific to the PopulationGLM class.
    """

    #######################
    # Test model.score
    #######################

    @pytest.mark.parametrize(
        "score_type", ["log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"]
    )
    def test_score_aggregation_ndim(self, score_type, poisson_population_GLM_model):
        """
        Test that the aggregate samples returns the right dimensional object.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        mn = model.score(X, y, score_type=score_type, aggregate_sample_scores=jnp.mean)
        mn_n = model.score(
            X,
            y,
            score_type=score_type,
            aggregate_sample_scores=lambda x: jnp.mean(x, axis=0),
        )
        assert mn.ndim == 0
        assert mn_n.ndim == 1

    #######################################
    # Compare with standard implementation
    #######################################

    def test_sklearn_clone(self, poisson_population_GLM_model):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.fit(X, y)
        cloned = sklearn.clone(model)
        assert cloned.feature_mask is None, "cloned GLM shouldn't have feature mask!"
        assert model.feature_mask is not None, "fit GLM should have feature mask!"

    @pytest.mark.parametrize(
        "mask, expectation",
        [
            (np.array([0, 1, 1] * 5).reshape(5, 3), does_not_raise()),
            (
                {"input_1": [0, 1, 0], "input_2": [1, 0, 1]},
                pytest.raises(
                    ValueError,
                    match="'feature_mask' of 'populationGLM' must be a 2-dimensional array",
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                does_not_raise(),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1.1])},
                pytest.raises(
                    ValueError, match="'feature_mask' must contain only 0s and 1s"
                ),
            ),
            (
                np.array([0.1, 1, 1] * 5).reshape(5, 3),
                pytest.raises(
                    ValueError, match="'feature_mask' must contain only 0s and 1s"
                ),
            ),
        ],
    )
    def test_feature_mask_setter(self, mask, expectation, poisson_population_GLM_model):
        _, _, model, _, _ = poisson_population_GLM_model
        with expectation:
            model.feature_mask = mask

    @pytest.mark.parametrize(
        "mask, expectation",
        [
            (np.array([0, 1, 1] * 5).reshape(5, 3), does_not_raise()),
            (
                np.array([0, 1, 1] * 4).reshape(4, 3),
                pytest.raises(ValueError, match="Inconsistent number of features"),
            ),
            (
                np.array([0, 1, 1, 1] * 5).reshape(5, 4),
                pytest.raises(ValueError, match="Inconsistent number of neurons"),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1]), "input_2": np.array([1, 0, 1, 0])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("attr_name", ["fit", "predict", "score"])
    def test_feature_mask_compatibility_fit(
        self, mask, expectation, attr_name, poisson_population_GLM_model
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.feature_mask = mask
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            if attr_name == "predict":
                getattr(model, attr_name)(X)
            else:
                getattr(model, attr_name)(X, y)

    @pytest.mark.parametrize(
        "mask, expectation",
        [
            (
                np.array([0, 1, 1] * 5).reshape(5, 3),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                np.array([0, 1, 1] * 4).reshape(4, 3),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                np.array([0, 1, 1, 1] * 5).reshape(5, 4),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                does_not_raise(),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1]), "input_2": np.array([1, 0, 1, 0])},
                pytest.raises(ValueError, match="Inconsistent number of neurons"),
            ),
            (
                {"input_1": np.array([0, 1, 0])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("attr_name", ["fit", "predict", "score"])
    def test_feature_mask_compatibility_fit_tree(
        self, mask, expectation, attr_name, poisson_population_GLM_model_pytree
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model_pytree
        model.feature_mask = mask
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            if attr_name == "predict":
                getattr(model, attr_name)(X)
            else:
                getattr(model, attr_name)(X, y)

    @pytest.mark.parametrize(
        "regularizer, regularizer_strength, solver_name, solver_kwargs",
        [
            (
                nmo.regularizer.UnRegularized(),
                None,
                "LBFGS",
                {"stepsize": 0.1, "tol": 10**-14},
            ),
            (
                nmo.regularizer.UnRegularized(),
                None,
                "GradientDescent",
                {"tol": 10**-14},
            ),
            (
                nmo.regularizer.Ridge(),
                1.0,
                "LBFGS",
                {"tol": 10**-14},
            ),
            (nmo.regularizer.Ridge(), 1.0, "LBFGS", {"stepsize": 0.1, "tol": 10**-14}),
            (
                nmo.regularizer.Lasso(),
                0.001,
                "ProximalGradient",
                {"tol": 10**-14},
            ),
        ],
    )
    @pytest.mark.parametrize(
        "mask",
        [
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                ]
            ),
            {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
        ],
    )
    def test_masked_fit_vs_loop(
        self,
        regularizer,
        regularizer_strength,
        solver_name,
        solver_kwargs,
        mask,
        poisson_population_GLM_model,
        poisson_population_GLM_model_pytree,
    ):
        jax.config.update("jax_enable_x64", True)
        if isinstance(mask, dict):
            X, y, _, true_params, firing_rate = poisson_population_GLM_model_pytree

            def map_neu(k, coef_):
                key_ind = {"input_1": [0, 1, 2], "input_2": [3, 4]}
                ind_array = np.zeros((0,), dtype=int)
                coef_stack = np.zeros((0,), dtype=int)
                for key, msk in mask.items():
                    if msk[k]:
                        ind_array = np.hstack((ind_array, key_ind[key]))
                        coef_stack = np.hstack((coef_stack, coef_[key]))
                return ind_array, coef_stack

        else:
            X, y, _, true_params, firing_rate = poisson_population_GLM_model

            def map_neu(k, coef_):
                ind_array = np.where(mask[:, k])[0]
                coef_stack = coef_
                return ind_array, coef_stack

        mask_bool = jax.tree_util.tree_map(lambda x: np.asarray(x.T, dtype=bool), mask)
        # fit pop glm
        kwargs = dict(
            feature_mask=mask,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        model = nmo.glm.PopulationGLM(**kwargs)
        model.fit(X, y)
        coef_vectorized = np.vstack(jax.tree_util.tree_leaves(model.coef_))

        coef_loop = np.zeros((5, 3))
        intercept_loop = np.zeros((3,))
        # loop over neuron
        for k in range(y.shape[1]):
            model_single_neu = nmo.glm.GLM(
                regularizer=regularizer,
                regularizer_strength=regularizer_strength,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
            )
            if isinstance(mask_bool, dict):
                X_neu = {}
                for key, xx in X.items():
                    if mask_bool[key][k]:
                        X_neu[key] = X[key]
                X_neu = FeaturePytree(**X_neu)
            else:
                X_neu = X[:, mask_bool[k]]

            model_single_neu.fit(X_neu, y[:, k])
            idx, coef = map_neu(k, model_single_neu.coef_)
            coef_loop[idx, k] = coef
            intercept_loop[k] = np.array(model_single_neu.intercept_)[0]
        print(f"\nMAX ERR: {np.abs(coef_loop - coef_vectorized).max()}")
        assert np.allclose(coef_loop, coef_vectorized, atol=10**-5, rtol=0)
