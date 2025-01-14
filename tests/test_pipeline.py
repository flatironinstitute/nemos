import joblib
import numpy as np
import pynapple as nap
import pytest
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV

from nemos import basis
from nemos.basis._transformer_basis import TransformerBasis


@pytest.mark.parametrize(
    "bas",
    [
        basis.MSplineEval(5),
        basis.BSplineEval(5),
        basis.CyclicBSplineEval(5),
        basis.OrthExponentialEval(5, decay_rates=np.arange(1, 6)),
        basis.RaisedCosineLinearEval(5),
    ],
)
def test_sklearn_transformer_pipeline(bas, poissonGLM_model_instantiation):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = TransformerBasis(bas).set_input_shape(*([1] * bas._n_input_dimensionality))
    pipe = pipeline.Pipeline([("eval", bas), ("fit", model)])

    pipe.fit(X[:, : bas.basis._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas",
    [
        basis.MSplineEval(5),
        basis.BSplineEval(5),
        basis.CyclicBSplineEval(5),
        basis.RaisedCosineLinearEval(5),
        basis.RaisedCosineLogEval(5),
    ],
)
def test_sklearn_transformer_pipeline_cv(bas, poissonGLM_model_instantiation):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = TransformerBasis(bas).set_input_shape(*([1] * bas._n_input_dimensionality))
    pipe = pipeline.Pipeline([("basis", bas), ("fit", model)])
    param_grid = dict(basis__n_basis_funcs=(4, 5, 10))
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, error_score="raise")
    gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


def test_sklearn_cv_clone(poisson_population_GLM_model):
    X, y, model, _, _ = poisson_population_GLM_model
    bas = basis.CyclicBSplineEval(5)
    bas = TransformerBasis(bas).set_input_shape(*([1] * bas._n_input_dimensionality))
    pipe = pipeline.Pipeline([("basis", bas), ("fit", model)])
    # if feature_mask isn't dropped by the cloning done by gridsearch cv, this will
    # error, because the shape of feature_mask doesn't match the shape of the output of
    # transformer basis with different number of basis funcs
    pipe.fit(X[:, : bas._n_input_dimensionality] ** 2, y)
    param_grid = dict(basis__n_basis_funcs=(4, 8))
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, error_score="raise")
    gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas",
    [
        basis.MSplineEval(5),
        basis.BSplineEval(5),
        basis.CyclicBSplineEval(5),
        basis.RaisedCosineLinearEval(5),
        basis.RaisedCosineLogEval(5),
    ],
)
def test_sklearn_transformer_pipeline_cv_multiprocess(
    bas, poissonGLM_model_instantiation
):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = TransformerBasis(bas).set_input_shape(*([1] * bas._n_input_dimensionality))
    pipe = pipeline.Pipeline([("basis", bas), ("fit", model)])
    param_grid = dict(basis__n_basis_funcs=(4, 5, 10))
    gridsearch = GridSearchCV(
        pipe, param_grid=param_grid, cv=3, n_jobs=3, error_score="raise"
    )
    # use threading instead of fork (this avoids conflicts with jax)
    with joblib.parallel_backend("threading"):
        gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas_cls",
    [
        basis.MSplineEval,
        basis.MSplineEval,
        basis.CyclicBSplineEval,
        basis.RaisedCosineLinearEval,
        basis.RaisedCosineLogEval,
    ],
)
def test_sklearn_transformer_pipeline_cv_directly_over_basis(
    bas_cls, poissonGLM_model_instantiation
):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = TransformerBasis(bas_cls(5))
    bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    pipe = pipeline.Pipeline([("transformerbasis", bas), ("fit", model)])
    param_grid = dict(
        transformerbasis__basis=(
            bas_cls(5).set_input_shape(*([1] * bas._n_input_dimensionality)),
            bas_cls(10).set_input_shape(*([1] * bas._n_input_dimensionality)),
            bas_cls(20).set_input_shape(*([1] * bas._n_input_dimensionality)),
        )
    )
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, error_score="raise")
    gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas_cls",
    [
        basis.MSplineEval,
        basis.MSplineEval,
        basis.CyclicBSplineEval,
        basis.RaisedCosineLinearEval,
        basis.RaisedCosineLogEval,
    ],
)
def test_sklearn_transformer_pipeline_cv_illegal_combination(
    bas_cls, poissonGLM_model_instantiation
):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = TransformerBasis(bas_cls(5))
    bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    pipe = pipeline.Pipeline([("transformerbasis", bas), ("fit", model)])
    param_grid = dict(
        transformerbasis__basis=(bas_cls(5), bas_cls(10), bas_cls(20)),
        transformerbasis__n_basis_funcs=(4, 5, 10),
    )
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, error_score="raise")
    with pytest.raises(
        ValueError,
        match="Set either new basis object or parameters for existing basis, not both.",
    ):
        gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas, expected_nans",
    [
        (basis.MSplineEval(5), 0),
        (basis.BSplineEval(5), 0),
        (basis.CyclicBSplineEval(5), 0),
        (basis.OrthExponentialEval(5, decay_rates=np.arange(1, 6)), 0),
        (basis.RaisedCosineLinearEval(5), 0),
        (basis.RaisedCosineLogEval(5), 0),
        (basis.RaisedCosineLogEval(5) + basis.MSplineEval(5), 0),
        (basis.MSplineConv(5, window_size=3), 6),
        (basis.BSplineConv(5, window_size=3), 6),
        (
            basis.CyclicBSplineConv(
                5, window_size=3, conv_kwargs=dict(predictor_causality="acausal")
            ),
            4,
        ),
        (
            basis.OrthExponentialConv(
                5, decay_rates=np.linspace(0.1, 1, 5), window_size=7
            ),
            14,
        ),
        (basis.RaisedCosineLinearConv(5, window_size=3), 6),
        (basis.RaisedCosineLogConv(5, window_size=3), 6),
        (
            basis.RaisedCosineLogConv(5, window_size=3) + basis.MSplineEval(5),
            6,
        ),
        (
            basis.RaisedCosineLogConv(5, window_size=3) * basis.MSplineEval(5),
            6,
        ),
    ],
)
def test_sklearn_transformer_pipeline_pynapple(
    bas, poissonGLM_model_instantiation, expected_nans
):
    X, y, model, _, _ = poissonGLM_model_instantiation

    # transform input to pynapple
    ep = nap.IntervalSet(start=[0, 20.5], end=[20, X.shape[0]])
    X_nap = nap.TsdFrame(t=np.arange(X.shape[0]), d=X, time_support=ep)
    y_nap = nap.Tsd(t=np.arange(X.shape[0]), d=y, time_support=ep)
    bas = TransformerBasis(bas).set_input_shape(*([1] * bas._n_input_dimensionality))

    # fit a pipeline & predict from pynapple
    pipe = pipeline.Pipeline([("eval", bas), ("fit", model)])
    pipe.fit(X_nap[:, : bas.basis._n_input_dimensionality] ** 2, y_nap)

    # get rate
    rate = pipe.predict(X_nap[:, : bas.basis._n_input_dimensionality] ** 2)
    # check rate is Tsd with same time info
    assert isinstance(rate, nap.Tsd)
    assert np.all(rate.t == X_nap.t)
    assert np.all(rate.time_support == X_nap.time_support)
    assert np.sum(np.isnan(rate.d)) == expected_nans
