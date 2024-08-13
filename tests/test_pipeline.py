import numpy as np
import pynapple as nap
import pytest
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV

from nemos import basis


@pytest.mark.parametrize(
    "bas",
    [
        basis.MSplineBasis(5),
        basis.BSplineBasis(5),
        basis.CyclicBSplineBasis(5),
        basis.OrthExponentialBasis(5, decay_rates=np.arange(1, 6)),
        basis.RaisedCosineBasisLinear(5),
    ],
)
def test_sklearn_transformer_pipeline(bas, poissonGLM_model_instantiation):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = basis.TransformerBasis(bas)
    pipe = pipeline.Pipeline([("eval", bas), ("fit", model)])

    pipe.fit(X[:, : bas._basis._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas",
    [
        basis.MSplineBasis(5),
        basis.BSplineBasis(5),
        basis.CyclicBSplineBasis(5),
        basis.RaisedCosineBasisLinear(5),
        basis.RaisedCosineBasisLog(5),
    ],
)
def test_sklearn_transformer_pipeline_cv(bas, poissonGLM_model_instantiation):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = basis.TransformerBasis(bas)
    pipe = pipeline.Pipeline([("basis", bas), ("fit", model)])
    param_grid = dict(basis__n_basis_funcs=(3, 5, 10))
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas",
    [
        basis.MSplineBasis(5),
        basis.BSplineBasis(5),
        basis.CyclicBSplineBasis(5),
        basis.RaisedCosineBasisLinear(5),
        basis.RaisedCosineBasisLog(5),
    ],
)
def test_sklearn_transformer_pipeline_cv_multiprocess(
    bas, poissonGLM_model_instantiation
):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = basis.TransformerBasis(bas)
    pipe = pipeline.Pipeline([("basis", bas), ("fit", model)])
    param_grid = dict(basis__n_basis_funcs=(3, 5, 10))
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=3)
    gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas_cls",
    [
        basis.MSplineBasis,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_sklearn_transformer_pipeline_cv_directly_over_basis(
    bas_cls, poissonGLM_model_instantiation
):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = basis.TransformerBasis(bas_cls(5))
    pipe = pipeline.Pipeline([("transformerbasis", bas), ("fit", model)])
    param_grid = dict(transformerbasis___basis=(bas_cls(5), bas_cls(10), bas_cls(20)))
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas_cls",
    [
        basis.MSplineBasis,
        basis.BSplineBasis,
        basis.CyclicBSplineBasis,
        basis.RaisedCosineBasisLinear,
        basis.RaisedCosineBasisLog,
    ],
)
def test_sklearn_transformer_pipeline_cv_illegal_combination(
    bas_cls, poissonGLM_model_instantiation
):
    X, y, model, _, _ = poissonGLM_model_instantiation
    bas = basis.TransformerBasis(bas_cls(5))
    pipe = pipeline.Pipeline([("transformerbasis", bas), ("fit", model)])
    param_grid = dict(
        transformerbasis___basis=(bas_cls(5), bas_cls(10), bas_cls(20)),
        transformerbasis__n_basis_funcs=(3, 5, 10),
    )
    gridsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    with pytest.raises(
        ValueError, match="Set either new _basis object or parameters for existing _basis, not both."
    ):
        gridsearch.fit(X[:, : bas._n_input_dimensionality] ** 2, y)


@pytest.mark.parametrize(
    "bas, expected_nans",
    [
        (basis.MSplineBasis(5), 0),
        (basis.BSplineBasis(5), 0),
        (basis.CyclicBSplineBasis(5), 0),
        (basis.OrthExponentialBasis(5, decay_rates=np.arange(1, 6)), 0),
        (basis.RaisedCosineBasisLinear(5), 0),
        (basis.RaisedCosineBasisLog(5), 0),
        (basis.RaisedCosineBasisLog(5) + basis.MSplineBasis(5), 0),
        (basis.MSplineBasis(5, mode="conv", window_size=3), 6),
        (basis.BSplineBasis(5, mode="conv", window_size=3), 6),
        (
            basis.CyclicBSplineBasis(
                5, mode="conv", window_size=3, predictor_causality="acausal"
            ),
            4,
        ),
        (
            basis.OrthExponentialBasis(
                5, decay_rates=np.linspace(0.1, 1, 5), mode="conv", window_size=7
            ),
            14,
        ),
        (basis.RaisedCosineBasisLinear(5, mode="conv", window_size=3), 6),
        (basis.RaisedCosineBasisLog(5, mode="conv", window_size=3), 6),
        (
            basis.RaisedCosineBasisLog(5, mode="conv", window_size=3)
            + basis.MSplineBasis(5),
            6,
        ),
        (
            basis.RaisedCosineBasisLog(5, mode="conv", window_size=3)
            * basis.MSplineBasis(5),
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
    bas = basis.TransformerBasis(bas)
    # fit a pipeline & predict from pynapple
    pipe = pipeline.Pipeline([("eval", bas), ("fit", model)])
    pipe.fit(X_nap[:, : bas._basis._n_input_dimensionality] ** 2, y_nap)

    # get rate
    rate = pipe.predict(X_nap[:, : bas._basis._n_input_dimensionality] ** 2)
    # check rate is Tsd with same time info
    assert isinstance(rate, nap.Tsd)
    assert np.all(rate.t == X_nap.t)
    assert np.all(rate.time_support == X_nap.time_support)
    assert np.sum(np.isnan(rate.d)) == expected_nans
