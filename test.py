import jax
import numpy as np
import jax.numpy as jnp
import nemos as nmo
from nemos.glm.params import GLMParams


def regr_data():
    np.random.seed(123)
    # define inputs and coeff
    n_samples, n_features = 50, 3
    X = np.random.normal(size=(n_samples, n_features))
    coef = np.random.normal(size=(n_features))
    # set y according to lin reg eqn
    y = X.dot(coef) + 0.1 * np.random.normal(size=(n_samples,))
    return X, y, coef


def linear_regression(regr_data):
    X, y, coef = regr_data
    # solve least-squares
    ols, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)

    # set the loss
    def loss(params, X, y):
        return jnp.power(y - jnp.dot(X, params), 2).mean()

    return X, y, coef, ols, loss


def ridge_regression(regr_data):
    X, y, coef = regr_data

    # solve least-squares
    yagu = np.hstack((y, np.zeros_like(coef)))
    Xagu = np.vstack((X, np.sqrt(0.5) * np.eye(coef.shape[0])))
    ridge, _, _, _ = np.linalg.lstsq(Xagu, yagu, rcond=-1)

    # set the loss
    def loss(params, XX, yy):
        return (
            jnp.power(yy - jnp.dot(XX, params), 2).sum()
            + 0.5 * jnp.power(params, 2).sum()
        )

    return X, y, coef, ridge, loss


def linear_regression_tree(linear_regression):
    X, y, coef, ols, loss = linear_regression
    X_tree = dict(input_1=X[..., :2], input_2=X[..., 2:])
    coef_tree = dict(input_1=coef[:2], input_2=coef[2:])
    ols_tree = dict(input_1=ols[:2], input_2=ols[2:])

    nmo.tree_utils.pytree_map_and_reduce(jnp.dot, sum, X_tree, coef_tree)

    def loss_tree(params, XX, yy):
        pred = nmo.tree_utils.pytree_map_and_reduce(jnp.dot, sum, XX, params)
        return jnp.power(yy - pred, 2).sum()

    return X_tree, y, coef_tree, ols_tree, loss_tree


def ridge_regression_tree(ridge_regression):
    X, y, coef, ridge, loss = ridge_regression
    X_tree = dict(input_1=X[..., :2], input_2=X[..., 2:])
    coef_tree = dict(input_1=coef[:2], input_2=coef[2:])
    ridge_tree = dict(input_1=ridge[:2], input_2=ridge[2:])

    def loss_tree(params, XX, yy):
        pred = nmo.tree_utils.pytree_map_and_reduce(jnp.dot, sum, XX, params)
        norm = (
            0.5
            * nmo.tree_utils.pytree_map_and_reduce(
                lambda x: jnp.power(x, 2).sum(), sum, params
            ).sum()
        )
        return jnp.power(yy - pred, 2).sum() + norm

    return X_tree, y, coef_tree, ridge_tree, loss_tree


def population_poissonGLM_model_instantiation():
    """Set up a population Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(500, 5))
    b_true = -2 * np.ones((3,))
    w_true = np.random.normal(size=(5, 3))
    observation_model = nmo.observation_models.PoissonObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(
        observation_model=observation_model, regularizer=regularizer
    )
    rate = jnp.exp(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, GLMParams(w_true, b_true), rate


def population_poissonGLM_model_instantiation_pytree(
    population_poissonGLM_model_instantiation,
):
    """Set up a population Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = population_poissonGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model, regularizer="Ridge"
    )
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


X_tree, y, model, true_params_tree, rate = (
    population_poissonGLM_model_instantiation_pytree(
        population_poissonGLM_model_instantiation()
    )
)


print(true_params_tree)
hess = model._get_hess_fn(true_params_tree)
# print(grad(true_params_tree, X_tree, y))
print(hess(true_params_tree, X_tree, y))
model.fit(X_tree, y)
