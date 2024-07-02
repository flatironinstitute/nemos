"""
# Pipelining and cross-validation with scikit-learn

In this demo we will show how to combine basis transformations and GLMs using scikit-learn's pipelines, and demonstrate how cross-validation can be used to fine-tune parameters of each step of the pipeline.
"""

# %%
# ## Let's start by some imports and creating toy data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
from _plotting_helpers import annotate_heatmap, despine, heatmap
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# %%
import nemos as nmo

# %%

# predictors, shape (n_samples, n_features)
X = np.random.uniform(low=0, high=1, size=(1000, 1))
# true coefficients, shape (n_features)
coef = np.array([5])
# observed counts, shape (n_samples, )
y = np.random.poisson(np.exp(np.matmul(X, coef))).astype(float)

# %%
fig, ax = plt.subplots()
ax.scatter(X.flatten(), y, alpha=0.2)
ax.set_xlabel("input")
ax.set_ylabel("spike count")
despine(ax=ax)

# %%
# ## Combining basis transformations and GLM in a pipeline

# %%
# ### `TransformerBasis` class
# The `TransformerBasis` wrapper class provides an API consistent with [scikit-learn's data transforms](https://scikit-learn.org/stable/data_transforms.html), allowing its use in pipelines as a transformation step.
#
# Instantiating a `TransformerBasis` can be done using the constructor directly or with `Basis.to_transformer()` and provides convenient access to the underlying `Basis` object's attributes:

# %%
bas = nmo.basis.RaisedCosineBasisLinear(5)
trans_bas_a = nmo.basis.TransformerBasis(bas)
trans_bas_b = bas.to_transformer()

print(bas.n_basis_funcs, trans_bas_a.n_basis_funcs, trans_bas_b.n_basis_funcs)

# %%
# ### Creating and fitting a pipeline
# Use this wrapper class to create a pipeline that first transforms the data, then fits a GLM on the transformed data:

# %%
pipeline = Pipeline(
    [
        (
            "transformerbasis",
            nmo.basis.TransformerBasis(nmo.basis.RaisedCosineBasisLinear(20)),
        ),
        (
            "glm",
            nmo.glm.GLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=1.0)),
        ),
    ]
)

pipeline.fit(X, y)

# %%
# ### Visualize the fit:

# %%
fig, ax = plt.subplots()

ax.scatter(X.flatten(), y, alpha=0.2, label="generated spike counts")
ax.set_xlabel("input")
ax.set_ylabel("spike count")

x = np.sort(X.flatten())
ax.plot(
    x,
    pipeline.predict(np.expand_dims(x, 1)),
    label="predicted rate",
    color="tab:orange",
)

ax.legend()
despine(ax=ax)

# %%
# There is some room for improvement, so in the next section we'll use cross-validation to tune the parameters of our pipeline.

# %%
# ## Hyperparameter-tuning with scikit-learn's gridsearch

# %%
# !!! warning
#     Please keep in mind that while `GLM.score` supports different ways of evaluating goodness-of-fit through the `score_type` argument, `pipeline.score(X, y, score_type="...")` does not propagate this, and uses the default value of `log-likelihood`.
#
#     To evaluate a pipeline, please create a custom scorer (e.g. `pseudo_r2` below) and call `my_custom_scorer(pipeline, X, y)`.

# %%
# ### Define parameter grid
# Let's define candidate values for the parameters of each step of the pipeline we want to cross-validate. In this case the number of basis functions in the transformation step and the ridge regularization's strength in the GLM fit:

# %%
param_grid = dict(
    glm__regularizer__regularizer_strength=(0.1, 0.01, 0.001, 1e-6),
    transformerbasis__n_basis_funcs=(3, 5, 10, 20, 100),
)

# %%
# ### Create custom scorer
# Create a custom scorer to evaluate models using the pseudo-R2 score instead of the log-likelihood which is the default in `GLM.score`:

# %%
from sklearn.metrics import make_scorer

# NOTE: the order of the arguments is reversed
pseudo_r2 = make_scorer(
    lambda y_true, y_pred: nmo.observation_models.PoissonObservations().pseudo_r2(
        y_pred, y_true
    )
)

# %%
# ### Run the grid search:

# %%
gridsearch = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring=pseudo_r2)

gridsearch.fit(X, y)

# %%
# ### Visualize the scores
# Let's take a look at the scores to see how the different parameter values influence the test score:

# %%
cvdf = pd.DataFrame(gridsearch.cv_results_)


fig, ax = plt.subplots()

im = heatmap(
    cvdf.pivot(
        index="param_transformerbasis__n_basis_funcs",
        columns="param_glm__regularizer__regularizer_strength",
        values="mean_test_score",
    ),
    param_grid["transformerbasis__n_basis_funcs"],
    param_grid["glm__regularizer__regularizer_strength"][
        ::-1
    ],  # note the reverse order
    ax=ax,
)
texts = annotate_heatmap(im, valfmt="{x:.3f}")
ax.set_xlabel("ridge regularization strength")
ax.set_ylabel("number of basis functions")

fig.tight_layout()

# %%
# ### Visualize the improved predictions
# Finally, visualize the predicted firing rates using the best model found by our grid-search, which gives a better fit than the randomly chosen parameter values we tried in the beginning:

# %%
fig, ax = plt.subplots()

ax.scatter(X.flatten(), y, alpha=0.2, label="generated spike counts")
ax.set_xlabel("input")
ax.set_ylabel("spike count")

x = np.sort(X.flatten())
ax.plot(
    x,
    gridsearch.best_estimator_.predict(np.expand_dims(x, 1)),
    label="predicted rate",
    color="tab:orange",
)

ax.legend()
despine(ax=ax)

# %%
